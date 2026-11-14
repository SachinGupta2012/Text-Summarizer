"""Apply the Phase 3 outlier policy and freeze training-ready artifacts."""

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from summarizer_ml.preprocessing.finalization import FinalizationReport
from summarizer_ml.preprocessing.outliers import (
    OutlierAuditReport,
    OutlierRecord,
)
from summarizer_ml.preprocessing.tokenization import (
    TokenAnalysisReport,
    TokenLengthRecord,
    summarize_token_length_records,
)

ProgressCallback = Callable[[str, int], None]


class PolicyAction(StrEnum):
    """Final Phase 3 disposition for a canonical record."""

    KEEP = "keep"
    LONG_TARGET = "long_target"
    QUARANTINE = "quarantine"
    REQUIRES_CHUNKING = "requires_chunking"


class PolicyReason(StrEnum):
    """Deterministic reason contributing to a non-keep disposition."""

    INPUT_BELOW_MINIMUM = "input_below_minimum"
    INPUT_EXCEEDS_DIRECT_LIMIT = "input_exceeds_direct_limit"
    SUMMARY_BELOW_MINIMUM = "summary_below_minimum"
    SUMMARY_EXCEEDS_DIRECT_LIMIT = "summary_exceeds_direct_limit"
    SUMMARY_LONGER_THAN_INPUT = "summary_longer_than_input"


class Phase3PolicyConfig(BaseModel):
    """Thresholds separating clean, suspicious, and model-routed records."""

    model_config = ConfigDict(frozen=True)

    direct_input_token_limit: int = Field(default=32_768, ge=1)
    direct_summary_token_limit: int = Field(default=2_048, ge=1)
    minimum_input_tokens: int = Field(default=64, ge=1)
    minimum_summary_tokens: int = Field(default=10, ge=1)
    quarantine_summary_longer_than_input: bool = True

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        """Require non-overlapping minimum and oversized ranges."""

        if self.minimum_input_tokens > self.direct_input_token_limit:
            raise ValueError("Minimum input tokens cannot exceed the direct input limit.")
        if self.minimum_summary_tokens > self.direct_summary_token_limit:
            raise ValueError("Minimum summary tokens cannot exceed the direct summary limit.")
        return self


class PolicyDecision(BaseModel):
    """Metadata-only routing decision for one non-keep record."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    dataset: str
    split: str
    action: PolicyAction
    reasons: tuple[PolicyReason, ...]
    input_tokens: int
    summary_tokens: int
    source_characters: int
    summary_characters: int


class ArtifactDigest(BaseModel):
    """Stable identity for one Phase 3 input or output artifact."""

    model_config = ConfigDict(frozen=True)

    path: str
    bytes: int
    sha256: str


class Phase3IntegrityGates(BaseModel):
    """Machine-readable post-policy integrity results."""

    model_config = ConfigDict(frozen=True)

    canonical_files_unchanged: bool
    canonical_record_count_matches: bool
    canonical_token_count_matches: bool
    decision_coverage_complete: bool
    training_record_token_ids_match: bool
    split_assignments_preserved: bool
    routed_records_absent_from_training: bool
    duplicate_record_ids: int
    cross_split_leakage_findings: int
    missing_token_records: int
    unexpected_token_records: int
    split_mismatches: int


class Phase3Manifest(BaseModel):
    """Frozen Phase 3 dataset contract consumed by model-data construction."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    policy: Phase3PolicyConfig
    tokenizer_id: str
    canonical_records: int
    canonical_token_records: int
    training_ready_records: int
    training_ready_token_records: int
    routed_records: int
    by_action: dict[str, int]
    by_action_and_dataset: dict[str, dict[str, int]]
    by_action_and_split: dict[str, dict[str, int]]
    training_ready_by_dataset: dict[str, int]
    training_ready_by_split: dict[str, int]
    integrity: Phase3IntegrityGates
    source_artifacts: tuple[ArtifactDigest, ...]
    output_artifacts: tuple[ArtifactDigest, ...]


class Phase3PolicyResult(BaseModel):
    """Location and manifest returned by the completed policy pipeline."""

    model_config = ConfigDict(frozen=True)

    output_directory: Path
    manifest: Phase3Manifest
    token_analysis: TokenAnalysisReport


def classify_outlier(
    record: OutlierRecord,
    config: Phase3PolicyConfig,
) -> tuple[PolicyAction, tuple[PolicyReason, ...]]:
    """Classify an audited record with conservative, deterministic precedence."""

    reasons: set[PolicyReason] = set()
    if record.input_tokens < config.minimum_input_tokens:
        reasons.add(PolicyReason.INPUT_BELOW_MINIMUM)
    if record.summary_tokens < config.minimum_summary_tokens:
        reasons.add(PolicyReason.SUMMARY_BELOW_MINIMUM)
    if record.input_tokens > config.direct_input_token_limit:
        reasons.add(PolicyReason.INPUT_EXCEEDS_DIRECT_LIMIT)
    if record.summary_tokens > config.direct_summary_token_limit:
        reasons.add(PolicyReason.SUMMARY_EXCEEDS_DIRECT_LIMIT)
    if config.quarantine_summary_longer_than_input and record.summary_tokens > record.input_tokens:
        reasons.add(PolicyReason.SUMMARY_LONGER_THAN_INPUT)

    ordered_reasons = tuple(sorted(reasons, key=str))
    quarantine_reasons = {
        PolicyReason.INPUT_BELOW_MINIMUM,
        PolicyReason.SUMMARY_BELOW_MINIMUM,
        PolicyReason.SUMMARY_LONGER_THAN_INPUT,
    }
    if reasons.intersection(quarantine_reasons):
        return PolicyAction.QUARANTINE, ordered_reasons
    if PolicyReason.SUMMARY_EXCEEDS_DIRECT_LIMIT in reasons:
        return PolicyAction.LONG_TARGET, ordered_reasons
    if PolicyReason.INPUT_EXCEEDS_DIRECT_LIMIT in reasons:
        return PolicyAction.REQUIRES_CHUNKING, ordered_reasons
    return PolicyAction.KEEP, ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_digest(path: Path, root: Path) -> ArtifactDigest:
    return ArtifactDigest(
        path=path.relative_to(root).as_posix(),
        bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _source_paths(final_directory: Path) -> tuple[Path, ...]:
    fixed = (
        final_directory / "finalization-report.json",
        final_directory / "analysis" / "token-analysis-report.json",
        final_directory / "analysis" / "token-lengths.jsonl",
        final_directory / "audit" / "outliers" / "outlier-audit-report.json",
        final_directory / "audit" / "outliers" / "outlier-records.jsonl",
    )
    paths = (*fixed, *sorted((final_directory / "processed").glob("*/*.jsonl")))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Phase 3 artifacts are missing: {missing}")
    return tuple(paths)


def _digests(paths: tuple[Path, ...], root: Path) -> tuple[ArtifactDigest, ...]:
    return tuple(_artifact_digest(path, root) for path in paths)


def _validate_policy_matches_audit(
    policy: Phase3PolicyConfig,
    audit: OutlierAuditReport,
) -> None:
    audit_config = audit.config
    requested = (
        policy.direct_input_token_limit,
        policy.direct_summary_token_limit,
        policy.minimum_input_tokens,
        policy.minimum_summary_tokens,
    )
    audited = (
        audit_config.extreme_input_tokens,
        audit_config.extreme_summary_tokens,
        audit_config.very_short_input_tokens,
        audit_config.very_short_summary_tokens,
    )
    if requested != audited:
        raise ValueError("Policy thresholds must match the completed outlier-audit thresholds.")


def _load_policy_decisions(
    path: Path,
    audit: OutlierAuditReport,
    config: Phase3PolicyConfig,
) -> dict[str, PolicyDecision]:
    selected = 0
    seen_ids: set[str] = set()
    decisions: dict[str, PolicyDecision] = {}
    with path.open(encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
            if not line.strip():
                continue
            try:
                record = OutlierRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid outlier record at {path}:{line_number}") from error
            selected += 1
            if record.record_id in seen_ids:
                raise ValueError("Outlier audit contains duplicate record IDs.")
            seen_ids.add(record.record_id)
            action, reasons = classify_outlier(record, config)
            if action is PolicyAction.KEEP:
                continue
            decisions[record.record_id] = PolicyDecision(
                record_id=record.record_id,
                dataset=record.dataset,
                split=record.split,
                action=action,
                reasons=reasons,
                input_tokens=record.input_tokens,
                summary_tokens=record.summary_tokens,
                source_characters=record.source_characters,
                summary_characters=record.summary_characters,
            )
    if selected != audit.selected_outlier_records:
        raise RuntimeError("Outlier report and outlier-record count do not match.")
    return decisions


def _create_integrity_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE canonical_record (
            record_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            split TEXT NOT NULL,
            action TEXT NOT NULL
        );
        CREATE TABLE token_record (
            record_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            split TEXT NOT NULL,
            action TEXT NOT NULL
        );
        CREATE TABLE source_occurrence (
            source_sha256 TEXT NOT NULL,
            split TEXT NOT NULL
        );
        CREATE INDEX source_occurrence_hash
            ON source_occurrence (source_sha256, split);
        CREATE TABLE group_occurrence (
            group_id TEXT NOT NULL,
            split TEXT NOT NULL
        );
        CREATE INDEX group_occurrence_id
            ON group_occurrence (group_id, split);
        """
    )


def _count_cross_split_leakage(connection: sqlite3.Connection) -> int:
    source_findings = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT source_sha256
            FROM source_occurrence
            GROUP BY source_sha256
            HAVING COUNT(DISTINCT split) > 1
        )
        """
    ).fetchone()
    group_findings = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT group_id
            FROM group_occurrence
            GROUP BY group_id
            HAVING COUNT(DISTINCT split) > 1
        )
        """
    ).fetchone()
    return int(source_findings[0]) + int(group_findings[0])


def _report_progress(
    callback: ProgressCallback | None,
    stage: str,
    count: int,
    progress_every: int,
    *,
    final: bool = False,
) -> None:
    if callback is not None and (final or count % progress_every == 0):
        callback(stage, count)


def _flush_record_index(
    connection: sqlite3.Connection,
    canonical_rows: list[tuple[str, str, str, str]],
    source_rows: list[tuple[str, str]],
    group_rows: list[tuple[str, str]],
) -> None:
    try:
        connection.executemany("INSERT INTO canonical_record VALUES (?, ?, ?, ?)", canonical_rows)
    except sqlite3.IntegrityError as error:
        raise RuntimeError("Canonical partitions contain duplicate record IDs.") from error
    connection.executemany("INSERT INTO source_occurrence VALUES (?, ?)", source_rows)
    connection.executemany("INSERT INTO group_occurrence VALUES (?, ?)", group_rows)
    canonical_rows.clear()
    source_rows.clear()
    group_rows.clear()


def _materialize_records(
    processed_root: Path,
    staging: Path,
    decisions: Mapping[str, PolicyDecision],
    connection: sqlite3.Connection,
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> tuple[
    int,
    int,
    Counter[str],
    dict[str, Counter[str]],
    dict[str, Counter[str]],
    Counter[str],
    Counter[str],
    set[str],
]:
    route_root = staging / "routing"
    route_root.mkdir(parents=True)
    route_files = {
        action: (route_root / f"{action.value}.jsonl").open("x", encoding="utf-8", newline="\n")
        for action in (
            PolicyAction.QUARANTINE,
            PolicyAction.REQUIRES_CHUNKING,
            PolicyAction.LONG_TARGET,
        )
    }
    canonical_count = 0
    training_count = 0
    by_action: Counter[str] = Counter()
    by_action_and_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    by_action_and_split: dict[str, Counter[str]] = defaultdict(Counter)
    training_by_dataset: Counter[str] = Counter()
    training_by_split: Counter[str] = Counter()
    seen_routed_ids: set[str] = set()
    canonical_rows: list[tuple[str, str, str, str]] = []
    source_rows: list[tuple[str, str]] = []
    group_rows: list[tuple[str, str]] = []
    try:
        for source_path in sorted(processed_root.glob("*/*.jsonl")):
            relative_path = source_path.relative_to(processed_root)
            destination_path = staging / "processed" / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                source_path.open(encoding="utf-8") as source_file,
                destination_path.open("x", encoding="utf-8", newline="\n") as destination_file,
            ):
                for line in source_file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    canonical_count += 1
                    record_id = str(record["record_id"])
                    dataset = str(record["dataset"])
                    split = str(record["split"])
                    decision = decisions.get(record_id)
                    action = decision.action if decision is not None else PolicyAction.KEEP
                    if decision is not None:
                        if decision.dataset != dataset or decision.split != split:
                            raise RuntimeError(
                                "Policy decision metadata does not match its record."
                            )
                        route_file = route_files[action]
                        route_file.write(
                            json.dumps(
                                decision.model_dump(mode="json"),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )
                        route_file.write("\n")
                        seen_routed_ids.add(record_id)
                    else:
                        destination_file.write(line if line.endswith("\n") else f"{line}\n")
                        training_count += 1
                        training_by_dataset[dataset] += 1
                        training_by_split[split] += 1
                        source_rows.append((str(record["source_sha256"]), split))
                        group_rows.append((str(record["group_id"]), split))

                    by_action[action.value] += 1
                    by_action_and_dataset[action.value][dataset] += 1
                    by_action_and_split[action.value][split] += 1
                    canonical_rows.append((record_id, dataset, split, action.value))
                    if len(canonical_rows) >= 1_000:
                        _flush_record_index(
                            connection,
                            canonical_rows,
                            source_rows,
                            group_rows,
                        )
                    _report_progress(
                        progress_callback,
                        "record_policy",
                        canonical_count,
                        progress_every,
                    )
        if canonical_rows:
            _flush_record_index(connection, canonical_rows, source_rows, group_rows)
        connection.commit()
    finally:
        for route_file in route_files.values():
            route_file.close()

    if canonical_count % progress_every:
        _report_progress(
            progress_callback,
            "record_policy",
            canonical_count,
            progress_every,
            final=True,
        )
    return (
        canonical_count,
        training_count,
        by_action,
        by_action_and_dataset,
        by_action_and_split,
        training_by_dataset,
        training_by_split,
        seen_routed_ids,
    )


def _filter_token_lengths(
    source_path: Path,
    destination_root: Path,
    decisions: Mapping[str, PolicyDecision],
    connection: sqlite3.Connection,
    tokenizer_id: str,
    thresholds: tuple[int, ...],
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> tuple[int, int, TokenAnalysisReport]:
    destination_root.mkdir(parents=True)
    destination_path = destination_root / "token-lengths.jsonl"
    canonical_token_count = 0
    training_token_count = 0
    token_rows: list[tuple[str, str, str, str]] = []

    def retained_measurements() -> Iterator[TokenLengthRecord]:
        nonlocal canonical_token_count, training_token_count
        with (
            source_path.open(encoding="utf-8") as source_file,
            destination_path.open("x", encoding="utf-8", newline="\n") as destination_file,
        ):
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    measurement = TokenLengthRecord.model_validate_json(line)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid token record at {source_path}:{line_number}"
                    ) from error
                canonical_token_count += 1
                decision = decisions.get(measurement.record_id)
                action = decision.action if decision is not None else PolicyAction.KEEP
                token_rows.append(
                    (
                        measurement.record_id,
                        measurement.dataset,
                        measurement.split,
                        action.value,
                    )
                )
                if len(token_rows) >= 1_000:
                    try:
                        connection.executemany(
                            "INSERT INTO token_record VALUES (?, ?, ?, ?)", token_rows
                        )
                    except sqlite3.IntegrityError as error:
                        raise RuntimeError(
                            "Token analysis contains duplicate record IDs."
                        ) from error
                    token_rows.clear()
                if action is PolicyAction.KEEP:
                    destination_file.write(line if line.endswith("\n") else f"{line}\n")
                    training_token_count += 1
                    yield measurement
                _report_progress(
                    progress_callback,
                    "token_policy",
                    canonical_token_count,
                    progress_every,
                )

    token_analysis = summarize_token_length_records(
        retained_measurements(),
        tokenizer_id,
        thresholds,
    )
    if token_rows:
        try:
            connection.executemany("INSERT INTO token_record VALUES (?, ?, ?, ?)", token_rows)
        except sqlite3.IntegrityError as error:
            raise RuntimeError("Token analysis contains duplicate record IDs.") from error
    connection.commit()
    if canonical_token_count % progress_every:
        _report_progress(
            progress_callback,
            "token_policy",
            canonical_token_count,
            progress_every,
            final=True,
        )
    (destination_root / "token-analysis-report.json").write_text(
        json.dumps(token_analysis.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return canonical_token_count, training_token_count, token_analysis


def _record_token_integrity(
    connection: sqlite3.Connection,
) -> tuple[int, int, int]:
    missing = connection.execute(
        """
        SELECT COUNT(*)
        FROM canonical_record AS records
        LEFT JOIN token_record AS tokens USING (record_id)
        WHERE tokens.record_id IS NULL
        """
    ).fetchone()
    unexpected = connection.execute(
        """
        SELECT COUNT(*)
        FROM token_record AS tokens
        LEFT JOIN canonical_record AS records USING (record_id)
        WHERE records.record_id IS NULL
        """
    ).fetchone()
    mismatches = connection.execute(
        """
        SELECT COUNT(*)
        FROM canonical_record AS records
        JOIN token_record AS tokens USING (record_id)
        WHERE records.dataset != tokens.dataset
           OR records.split != tokens.split
           OR records.action != tokens.action
        """
    ).fetchone()
    return int(missing[0]), int(unexpected[0]), int(mismatches[0])


def _nested_counts(values: Mapping[str, Counter[str]]) -> dict[str, dict[str, int]]:
    return {action: dict(sorted(counts.items())) for action, counts in sorted(values.items())}


def _output_paths(staging: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in staging.rglob("*")
            if path.is_file()
            and path.name not in {"phase3-manifest.json", "policy-integrity.sqlite3"}
        )
    )


def materialize_training_ready_partitions(
    final_directory: Path,
    output_directory: Path | None = None,
    config: Phase3PolicyConfig | None = None,
    *,
    progress_every: int = 50_000,
    progress_callback: ProgressCallback | None = None,
) -> Phase3PolicyResult:
    """Apply outlier policy without mutating the canonical leakage-safe data."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive.")
    active_config = config or Phase3PolicyConfig()
    destination = output_directory or final_directory / "training-ready"
    manifest_path = destination / "phase3-manifest.json"
    token_report_path = destination / "analysis" / "token-analysis-report.json"
    if manifest_path.is_file() and token_report_path.is_file():
        manifest = Phase3Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.policy != active_config:
            raise ValueError("Existing Phase 3 policy output used a different configuration.")
        return Phase3PolicyResult(
            output_directory=destination,
            manifest=manifest,
            token_analysis=TokenAnalysisReport.model_validate_json(
                token_report_path.read_text(encoding="utf-8")
            ),
        )
    if destination.exists():
        raise FileExistsError(f"Policy output is incomplete or non-empty: {destination}")
    staging = destination.with_name(f"{destination.name}.staging")
    if staging.exists():
        raise FileExistsError(f"A previous policy staging directory exists: {staging}")

    source_paths = _source_paths(final_directory)
    source_digests_before = _digests(source_paths, final_directory)
    finalization_report = FinalizationReport.model_validate_json(
        (final_directory / "finalization-report.json").read_text(encoding="utf-8")
    )
    if finalization_report.leakage_findings_after:
        raise RuntimeError("Policy requires leakage-safe canonical partitions.")
    source_token_analysis = TokenAnalysisReport.model_validate_json(
        (final_directory / "analysis" / "token-analysis-report.json").read_text(encoding="utf-8")
    )
    outlier_root = final_directory / "audit" / "outliers"
    outlier_report = OutlierAuditReport.model_validate_json(
        (outlier_root / "outlier-audit-report.json").read_text(encoding="utf-8")
    )
    _validate_policy_matches_audit(active_config, outlier_report)
    if outlier_report.total_processed_records != finalization_report.final_records:
        raise RuntimeError("Outlier audit and finalization record counts do not match.")
    decisions = _load_policy_decisions(
        outlier_root / "outlier-records.jsonl",
        outlier_report,
        active_config,
    )

    staging.mkdir(parents=True)
    index_path = staging / "policy-integrity.sqlite3"
    connection = sqlite3.connect(index_path)
    try:
        _create_integrity_index(connection)
        (
            canonical_count,
            training_count,
            by_action,
            by_action_and_dataset,
            by_action_and_split,
            training_by_dataset,
            training_by_split,
            seen_routed_ids,
        ) = _materialize_records(
            final_directory / "processed",
            staging,
            decisions,
            connection,
            progress_every,
            progress_callback,
        )
        canonical_token_count, training_token_count, token_analysis = _filter_token_lengths(
            final_directory / "analysis" / "token-lengths.jsonl",
            staging / "analysis",
            decisions,
            connection,
            source_token_analysis.tokenizer_id,
            source_token_analysis.thresholds,
            progress_every,
            progress_callback,
        )
        missing_tokens, unexpected_tokens, split_mismatches = _record_token_integrity(connection)
        leakage_findings = _count_cross_split_leakage(connection)
    finally:
        connection.close()
    index_path.unlink(missing_ok=True)

    source_digests_after = _digests(source_paths, final_directory)
    integrity = Phase3IntegrityGates(
        canonical_files_unchanged=source_digests_before == source_digests_after,
        canonical_record_count_matches=canonical_count == finalization_report.final_records,
        canonical_token_count_matches=(
            canonical_token_count == finalization_report.final_token_records
        ),
        decision_coverage_complete=seen_routed_ids == set(decisions),
        training_record_token_ids_match=(
            training_count == training_token_count
            and missing_tokens == 0
            and unexpected_tokens == 0
        ),
        split_assignments_preserved=split_mismatches == 0,
        routed_records_absent_from_training=(training_count + len(decisions) == canonical_count),
        duplicate_record_ids=0,
        cross_split_leakage_findings=leakage_findings,
        missing_token_records=missing_tokens,
        unexpected_token_records=unexpected_tokens,
        split_mismatches=split_mismatches,
    )
    failed_gates = [
        field
        for field, value in integrity.model_dump().items()
        if (type(value) is bool and not value) or (type(value) is int and value != 0)
    ]
    if failed_gates:
        raise RuntimeError(f"Phase 3 policy integrity gates failed: {failed_gates}")

    output_digests = _digests(_output_paths(staging), staging)
    manifest = Phase3Manifest(
        policy=active_config,
        tokenizer_id=source_token_analysis.tokenizer_id,
        canonical_records=canonical_count,
        canonical_token_records=canonical_token_count,
        training_ready_records=training_count,
        training_ready_token_records=training_token_count,
        routed_records=len(decisions),
        by_action=dict(sorted(by_action.items())),
        by_action_and_dataset=_nested_counts(by_action_and_dataset),
        by_action_and_split=_nested_counts(by_action_and_split),
        training_ready_by_dataset=dict(sorted(training_by_dataset.items())),
        training_ready_by_split=dict(sorted(training_by_split.items())),
        integrity=integrity,
        source_artifacts=source_digests_before,
        output_artifacts=output_digests,
    )
    (staging / "phase3-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(destination)
    return Phase3PolicyResult(
        output_directory=destination,
        manifest=manifest,
        token_analysis=token_analysis,
    )
