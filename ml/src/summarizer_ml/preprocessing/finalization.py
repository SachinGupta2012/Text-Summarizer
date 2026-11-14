"""Materialize leakage-safe Phase 3 partitions from diagnostic artifacts."""

import json
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from summarizer_ml.preprocessing.deduplication import LeakageFinding, LeakageKind
from summarizer_ml.preprocessing.models import DatasetName, Split
from summarizer_ml.preprocessing.streaming import ProcessingReport
from summarizer_ml.preprocessing.tokenization import (
    TokenAnalysisReport,
    TokenLengthRecord,
    summarize_token_length_records,
)

_SPLIT_PRIORITY = {
    Split.TRAIN: 1,
    Split.VALIDATION: 2,
    Split.CA_TEST: 3,
    Split.TEST: 4,
}


class LeakageRemoval(BaseModel):
    """One lower-priority record removed to protect an evaluation split."""

    model_config = ConfigDict(frozen=True)

    finding_kind: LeakageKind
    finding_key: str
    removed_record_id: str
    removed_dataset: DatasetName
    removed_split: Split
    retained_split: Split
    retained_record_ids: tuple[str, ...]


class FinalizationReport(BaseModel):
    """Counts and gates for leakage-safe final partitions."""

    model_config = ConfigDict(frozen=True)

    diagnostic_accepted_records: int
    diagnostic_rejected_records: int
    diagnostic_exact_duplicates: int
    leakage_findings_before: int
    leakage_removals: int
    final_records: int
    final_token_records: int
    leakage_findings_after: int
    final_by_dataset: dict[str, int]
    final_by_split: dict[str, int]
    quality_flags: dict[str, int]


class FinalizationResult(BaseModel):
    """Paths and reports produced by leakage-safe materialization."""

    model_config = ConfigDict(frozen=True)

    final_directory: Path
    report: FinalizationReport
    token_analysis: TokenAnalysisReport


@dataclass(frozen=True, slots=True)
class _RecordLocation:
    record_id: str
    dataset: DatasetName
    split: Split


def _read_findings(path: Path) -> tuple[LeakageFinding, ...]:
    if not path.exists():
        return ()
    findings: list[LeakageFinding] = []
    with path.open(encoding="utf-8") as findings_file:
        for line_number, line in enumerate(findings_file, start=1):
            if not line.strip():
                continue
            try:
                findings.append(LeakageFinding.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"Invalid leakage finding at {path}:{line_number}") from error
    return tuple(findings)


def _find_record_locations(
    processed_root: Path,
    target_record_ids: set[str],
) -> dict[str, _RecordLocation]:
    if not target_record_ids:
        return {}
    locations: dict[str, _RecordLocation] = {}
    for path in sorted(processed_root.glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as records_file:
            for line in records_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                record_id = str(record.get("record_id", ""))
                if record_id not in target_record_ids:
                    continue
                locations[record_id] = _RecordLocation(
                    record_id=record_id,
                    dataset=DatasetName(str(record["dataset"])),
                    split=Split(str(record["split"])),
                )

    missing = target_record_ids.difference(locations)
    if missing:
        raise ValueError(f"Leakage audit references {len(missing)} missing processed records.")
    return locations


def _plan_removals(
    findings: tuple[LeakageFinding, ...],
    locations: Mapping[str, _RecordLocation],
) -> tuple[LeakageRemoval, ...]:
    removals: list[LeakageRemoval] = []
    for finding in findings:
        finding_locations = [locations[record_id] for record_id in finding.record_ids]
        retained_split = max(
            {location.split for location in finding_locations},
            key=_SPLIT_PRIORITY.__getitem__,
        )
        retained_record_ids = tuple(
            sorted(
                location.record_id
                for location in finding_locations
                if location.split is retained_split
            )
        )
        for location in finding_locations:
            if location.split is retained_split:
                continue
            removals.append(
                LeakageRemoval(
                    finding_kind=finding.kind,
                    finding_key=finding.key,
                    removed_record_id=location.record_id,
                    removed_dataset=location.dataset,
                    removed_split=location.split,
                    retained_split=retained_split,
                    retained_record_ids=retained_record_ids,
                )
            )
    return tuple(
        sorted(
            removals,
            key=lambda removal: (
                removal.removed_record_id,
                removal.finding_kind,
                removal.finding_key,
            ),
        )
    )


def _create_leakage_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE source_occurrence (
            source_sha256 TEXT NOT NULL,
            split TEXT NOT NULL,
            record_id TEXT NOT NULL
        );
        CREATE INDEX source_occurrence_hash
            ON source_occurrence (source_sha256, split);
        CREATE TABLE group_occurrence (
            group_id TEXT NOT NULL,
            split TEXT NOT NULL,
            record_id TEXT NOT NULL
        );
        CREATE INDEX group_occurrence_id
            ON group_occurrence (group_id, split);
        """
    )


def _count_index_leakage(connection: sqlite3.Connection) -> int:
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


def _materialize_partitions(
    processed_root: Path,
    destination_root: Path,
    removed_record_ids: set[str],
    index_path: Path,
) -> tuple[int, dict[str, int], dict[str, int], dict[str, int], int]:
    final_by_dataset: Counter[str] = Counter()
    final_by_split: Counter[str] = Counter()
    quality_flags: Counter[str] = Counter()
    final_records = 0
    connection = sqlite3.connect(index_path)
    try:
        _create_leakage_index(connection)
        source_rows: list[tuple[str, str, str]] = []
        group_rows: list[tuple[str, str, str]] = []
        for source_path in sorted(processed_root.glob("*/*.jsonl")):
            relative_path = source_path.relative_to(processed_root)
            destination_path = destination_root / "processed" / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                source_path.open(encoding="utf-8") as source_file,
                destination_path.open("x", encoding="utf-8", newline="\n") as destination_file,
            ):
                for line in source_file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record_id = str(record["record_id"])
                    if record_id in removed_record_ids:
                        continue
                    destination_file.write(line if line.endswith("\n") else f"{line}\n")
                    dataset = str(record["dataset"])
                    split = str(record["split"])
                    final_records += 1
                    final_by_dataset[dataset] += 1
                    final_by_split[split] += 1
                    flags = record.get("quality_flags", [])
                    if isinstance(flags, list):
                        quality_flags.update(str(flag) for flag in flags)
                    source_rows.append((str(record["source_sha256"]), split, record_id))
                    group_rows.append((str(record["group_id"]), split, record_id))
                    if len(source_rows) >= 1_000:
                        connection.executemany(
                            "INSERT INTO source_occurrence VALUES (?, ?, ?)", source_rows
                        )
                        connection.executemany(
                            "INSERT INTO group_occurrence VALUES (?, ?, ?)", group_rows
                        )
                        source_rows.clear()
                        group_rows.clear()

        if source_rows:
            connection.executemany("INSERT INTO source_occurrence VALUES (?, ?, ?)", source_rows)
            connection.executemany("INSERT INTO group_occurrence VALUES (?, ?, ?)", group_rows)
        connection.commit()
        leakage_findings_after = _count_index_leakage(connection)
    finally:
        connection.close()

    return (
        final_records,
        dict(sorted(final_by_dataset.items())),
        dict(sorted(final_by_split.items())),
        dict(sorted(quality_flags.items())),
        leakage_findings_after,
    )


def _filter_token_lengths(
    source_analysis_root: Path,
    destination_analysis_root: Path,
    removed_record_ids: set[str],
) -> tuple[int, TokenAnalysisReport]:
    source_report_path = source_analysis_root / "token-analysis-report.json"
    source_lengths_path = source_analysis_root / "token-lengths.jsonl"
    source_report = TokenAnalysisReport.model_validate_json(
        source_report_path.read_text(encoding="utf-8")
    )
    destination_analysis_root.mkdir(parents=True, exist_ok=True)
    destination_lengths_path = destination_analysis_root / "token-lengths.jsonl"

    retained_count = 0

    def retained_measurements() -> Iterator[TokenLengthRecord]:
        nonlocal retained_count
        with (
            source_lengths_path.open(encoding="utf-8") as source_file,
            destination_lengths_path.open("x", encoding="utf-8", newline="\n") as destination,
        ):
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    measurement = TokenLengthRecord.model_validate_json(line)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid token-length record at {source_lengths_path}:{line_number}"
                    ) from error
                if measurement.record_id in removed_record_ids:
                    continue
                destination.write(line if line.endswith("\n") else f"{line}\n")
                retained_count += 1
                yield measurement

    final_report = summarize_token_length_records(
        retained_measurements(),
        source_report.tokenizer_id,
        source_report.thresholds,
    )
    (destination_analysis_root / "token-analysis-report.json").write_text(
        json.dumps(final_report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return retained_count, final_report


def _write_removal_audit(path: Path, removals: tuple[LeakageRemoval, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as audit_file:
        for removal in removals:
            audit_file.write(
                json.dumps(removal.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            )
            audit_file.write("\n")


def materialize_leakage_safe_partitions(
    diagnostic_directory: Path,
    final_directory: Path | None = None,
) -> FinalizationResult:
    """Create final partitions using evaluation-first cross-split precedence."""

    destination = final_directory or diagnostic_directory / "final"
    report_path = destination / "finalization-report.json"
    token_report_path = destination / "analysis" / "token-analysis-report.json"
    if report_path.exists() and token_report_path.exists():
        return FinalizationResult(
            final_directory=destination,
            report=FinalizationReport.model_validate_json(report_path.read_text(encoding="utf-8")),
            token_analysis=TokenAnalysisReport.model_validate_json(
                token_report_path.read_text(encoding="utf-8")
            ),
        )
    if destination.exists():
        raise FileExistsError(f"Final directory is incomplete or non-empty: {destination}")

    processed_root = diagnostic_directory / "processed"
    processing_report_path = diagnostic_directory / "processing-report.json"
    if not processed_root.is_dir() or not processing_report_path.is_file():
        raise FileNotFoundError("Diagnostic processed partitions or report are missing.")

    processing_report = ProcessingReport.model_validate_json(
        processing_report_path.read_text(encoding="utf-8")
    )
    findings = _read_findings(diagnostic_directory / "audit" / "split-leakage.jsonl")
    target_record_ids = {record_id for finding in findings for record_id in finding.record_ids}
    locations = _find_record_locations(processed_root, target_record_ids)
    removals = _plan_removals(findings, locations)
    removed_record_ids = {removal.removed_record_id for removal in removals}

    staging = destination.with_name(f"{destination.name}.staging")
    if staging.exists():
        raise FileExistsError(f"A previous finalization staging directory exists: {staging}")
    staging.mkdir(parents=True)
    index_path = staging / "finalization-index.sqlite3"

    (
        final_records,
        final_by_dataset,
        final_by_split,
        quality_flags,
        leakage_findings_after,
    ) = _materialize_partitions(
        processed_root,
        staging,
        removed_record_ids,
        index_path,
    )
    index_path.unlink(missing_ok=True)
    if leakage_findings_after:
        raise RuntimeError(
            f"Leakage-safe finalization found {leakage_findings_after} remaining overlaps."
        )

    final_token_records, token_analysis = _filter_token_lengths(
        diagnostic_directory / "analysis",
        staging / "analysis",
        removed_record_ids,
    )
    expected_final_records = processing_report.accepted_records - len(removed_record_ids)
    if final_records != expected_final_records or final_token_records != final_records:
        raise RuntimeError(
            "Final partition, diagnostic, and token-length record counts do not reconcile."
        )

    _write_removal_audit(staging / "audit" / "leakage-removals.jsonl", removals)
    report = FinalizationReport(
        diagnostic_accepted_records=processing_report.accepted_records,
        diagnostic_rejected_records=processing_report.rejected_records,
        diagnostic_exact_duplicates=processing_report.exact_duplicates,
        leakage_findings_before=len(findings),
        leakage_removals=len(removed_record_ids),
        final_records=final_records,
        final_token_records=final_token_records,
        leakage_findings_after=leakage_findings_after,
        final_by_dataset=final_by_dataset,
        final_by_split=final_by_split,
        quality_flags=quality_flags,
    )
    (staging / "finalization-report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(destination)
    return FinalizationResult(
        final_directory=destination,
        report=report,
        token_analysis=token_analysis,
    )
