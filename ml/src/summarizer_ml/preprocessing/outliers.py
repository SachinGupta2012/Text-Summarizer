"""Metadata-only outlier audit for leakage-safe Phase 3 partitions."""

import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from summarizer_ml.preprocessing.finalization import FinalizationReport
from summarizer_ml.preprocessing.tokenization import TokenLengthRecord

ProgressCallback = Callable[[str, int], None]
_OVERALL = "__overall__"
_METADATA_FIELDS = {
    "conversation_id",
    "meeting_id",
    "query_index",
    "query_list",
    "reference_count",
    "reference_index",
    "summary_missing",
    "title",
}


class OutlierFlag(StrEnum):
    """Conditions requiring a data-policy decision before training."""

    EXTREME_INPUT = "extreme_input"
    EXTREME_SUMMARY = "extreme_summary"
    SUMMARY_LONGER_IN_CHARACTERS = "summary_longer_in_characters"
    SUMMARY_LONGER_IN_TOKENS = "summary_longer_in_tokens"
    TOP_INPUT = "top_input"
    TOP_SUMMARY = "top_summary"
    TOP_SUMMARY_RATIO = "top_summary_ratio"
    VERY_SHORT_INPUT = "very_short_input"
    VERY_SHORT_SUMMARY = "very_short_summary"


class OutlierAuditConfig(BaseModel):
    """Non-destructive thresholds used to select records for review."""

    model_config = ConfigDict(frozen=True)

    extreme_input_tokens: int = Field(default=32_768, ge=1)
    extreme_summary_tokens: int = Field(default=2_048, ge=1)
    very_short_input_tokens: int = Field(default=64, ge=1)
    very_short_summary_tokens: int = Field(default=10, ge=1)
    top_k_per_metric: int = Field(default=10, ge=1, le=100)


class TokenObservation(BaseModel):
    """Record selected as a top token-length outlier."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    dataset: str
    tokens: int


class RatioObservation(BaseModel):
    """Record selected for a high summary-to-input token ratio."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    dataset: str
    ratio: float


class DatasetOutlierStatistics(BaseModel):
    """Outlier counts and top records for one dataset or the full mixture."""

    model_config = ConfigDict(frozen=True)

    total_records: int
    selected_records: int
    flags: dict[str, int]
    top_input_tokens: tuple[TokenObservation, ...]
    top_summary_tokens: tuple[TokenObservation, ...]
    top_summary_to_input_ratio: tuple[RatioObservation, ...]


class OutlierAuditReport(BaseModel):
    """Complete metadata-only outlier audit summary."""

    model_config = ConfigDict(frozen=True)

    config: OutlierAuditConfig
    total_processed_records: int
    total_token_records: int
    selected_outlier_records: int
    flags: dict[str, int]
    overall: DatasetOutlierStatistics
    by_dataset: dict[str, DatasetOutlierStatistics]


class OutlierRecord(BaseModel):
    """Metadata and length measurements for one selected record."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    dataset: str
    split: str
    source_id: str
    group_id: str
    query_type: str
    source_tokens: int
    query_tokens: int
    input_tokens: int
    summary_tokens: int
    source_characters: int
    summary_characters: int
    summary_to_input_ratio: float
    quality_flags: tuple[str, ...]
    audit_flags: tuple[OutlierFlag, ...]
    query: str | None
    metadata: dict[str, object]


class OutlierAuditResult(BaseModel):
    """Paths and report from one completed outlier audit."""

    model_config = ConfigDict(frozen=True)

    output_directory: Path
    report: OutlierAuditReport


TokenHeap = list[tuple[int, str, str]]
RatioHeap = list[tuple[float, str, str]]


@dataclass(frozen=True, slots=True)
class _ProcessedMetadata:
    dataset: str
    split: str
    source_id: str
    group_id: str
    query_type: str
    source_characters: int
    summary_characters: int
    quality_flags: tuple[str, ...]
    query: str | None
    metadata: dict[str, object]


def _push_top_token(
    heap: TokenHeap,
    tokens: int,
    record_id: str,
    dataset: str,
    limit: int,
) -> None:
    candidate = (tokens, record_id, dataset)
    if len(heap) < limit:
        heapq.heappush(heap, candidate)
    elif candidate > heap[0]:
        heapq.heapreplace(heap, candidate)


def _push_top_ratio(
    heap: RatioHeap,
    ratio: float,
    record_id: str,
    dataset: str,
    limit: int,
) -> None:
    candidate = (ratio, record_id, dataset)
    if len(heap) < limit:
        heapq.heappush(heap, candidate)
    elif candidate > heap[0]:
        heapq.heapreplace(heap, candidate)


def _mark(
    record_id: str,
    dataset: str,
    flag: OutlierFlag,
    flags_by_record: dict[str, set[OutlierFlag]],
    dataset_by_record: dict[str, str],
) -> None:
    flags_by_record[record_id].add(flag)
    dataset_by_record[record_id] = dataset


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


def _scan_token_lengths(
    path: Path,
    config: OutlierAuditConfig,
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> tuple[
    int,
    Counter[str],
    dict[str, set[OutlierFlag]],
    dict[str, str],
    dict[str, TokenHeap],
    dict[str, TokenHeap],
    dict[str, RatioHeap],
]:
    total_records = 0
    records_by_dataset: Counter[str] = Counter()
    flags_by_record: dict[str, set[OutlierFlag]] = defaultdict(set)
    dataset_by_record: dict[str, str] = {}
    top_inputs: dict[str, TokenHeap] = defaultdict(list)
    top_summaries: dict[str, TokenHeap] = defaultdict(list)
    top_ratios: dict[str, RatioHeap] = defaultdict(list)

    with path.open(encoding="utf-8") as lengths_file:
        for line_number, line in enumerate(lengths_file, start=1):
            if not line.strip():
                continue
            try:
                measurement = TokenLengthRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid token-length record at {path}:{line_number}") from error
            total_records += 1
            dataset = measurement.dataset
            records_by_dataset[dataset] += 1
            ratio = measurement.summary_tokens / max(1, measurement.input_tokens)
            for group in (dataset, _OVERALL):
                _push_top_token(
                    top_inputs[group],
                    measurement.input_tokens,
                    measurement.record_id,
                    dataset,
                    config.top_k_per_metric,
                )
                _push_top_token(
                    top_summaries[group],
                    measurement.summary_tokens,
                    measurement.record_id,
                    dataset,
                    config.top_k_per_metric,
                )
                _push_top_ratio(
                    top_ratios[group],
                    ratio,
                    measurement.record_id,
                    dataset,
                    config.top_k_per_metric,
                )

            if measurement.input_tokens > config.extreme_input_tokens:
                _mark(
                    measurement.record_id,
                    dataset,
                    OutlierFlag.EXTREME_INPUT,
                    flags_by_record,
                    dataset_by_record,
                )
            if measurement.summary_tokens > config.extreme_summary_tokens:
                _mark(
                    measurement.record_id,
                    dataset,
                    OutlierFlag.EXTREME_SUMMARY,
                    flags_by_record,
                    dataset_by_record,
                )
            if measurement.input_tokens < config.very_short_input_tokens:
                _mark(
                    measurement.record_id,
                    dataset,
                    OutlierFlag.VERY_SHORT_INPUT,
                    flags_by_record,
                    dataset_by_record,
                )
            if measurement.summary_tokens < config.very_short_summary_tokens:
                _mark(
                    measurement.record_id,
                    dataset,
                    OutlierFlag.VERY_SHORT_SUMMARY,
                    flags_by_record,
                    dataset_by_record,
                )
            if measurement.summary_tokens > measurement.input_tokens:
                _mark(
                    measurement.record_id,
                    dataset,
                    OutlierFlag.SUMMARY_LONGER_IN_TOKENS,
                    flags_by_record,
                    dataset_by_record,
                )
            _report_progress(
                progress_callback,
                "token_scan",
                total_records,
                progress_every,
            )

    if total_records % progress_every:
        _report_progress(
            progress_callback,
            "token_scan",
            total_records,
            progress_every,
            final=True,
        )
    return (
        total_records,
        records_by_dataset,
        flags_by_record,
        dataset_by_record,
        top_inputs,
        top_summaries,
        top_ratios,
    )


def _mark_top_records(
    top_inputs: Mapping[str, TokenHeap],
    top_summaries: Mapping[str, TokenHeap],
    top_ratios: Mapping[str, RatioHeap],
    flags_by_record: dict[str, set[OutlierFlag]],
    dataset_by_record: dict[str, str],
) -> None:
    for heaps, flag in (
        (top_inputs, OutlierFlag.TOP_INPUT),
        (top_summaries, OutlierFlag.TOP_SUMMARY),
        (top_ratios, OutlierFlag.TOP_SUMMARY_RATIO),
    ):
        for group, heap in heaps.items():
            if group == _OVERALL:
                continue
            for _, record_id, dataset in heap:
                _mark(record_id, dataset, flag, flags_by_record, dataset_by_record)


def _scan_processed_metadata(
    processed_root: Path,
    flags_by_record: dict[str, set[OutlierFlag]],
    dataset_by_record: dict[str, str],
    expected_records: int,
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> tuple[int, dict[str, _ProcessedMetadata]]:
    processed_count = 0
    metadata_by_record: dict[str, _ProcessedMetadata] = {}
    for path in sorted(processed_root.glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as records_file:
            for line in records_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                processed_count += 1
                record_id = str(record["record_id"])
                dataset = str(record["dataset"])
                quality_flags = tuple(str(flag) for flag in record.get("quality_flags", []))
                if "summary_longer_than_source" in quality_flags:
                    _mark(
                        record_id,
                        dataset,
                        OutlierFlag.SUMMARY_LONGER_IN_CHARACTERS,
                        flags_by_record,
                        dataset_by_record,
                    )
                if record_id in flags_by_record:
                    raw_metadata = record.get("metadata", {})
                    metadata = (
                        {
                            key: value
                            for key, value in raw_metadata.items()
                            if key in _METADATA_FIELDS
                        }
                        if isinstance(raw_metadata, dict)
                        else {}
                    )
                    raw_query = record.get("query")
                    metadata_by_record[record_id] = _ProcessedMetadata(
                        dataset=dataset,
                        split=str(record["split"]),
                        source_id=str(record["source_id"]),
                        group_id=str(record["group_id"]),
                        query_type=str(record["query_type"]),
                        source_characters=len(str(record["source_text"])),
                        summary_characters=len(str(record["summary_text"])),
                        quality_flags=quality_flags,
                        query=raw_query if isinstance(raw_query, str) else None,
                        metadata=metadata,
                    )
                _report_progress(
                    progress_callback,
                    "partition_scan",
                    processed_count,
                    progress_every,
                )

    if processed_count % progress_every:
        _report_progress(
            progress_callback,
            "partition_scan",
            processed_count,
            progress_every,
            final=True,
        )
    if processed_count != expected_records:
        raise RuntimeError("Processed partition and finalization record counts do not match.")
    missing_metadata = set(flags_by_record).difference(metadata_by_record)
    if missing_metadata:
        raise RuntimeError(f"Metadata is missing for {len(missing_metadata)} selected records.")
    return processed_count, metadata_by_record


def _load_selected_measurements(
    path: Path,
    selected_record_ids: set[str],
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> dict[str, TokenLengthRecord]:
    measurements: dict[str, TokenLengthRecord] = {}
    scanned = 0
    with path.open(encoding="utf-8") as lengths_file:
        for line in lengths_file:
            if not line.strip():
                continue
            scanned += 1
            measurement = TokenLengthRecord.model_validate_json(line)
            if measurement.record_id in selected_record_ids:
                measurements[measurement.record_id] = measurement
            _report_progress(
                progress_callback,
                "token_join",
                scanned,
                progress_every,
            )
    if scanned % progress_every:
        _report_progress(
            progress_callback,
            "token_join",
            scanned,
            progress_every,
            final=True,
        )
    missing = selected_record_ids.difference(measurements)
    if missing:
        raise RuntimeError(f"Token measurements are missing for {len(missing)} selected records.")
    return measurements


def _token_observations(heap: TokenHeap) -> tuple[TokenObservation, ...]:
    return tuple(
        TokenObservation(record_id=record_id, dataset=dataset, tokens=tokens)
        for tokens, record_id, dataset in sorted(heap, reverse=True)
    )


def _ratio_observations(heap: RatioHeap) -> tuple[RatioObservation, ...]:
    return tuple(
        RatioObservation(record_id=record_id, dataset=dataset, ratio=ratio)
        for ratio, record_id, dataset in sorted(heap, reverse=True)
    )


def _build_statistics(
    group: str,
    total_records: int,
    flags_by_record: Mapping[str, set[OutlierFlag]],
    dataset_by_record: Mapping[str, str],
    top_inputs: Mapping[str, TokenHeap],
    top_summaries: Mapping[str, TokenHeap],
    top_ratios: Mapping[str, RatioHeap],
) -> DatasetOutlierStatistics:
    selected_ids = {
        record_id
        for record_id in flags_by_record
        if group == _OVERALL or dataset_by_record[record_id] == group
    }
    flag_counts: Counter[str] = Counter()
    for record_id in selected_ids:
        flag_counts.update(flag.value for flag in flags_by_record[record_id])
    return DatasetOutlierStatistics(
        total_records=total_records,
        selected_records=len(selected_ids),
        flags=dict(sorted(flag_counts.items())),
        top_input_tokens=_token_observations(top_inputs[group]),
        top_summary_tokens=_token_observations(top_summaries[group]),
        top_summary_to_input_ratio=_ratio_observations(top_ratios[group]),
    )


def audit_outliers(
    final_directory: Path,
    output_directory: Path | None = None,
    config: OutlierAuditConfig | None = None,
    *,
    progress_every: int = 50_000,
    progress_callback: ProgressCallback | None = None,
) -> OutlierAuditResult:
    """Audit final partitions without modifying or excluding any record."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive.")
    active_config = config or OutlierAuditConfig()
    destination = output_directory or final_directory / "audit" / "outliers"
    report_path = destination / "outlier-audit-report.json"
    records_path = destination / "outlier-records.jsonl"
    if report_path.exists() and records_path.exists():
        report = OutlierAuditReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        if report.config != active_config:
            raise ValueError("Existing outlier audit used a different configuration.")
        return OutlierAuditResult(output_directory=destination, report=report)
    if destination.exists():
        raise FileExistsError(f"Outlier audit directory is incomplete or non-empty: {destination}")
    staging = destination.with_name(f"{destination.name}.staging")
    if staging.exists():
        raise FileExistsError(f"A previous outlier-audit staging directory exists: {staging}")

    finalization_report_path = final_directory / "finalization-report.json"
    processed_root = final_directory / "processed"
    token_lengths_path = final_directory / "analysis" / "token-lengths.jsonl"
    if not finalization_report_path.is_file() or not processed_root.is_dir():
        raise FileNotFoundError("Leakage-safe final partitions or report are missing.")
    if not token_lengths_path.is_file():
        raise FileNotFoundError("Final token-length records are missing.")
    finalization_report = FinalizationReport.model_validate_json(
        finalization_report_path.read_text(encoding="utf-8")
    )
    if finalization_report.leakage_findings_after:
        raise RuntimeError("Cannot audit final partitions that still contain leakage.")

    (
        token_count,
        records_by_dataset,
        flags_by_record,
        dataset_by_record,
        top_inputs,
        top_summaries,
        top_ratios,
    ) = _scan_token_lengths(
        token_lengths_path,
        active_config,
        progress_every,
        progress_callback,
    )
    _mark_top_records(
        top_inputs,
        top_summaries,
        top_ratios,
        flags_by_record,
        dataset_by_record,
    )
    processed_count, metadata_by_record = _scan_processed_metadata(
        processed_root,
        flags_by_record,
        dataset_by_record,
        finalization_report.final_records,
        progress_every,
        progress_callback,
    )
    if token_count != finalization_report.final_token_records:
        raise RuntimeError("Token-length and finalization record counts do not match.")

    selected_record_ids = set(flags_by_record)
    measurements = _load_selected_measurements(
        token_lengths_path,
        selected_record_ids,
        progress_every,
        progress_callback,
    )
    outlier_records: list[OutlierRecord] = []
    for record_id in selected_record_ids:
        measurement = measurements[record_id]
        metadata = metadata_by_record[record_id]
        outlier_records.append(
            OutlierRecord(
                record_id=record_id,
                dataset=metadata.dataset,
                split=metadata.split,
                source_id=metadata.source_id,
                group_id=metadata.group_id,
                query_type=metadata.query_type,
                source_tokens=measurement.source_tokens,
                query_tokens=measurement.query_tokens,
                input_tokens=measurement.input_tokens,
                summary_tokens=measurement.summary_tokens,
                source_characters=metadata.source_characters,
                summary_characters=metadata.summary_characters,
                summary_to_input_ratio=(
                    measurement.summary_tokens / max(1, measurement.input_tokens)
                ),
                quality_flags=metadata.quality_flags,
                audit_flags=tuple(sorted(flags_by_record[record_id], key=str)),
                query=metadata.query,
                metadata=metadata.metadata,
            )
        )

    by_dataset = {
        dataset: _build_statistics(
            dataset,
            count,
            flags_by_record,
            dataset_by_record,
            top_inputs,
            top_summaries,
            top_ratios,
        )
        for dataset, count in sorted(records_by_dataset.items())
    }
    overall = _build_statistics(
        _OVERALL,
        token_count,
        flags_by_record,
        dataset_by_record,
        top_inputs,
        top_summaries,
        top_ratios,
    )
    report = OutlierAuditReport(
        config=active_config,
        total_processed_records=processed_count,
        total_token_records=token_count,
        selected_outlier_records=len(outlier_records),
        flags=overall.flags,
        overall=overall,
        by_dataset=by_dataset,
    )

    staging.mkdir(parents=True)
    with (staging / "outlier-records.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    ) as records_file:
        for record in sorted(
            outlier_records,
            key=lambda item: (item.dataset, item.split, item.record_id),
        ):
            records_file.write(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            )
            records_file.write("\n")
    (staging / "outlier-audit-report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(destination)
    return OutlierAuditResult(output_directory=destination, report=report)
