"""Model-token length analysis without importing a model framework locally."""

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from math import ceil
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from summarizer_ml.preprocessing.models import ProcessedRecord


class Tokenizer(Protocol):
    """Subset of the Hugging Face tokenizer interface used by Phase 3."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...

    def __call__(
        self,
        text: list[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
        return_length: bool = False,
    ) -> Mapping[str, object]: ...


class TokenLengthRecord(BaseModel):
    """Token counts for one canonical record."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    dataset: str
    split: str
    source_tokens: int
    query_tokens: int
    input_tokens: int
    summary_tokens: int


class LengthDistribution(BaseModel):
    """Summary statistics for one collection of token lengths."""

    model_config = ConfigDict(frozen=True)

    count: int
    minimum: int
    maximum: int
    mean: float
    p50: int
    p90: int
    p95: int
    p99: int
    over_threshold: dict[str, int]


class DatasetTokenStatistics(BaseModel):
    """Input and target distributions for one dataset or the full mixture."""

    model_config = ConfigDict(frozen=True)

    input_tokens: LengthDistribution
    summary_tokens: LengthDistribution


class TokenAnalysisReport(BaseModel):
    """Per-dataset and overall Qwen tokenizer statistics."""

    model_config = ConfigDict(frozen=True)

    tokenizer_id: str
    thresholds: tuple[int, ...]
    overall: DatasetTokenStatistics
    by_dataset: dict[str, DatasetTokenStatistics]


def measure_token_lengths(record: ProcessedRecord, tokenizer: Tokenizer) -> TokenLengthRecord:
    """Measure source, optional query, combined input, and summary token lengths."""

    source_tokens = len(tokenizer.encode(record.source_text, add_special_tokens=False))
    query_text = record.query or ""
    query_tokens = len(tokenizer.encode(query_text, add_special_tokens=False)) if query_text else 0
    input_tokens = source_tokens
    if query_text:
        input_text = f"{query_text}\n\n{record.source_text}"
        input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
    summary_tokens = len(tokenizer.encode(record.summary_text, add_special_tokens=False))
    return TokenLengthRecord(
        record_id=record.record_id,
        dataset=record.dataset.value,
        split=record.split.value,
        source_tokens=source_tokens,
        query_tokens=query_tokens,
        input_tokens=input_tokens,
        summary_tokens=summary_tokens,
    )


def _percentile(sorted_values: list[int], quantile: float) -> int:
    index = max(0, ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[index]


def _distribution(values: list[int], thresholds: tuple[int, ...]) -> LengthDistribution:
    if not values:
        raise ValueError("Cannot summarize an empty token-length collection.")
    ordered = sorted(values)
    return LengthDistribution(
        count=len(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=sum(ordered) / len(ordered),
        p50=_percentile(ordered, 0.50),
        p90=_percentile(ordered, 0.90),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
        over_threshold={
            str(threshold): sum(value > threshold for value in ordered) for threshold in thresholds
        },
    )


def analyze_token_lengths(
    records: Iterable[ProcessedRecord],
    tokenizer: Tokenizer,
    tokenizer_id: str,
    thresholds: tuple[int, ...] = (4_096, 8_192, 16_384, 32_768),
) -> tuple[tuple[TokenLengthRecord, ...], TokenAnalysisReport]:
    """Measure records and produce overall plus per-dataset distributions."""

    measurements: list[TokenLengthRecord] = []
    for record in records:
        measurement = measure_token_lengths(record, tokenizer)
        measurements.append(measurement)

    if not measurements:
        raise ValueError("No records were supplied for token analysis.")
    report = summarize_token_length_records(measurements, tokenizer_id, thresholds)
    return tuple(measurements), report


def summarize_token_length_records(
    measurements: Iterable[TokenLengthRecord],
    tokenizer_id: str,
    thresholds: tuple[int, ...] = (4_096, 8_192, 16_384, 32_768),
) -> TokenAnalysisReport:
    """Build token distributions from an existing measurement stream."""

    grouped_inputs: dict[str, list[int]] = defaultdict(list)
    grouped_summaries: dict[str, list[int]] = defaultdict(list)
    measurement_count = 0
    for measurement in measurements:
        grouped_inputs[measurement.dataset].append(measurement.input_tokens)
        grouped_summaries[measurement.dataset].append(measurement.summary_tokens)
        measurement_count += 1
    if measurement_count == 0:
        raise ValueError("No token-length records were supplied for analysis.")
    return _build_token_analysis_report(
        grouped_inputs,
        grouped_summaries,
        tokenizer_id,
        thresholds,
    )


def analyze_token_lengths_stream(
    records: Iterable[ProcessedRecord],
    tokenizer: Tokenizer,
    tokenizer_id: str,
    output_directory: Path,
    thresholds: tuple[int, ...] = (4_096, 8_192, 16_384, 32_768),
    *,
    batch_size: int = 256,
    progress_every: int = 10_000,
    progress_callback: Callable[[int], None] | None = None,
    resume: bool = True,
) -> TokenAnalysisReport:
    """Batch tokenization, checkpoint lengths, and resume an interrupted analysis."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if progress_every < 1:
        raise ValueError("progress_every must be positive.")

    output_directory.mkdir(parents=True, exist_ok=True)
    lengths_path = output_directory / "token-lengths.jsonl"
    report_path = output_directory / "token-analysis-report.json"
    if report_path.exists():
        report = TokenAnalysisReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        if report.tokenizer_id != tokenizer_id or report.thresholds != thresholds:
            raise ValueError("Completed token analysis does not match the requested configuration.")
        if progress_callback is not None:
            progress_callback(report.overall.input_tokens.count)
        return report
    if lengths_path.exists() and not resume:
        raise FileExistsError("Token-length checkpoint already exists.")

    grouped_inputs: dict[str, list[int]] = defaultdict(list)
    grouped_summaries: dict[str, list[int]] = defaultdict(list)
    measured_record_ids: set[str] = set()
    if lengths_path.exists():
        with lengths_path.open(encoding="utf-8") as checkpoint_file:
            for line_number, line in enumerate(checkpoint_file, start=1):
                if not line.strip():
                    continue
                try:
                    measurement = TokenLengthRecord.model_validate_json(line)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid token-length checkpoint at line {line_number}."
                    ) from error
                if measurement.record_id in measured_record_ids:
                    raise ValueError("Token-length checkpoint contains duplicate record IDs.")
                measured_record_ids.add(measurement.record_id)
                grouped_inputs[measurement.dataset].append(measurement.input_tokens)
                grouped_summaries[measurement.dataset].append(measurement.summary_tokens)

    measurement_count = len(measured_record_ids)
    last_reported_count = measurement_count if progress_callback is not None else -1
    if progress_callback is not None and measurement_count:
        progress_callback(measurement_count)
    next_progress = ((measurement_count // progress_every) + 1) * progress_every

    unmeasured_records = (
        record for record in records if record.record_id not in measured_record_ids
    )
    file_mode = "a" if lengths_path.exists() else "x"
    with lengths_path.open(file_mode, encoding="utf-8", newline="\n") as lengths_file:
        for batch in _batched(unmeasured_records, batch_size):
            for measurement in _measure_batch(batch, tokenizer):
                lengths_file.write(
                    json.dumps(
                        measurement.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                lengths_file.write("\n")
                grouped_inputs[measurement.dataset].append(measurement.input_tokens)
                grouped_summaries[measurement.dataset].append(measurement.summary_tokens)
                measurement_count += 1
            lengths_file.flush()
            if progress_callback is not None and measurement_count >= next_progress:
                progress_callback(measurement_count)
                last_reported_count = measurement_count
                while next_progress <= measurement_count:
                    next_progress += progress_every

    if progress_callback is not None and measurement_count != last_reported_count:
        progress_callback(measurement_count)

    if measurement_count == 0:
        lengths_path.unlink()
        raise ValueError("No records were supplied for token analysis.")

    report = _build_token_analysis_report(
        grouped_inputs,
        grouped_summaries,
        tokenizer_id,
        thresholds,
    )
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _build_token_analysis_report(
    grouped_inputs: Mapping[str, list[int]],
    grouped_summaries: Mapping[str, list[int]],
    tokenizer_id: str,
    thresholds: tuple[int, ...],
) -> TokenAnalysisReport:
    all_inputs = [value for values in grouped_inputs.values() for value in values]
    all_summaries = [value for values in grouped_summaries.values() for value in values]
    return TokenAnalysisReport(
        tokenizer_id=tokenizer_id,
        thresholds=thresholds,
        overall=DatasetTokenStatistics(
            input_tokens=_distribution(all_inputs, thresholds),
            summary_tokens=_distribution(all_summaries, thresholds),
        ),
        by_dataset={
            dataset: DatasetTokenStatistics(
                input_tokens=_distribution(grouped_inputs[dataset], thresholds),
                summary_tokens=_distribution(grouped_summaries[dataset], thresholds),
            )
            for dataset in sorted(grouped_inputs)
        },
    )


def _batched(
    records: Iterable[ProcessedRecord],
    batch_size: int,
) -> Iterator[tuple[ProcessedRecord, ...]]:
    batch: list[ProcessedRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _batch_lengths(texts: list[str], tokenizer: Tokenizer) -> list[int]:
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_length=True,
    )
    raw_lengths = encoded.get("length")
    if (
        not isinstance(raw_lengths, Sequence)
        or isinstance(raw_lengths, (str, bytes))
        or len(raw_lengths) != len(texts)
    ):
        raise ValueError("Tokenizer returned invalid batch lengths.")
    lengths: list[int] = []
    for length in raw_lengths:
        if not isinstance(length, int) or isinstance(length, bool):
            raise ValueError("Tokenizer returned a non-integer batch length.")
        lengths.append(length)
    return lengths


def _measure_batch(
    records: tuple[ProcessedRecord, ...],
    tokenizer: Tokenizer,
) -> tuple[TokenLengthRecord, ...]:
    source_lengths = _batch_lengths([record.source_text for record in records], tokenizer)
    summary_lengths = _batch_lengths([record.summary_text for record in records], tokenizer)
    query_lengths = [0] * len(records)
    input_lengths = list(source_lengths)

    query_indexes = [index for index, record in enumerate(records) if record.query]
    if query_indexes:
        queries = [records[index].query or "" for index in query_indexes]
        combined_inputs = [
            f"{records[index].query}\n\n{records[index].source_text}" for index in query_indexes
        ]
        measured_queries = _batch_lengths(queries, tokenizer)
        measured_inputs = _batch_lengths(combined_inputs, tokenizer)
        for index, query_length, input_length in zip(
            query_indexes,
            measured_queries,
            measured_inputs,
            strict=True,
        ):
            query_lengths[index] = query_length
            input_lengths[index] = input_length

    return tuple(
        TokenLengthRecord(
            record_id=record.record_id,
            dataset=record.dataset.value,
            split=record.split.value,
            source_tokens=source_lengths[index],
            query_tokens=query_lengths[index],
            input_tokens=input_lengths[index],
            summary_tokens=summary_lengths[index],
        )
        for index, record in enumerate(records)
    )
