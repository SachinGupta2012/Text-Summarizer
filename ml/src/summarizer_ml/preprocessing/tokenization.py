"""Model-token length analysis without importing a model framework locally."""

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from math import ceil
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from summarizer_ml.preprocessing.models import ProcessedRecord


class Tokenizer(Protocol):
    """Subset of the Hugging Face tokenizer interface used by Phase 3."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...


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
    input_text = f"{query_text}\n\n{record.source_text}" if query_text else record.source_text
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
    grouped_inputs: dict[str, list[int]] = defaultdict(list)
    grouped_summaries: dict[str, list[int]] = defaultdict(list)
    for record in records:
        measurement = measure_token_lengths(record, tokenizer)
        measurements.append(measurement)
        grouped_inputs[measurement.dataset].append(measurement.input_tokens)
        grouped_summaries[measurement.dataset].append(measurement.summary_tokens)

    if not measurements:
        raise ValueError("No records were supplied for token analysis.")

    all_inputs = [measurement.input_tokens for measurement in measurements]
    all_summaries = [measurement.summary_tokens for measurement in measurements]
    by_dataset = {
        dataset: DatasetTokenStatistics(
            input_tokens=_distribution(grouped_inputs[dataset], thresholds),
            summary_tokens=_distribution(grouped_summaries[dataset], thresholds),
        )
        for dataset in sorted(grouped_inputs)
    }
    report = TokenAnalysisReport(
        tokenizer_id=tokenizer_id,
        thresholds=thresholds,
        overall=DatasetTokenStatistics(
            input_tokens=_distribution(all_inputs, thresholds),
            summary_tokens=_distribution(all_summaries, thresholds),
        ),
        by_dataset=by_dataset,
    )
    return tuple(measurements), report


def analyze_token_lengths_stream(
    records: Iterable[ProcessedRecord],
    tokenizer: Tokenizer,
    tokenizer_id: str,
    output_directory: Path,
    thresholds: tuple[int, ...] = (4_096, 8_192, 16_384, 32_768),
) -> TokenAnalysisReport:
    """Write per-record lengths while retaining only integer distributions in memory."""

    output_directory.mkdir(parents=True, exist_ok=True)
    lengths_path = output_directory / "token-lengths.jsonl"
    report_path = output_directory / "token-analysis-report.json"
    if lengths_path.exists() or report_path.exists():
        raise FileExistsError("Token-analysis outputs already exist.")

    grouped_inputs: dict[str, list[int]] = defaultdict(list)
    grouped_summaries: dict[str, list[int]] = defaultdict(list)
    measurement_count = 0
    with lengths_path.open("x", encoding="utf-8", newline="\n") as lengths_file:
        for record in records:
            measurement = measure_token_lengths(record, tokenizer)
            lengths_file.write(
                json.dumps(measurement.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            )
            lengths_file.write("\n")
            grouped_inputs[measurement.dataset].append(measurement.input_tokens)
            grouped_summaries[measurement.dataset].append(measurement.summary_tokens)
            measurement_count += 1

    if measurement_count == 0:
        lengths_path.unlink()
        raise ValueError("No records were supplied for token analysis.")

    all_inputs = [value for values in grouped_inputs.values() for value in values]
    all_summaries = [value for values in grouped_summaries.values() for value in values]
    report = TokenAnalysisReport(
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
    report_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
