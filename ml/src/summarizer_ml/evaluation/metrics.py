"""Transparent lexical, semantic, and runtime metrics for Phase 5."""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from summarizer_ml.evaluation.models import (
    BaselinePrediction,
    BenchmarkExample,
    MetricSummary,
    Phase5Config,
    Phase5IntegrityGates,
    Phase5Report,
    Phase5Result,
)

_WORD = re.compile(r"\w+", flags=re.UNICODE)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(value.casefold() for value in _WORD.findall(text))


def _ngram_f1(prediction: tuple[str, ...], reference: tuple[str, ...], size: int) -> float:
    if len(prediction) < size or len(reference) < size:
        return 0.0
    predicted = Counter(zip(*(prediction[index:] for index in range(size)), strict=False))
    expected = Counter(zip(*(reference[index:] for index in range(size)), strict=False))
    overlap = sum((predicted & expected).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def _lcs_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0]
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def _rouge_l_f1(prediction: tuple[str, ...], reference: tuple[str, ...]) -> float:
    if not prediction or not reference:
        return 0.0
    overlap = _lcs_length(prediction, reference)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def _mean(values: Sequence[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summarize(
    pairs: Sequence[tuple[BenchmarkExample, BaselinePrediction]],
    bertscore: Mapping[str, float] | None,
) -> MetricSummary:
    rouge1: list[float] = []
    rouge2: list[float] = []
    rouge_l: list[float] = []
    semantic: list[float] = []
    prediction_tokens: list[int] = []
    reference_tokens: list[int] = []
    latencies: list[float] = []
    empty = 0
    for example, prediction in pairs:
        predicted = _tokens(prediction.prediction)
        reference = _tokens(example.reference_summary)
        rouge1.append(_ngram_f1(predicted, reference, 1))
        rouge2.append(_ngram_f1(predicted, reference, 2))
        rouge_l.append(_rouge_l_f1(predicted, reference))
        if bertscore is not None:
            semantic.append(bertscore[example.record_id])
        prediction_tokens.append(prediction.generated_tokens)
        reference_tokens.append(example.reference_tokens)
        latencies.append(prediction.latency_seconds)
        empty += int(not prediction.prediction.strip())
    return MetricSummary(
        records=len(pairs),
        rouge1_f1=_mean(rouge1),
        rouge2_f1=_mean(rouge2),
        rouge_l_f1=_mean(rouge_l),
        bertscore_f1=_mean(semantic) if semantic else None,
        mean_prediction_tokens=_mean(prediction_tokens),
        mean_reference_tokens=_mean(reference_tokens),
        mean_latency_seconds=_mean(latencies),
        empty_predictions=empty,
    )


def evaluate_baseline(
    selection: Sequence[BenchmarkExample],
    predictions: Sequence[BaselinePrediction],
    output_directory: Path,
    config: Phase5Config,
    phase4_manifest_sha256: str,
    model_revision: str,
    *,
    bertscore: Mapping[str, float] | None = None,
) -> Phase5Result:
    """Validate prediction identity and freeze aggregate benchmark metrics."""

    selected = {example.record_id: example for example in selection}
    predicted = {prediction.record_id: prediction for prediction in predictions}
    prediction_ids_unique = len(predicted) == len(predictions)
    identity_matches = all(
        record_id in selected
        and prediction.example_id == selected[record_id].example_id
        and prediction.dataset == selected[record_id].dataset
        and prediction.domain == selected[record_id].domain
        and prediction.split == selected[record_id].split
        for record_id, prediction in predicted.items()
    )
    coverage_complete = set(predicted) == set(selected)
    held_out_only = all(example.split in config.eligible_splits for example in selection)
    if bertscore is not None and set(bertscore) != set(selected):
        raise ValueError("BERTScore coverage does not match the benchmark selection.")
    if not prediction_ids_unique or not identity_matches or not coverage_complete:
        raise RuntimeError("Baseline prediction integrity failed.")
    pairs = tuple((example, predicted[example.record_id]) for example in selection)
    groups: dict[str, dict[str, list[tuple[BenchmarkExample, BaselinePrediction]]]] = {
        "dataset": defaultdict(list),
        "domain": defaultdict(list),
        "split": defaultdict(list),
    }
    for pair in pairs:
        example = pair[0]
        groups["dataset"][example.dataset].append(pair)
        groups["domain"][example.domain].append(pair)
        groups["split"][example.split].append(pair)
    overall = _summarize(pairs, bertscore)
    integrity = Phase5IntegrityGates(
        selection_uses_only_held_out_splits=held_out_only,
        prediction_coverage_complete=coverage_complete,
        prediction_ids_unique=prediction_ids_unique,
        prediction_identity_matches_selection=identity_matches,
        empty_predictions=overall.empty_predictions,
    )
    failed = [
        field
        for field, value in integrity.model_dump().items()
        if (type(value) is bool and not value) or (type(value) is int and value != 0)
    ]
    if failed:
        raise RuntimeError(f"Phase 5 integrity gates failed: {failed}")
    report = Phase5Report(
        config=config,
        phase4_manifest_sha256=phase4_manifest_sha256,
        model_revision=model_revision,
        selected_records=len(selection),
        overall=overall,
        by_dataset={
            key: _summarize(value, bertscore) for key, value in sorted(groups["dataset"].items())
        },
        by_domain={
            key: _summarize(value, bertscore) for key, value in sorted(groups["domain"].items())
        },
        by_split={
            key: _summarize(value, bertscore) for key, value in sorted(groups["split"].items())
        },
        integrity=integrity,
        selection_path=output_directory / "benchmark-selection.jsonl",
        predictions_path=output_directory / "base-predictions.jsonl",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "baseline-report.json"
    temporary = report_path.with_name(f"{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return Phase5Result(output_directory=output_directory, report=report)
