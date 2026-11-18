"""Staged, paired Phase 7 evaluation of the QLoRA pilot adapter."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from summarizer_ml.evaluation.benchmark import load_predictions
from summarizer_ml.evaluation.metrics import (
    prediction_pair_metrics,
    summarize_predictions,
)
from summarizer_ml.evaluation.models import (
    BaselinePrediction,
    BenchmarkExample,
    ConfidenceInterval,
    MetricDelta,
    MetricSummary,
    Phase7Config,
    Phase7Decision,
    Phase7IntegrityGates,
    Phase7Report,
    Phase7Result,
    Phase7Stage,
)


def sha256_file(path: Path) -> str:
    """Return the stable SHA-256 identity of an artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(seed: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def select_phase7_examples(
    selection: Sequence[BenchmarkExample],
    config: Phase7Config,
    stage: Phase7Stage,
) -> tuple[BenchmarkExample, ...]:
    """Select a deterministic balanced screen or the complete frozen benchmark."""

    if stage is Phase7Stage.FULL:
        return tuple(selection)
    grouped: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in selection:
        grouped[example.dataset].append(example)
    selected: list[BenchmarkExample] = []
    for dataset, examples in sorted(grouped.items()):
        if len(examples) < config.screen_records_per_dataset:
            raise ValueError(
                f"Dataset {dataset} has fewer than {config.screen_records_per_dataset} records."
            )
        ranked = sorted(
            examples,
            key=lambda item: (_rank(config.screening_seed, item.record_id), item.record_id),
        )
        selected.extend(ranked[: config.screen_records_per_dataset])
    return tuple(sorted(selected, key=lambda item: (item.dataset, item.record_id)))


def prediction_owner(record_id: str, world_size: int) -> int:
    """Assign one record to a stable worker independent of evaluation stage."""

    if world_size < 1:
        raise ValueError("World size must be positive.")
    digest = hashlib.sha256(record_id.encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % world_size


def select_worker_examples(
    selection: Sequence[BenchmarkExample],
    rank: int,
    world_size: int,
) -> tuple[BenchmarkExample, ...]:
    """Return the deterministic, disjoint records owned by one inference worker."""

    if rank < 0 or rank >= world_size:
        raise ValueError("Worker rank is outside the active world size.")
    return tuple(
        example for example in selection if prediction_owner(example.record_id, world_size) == rank
    )


def merge_prediction_shards(
    selection: Sequence[BenchmarkExample],
    shard_paths: Sequence[Path],
    output_path: Path,
) -> tuple[BaselinePrediction, ...]:
    """Merge disjoint resumable worker checkpoints into benchmark order."""

    selected = {example.record_id: example for example in selection}
    merged: dict[str, BaselinePrediction] = {}
    for path in shard_paths:
        for prediction in load_predictions(path):
            if prediction.record_id not in selected:
                continue
            if prediction.record_id in merged:
                raise ValueError("Prediction shards contain a duplicate record ID.")
            example = selected[prediction.record_id]
            if (
                prediction.example_id != example.example_id
                or prediction.dataset != example.dataset
                or prediction.domain != example.domain
                or prediction.split != example.split
            ):
                raise ValueError("Prediction shard identity does not match the benchmark.")
            merged[prediction.record_id] = prediction
    if set(merged) != set(selected):
        missing = len(set(selected).difference(merged))
        raise RuntimeError(f"Prediction shard coverage is incomplete: {missing} missing.")
    ordered = tuple(merged[example.record_id] for example in selection)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for prediction in ordered:
            output.write(
                json.dumps(
                    prediction.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            output.write("\n")
    temporary.replace(output_path)
    return ordered


def _delta(base: MetricSummary, adapter: MetricSummary) -> MetricDelta:
    bertscore_delta = None
    if base.bertscore_f1 is not None and adapter.bertscore_f1 is not None:
        bertscore_delta = adapter.bertscore_f1 - base.bertscore_f1
    return MetricDelta(
        rouge1_f1=adapter.rouge1_f1 - base.rouge1_f1,
        rouge2_f1=adapter.rouge2_f1 - base.rouge2_f1,
        rouge_l_f1=adapter.rouge_l_f1 - base.rouge_l_f1,
        bertscore_f1=bertscore_delta,
        mean_prediction_tokens=(adapter.mean_prediction_tokens - base.mean_prediction_tokens),
        mean_latency_seconds=adapter.mean_latency_seconds - base.mean_latency_seconds,
        empty_predictions=adapter.empty_predictions - base.empty_predictions,
        max_token_predictions=(adapter.max_token_predictions - base.max_token_predictions),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile of an empty sequence.")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _paired_intervals(
    selection: Sequence[BenchmarkExample],
    base_predictions: Mapping[str, BaselinePrediction],
    adapter_predictions: Mapping[str, BaselinePrediction],
    base_bertscore: Mapping[str, float],
    adapter_bertscore: Mapping[str, float],
    config: Phase7Config,
    stage: Phase7Stage,
) -> dict[str, ConfidenceInterval]:
    differences: dict[str, list[float]] = defaultdict(list)
    for example in selection:
        record_id = example.record_id
        base_values = prediction_pair_metrics(
            example,
            base_predictions[record_id],
            base_bertscore[record_id],
        )
        adapter_values = prediction_pair_metrics(
            example,
            adapter_predictions[record_id],
            adapter_bertscore[record_id],
        )
        for metric in ("rouge1_f1", "rouge2_f1", "rouge_l_f1", "bertscore_f1"):
            differences[metric].append(adapter_values[metric] - base_values[metric])

    randomizer = random.Random(f"{config.screening_seed}:{stage.value}")
    alpha = (1 - config.confidence_level) / 2
    intervals: dict[str, ConfidenceInterval] = {}
    for metric, values in sorted(differences.items()):
        simulations = []
        for _ in range(config.bootstrap_samples):
            simulations.append(
                sum(values[randomizer.randrange(len(values))] for _ in values) / len(values)
            )
        intervals[metric] = ConfidenceInterval(
            lower=_quantile(simulations, alpha),
            point=sum(values) / len(values),
            upper=_quantile(simulations, 1 - alpha),
        )
    return intervals


def _decision(
    stage: Phase7Stage,
    delta: MetricDelta,
    by_dataset: Mapping[str, MetricDelta],
    adapter: MetricSummary,
    config: Phase7Config,
) -> tuple[Phase7Decision, tuple[str, ...]]:
    regressions = tuple(
        dataset
        for dataset, values in sorted(by_dataset.items())
        if values.rouge_l_f1 < -config.maximum_dataset_regression
        or (
            values.bertscore_f1 is not None
            and values.bertscore_f1 < -config.maximum_dataset_regression
        )
    )
    meaningful = (
        delta.rouge_l_f1 >= config.minimum_rouge_l_gain
        or delta.rouge2_f1 >= config.minimum_rouge2_gain
        or (delta.bertscore_f1 is not None and delta.bertscore_f1 >= config.minimum_bertscore_gain)
    )
    reasons = [
        f"rouge_l_delta={delta.rouge_l_f1:.6f}",
        f"rouge2_delta={delta.rouge2_f1:.6f}",
        f"bertscore_delta={delta.bertscore_f1 if delta.bertscore_f1 is not None else 'missing'}",
        f"major_dataset_regressions={','.join(regressions) if regressions else 'none'}",
    ]
    if adapter.empty_predictions:
        reasons.append(f"adapter_empty_predictions={adapter.empty_predictions}")
        return (
            Phase7Decision.STOP if stage is Phase7Stage.SCREEN else Phase7Decision.REJECT_PILOT,
            tuple(reasons),
        )
    if stage is Phase7Stage.SCREEN:
        if meaningful and len(regressions) <= 1:
            reasons.append("screen_has_a_meaningful_gain")
            return Phase7Decision.CONTINUE, tuple(reasons)
        reasons.append("screen_did_not_meet_the_continuation_gate")
        return Phase7Decision.STOP, tuple(reasons)

    accepted = (
        delta.rouge_l_f1 >= config.minimum_rouge_l_gain
        and delta.rouge2_f1 >= 0
        and delta.bertscore_f1 is not None
        and delta.bertscore_f1 >= config.minimum_bertscore_gain
        and len(regressions) <= 1
    )
    reasons.append(
        "full_benchmark_acceptance_gate_passed"
        if accepted
        else "full_benchmark_acceptance_gate_failed"
    )
    return (
        Phase7Decision.ACCEPT_PILOT if accepted else Phase7Decision.REJECT_PILOT,
        tuple(reasons),
    )


def evaluate_adapter_comparison(
    full_selection: Sequence[BenchmarkExample],
    evaluated_selection: Sequence[BenchmarkExample],
    base_predictions: Sequence[BaselinePrediction],
    adapter_predictions: Sequence[BaselinePrediction],
    base_bertscore: Mapping[str, float],
    adapter_bertscore: Mapping[str, float],
    output_directory: Path,
    config: Phase7Config,
    stage: Phase7Stage,
    *,
    max_new_tokens: int,
    benchmark_selection_path: Path,
    base_predictions_path: Path,
    adapter_path: Path,
    adapter_predictions_path: Path,
) -> Phase7Result:
    """Create the paired report and enforce the automatic Phase 7 gate."""

    full_by_id = {example.record_id: example for example in full_selection}
    evaluated_by_id = {example.record_id: example for example in evaluated_selection}
    base_by_id = {prediction.record_id: prediction for prediction in base_predictions}
    adapter_by_id = {prediction.record_id: prediction for prediction in adapter_predictions}
    adapter_ids_unique = len(adapter_by_id) == len(adapter_predictions)
    base_coverage = set(evaluated_by_id).issubset(base_by_id)
    adapter_coverage = set(evaluated_by_id) == set(adapter_by_id)
    identities_match = all(
        record_id in full_by_id
        and prediction.example_id == full_by_id[record_id].example_id
        and prediction.dataset == full_by_id[record_id].dataset
        and prediction.domain == full_by_id[record_id].domain
        and prediction.split == full_by_id[record_id].split
        for record_id, prediction in adapter_by_id.items()
    )
    held_out = all(example.split in {"test", "ca_test"} for example in evaluated_selection)
    if not (
        len(full_by_id) == len(full_selection)
        and len(evaluated_by_id) == len(evaluated_selection)
        and base_coverage
        and adapter_coverage
        and adapter_ids_unique
        and identities_match
        and held_out
    ):
        raise RuntimeError("Phase 7 paired-comparison identity gate failed.")
    target_ids = set(evaluated_by_id)
    if set(base_bertscore) != target_ids or set(adapter_bertscore) != target_ids:
        raise ValueError("Phase 7 BERTScore coverage does not match the evaluated selection.")

    ordered_base = tuple(base_by_id[item.record_id] for item in evaluated_selection)
    base_summary = summarize_predictions(
        evaluated_selection,
        ordered_base,
        bertscore=base_bertscore,
        max_new_tokens=max_new_tokens,
    )
    adapter_summary = summarize_predictions(
        evaluated_selection,
        adapter_predictions,
        bertscore=adapter_bertscore,
        max_new_tokens=max_new_tokens,
    )
    by_dataset_examples: dict[str, list[BenchmarkExample]] = defaultdict(list)
    for example in evaluated_selection:
        by_dataset_examples[example.dataset].append(example)
    by_dataset_delta: dict[str, MetricDelta] = {}
    for dataset, examples in sorted(by_dataset_examples.items()):
        dataset_ids = {example.record_id for example in examples}
        base_subset = tuple(base_by_id[example.record_id] for example in examples)
        adapter_subset = tuple(adapter_by_id[example.record_id] for example in examples)
        base_dataset_summary = summarize_predictions(
            examples,
            base_subset,
            bertscore={key: base_bertscore[key] for key in dataset_ids},
            max_new_tokens=max_new_tokens,
        )
        adapter_dataset_summary = summarize_predictions(
            examples,
            adapter_subset,
            bertscore={key: adapter_bertscore[key] for key in dataset_ids},
            max_new_tokens=max_new_tokens,
        )
        by_dataset_delta[dataset] = _delta(base_dataset_summary, adapter_dataset_summary)

    overall_delta = _delta(base_summary, adapter_summary)
    intervals = _paired_intervals(
        evaluated_selection,
        base_by_id,
        adapter_by_id,
        base_bertscore,
        adapter_bertscore,
        config,
        stage,
    )
    decision, reasons = _decision(
        stage,
        overall_delta,
        by_dataset_delta,
        adapter_summary,
        config,
    )
    integrity = Phase7IntegrityGates(
        frozen_selection_unchanged=len(full_selection) == len(full_by_id),
        base_prediction_coverage_complete=base_coverage,
        adapter_prediction_coverage_complete=adapter_coverage,
        prediction_identities_match=identities_match,
        adapter_prediction_ids_unique=adapter_ids_unique,
        evaluation_uses_only_held_out_splits=held_out,
        adapter_empty_predictions=adapter_summary.empty_predictions,
    )
    report = Phase7Report(
        stage=stage,
        config=config,
        benchmark_selection_sha256=sha256_file(benchmark_selection_path),
        base_predictions_sha256=sha256_file(base_predictions_path),
        adapter_sha256=sha256_file(adapter_path),
        records=len(evaluated_selection),
        records_by_dataset=dict(
            sorted(Counter(item.dataset for item in evaluated_selection).items())
        ),
        base_overall=base_summary,
        adapter_overall=adapter_summary,
        overall_delta=overall_delta,
        by_dataset_delta=by_dataset_delta,
        paired_intervals=intervals,
        decision=decision,
        decision_reasons=reasons,
        integrity=integrity,
        adapter_predictions_path=adapter_predictions_path,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / f"{stage.value}-comparison-report.json"
    temporary = report_path.with_name(f"{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return Phase7Result(output_directory=output_directory, report=report)
