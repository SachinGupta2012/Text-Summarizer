"""Phase 8 capped-output validation and deployable adapter packaging."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from summarizer_ml.evaluation.comparison import (
    evaluate_adapter_comparison,
    metric_delta,
    sha256_file,
)
from summarizer_ml.evaluation.metrics import (
    prediction_pair_metrics,
    summarize_predictions,
)
from summarizer_ml.evaluation.models import (
    BaselinePrediction,
    BenchmarkExample,
    CapAuditSummary,
    GenerationPolicy,
    MetricSummary,
    ModelPackageManifest,
    Phase7Decision,
    Phase7Report,
    Phase7Stage,
    Phase8Config,
    Phase8IntegrityGates,
    Phase8PolicyDecision,
    Phase8Report,
)

_WORD = re.compile(r"\w+", flags=re.UNICODE)
_COMPLETE_ENDINGS = (".", "?", "!", '"', "'", ")", "]")


def _prediction_map(
    predictions: Sequence[BaselinePrediction],
) -> dict[str, BaselinePrediction]:
    mapped = {prediction.record_id: prediction for prediction in predictions}
    if len(mapped) != len(predictions):
        raise ValueError("Predictions contain duplicate record IDs.")
    return mapped


def _write_predictions(
    selection: Sequence[BenchmarkExample],
    predictions: Mapping[str, BaselinePrediction],
    path: Path,
) -> tuple[BaselinePrediction, ...]:
    if set(predictions) != {example.record_id for example in selection}:
        raise ValueError("Prediction coverage does not match the frozen benchmark.")
    for example in selection:
        prediction = predictions[example.record_id]
        if (
            prediction.example_id != example.example_id
            or prediction.dataset != example.dataset
            or prediction.domain != example.domain
            or prediction.split != example.split
        ):
            raise ValueError("Prediction identity does not match the frozen benchmark.")
    ordered = tuple(predictions[example.record_id] for example in selection)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
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
    temporary.replace(path)
    return ordered


def _repetition_fraction(text: str, ngram_size: int) -> float:
    tokens = tuple(value.casefold() for value in _WORD.findall(text))
    if len(tokens) < ngram_size:
        return 0.0
    ngrams = tuple(
        tokens[index : index + ngram_size] for index in range(len(tokens) - ngram_size + 1)
    )
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(ngrams)


def audit_predictions(
    selection: Sequence[BenchmarkExample],
    predictions: Sequence[BaselinePrediction],
    token_limit: int,
    output_directory: Path,
    label: str,
    config: Phase8Config,
) -> CapAuditSummary:
    """Audit caps, endings, repetition, and compression without GPU work."""

    predicted = _prediction_map(predictions)
    if set(predicted) != {example.record_id for example in selection}:
        raise ValueError("Audit prediction coverage does not match the benchmark.")
    output_directory.mkdir(parents=True, exist_ok=True)
    records_path = output_directory / f"{label}-cap-audit-records.jsonl"
    capped_by_dataset: Counter[str] = Counter()
    incomplete_endings = 0
    high_repetition = 0
    ratios: list[float] = []
    capped_records = 0
    temporary = records_path.with_name(f"{records_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for example in selection:
            prediction = predicted[example.record_id]
            capped = prediction.generated_tokens >= token_limit
            repetition = _repetition_fraction(
                prediction.prediction,
                config.repetition_ngram_size,
            )
            incomplete = bool(
                prediction.prediction.strip()
            ) and not prediction.prediction.rstrip().endswith(_COMPLETE_ENDINGS)
            ratio = prediction.generated_tokens / max(example.prompt_tokens, 1)
            ratios.append(ratio)
            capped_records += int(capped)
            incomplete_endings += int(incomplete)
            high_repetition += int(repetition >= config.repetition_fraction_threshold)
            if capped:
                capped_by_dataset[example.dataset] += 1
            output.write(
                json.dumps(
                    {
                        "record_id": example.record_id,
                        "dataset": example.dataset,
                        "prompt_tokens": example.prompt_tokens,
                        "generated_tokens": prediction.generated_tokens,
                        "capped": capped,
                        "incomplete_ending": incomplete,
                        "repetition_fraction": repetition,
                        "output_to_prompt_ratio": ratio,
                    },
                    sort_keys=True,
                )
            )
            output.write("\n")
    temporary.replace(records_path)
    count = len(selection)
    return CapAuditSummary(
        records=count,
        token_limit=token_limit,
        capped_records=capped_records,
        capped_rate=capped_records / count if count else 0,
        capped_by_dataset=dict(sorted(capped_by_dataset.items())),
        incomplete_endings=incomplete_endings,
        high_repetition_records=high_repetition,
        mean_output_to_prompt_ratio=sum(ratios) / len(ratios) if ratios else 0,
        audit_records_path=records_path,
    )


def capped_record_ids(
    predictions: Sequence[BaselinePrediction],
    token_limit: int,
) -> tuple[str, ...]:
    """Return stable record IDs requiring a larger generation-budget test."""

    return tuple(
        sorted(
            prediction.record_id
            for prediction in predictions
            if prediction.generated_tokens >= token_limit
        )
    )


def overlay_repairs(
    selection: Sequence[BenchmarkExample],
    original_predictions: Sequence[BaselinePrediction],
    repair_predictions: Sequence[BaselinePrediction],
    expected_repair_ids: Sequence[str],
    output_path: Path,
) -> tuple[BaselinePrediction, ...]:
    """Replace exactly the expected capped records and preserve all other outputs."""

    original = _prediction_map(original_predictions)
    repairs = _prediction_map(repair_predictions)
    expected = set(expected_repair_ids)
    if set(repairs) != expected:
        raise ValueError("Repair prediction coverage does not match the capped record set.")
    if not expected.issubset(original):
        raise ValueError("Repair records are outside the original prediction set.")
    original.update(repairs)
    return _write_predictions(selection, original, output_path)


def _choose_policy(
    original_metrics: MetricSummary,
    candidate_metrics: MetricSummary,
    original_audit: CapAuditSummary,
    candidate_audit: CapAuditSummary,
    config: Phase8Config,
) -> tuple[Phase8PolicyDecision, tuple[str, ...]]:
    delta = metric_delta(original_metrics, candidate_metrics)
    reduces_caps = candidate_audit.capped_records < original_audit.capped_records
    quality_preserved = (
        delta.rouge_l_f1 >= -config.maximum_quality_regression
        and delta.rouge2_f1 >= -config.maximum_quality_regression
        and delta.bertscore_f1 is not None
        and delta.bertscore_f1 >= -config.maximum_quality_regression
    )
    reasons = (
        f"capped_records={original_audit.capped_records}->{candidate_audit.capped_records}",
        f"rouge_l_delta={delta.rouge_l_f1:.6f}",
        f"rouge2_delta={delta.rouge2_f1:.6f}",
        f"bertscore_delta={delta.bertscore_f1 if delta.bertscore_f1 is not None else 'missing'}",
        f"quality_preserved={quality_preserved}",
    )
    if reduces_caps and quality_preserved:
        return Phase8PolicyDecision.EXTENDED_LIMIT, reasons
    return Phase8PolicyDecision.ORIGINAL_LIMIT_WITH_CHUNKING_ROUTE, reasons


def _write_route(
    selection: Sequence[BenchmarkExample],
    predictions: Sequence[BaselinePrediction],
    limit: int,
    path: Path,
) -> tuple[str, ...]:
    predicted = _prediction_map(predictions)
    routed: list[str] = []
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for example in selection:
            prediction = predicted[example.record_id]
            if prediction.generated_tokens < limit:
                continue
            routed.append(example.record_id)
            output.write(
                json.dumps(
                    {
                        "record_id": example.record_id,
                        "dataset": example.dataset,
                        "prompt_tokens": example.prompt_tokens,
                        "generated_tokens": prediction.generated_tokens,
                        "reason": "generation_reached_token_limit",
                        "action": "hierarchical_chunking",
                    },
                    sort_keys=True,
                )
            )
            output.write("\n")
    temporary.replace(path)
    return tuple(routed)


def _write_qualitative_audit(
    selection: Sequence[BenchmarkExample],
    base_predictions: Sequence[BaselinePrediction],
    final_predictions: Sequence[BaselinePrediction],
    base_bertscore: Mapping[str, float],
    final_bertscore: Mapping[str, float],
    final_limit: int,
    path: Path,
    config: Phase8Config,
) -> None:
    base = _prediction_map(base_predictions)
    final = _prediction_map(final_predictions)
    selected: list[BenchmarkExample] = []
    datasets = sorted({example.dataset for example in selection})
    for dataset in datasets:
        candidates = [example for example in selection if example.dataset == dataset]
        candidates.sort(
            key=lambda item: (
                final[item.record_id].generated_tokens < final_limit,
                hashlib.sha256(item.record_id.encode()).hexdigest(),
            )
        )
        selected.extend(candidates[: config.qualitative_records_per_dataset])
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for example in selected:
            record_id = example.record_id
            base_values = prediction_pair_metrics(
                example,
                base[record_id],
                base_bertscore[record_id],
            )
            final_values = prediction_pair_metrics(
                example,
                final[record_id],
                final_bertscore[record_id],
            )
            user_content = next(
                (message.content for message in reversed(example.prompt) if message.role == "user"),
                "",
            )
            output.write(
                json.dumps(
                    {
                        "record_id": record_id,
                        "dataset": example.dataset,
                        "source_excerpt": user_content[:2_000],
                        "reference_summary": example.reference_summary,
                        "base_prediction": base[record_id].prediction,
                        "final_prediction": final[record_id].prediction,
                        "base_metrics": base_values,
                        "final_metrics": final_values,
                        "reached_final_limit": final[record_id].generated_tokens >= final_limit,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            output.write("\n")
    temporary.replace(path)


def finalize_phase8(
    selection: Sequence[BenchmarkExample],
    base_predictions: Sequence[BaselinePrediction],
    original_predictions: Sequence[BaselinePrediction],
    candidate_predictions: Sequence[BaselinePrediction],
    base_bertscore: Mapping[str, float],
    original_bertscore: Mapping[str, float],
    candidate_bertscore: Mapping[str, float],
    phase7_report: Phase7Report,
    output_directory: Path,
    config: Phase8Config,
    *,
    benchmark_selection_path: Path,
    base_predictions_path: Path,
    adapter_path: Path,
    phase7_report_path: Path,
    model_id: str,
) -> Phase8Report:
    """Select the safe generation policy and freeze final evaluation artifacts."""

    output_directory.mkdir(parents=True, exist_ok=True)
    adapter_hash = sha256_file(adapter_path)
    original_audit = audit_predictions(
        selection,
        original_predictions,
        config.original_max_new_tokens,
        output_directory,
        "original",
        config,
    )
    candidate_audit = audit_predictions(
        selection,
        candidate_predictions,
        config.candidate_max_new_tokens,
        output_directory,
        "candidate",
        config,
    )
    original_metrics = summarize_predictions(
        selection,
        original_predictions,
        bertscore=original_bertscore,
        max_new_tokens=config.original_max_new_tokens,
    )
    candidate_metrics = summarize_predictions(
        selection,
        candidate_predictions,
        bertscore=candidate_bertscore,
        max_new_tokens=config.candidate_max_new_tokens,
    )
    policy_decision, policy_reasons = _choose_policy(
        original_metrics,
        candidate_metrics,
        original_audit,
        candidate_audit,
        config,
    )
    if policy_decision is Phase8PolicyDecision.EXTENDED_LIMIT:
        selected_predictions = tuple(candidate_predictions)
        selected_bertscore = dict(candidate_bertscore)
        selected_metrics = candidate_metrics
        selected_limit = config.candidate_max_new_tokens
    else:
        selected_predictions = tuple(original_predictions)
        selected_bertscore = dict(original_bertscore)
        selected_metrics = original_metrics
        selected_limit = config.original_max_new_tokens

    final_predictions_path = output_directory / "final-predictions.jsonl"
    final_predictions = _write_predictions(
        selection,
        _prediction_map(selected_predictions),
        final_predictions_path,
    )
    final_bertscore_path = output_directory / "final-bertscore.jsonl"
    with final_bertscore_path.open("w", encoding="utf-8", newline="\n") as output:
        for example in selection:
            output.write(
                json.dumps(
                    {"record_id": example.record_id, "f1": selected_bertscore[example.record_id]}
                )
            )
            output.write("\n")

    route_path = output_directory / "chunking-route.jsonl"
    routed_ids = _write_route(selection, final_predictions, selected_limit, route_path)
    routed_by_dataset = Counter(
        example.dataset for example in selection if example.record_id in set(routed_ids)
    )
    policy = GenerationPolicy(
        model_id=model_id,
        adapter_sha256=adapter_hash,
        max_new_tokens=selected_limit,
        direct_input_token_limit=config.direct_input_token_limit,
        policy_decision=policy_decision,
    )
    policy_path = output_directory / "generation-policy.json"
    policy_path.write_text(policy.model_dump_json(indent=2) + "\n", encoding="utf-8")

    comparison = evaluate_adapter_comparison(
        selection,
        selection,
        base_predictions,
        final_predictions,
        base_bertscore,
        selected_bertscore,
        output_directory,
        phase7_report.config,
        Phase7Stage.FULL,
        max_new_tokens=selected_limit,
        benchmark_selection_path=benchmark_selection_path,
        base_predictions_path=base_predictions_path,
        adapter_path=adapter_path,
        adapter_predictions_path=final_predictions_path,
    )
    qualitative_path = output_directory / "qualitative-audit.jsonl"
    _write_qualitative_audit(
        selection,
        base_predictions,
        final_predictions,
        base_bertscore,
        selected_bertscore,
        selected_limit,
        qualitative_path,
        config,
    )
    expected_repairs = set(capped_record_ids(original_predictions, config.original_max_new_tokens))
    candidate_map = _prediction_map(candidate_predictions)
    original_map = _prediction_map(original_predictions)
    repaired_ids = {
        record_id
        for record_id in original_map
        if candidate_map[record_id] != original_map[record_id]
    }
    integrity = Phase8IntegrityGates(
        phase7_adapter_identity_matches=adapter_hash == phase7_report.adapter_sha256,
        original_prediction_coverage_complete=len(original_predictions) == len(selection),
        repair_coverage_complete=repaired_ids == expected_repairs,
        final_prediction_coverage_complete=len(final_predictions) == len(selection),
        benchmark_identity_preserved=(
            phase7_report.benchmark_selection_sha256 == sha256_file(benchmark_selection_path)
        ),
        final_empty_predictions=selected_metrics.empty_predictions,
        remaining_caps_routed=len(routed_ids) == selected_metrics.max_token_predictions,
        final_comparison_accepted=(comparison.report.decision is Phase7Decision.ACCEPT_PILOT),
    )
    failed = [
        key
        for key, value in integrity.model_dump().items()
        if (type(value) is bool and not value) or (type(value) is int and value != 0)
    ]
    if failed:
        raise RuntimeError(f"Phase 8 integrity gates failed: {failed}")
    report = Phase8Report(
        config=config,
        adapter_sha256=adapter_hash,
        phase7_report_sha256=sha256_file(phase7_report_path),
        records=len(selection),
        repaired_records=len(expected_repairs),
        original_audit=original_audit,
        candidate_audit=candidate_audit,
        original_metrics=original_metrics,
        candidate_metrics=candidate_metrics,
        selected_metrics=selected_metrics,
        candidate_delta_from_original=metric_delta(original_metrics, candidate_metrics),
        policy_decision=policy_decision,
        policy_reasons=policy_reasons,
        final_comparison_decision=comparison.report.decision,
        remaining_caps=len(routed_ids),
        remaining_caps_by_dataset=dict(sorted(routed_by_dataset.items())),
        integrity=integrity,
        final_predictions_path=final_predictions_path,
        chunking_route_path=route_path,
        generation_policy_path=policy_path,
    )
    report_path = output_directory / "phase8-final-report.json"
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report


def create_model_package(
    adapter_directory: Path,
    tokenizer_directory: Path,
    generation_policy_path: Path,
    phase8_report_path: Path,
    package_directory: Path,
    archive_path: Path,
    *,
    model_id: str,
    model_revision: str,
) -> ModelPackageManifest:
    """Create and hash the compact adapter package used by inference services."""

    if not adapter_directory.is_dir() or not tokenizer_directory.is_dir():
        raise FileNotFoundError("Adapter or tokenizer directory is missing.")
    source_files = tuple(
        sorted(
            [path for path in adapter_directory.iterdir() if path.is_file()]
            + [path for path in tokenizer_directory.iterdir() if path.is_file()],
            key=lambda item: item.name,
        )
    )
    if not source_files:
        raise FileNotFoundError("Adapter or tokenizer package inputs are missing.")
    if len({path.name for path in source_files}) != len(source_files):
        raise ValueError("Adapter and tokenizer inputs contain a filename collision.")
    staging_directory = package_directory.with_name(f"{package_directory.name}.tmp")
    if staging_directory.exists():
        shutil.rmtree(staging_directory)
    staging_directory.mkdir(parents=True)
    copied: list[Path] = []
    for source in source_files:
        destination = staging_directory / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    for source, name in (
        (generation_policy_path, "generation-policy.json"),
        (phase8_report_path, "evaluation-report.json"),
    ):
        destination = staging_directory / name
        shutil.copy2(source, destination)
        copied.append(destination)
    adapter_path = staging_directory / "adapter_model.safetensors"
    files = {path.name: sha256_file(path) for path in sorted(copied)}
    manifest = ModelPackageManifest(
        model_id=model_id,
        model_revision=model_revision,
        adapter_sha256=sha256_file(adapter_path),
        phase8_report_sha256=sha256_file(phase8_report_path),
        files=files,
    )
    manifest_path = staging_directory / "model-manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if package_directory.exists():
        shutil.rmtree(package_directory)
    staging_directory.replace(package_directory)
    copied = [package_directory / path.name for path in copied]
    manifest_path = package_directory / manifest_path.name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(f"{archive_path.name}.tmp")
    with tarfile.open(temporary, mode="w:gz", compresslevel=3) as archive:
        for path in (*sorted(copied), manifest_path):
            archive.add(path, arcname=f"qwen3-summarizer-adapter-v1/{path.name}", recursive=False)
    temporary.replace(archive_path)
    return manifest
