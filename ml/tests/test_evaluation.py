"""Tests for the frozen and resumable Phase 5 baseline benchmark."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from summarizer_ml.evaluation import (
    BaselinePrediction,
    BenchmarkExample,
    GeneratedSummary,
    Phase5Config,
    Phase7Config,
    Phase7Decision,
    Phase7Stage,
    evaluate_adapter_comparison,
    evaluate_baseline,
    generate_baseline_predictions,
    load_benchmark_selection,
    merge_prediction_shards,
    prepare_benchmark_selection,
    regenerate_capped_predictions,
    select_phase7_examples,
    select_worker_examples,
)
from summarizer_ml.training_data import (
    ChatMessage,
    ChatRole,
    LengthStatistics,
    Phase4Config,
    Phase4IntegrityGates,
    Phase4Manifest,
    SftExample,
    SourceStrategy,
)


def _identifier(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _example(dataset: str, split: str, index: int) -> SftExample:
    record_id = _identifier(f"{dataset}:{split}:{index}")
    return SftExample(
        example_id=_identifier(f"example:{record_id}"),
        record_id=record_id,
        source_id=f"source-{index}",
        group_id=f"{dataset}:group-{index}",
        dataset=dataset,
        domain="general_article" if dataset == "cnn_dailymail" else "legal_document",
        split=split,
        query_type="document",
        source_strategy=SourceStrategy.COMPLETE,
        prompt=(
            ChatMessage(role=ChatRole.SYSTEM, content="Summarize faithfully."),
            ChatMessage(role=ChatRole.USER, content=f"Document {dataset} {index}"),
        ),
        completion=(ChatMessage(role=ChatRole.ASSISTANT, content=f"Reference summary {index}"),),
        original_source_tokens=20,
        used_source_tokens=20,
        prompt_tokens=30,
        completion_tokens=4,
        total_tokens=34,
    )


def _write_phase4(root: Path) -> None:
    examples = [
        *(_example("cnn_dailymail", "test", index) for index in range(12)),
        *(_example("billsum", "ca_test", index) for index in range(12)),
        _example("cnn_dailymail", "train", 100),
        _example("billsum", "validation", 100),
    ]
    for split in {example.split for example in examples}:
        path = root / "shards" / split / f"{split}-00000.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        selected = [example for example in examples if example.split == split]
        path.write_text(
            "".join(
                json.dumps(example.model_dump(mode="json"), sort_keys=True) + "\n"
                for example in selected
            ),
            encoding="utf-8",
        )
    integrity = Phase4IntegrityGates(
        phase3_integrity_passed=True,
        selection_coverage_complete=True,
        generated_or_excluded_complete=True,
        evaluation_splits_preserved=True,
        train_caps_respected=True,
        split_assignments_preserved=True,
        sequence_limit_violations=0,
        duplicate_example_ids=0,
        duplicate_record_ids=0,
        cross_split_leakage_findings=0,
    )
    statistics = LengthStatistics(
        count=len(examples),
        minimum=34,
        maximum=34,
        mean=34,
        p50=34,
        p95=34,
        p99=34,
    )
    manifest = Phase4Manifest(
        config=Phase4Config(),
        phase3_manifest_sha256=_identifier("phase3"),
        candidate_records=len(examples),
        selected_records=len(examples),
        generated_examples=len(examples),
        sampling_dropped_records=0,
        phase3_long_target_exclusions=0,
        construction_exclusions=0,
        candidates_by_dataset={},
        candidates_by_split={},
        selected_by_dataset={},
        selected_by_split={},
        examples_by_dataset={},
        examples_by_split={},
        source_strategies={"complete": len(examples)},
        total_token_statistics=statistics,
        integrity=integrity,
        output_artifacts=(),
    )
    (root / "phase4-manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )


class FakeGenerator:
    """Deterministic backend that can simulate an interrupted GPU run."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    @property
    def model_revision(self) -> str:
        return "fake-revision"

    def generate(self, examples: Sequence[BenchmarkExample]) -> Sequence[GeneratedSummary]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("planned generation interruption")
        return tuple(
            GeneratedSummary(
                text=example.reference_summary,
                generated_tokens=example.reference_tokens,
                latency_seconds=0.1,
            )
            for example in examples
        )


def test_phase5_selection_generation_resume_and_metrics(tmp_path: Path) -> None:
    phase4 = tmp_path / "phase4"
    output = tmp_path / "phase5"
    _write_phase4(phase4)
    config = Phase5Config(
        benchmark_records_per_dataset=10,
        inference_batch_size=4,
    )

    manifest = prepare_benchmark_selection(phase4, output, config)
    repeated = prepare_benchmark_selection(phase4, output, config)
    selection = load_benchmark_selection(output / "benchmark-selection.jsonl")

    assert repeated == manifest
    assert manifest.eligible_records == 24
    assert manifest.selected_records == 20
    assert manifest.selected_by_dataset == {"billsum": 10, "cnn_dailymail": 10}
    assert set(manifest.selected_by_split) == {"ca_test", "test"}

    predictions_path = output / "base-predictions.jsonl"
    with pytest.raises(RuntimeError, match="planned generation interruption"):
        generate_baseline_predictions(
            selection,
            predictions_path,
            FakeGenerator(fail_on_call=2),
            config,
        )
    assert len(predictions_path.read_text(encoding="utf-8").splitlines()) == 4

    progress: list[tuple[str, int, int]] = []
    generator = FakeGenerator()
    predictions = generate_baseline_predictions(
        selection,
        predictions_path,
        generator,
        config,
        progress_every=5,
        progress_callback=lambda stage, count, total: progress.append((stage, count, total)),
    )
    assert len(predictions) == 20
    assert progress[0] == ("generation_resume", 4, 20)
    assert progress[-1] == ("generation", 20, 20)

    checkpoint_rows = [
        json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in checkpoint_rows[:2]:
        row["prediction"] = "truncated legacy prediction"
        row["generated_tokens"] = 768
        row["generation_max_new_tokens"] = None
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in checkpoint_rows),
        encoding="utf-8",
    )

    repair_generator = FakeGenerator()
    predictions, repaired_ids = regenerate_capped_predictions(
        selection,
        predictions_path,
        repair_generator,
        config,
        previous_max_new_tokens=768,
    )
    assert len(repaired_ids) == 2
    assert repair_generator.calls == 1
    assert all(
        prediction.generation_max_new_tokens == config.max_new_tokens
        for prediction in predictions
        if prediction.record_id in repaired_ids
    )
    predictions, repeated_repairs = regenerate_capped_predictions(
        selection,
        predictions_path,
        repair_generator,
        config,
        previous_max_new_tokens=768,
    )
    assert repeated_repairs == ()
    assert repair_generator.calls == 1

    subset_generator = FakeGenerator()
    subset_predictions = generate_baseline_predictions(
        selection[:4],
        predictions_path,
        subset_generator,
        config,
        allow_checkpoint_superset=True,
    )
    assert len(subset_predictions) == 4
    assert subset_generator.calls == 0

    result = evaluate_baseline(
        selection,
        predictions,
        output,
        config,
        manifest.phase4_manifest_sha256,
        generator.model_revision,
        bertscore={example.record_id: 0.8 for example in selection},
    )
    assert result.report.selected_records == 20
    assert result.report.overall.rouge1_f1 == 1
    assert result.report.overall.rouge2_f1 == 1
    assert result.report.overall.rouge_l_f1 == 1
    assert result.report.overall.bertscore_f1 == pytest.approx(0.8)
    assert result.report.integrity.prediction_coverage_complete
    assert (output / "baseline-report.json").is_file()


def _prediction(example: BenchmarkExample, text: str) -> BaselinePrediction:
    return BaselinePrediction(
        example_id=example.example_id,
        record_id=example.record_id,
        dataset=example.dataset,
        domain=example.domain,
        split=example.split,
        prediction=text,
        generated_tokens=len(text.split()),
        latency_seconds=0.1,
        generation_max_new_tokens=1_024,
    )


def test_phase7_screen_shards_and_paired_acceptance(tmp_path: Path) -> None:
    phase4 = tmp_path / "phase4"
    phase5 = tmp_path / "phase5"
    phase7 = tmp_path / "phase7"
    adapter = tmp_path / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter")
    _write_phase4(phase4)
    phase5_config = Phase5Config(
        benchmark_records_per_dataset=10,
        inference_batch_size=4,
    )
    manifest = prepare_benchmark_selection(phase4, phase5, phase5_config)
    full_selection = load_benchmark_selection(manifest.selection_path)
    config = Phase7Config(
        screen_records_per_dataset=5,
        bootstrap_samples=200,
    )
    screen = select_phase7_examples(full_selection, config, Phase7Stage.SCREEN)
    assert len(screen) == 10
    rank_zero = select_worker_examples(screen, rank=0, world_size=2)
    rank_one = select_worker_examples(screen, rank=1, world_size=2)
    assert {item.record_id for item in rank_zero}.isdisjoint(item.record_id for item in rank_one)
    assert {item.record_id for item in (*rank_zero, *rank_one)} == {
        item.record_id for item in screen
    }

    base_predictions = tuple(_prediction(item, "unrelated output") for item in full_selection)
    adapter_predictions = tuple(_prediction(item, item.reference_summary) for item in screen)
    shard_paths = (phase7 / "rank-00.jsonl", phase7 / "rank-01.jsonl")
    for path, worker_examples in zip(shard_paths, (rank_zero, rank_one), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                prediction.model_dump_json() + "\n"
                for prediction in adapter_predictions
                if prediction.record_id in {item.record_id for item in worker_examples}
            ),
            encoding="utf-8",
        )
    merged_path = phase7 / "screen-predictions.jsonl"
    merged = merge_prediction_shards(screen, shard_paths, merged_path)
    assert len(merged) == len(screen)

    benchmark_path = phase5 / "benchmark-selection.jsonl"
    base_path = phase5 / "base-predictions.jsonl"
    base_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in base_predictions),
        encoding="utf-8",
    )
    screen_ids = {item.record_id for item in screen}
    result = evaluate_adapter_comparison(
        full_selection,
        screen,
        base_predictions,
        merged,
        {record_id: 0.1 for record_id in screen_ids},
        {record_id: 0.8 for record_id in screen_ids},
        phase7,
        config,
        Phase7Stage.SCREEN,
        max_new_tokens=1_024,
        benchmark_selection_path=benchmark_path,
        base_predictions_path=base_path,
        adapter_path=adapter,
        adapter_predictions_path=merged_path,
    )
    assert result.report.decision is Phase7Decision.CONTINUE
    assert result.report.overall_delta.rouge_l_f1 > 0
    assert result.report.integrity.adapter_prediction_coverage_complete
    assert (phase7 / "screen-comparison-report.json").is_file()
