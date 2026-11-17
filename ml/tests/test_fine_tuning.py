"""Tests for Phase 6 optimized QLoRA preparation."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from summarizer_ml.fine_tuning import (
    Phase6Config,
    Phase6RunMode,
    gradient_accumulation_steps,
    optimizer_steps,
    prepare_phase6_selection,
    project_runtime,
    tokenize_sft_batch,
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
        source_id=f"source-{dataset}-{index}",
        group_id=f"{dataset}:group-{index}",
        dataset=dataset,
        domain="general_article",
        split=split,
        query_type="document",
        source_strategy=SourceStrategy.COMPLETE,
        prompt=(
            ChatMessage(role=ChatRole.SYSTEM, content="Summarize faithfully."),
            ChatMessage(role=ChatRole.USER, content=f"Document {dataset} {index}"),
        ),
        completion=(ChatMessage(role=ChatRole.ASSISTANT, content=f"Summary {index}"),),
        original_source_tokens=20,
        used_source_tokens=20,
        prompt_tokens=10,
        completion_tokens=3,
        total_tokens=13,
    )


def _write_phase4(root: Path) -> tuple[SftExample, ...]:
    datasets = ("billsum", "cnn_dailymail", "govreport")
    examples = tuple(
        [
            *(
                _example(dataset, "train", index)
                for dataset in datasets
                for index in range(10)
            ),
            *(
                _example(dataset, "validation", index)
                for dataset in datasets
                for index in range(5)
            ),
            *(_example(dataset, "test", index) for dataset in datasets for index in range(2)),
        ]
    )
    for split in {example.split for example in examples}:
        path = root / "shards" / split / f"{split}-00000.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(
                json.dumps(example.model_dump(mode="json"), sort_keys=True) + "\n"
                for example in examples
                if example.split == split
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
        count=len(examples), minimum=13, maximum=13, mean=13, p50=13, p95=13, p99=13
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
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return examples


def _write_benchmark(path: Path, records: Sequence[SftExample]) -> None:
    path.write_text(
        "".join(json.dumps({"record_id": item.record_id}) + "\n" for item in records),
        encoding="utf-8",
    )


def _read_examples(path: Path) -> tuple[SftExample, ...]:
    return tuple(
        SftExample.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


class FakeTokenizer:
    """Whitespace tokenizer with a prefix-compatible chat template."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert not tokenize
        rendered = " ".join(
            f"<{message['role']}> {message['content']}" for message in conversation
        )
        if add_generation_prompt:
            rendered += " <assistant>"
        return rendered

    def __call__(
        self,
        text: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, list[list[int]]]:
        assert not add_special_tokens and not padding and not truncation
        rows: list[list[int]] = []
        for value in text:
            row: list[int] = []
            for token in value.split():
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary) + 1
                row.append(self.vocabulary[token])
            rows.append(row)
        return {"input_ids": rows}


def test_phase6_selection_is_balanced_leakage_safe_and_idempotent(tmp_path: Path) -> None:
    phase4 = tmp_path / "phase4"
    examples = _write_phase4(phase4)
    benchmark = tmp_path / "benchmark.jsonl"
    _write_benchmark(benchmark, [item for item in examples if item.split == "test"])
    config = Phase6Config(
        smoke_train_records=6,
        smoke_validation_records=3,
        calibration_train_records=9,
        calibration_validation_records=6,
        pilot_train_records=12,
        validation_records_per_dataset=2,
    )
    output = tmp_path / "selection-smoke"

    result = prepare_phase6_selection(
        phase4,
        output,
        config,
        Phase6RunMode.SMOKE,
        benchmark_selection_path=benchmark,
    )
    repeated = prepare_phase6_selection(
        phase4,
        output,
        config,
        Phase6RunMode.SMOKE,
        benchmark_selection_path=benchmark,
    )

    assert repeated == result
    assert result.manifest.train_records == 6
    assert result.manifest.validation_records == 3
    assert result.manifest.train_by_dataset == {
        "billsum": 2,
        "cnn_dailymail": 2,
        "govreport": 2,
    }
    assert all(item.split == "train" for item in _read_examples(result.manifest.train_path))
    assert all(
        item.split == "validation" for item in _read_examples(result.manifest.validation_path)
    )
    assert result.manifest.integrity.benchmark_overlap == 0

    full = prepare_phase6_selection(
        phase4,
        tmp_path / "selection-full",
        config,
        Phase6RunMode.FULL,
        benchmark_selection_path=benchmark,
    )
    assert full.manifest.train_records == 30
    assert full.manifest.validation_records == 6


def test_phase6_rejects_benchmark_training_overlap(tmp_path: Path) -> None:
    phase4 = tmp_path / "phase4"
    examples = _write_phase4(phase4)
    benchmark = tmp_path / "benchmark.jsonl"
    train_example = next(item for item in examples if item.split == "train")
    _write_benchmark(benchmark, [train_example])
    config = Phase6Config(
        smoke_train_records=30,
        calibration_train_records=30,
        pilot_train_records=30,
    )

    with pytest.raises(RuntimeError, match="benchmark_overlap"):
        prepare_phase6_selection(
            phase4,
            tmp_path / "selection",
            config,
            Phase6RunMode.SMOKE,
            benchmark_selection_path=benchmark,
        )


def test_phase6_tokenization_masks_prompt_and_runtime_math() -> None:
    example = _example("cnn_dailymail", "train", 1)
    tokenized = tokenize_sft_batch((example,), FakeTokenizer(), max_sequence_tokens=128)

    assert len(tokenized) == 1
    assert tokenized[0].length == len(tokenized[0].input_ids)
    assert tokenized[0].labels[-2:] == tokenized[0].input_ids[-2:]
    assert all(value == -100 for value in tokenized[0].labels[:-2])
    assert tokenized[0].attention_mask == (1,) * tokenized[0].length

    config = Phase6Config()
    assert gradient_accumulation_steps(config, world_size=2) == 8
    assert gradient_accumulation_steps(config, world_size=1) == 16
    assert optimizer_steps(8_000, config, world_size=2) == 500
    projection = project_runtime(20, 200, 500)
    assert projection.seconds_per_step == 10
    assert projection.projected_hours == pytest.approx(5_000 / 3_600)
