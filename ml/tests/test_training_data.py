"""Tests for model-specific Qwen3 SFT dataset construction."""

import json
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

import pytest
from summarizer_ml.preprocessing.models import (
    DatasetName,
    Domain,
    ProcessedRecord,
    QueryType,
    RawRecord,
    Split,
)
from summarizer_ml.preprocessing.pipeline import preprocess_record
from summarizer_ml.preprocessing.policy import (
    Phase3IntegrityGates,
    Phase3Manifest,
    Phase3PolicyConfig,
    PolicyAction,
    PolicyDecision,
    PolicyReason,
)
from summarizer_ml.training_data import (
    ConstructionExclusion,
    ConstructionExclusionReason,
    Phase4Config,
    SourceStrategy,
    build_prompt,
    build_sft_example,
    construct_sft_dataset,
    create_phase3_handoff,
)


class WordChatTokenizer:
    """Reversible word tokenizer with a deterministic chat template."""

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.tokens: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        for word in text.split():
            if word not in self.vocabulary:
                token_id = len(self.vocabulary) + 1
                self.vocabulary[word] = token_id
                self.tokens[token_id] = word
            ids.append(self.vocabulary[word])
        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return " ".join(self.tokens[token_id] for token_id in token_ids)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int] | str:
        parts = [f"<{message['role']}> {message['content']}" for message in conversation]
        if add_generation_prompt:
            parts.append("<assistant>")
        rendered = "\n".join(parts)
        return self.encode(rendered) if tokenize else rendered

    def __call__(
        self,
        text: list[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
    ) -> dict[str, object]:
        del padding, truncation
        return {
            "input_ids": [
                self.encode(value, add_special_tokens=add_special_tokens) for value in text
            ]
        }


class InterruptingChatTokenizer(WordChatTokenizer):
    """Fail on a chosen batch-tokenization call to exercise resume behavior."""

    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self.fail_on_call = fail_on_call
        self.calls = 0

    def __call__(
        self,
        text: list[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
    ) -> dict[str, object]:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("planned tokenizer interruption")
        return super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
        )


class TensorLikeIds:
    """Small test double for tensor and NumPy chat-template outputs."""

    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[list[int]]:
        return [self.values]


class TensorChatTokenizer:
    """Tokenizer returning a tensor-like singleton batch."""

    def __init__(self) -> None:
        self.base = WordChatTokenizer()

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self.base.encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self.base.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(
        self,
        text: list[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
    ) -> dict[str, object]:
        return self.base(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
        )

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> TensorLikeIds | str:
        values = self.base.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(values, str):
            return values
        return TensorLikeIds(values)


class MappingChatTokenizer:
    """Tokenizer returning a BatchEncoding-style mapping."""

    def __init__(self) -> None:
        self.base = WordChatTokenizer()

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self.base.encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        return self.base.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def __call__(
        self,
        text: list[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
    ) -> dict[str, object]:
        return self.base(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
        )

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> dict[str, TensorLikeIds] | str:
        values = self.base.apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(values, str):
            return values
        return {"input_ids": TensorLikeIds(values)}


def _processed(
    source_id: str,
    *,
    dataset: DatasetName = DatasetName.CNN_DAILYMAIL,
    domain: Domain = Domain.GENERAL_ARTICLE,
    split: Split = Split.TRAIN,
    source_words: int = 20,
    summary_words: int = 5,
    query: str | None = None,
    query_type: QueryType = QueryType.DOCUMENT,
) -> ProcessedRecord:
    raw = RawRecord(
        dataset=dataset,
        domain=domain,
        split=split,
        source_id=source_id,
        group_id=f"{dataset.value}:{source_id}",
        source_text=" ".join(f"source-{source_id}-{index}" for index in range(source_words)),
        summary_text=" ".join(f"summary-{source_id}-{index}" for index in range(summary_words)),
        query=query,
        query_type=query_type,
    )
    result = preprocess_record(raw)
    assert isinstance(result, ProcessedRecord)
    return result


def _write_records(root: Path, records: list[ProcessedRecord]) -> None:
    handles: dict[tuple[str, str], TextIO] = {}
    try:
        for record in records:
            key = (record.dataset.value, record.split.value)
            if key not in handles:
                path = root / key[0] / f"{key[1]}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[key] = path.open("x", encoding="utf-8", newline="\n")
            output = handles[key]
            output.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
            output.write("\n")
    finally:
        for output in handles.values():
            output.close()


def _phase3_manifest(training_ready_records: int) -> Phase3Manifest:
    return Phase3Manifest(
        policy=Phase3PolicyConfig(),
        tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
        canonical_records=training_ready_records + 2,
        canonical_token_records=training_ready_records + 2,
        training_ready_records=training_ready_records,
        training_ready_token_records=training_ready_records,
        routed_records=2,
        by_action={
            "keep": training_ready_records,
            "long_target": 1,
            "requires_chunking": 1,
        },
        by_action_and_dataset={},
        by_action_and_split={},
        training_ready_by_dataset={},
        training_ready_by_split={},
        integrity=Phase3IntegrityGates(
            canonical_files_unchanged=True,
            canonical_record_count_matches=True,
            canonical_token_count_matches=True,
            decision_coverage_complete=True,
            training_record_token_ids_match=True,
            split_assignments_preserved=True,
            routed_records_absent_from_training=True,
            duplicate_record_ids=0,
            cross_split_leakage_findings=0,
            missing_token_records=0,
            unexpected_token_records=0,
            split_mismatches=0,
        ),
        source_artifacts=(),
        output_artifacts=(),
    )


def _decision(record: ProcessedRecord, action: PolicyAction) -> PolicyDecision:
    reason = PolicyReason.INPUT_EXCEEDS_DIRECT_LIMIT
    if action is PolicyAction.LONG_TARGET:
        reason = PolicyReason.SUMMARY_EXCEEDS_DIRECT_LIMIT
    return PolicyDecision(
        record_id=record.record_id,
        dataset=record.dataset.value,
        split=record.split.value,
        action=action,
        reasons=(reason,),
        input_tokens=40_000,
        summary_tokens=100,
        source_characters=len(record.source_text),
        summary_characters=len(record.summary_text),
    )


def test_prompt_is_domain_and_query_aware() -> None:
    record = _processed(
        "meeting",
        dataset=DatasetName.QMSUM,
        domain=Domain.MEETING,
        query="What budget decision was made?",
        query_type=QueryType.SPECIFIC_MEETING,
    )

    prompt = build_prompt(record, record.source_text)

    assert prompt[0].role.value == "system"
    assert "What budget decision was made?" in prompt[1].content
    assert "<source>" in prompt[1].content


def test_sft_example_compacts_source_without_truncating_completion() -> None:
    tokenizer = WordChatTokenizer()
    record = _processed("long", source_words=800, summary_words=8)
    config = Phase4Config(
        max_sequence_tokens=512,
        minimum_source_tokens=32,
        max_train_records_per_dataset=10,
        records_per_shard=100,
    )

    outcome = build_sft_example(record, tokenizer, config)

    assert not isinstance(outcome, ConstructionExclusion)
    assert outcome.source_strategy is SourceStrategy.UNIFORM_WINDOWS
    assert outcome.total_tokens <= 512
    assert outcome.completion[0].content == record.summary_text
    assert "Document window" in outcome.prompt[1].content


def test_sft_example_normalizes_chat_token_id_containers() -> None:
    for tokenizer in (TensorChatTokenizer(), MappingChatTokenizer()):
        outcome = build_sft_example(
            _processed(type(tokenizer).__name__),
            tokenizer,
            Phase4Config(
                max_sequence_tokens=512,
                minimum_source_tokens=32,
                max_train_records_per_dataset=10,
                records_per_shard=100,
            ),
        )

        assert not isinstance(outcome, ConstructionExclusion)
        assert outcome.total_tokens > 0


def test_sft_example_routes_target_that_cannot_fit() -> None:
    tokenizer = WordChatTokenizer()
    record = _processed("target", source_words=40, summary_words=500)
    config = Phase4Config(
        max_sequence_tokens=512,
        minimum_source_tokens=32,
        max_train_records_per_dataset=10,
        records_per_shard=100,
    )

    outcome = build_sft_example(record, tokenizer, config)

    assert isinstance(outcome, ConstructionExclusion)
    assert outcome.reason is ConstructionExclusionReason.INSUFFICIENT_SOURCE_BUDGET


def test_construct_sft_dataset_balances_train_and_preserves_evaluation(
    tmp_path: Path,
) -> None:
    final_directory = tmp_path / "final"
    training_ready = final_directory / "training-ready"
    ready_records = [
        _processed("cnn-1"),
        _processed("cnn-2"),
        _processed("cnn-3"),
        _processed(
            "bill-1",
            dataset=DatasetName.BILLSUM,
            domain=Domain.LEGAL_DOCUMENT,
        ),
        _processed("cnn-validation", split=Split.VALIDATION),
        _processed("cnn-validation-2", split=Split.VALIDATION),
    ]
    routed = _processed(
        "report-routed",
        dataset=DatasetName.GOVREPORT,
        domain=Domain.LONG_REPORT,
        source_words=80,
    )
    long_target = _processed(
        "report-target",
        dataset=DatasetName.GOVREPORT,
        domain=Domain.LONG_REPORT,
    )
    _write_records(training_ready / "processed", ready_records)
    _write_records(final_directory / "processed", [routed, long_target])
    (training_ready / "phase3-manifest.json").write_text(
        _phase3_manifest(len(ready_records)).model_dump_json(indent=2),
        encoding="utf-8",
    )
    route_root = training_ready / "routing"
    route_root.mkdir(parents=True)
    (route_root / "requires_chunking.jsonl").write_text(
        _decision(routed, PolicyAction.REQUIRES_CHUNKING).model_dump_json() + "\n",
        encoding="utf-8",
    )
    (route_root / "long_target.jsonl").write_text(
        _decision(long_target, PolicyAction.LONG_TARGET).model_dump_json() + "\n",
        encoding="utf-8",
    )

    progress: list[tuple[str, int]] = []
    output_directory = tmp_path / "phase4"
    config = Phase4Config(
        max_sequence_tokens=512,
        minimum_source_tokens=32,
        max_train_records_per_dataset=2,
        max_evaluation_records_per_dataset_split=1,
        construction_batch_size=2,
        records_per_shard=100,
    )
    with pytest.raises(RuntimeError, match="planned tokenizer interruption"):
        construct_sft_dataset(
            training_ready,
            output_directory,
            InterruptingChatTokenizer(fail_on_call=3),
            config,
            progress_every=2,
            progress_callback=lambda stage, count: progress.append((stage, count)),
        )
    checkpoint = json.loads(
        (tmp_path / "phase4.staging" / "build-checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["completed_records"] == 2

    progress.clear()
    result = construct_sft_dataset(
        training_ready,
        output_directory,
        WordChatTokenizer(),
        config,
        progress_every=2,
        progress_callback=lambda stage, count: progress.append((stage, count)),
    )

    assert result.manifest.candidate_records == 7
    assert result.manifest.selected_records == 5
    assert result.manifest.generated_examples == 5
    assert result.manifest.sampling_dropped_records == 2
    assert result.manifest.phase3_long_target_exclusions == 1
    assert result.manifest.construction_exclusions == 0
    assert result.manifest.selected_by_dataset == {
        "billsum": 1,
        "cnn_dailymail": 3,
        "govreport": 1,
    }
    assert result.manifest.examples_by_split == {"train": 4, "validation": 1}
    assert result.manifest.integrity.cross_split_leakage_findings == 0
    assert result.manifest.total_token_statistics.maximum <= 512
    assert progress == [
        ("candidate_resume", 7),
        ("example_resume", 2),
        ("example_build", 4),
        ("example_build", 5),
        ("finalize", 5),
    ]
    shard_lines = [
        line
        for path in (output_directory / "shards").rglob("*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(shard_lines) == 5
    first = json.loads(shard_lines[0])
    assert first["prompt"][0]["role"] == "system"
    assert first["completion"][0]["role"] == "assistant"
    repeated = construct_sft_dataset(
        training_ready,
        output_directory,
        WordChatTokenizer(),
        config,
    )
    assert repeated == result

    archive_path = tmp_path / "phase3-for-phase4.tar.gz"
    handoff = create_phase3_handoff(final_directory, archive_path)
    assert handoff.manifest.training_ready_records == 6
    assert handoff.manifest.routed_source_records == 1
    assert handoff.archive_bytes > 0
    assert len(handoff.archive_sha256) == 64
    with tarfile.open(archive_path, mode="r:gz") as archive:
        names = set(archive.getnames())
    assert "phase3-handoff/training-ready/phase3-manifest.json" in names
    assert "phase3-handoff/processed/govreport/train.jsonl" in names
