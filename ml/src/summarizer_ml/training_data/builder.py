"""Build deterministic Qwen3 prompt-completion shards from Phase 3 artifacts."""

import hashlib
import heapq
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from math import ceil
from numbers import Integral
from pathlib import Path
from typing import Protocol, TextIO

from pydantic import BaseModel, ConfigDict, Field

from summarizer_ml.preprocessing.models import ProcessedRecord, Split
from summarizer_ml.preprocessing.policy import (
    ArtifactDigest,
    Phase3Manifest,
    PolicyDecision,
)
from summarizer_ml.preprocessing.tokenization import TokenLengthRecord
from summarizer_ml.training_data.models import (
    ChatMessage,
    ChatRole,
    ConstructionExclusion,
    ConstructionExclusionReason,
    LengthStatistics,
    Phase4Config,
    Phase4IntegrityGates,
    Phase4Manifest,
    Phase4Result,
    SftExample,
    SourceStrategy,
)
from summarizer_ml.training_data.prompts import build_prompt

ProgressCallback = Callable[[str, int], None]


class _SelectionCheckpoint(BaseModel):
    """Deterministic selection result reused when Phase 4 resumes."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    config_sha256: str
    phase3_manifest_sha256: str
    selected_record_ids: tuple[str, ...]
    candidate_records: int
    candidates_by_dataset: dict[str, int]
    candidates_by_split: dict[str, int]
    train_candidates_by_dataset: dict[str, int]
    selected_train_by_dataset: dict[str, int]
    selected_by_dataset: dict[str, int]
    selected_by_split: dict[str, int]


class _BuildCheckpoint(BaseModel):
    """Atomic progress marker for resumable SFT example construction."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    config_sha256: str
    phase3_manifest_sha256: str
    selected_records: int
    completed_records: int = Field(default=0, ge=0)
    part_counts: tuple[int, ...] = ()


class _PartRecord(BaseModel):
    """One resumable result stored before final shards are assembled."""

    model_config = ConfigDict(frozen=True)

    source_sha256: str
    example: SftExample | None = None
    exclusion: ConstructionExclusion | None = None

    def model_post_init(self, __context: object) -> None:
        if (self.example is None) == (self.exclusion is None):
            raise ValueError("A Phase 4 part must contain exactly one result.")


class ChatTokenizer(Protocol):
    """Tokenizer surface required by the Phase 4 builder."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str: ...

    def __call__(
        self,
        text: list[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
    ) -> Mapping[str, object]: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> object: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_digest(path: Path, root: Path) -> ArtifactDigest:
    return ArtifactDigest(
        path=path.relative_to(root).as_posix(),
        bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _phase3_passed(manifest: Phase3Manifest) -> bool:
    values = manifest.integrity.model_dump()
    return all((value if type(value) is bool else value == 0) for value in values.values())


def _read_route_ids(path: Path) -> tuple[set[str], int]:
    record_ids: set[str] = set()
    count = 0
    with path.open(encoding="utf-8") as route_file:
        for line_number, line in enumerate(route_file, start=1):
            if not line.strip():
                continue
            try:
                decision = PolicyDecision.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid Phase 3 route at {path}:{line_number}") from error
            count += 1
            if decision.record_id in record_ids:
                raise ValueError(f"Duplicate routed record ID in {path}")
            record_ids.add(decision.record_id)
    return record_ids, count


def _iter_processed(root: Path) -> Iterator[ProcessedRecord]:
    for path in sorted(root.glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as records_file:
            for line_number, line in enumerate(records_file, start=1):
                if not line.strip():
                    continue
                try:
                    yield ProcessedRecord.model_validate_json(line)
                except ValueError as error:
                    raise ValueError(f"Invalid processed record at {path}:{line_number}") from error


def _iter_candidates(
    training_ready_directory: Path,
    routed_record_ids: set[str],
) -> Iterator[ProcessedRecord]:
    yield from _iter_processed(training_ready_directory / "processed")
    if not routed_record_ids:
        return
    canonical_root = training_ready_directory.parent / "processed"
    found: set[str] = set()
    for record in _iter_processed(canonical_root):
        if record.record_id in routed_record_ids:
            found.add(record.record_id)
            yield record
    missing = routed_record_ids.difference(found)
    if missing:
        raise RuntimeError(f"Canonical records are missing for {len(missing)} chunking routes.")


def _sampling_rank(seed: str, record_id: str) -> int:
    value = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    return int.from_bytes(value, byteorder="big", signed=False)


def _config_sha256(config: Phase4Config) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_model(path: Path, model: BaseModel) -> None:
    _atomic_write(
        path,
        json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )


def _select_records(
    training_ready_directory: Path,
    routed_record_ids: set[str],
    config: Phase4Config,
    config_hash: str,
    phase3_hash: str,
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> _SelectionCheckpoint:
    heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
    bucket_metadata: dict[str, tuple[str, str]] = {}
    seen_ids: set[str] = set()
    candidates_by_dataset: Counter[str] = Counter()
    candidates_by_split: Counter[str] = Counter()
    train_candidates_by_dataset: Counter[str] = Counter()
    candidate_count = 0
    for record in _iter_candidates(training_ready_directory, routed_record_ids):
        candidate_count += 1
        if record.record_id in seen_ids:
            raise RuntimeError("Phase 4 candidates contain duplicate record IDs.")
        seen_ids.add(record.record_id)
        dataset = record.dataset.value
        split = record.split.value
        candidates_by_dataset[dataset] += 1
        candidates_by_split[split] += 1
        if record.split is Split.TRAIN:
            train_candidates_by_dataset[dataset] += 1
            bucket = f"train:{dataset}"
            capacity = config.max_train_records_per_dataset
        else:
            bucket = f"evaluation:{dataset}:{split}"
            capacity = config.max_evaluation_records_per_dataset_split
        bucket_metadata[bucket] = (dataset, split)
        rank = _sampling_rank(config.sampling_seed, record.record_id)
        candidate = (-rank, record.record_id)
        heap = heaps[bucket]
        if len(heap) < capacity:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)
        _report_progress(
            progress_callback,
            "candidate_scan",
            candidate_count,
            progress_every,
        )
    if candidate_count % progress_every:
        _report_progress(
            progress_callback,
            "candidate_scan",
            candidate_count,
            progress_every,
            final=True,
        )
    selected_record_ids = tuple(
        sorted(record_id for heap in heaps.values() for _, record_id in heap)
    )
    selected_by_dataset: Counter[str] = Counter()
    selected_by_split: Counter[str] = Counter()
    selected_train_by_dataset: Counter[str] = Counter()
    for bucket, heap in heaps.items():
        dataset, split = bucket_metadata[bucket]
        selected_by_dataset[dataset] += len(heap)
        selected_by_split[split] += len(heap)
        if split == Split.TRAIN.value:
            selected_train_by_dataset[dataset] += len(heap)
    return _SelectionCheckpoint(
        config_sha256=config_hash,
        phase3_manifest_sha256=phase3_hash,
        selected_record_ids=selected_record_ids,
        candidate_records=candidate_count,
        candidates_by_dataset=dict(sorted(candidates_by_dataset.items())),
        candidates_by_split=dict(sorted(candidates_by_split.items())),
        train_candidates_by_dataset=dict(sorted(train_candidates_by_dataset.items())),
        selected_train_by_dataset=dict(sorted(selected_train_by_dataset.items())),
        selected_by_dataset=dict(sorted(selected_by_dataset.items())),
        selected_by_split=dict(sorted(selected_by_split.items())),
    )


def _message_dicts(messages: tuple[ChatMessage, ...]) -> list[dict[str, str]]:
    return [{"role": message.role.value, "content": message.content} for message in messages]


def _token_rows(raw_ids: object, expected_rows: int) -> tuple[tuple[int, ...], ...]:
    if isinstance(raw_ids, Mapping):
        raw_ids = raw_ids.get("input_ids")
    tolist = getattr(raw_ids, "tolist", None)
    if callable(tolist):
        raw_ids = tolist()
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
        raise ValueError("Tokenizer returned invalid token IDs.")
    if expected_rows == 1 and (not raw_ids or isinstance(raw_ids[0], Integral)):
        raw_ids = [raw_ids]
    if len(raw_ids) != expected_rows:
        raise ValueError("Tokenizer returned an unexpected number of token rows.")
    rows: list[tuple[int, ...]] = []
    for raw_row in raw_ids:
        if isinstance(raw_row, (str, bytes)) or not isinstance(raw_row, Sequence):
            raise ValueError("Tokenizer returned an invalid token row.")
        row: list[int] = []
        for value in raw_row:
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise ValueError("Tokenizer returned a non-integer token ID.")
            row.append(int(value))
        rows.append(tuple(row))
    return tuple(rows)


def _batch_token_ids(
    tokenizer: ChatTokenizer,
    texts: list[str],
) -> tuple[tuple[int, ...], ...]:
    if not texts:
        return ()
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return _token_rows(encoded, len(texts))


def _chat_text(
    tokenizer: ChatTokenizer,
    messages: tuple[ChatMessage, ...],
    *,
    add_generation_prompt: bool,
) -> str:
    rendered = tokenizer.apply_chat_template(
        _message_dicts(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Tokenizer returned invalid rendered chat text.")
    return rendered


def _uniform_windows(
    source_ids: tuple[int, ...],
    token_budget: int,
    windows: int,
    tokenizer: ChatTokenizer,
) -> str:
    if token_budget >= len(source_ids):
        return tokenizer.decode(source_ids, skip_special_tokens=True)
    active_windows = min(windows, max(1, token_budget // 32))
    base_size, remainder = divmod(token_budget, active_windows)
    sizes = [base_size + (1 if index < remainder else 0) for index in range(active_windows)]
    spans: list[tuple[int, int]] = []
    for index, size in enumerate(sizes):
        if active_windows == 1:
            start = 0
        else:
            start = round(index * (len(source_ids) - size) / (active_windows - 1))
        spans.append((start, start + size))
    sections = [
        (
            f"[Document window {index + 1} of {active_windows}]\n"
            f"{tokenizer.decode(source_ids[start:end], skip_special_tokens=True)}"
        )
        for index, (start, end) in enumerate(spans)
    ]
    return "\n\n".join(sections)


def _render_batch(
    records: Sequence[ProcessedRecord],
    source_texts: Sequence[str],
    tokenizer: ChatTokenizer,
) -> tuple[tuple[tuple[ChatMessage, ...], tuple[ChatMessage, ...], int, int, int], ...]:
    if len(records) != len(source_texts):
        raise ValueError("Records and source texts must have identical lengths.")
    prompts = tuple(
        build_prompt(record, source_text)
        for record, source_text in zip(records, source_texts, strict=True)
    )
    completions = tuple(
        (ChatMessage(role=ChatRole.ASSISTANT, content=record.summary_text),) for record in records
    )
    prompt_texts = [_chat_text(tokenizer, prompt, add_generation_prompt=True) for prompt in prompts]
    full_texts = [
        _chat_text(
            tokenizer,
            (*prompt, *completion),
            add_generation_prompt=False,
        )
        for prompt, completion in zip(prompts, completions, strict=True)
    ]
    token_rows = _batch_token_ids(tokenizer, [*prompt_texts, *full_texts])
    prompt_rows = token_rows[: len(records)]
    full_rows = token_rows[len(records) :]
    rendered: list[tuple[tuple[ChatMessage, ...], tuple[ChatMessage, ...], int, int, int]] = []
    for prompt, completion, prompt_ids, full_ids in zip(
        prompts,
        completions,
        prompt_rows,
        full_rows,
        strict=True,
    ):
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "Chat template generation prompt is not a prefix of the completed conversation."
            )
        completion_tokens = len(full_ids) - len(prompt_ids)
        if completion_tokens < 1:
            raise ValueError("Chat template produced an empty completion span.")
        rendered.append((prompt, completion, len(prompt_ids), completion_tokens, len(full_ids)))
    return tuple(rendered)


def _render_lengths(
    record: ProcessedRecord,
    source_text: str,
    tokenizer: ChatTokenizer,
) -> tuple[tuple[ChatMessage, ...], tuple[ChatMessage, ...], int, int, int]:
    return _render_batch((record,), (source_text,), tokenizer)[0]


def _example_id(record: ProcessedRecord, config: Phase4Config) -> str:
    identity = f"{config.prompt_version}:{config.max_sequence_tokens}:{record.record_id}"
    return hashlib.sha256(identity.encode()).hexdigest()


def _complete_example(
    record: ProcessedRecord,
    config: Phase4Config,
    rendered: tuple[tuple[ChatMessage, ...], tuple[ChatMessage, ...], int, int, int],
    original_source_tokens: int,
    used_source_tokens: int,
    strategy: SourceStrategy,
) -> SftExample | ConstructionExclusion:
    prompt, completion, prompt_tokens, completion_tokens, total_tokens = rendered
    return SftExample(
        example_id=_example_id(record, config),
        record_id=record.record_id,
        source_id=record.source_id,
        group_id=record.group_id,
        dataset=record.dataset.value,
        domain=record.domain.value,
        split=record.split.value,
        query_type=record.query_type.value,
        source_strategy=strategy,
        prompt=prompt,
        completion=completion,
        original_source_tokens=original_source_tokens,
        used_source_tokens=used_source_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _fit_long_source(
    record: ProcessedRecord,
    tokenizer: ChatTokenizer,
    config: Phase4Config,
    source_ids: tuple[int, ...],
    initial_render: tuple[tuple[ChatMessage, ...], tuple[ChatMessage, ...], int, int, int],
) -> SftExample | ConstructionExclusion:
    minimum_budget = min(config.minimum_source_tokens, len(source_ids))
    prompt_overhead = max(0, initial_render[2] - len(source_ids))
    available = (
        config.max_sequence_tokens
        - initial_render[3]
        - prompt_overhead
        - config.source_budget_reserve_tokens
    )
    budget = min(len(source_ids) - 1, max(minimum_budget, available))
    rendered = initial_render
    used_source_text = record.source_text

    for _ in range(config.source_fit_attempts):
        used_source_text = _uniform_windows(
            source_ids,
            budget,
            config.source_windows,
            tokenizer,
        )
        rendered = _render_lengths(record, used_source_text, tokenizer)
        if rendered[4] <= config.max_sequence_tokens:
            return _complete_example(
                record,
                config,
                rendered,
                len(source_ids),
                budget,
                SourceStrategy.UNIFORM_WINDOWS,
            )
        overflow = rendered[4] - config.max_sequence_tokens
        reduced = max(
            minimum_budget,
            budget - overflow - config.source_budget_reserve_tokens,
        )
        if reduced == budget:
            break
        budget = reduced

    if budget != minimum_budget:
        budget = minimum_budget
        used_source_text = _uniform_windows(
            source_ids,
            budget,
            config.source_windows,
            tokenizer,
        )
        rendered = _render_lengths(record, used_source_text, tokenizer)
        if rendered[4] <= config.max_sequence_tokens:
            return _complete_example(
                record,
                config,
                rendered,
                len(source_ids),
                budget,
                SourceStrategy.UNIFORM_WINDOWS,
            )

    reason = ConstructionExclusionReason.INSUFFICIENT_SOURCE_BUDGET
    if rendered[3] >= config.max_sequence_tokens:
        reason = ConstructionExclusionReason.TARGET_EXCEEDS_CONTEXT
    return ConstructionExclusion(
        record_id=record.record_id,
        dataset=record.dataset.value,
        split=record.split.value,
        reason=reason,
        source_tokens=len(source_ids),
        completion_tokens=rendered[3],
        minimum_total_tokens=rendered[4],
    )


def build_sft_examples(
    records: Sequence[ProcessedRecord],
    tokenizer: ChatTokenizer,
    config: Phase4Config,
    source_token_lengths: Mapping[str, int] | None = None,
) -> tuple[SftExample | ConstructionExclusion, ...]:
    """Build a tokenizer batch while retaining exact per-example length checks."""

    if not records:
        return ()
    rendered = _render_batch(
        records,
        tuple(record.source_text for record in records),
        tokenizer,
    )
    known_lengths = source_token_lengths or {}
    required_source_indexes = [
        index
        for index, (record, initial) in enumerate(zip(records, rendered, strict=True))
        if record.record_id not in known_lengths or initial[4] > config.max_sequence_tokens
    ]
    source_rows: dict[int, tuple[int, ...]] = {}
    if required_source_indexes:
        encoded_rows = _batch_token_ids(
            tokenizer,
            [records[index].source_text for index in required_source_indexes],
        )
        source_rows = dict(zip(required_source_indexes, encoded_rows, strict=True))

    outcomes: list[SftExample | ConstructionExclusion] = []
    for index, (record, initial) in enumerate(zip(records, rendered, strict=True)):
        source_ids = source_rows.get(index)
        original_source_tokens = known_lengths.get(record.record_id)
        if source_ids is not None:
            if not source_ids:
                raise ValueError("Processed record produced an empty source token sequence.")
            if original_source_tokens is not None and original_source_tokens != len(source_ids):
                raise ValueError("Phase 3 and Phase 4 source token lengths do not match.")
            original_source_tokens = len(source_ids)
        if original_source_tokens is None:
            raise RuntimeError("Source token length was not measured.")
        if initial[4] <= config.max_sequence_tokens:
            outcomes.append(
                _complete_example(
                    record,
                    config,
                    initial,
                    original_source_tokens,
                    original_source_tokens,
                    SourceStrategy.COMPLETE,
                )
            )
            continue
        if source_ids is None:
            raise RuntimeError("Long source token IDs were not materialized.")
        outcomes.append(_fit_long_source(record, tokenizer, config, source_ids, initial))
    return tuple(outcomes)


def build_sft_example(
    record: ProcessedRecord,
    tokenizer: ChatTokenizer,
    config: Phase4Config,
) -> SftExample | ConstructionExclusion:
    """Fit one record without truncating its reference summary."""

    return build_sft_examples((record,), tokenizer, config)[0]


class _ShardWriter:
    def __init__(self, root: Path, records_per_shard: int) -> None:
        self.root = root
        self.records_per_shard = records_per_shard
        self._files: dict[str, TextIO] = {}
        self._counts: Counter[str] = Counter()
        self._indexes: Counter[str] = Counter()

    def write(self, example: SftExample) -> None:
        split = example.split
        if self._counts[split] % self.records_per_shard == 0:
            previous = self._files.pop(split, None)
            if previous is not None:
                previous.close()
            split_root = self.root / split
            split_root.mkdir(parents=True, exist_ok=True)
            path = split_root / f"{split}-{self._indexes[split]:05d}.jsonl"
            self._files[split] = path.open("x", encoding="utf-8", newline="\n")
            self._indexes[split] += 1
        output = self._files[split]
        output.write(
            json.dumps(example.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        )
        output.write("\n")
        self._counts[split] += 1

    def close(self) -> None:
        for output in self._files.values():
            output.close()
        self._files.clear()


def _create_integrity_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE example (
            example_id TEXT PRIMARY KEY,
            record_id TEXT UNIQUE NOT NULL,
            source_sha256 TEXT NOT NULL,
            group_id TEXT NOT NULL,
            split TEXT NOT NULL
        );
        CREATE INDEX example_source ON example (source_sha256, split);
        CREATE INDEX example_group ON example (group_id, split);
        """
    )


def _flush_examples(
    connection: sqlite3.Connection,
    rows: list[tuple[str, str, str, str, str]],
) -> None:
    try:
        connection.executemany("INSERT INTO example VALUES (?, ?, ?, ?, ?)", rows)
    except sqlite3.IntegrityError as error:
        raise RuntimeError("Phase 4 produced duplicate example or record IDs.") from error
    rows.clear()


def _count_leakage(connection: sqlite3.Connection) -> int:
    source = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT source_sha256 FROM example
            GROUP BY source_sha256
            HAVING COUNT(DISTINCT split) > 1
        )
        """
    ).fetchone()
    groups = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT group_id FROM example
            GROUP BY group_id
            HAVING COUNT(DISTINCT split) > 1
        )
        """
    ).fetchone()
    return int(source[0]) + int(groups[0])


def _percentile(ordered: list[int], quantile: float) -> int:
    return ordered[max(0, ceil(quantile * len(ordered)) - 1)]


def _length_statistics(values: list[int]) -> LengthStatistics:
    if not values:
        raise ValueError("Phase 4 generated no SFT examples.")
    ordered = sorted(values)
    return LengthStatistics(
        count=len(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=sum(ordered) / len(ordered),
        p50=_percentile(ordered, 0.50),
        p95=_percentile(ordered, 0.95),
        p99=_percentile(ordered, 0.99),
    )


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


def _copy_long_target_routes(source: Path, destination: Path) -> int:
    count = 0
    with (
        source.open(encoding="utf-8") as source_file,
        destination.open("x", encoding="utf-8", newline="\n") as destination_file,
    ):
        for line in source_file:
            if not line.strip():
                continue
            PolicyDecision.model_validate_json(line)
            destination_file.write(line if line.endswith("\n") else f"{line}\n")
            count += 1
    return count


def _output_digests(staging: Path) -> tuple[ArtifactDigest, ...]:
    paths = sorted(
        path
        for path in staging.rglob("*")
        if path.is_file() and path.name not in {"phase4-manifest.json", "phase4-integrity.sqlite3"}
    )
    return tuple(_artifact_digest(path, staging) for path in paths)


def _load_source_token_lengths(
    training_ready_directory: Path,
    selected_record_ids: set[str],
) -> dict[str, int]:
    path = training_ready_directory / "analysis" / "token-lengths.jsonl"
    if not path.is_file():
        return {}
    lengths: dict[str, int] = {}
    with path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                measurement = TokenLengthRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid Phase 3 token length at {path}:{line_number}") from error
            if measurement.record_id not in selected_record_ids:
                continue
            if measurement.record_id in lengths:
                raise ValueError("Phase 3 token lengths contain duplicate record IDs.")
            lengths[measurement.record_id] = measurement.source_tokens
    return lengths


def _load_or_create_selection(
    staging: Path,
    training_ready_directory: Path,
    routed_record_ids: set[str],
    config: Phase4Config,
    config_hash: str,
    phase3_hash: str,
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> _SelectionCheckpoint:
    path = staging / "selection-checkpoint.json"
    if path.is_file():
        selection = _SelectionCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        if selection.config_sha256 != config_hash:
            raise ValueError("Phase 4 selection checkpoint used a different configuration.")
        if selection.phase3_manifest_sha256 != phase3_hash:
            raise ValueError("Phase 4 selection checkpoint used different Phase 3 inputs.")
        if progress_callback is not None:
            progress_callback("candidate_resume", selection.candidate_records)
        return selection
    selection = _select_records(
        training_ready_directory,
        routed_record_ids,
        config,
        config_hash,
        phase3_hash,
        progress_every,
        progress_callback,
    )
    _write_model(path, selection)
    return selection


def _load_or_create_build_checkpoint(
    staging: Path,
    selection: _SelectionCheckpoint,
) -> _BuildCheckpoint:
    path = staging / "build-checkpoint.json"
    parts_root = staging / "parts"
    parts_root.mkdir(exist_ok=True)
    for temporary in parts_root.glob("*.tmp"):
        temporary.unlink()
    if path.is_file():
        checkpoint = _BuildCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        if checkpoint.config_sha256 != selection.config_sha256:
            raise ValueError("Phase 4 build checkpoint used a different configuration.")
        if checkpoint.phase3_manifest_sha256 != selection.phase3_manifest_sha256:
            raise ValueError("Phase 4 build checkpoint used different Phase 3 inputs.")
        if checkpoint.selected_records != len(selection.selected_record_ids):
            raise ValueError("Phase 4 build checkpoint selection size changed.")
        if checkpoint.completed_records != sum(checkpoint.part_counts):
            raise ValueError("Phase 4 build checkpoint counts are inconsistent.")
    else:
        checkpoint = _BuildCheckpoint(
            config_sha256=selection.config_sha256,
            phase3_manifest_sha256=selection.phase3_manifest_sha256,
            selected_records=len(selection.selected_record_ids),
        )
        _write_model(path, checkpoint)

    expected_names = {f"part-{index:06d}.jsonl" for index in range(len(checkpoint.part_counts))}
    for part in parts_root.glob("part-*.jsonl"):
        if part.name not in expected_names:
            part.unlink()
    missing = [name for name in sorted(expected_names) if not (parts_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Phase 4 checkpoint is missing part files: {missing[:3]}")
    return checkpoint


def _write_part(
    path: Path,
    records: Sequence[ProcessedRecord],
    outcomes: Sequence[SftExample | ConstructionExclusion],
) -> None:
    if len(records) != len(outcomes):
        raise ValueError("Phase 4 batch records and outcomes do not match.")
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        for record, outcome in zip(records, outcomes, strict=True):
            part = _PartRecord(
                source_sha256=record.source_sha256,
                example=outcome if isinstance(outcome, SftExample) else None,
                exclusion=outcome if isinstance(outcome, ConstructionExclusion) else None,
            )
            output.write(
                json.dumps(part.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            )
            output.write("\n")
        output.flush()
    temporary.replace(path)


def _build_resumable_parts(
    staging: Path,
    training_ready_directory: Path,
    routed_record_ids: set[str],
    selection: _SelectionCheckpoint,
    checkpoint: _BuildCheckpoint,
    tokenizer: ChatTokenizer,
    config: Phase4Config,
    source_token_lengths: Mapping[str, int],
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> _BuildCheckpoint:
    selected_ids = set(selection.selected_record_ids)
    completed = checkpoint.completed_records
    if progress_callback is not None and completed:
        progress_callback("example_resume", completed)
    next_progress = ((completed // progress_every) + 1) * progress_every
    selected_seen = 0
    batch: list[ProcessedRecord] = []

    def commit_batch(active_checkpoint: _BuildCheckpoint) -> _BuildCheckpoint:
        nonlocal completed, next_progress
        outcomes = build_sft_examples(batch, tokenizer, config, source_token_lengths)
        part_index = len(active_checkpoint.part_counts)
        part_path = staging / "parts" / f"part-{part_index:06d}.jsonl"
        _write_part(part_path, batch, outcomes)
        completed += len(batch)
        updated = _BuildCheckpoint(
            config_sha256=active_checkpoint.config_sha256,
            phase3_manifest_sha256=active_checkpoint.phase3_manifest_sha256,
            selected_records=active_checkpoint.selected_records,
            completed_records=completed,
            part_counts=(*active_checkpoint.part_counts, len(batch)),
        )
        _write_model(staging / "build-checkpoint.json", updated)
        batch.clear()
        if progress_callback is not None and completed >= next_progress:
            progress_callback("example_build", completed)
            while next_progress <= completed:
                next_progress += progress_every
        return updated

    active_checkpoint = checkpoint
    for record in _iter_candidates(training_ready_directory, routed_record_ids):
        if record.record_id not in selected_ids:
            continue
        if selected_seen < checkpoint.completed_records:
            selected_seen += 1
            continue
        selected_seen += 1
        batch.append(record)
        if len(batch) >= config.construction_batch_size:
            active_checkpoint = commit_batch(active_checkpoint)
    if batch:
        active_checkpoint = commit_batch(active_checkpoint)
    if selected_seen != len(selected_ids):
        raise RuntimeError("Phase 4 selected record coverage is incomplete.")
    if completed != len(selected_ids):
        raise RuntimeError("Phase 4 resumable build did not complete every selection.")
    if progress_callback is not None and completed % progress_every:
        progress_callback("example_build", completed)
    return active_checkpoint


def _assemble_parts(
    staging: Path,
    output_directory: Path,
    training_ready_directory: Path,
    selection: _SelectionCheckpoint,
    checkpoint: _BuildCheckpoint,
    phase3_manifest: Phase3Manifest,
    phase3_hash: str,
    config: Phase4Config,
    long_target_count: int,
) -> Phase4Result:
    finalizing = staging / "finalizing"
    if finalizing.exists():
        shutil.rmtree(finalizing)
    finalizing.mkdir()
    exclusions_root = finalizing / "exclusions"
    exclusions_root.mkdir()
    copied_long_targets = _copy_long_target_routes(
        training_ready_directory / "routing" / "long_target.jsonl",
        exclusions_root / "long-target.jsonl",
    )
    if copied_long_targets != long_target_count:
        raise RuntimeError("Phase 3 long-target route count changed during Phase 4.")
    construction_file = (exclusions_root / "construction.jsonl").open(
        "x", encoding="utf-8", newline="\n"
    )
    writer = _ShardWriter(finalizing / "shards", config.records_per_shard)
    index_path = finalizing / "phase4-integrity.sqlite3"
    connection = sqlite3.connect(index_path)
    _create_integrity_index(connection)
    rows: list[tuple[str, str, str, str, str]] = []
    selected_seen = 0
    generated = 0
    construction_exclusions = 0
    sequence_violations = 0
    split_mismatches = 0
    examples_by_dataset: Counter[str] = Counter()
    examples_by_split: Counter[str] = Counter()
    excluded_by_split: Counter[str] = Counter()
    source_strategies: Counter[str] = Counter()
    total_lengths: list[int] = []
    try:
        for part_index, expected_count in enumerate(checkpoint.part_counts):
            part_path = staging / "parts" / f"part-{part_index:06d}.jsonl"
            part_count = 0
            with part_path.open(encoding="utf-8") as part_file:
                for line_number, line in enumerate(part_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        part = _PartRecord.model_validate_json(line)
                    except ValueError as error:
                        raise ValueError(
                            f"Invalid Phase 4 part at {part_path}:{line_number}"
                        ) from error
                    part_count += 1
                    selected_seen += 1
                    if part.exclusion is not None:
                        construction_file.write(
                            json.dumps(part.exclusion.model_dump(mode="json"), sort_keys=True)
                        )
                        construction_file.write("\n")
                        construction_exclusions += 1
                        excluded_by_split[part.exclusion.split] += 1
                        continue
                    if part.example is None:
                        raise RuntimeError("Phase 4 part has no outcome.")
                    example = part.example
                    if example.total_tokens > config.max_sequence_tokens:
                        sequence_violations += 1
                    writer.write(example)
                    generated += 1
                    examples_by_dataset[example.dataset] += 1
                    examples_by_split[example.split] += 1
                    source_strategies[example.source_strategy.value] += 1
                    total_lengths.append(example.total_tokens)
                    rows.append(
                        (
                            example.example_id,
                            example.record_id,
                            part.source_sha256,
                            example.group_id,
                            example.split,
                        )
                    )
                    if len(rows) >= 1_000:
                        _flush_examples(connection, rows)
            if part_count != expected_count:
                raise RuntimeError(f"Phase 4 part count changed: {part_path}")
        if rows:
            _flush_examples(connection, rows)
        connection.commit()
        leakage_findings = _count_leakage(connection)
    finally:
        writer.close()
        construction_file.close()
        connection.close()
    index_path.unlink(missing_ok=True)

    selected_by_split = Counter(selection.selected_by_split)
    evaluation_preserved = all(
        examples_by_split.get(split, 0) + excluded_by_split.get(split, 0) == count
        for split, count in selected_by_split.items()
        if split != Split.TRAIN.value
    )
    train_caps_respected = all(
        count <= config.max_train_records_per_dataset
        for count in selection.selected_train_by_dataset.values()
    )
    integrity = Phase4IntegrityGates(
        phase3_integrity_passed=_phase3_passed(phase3_manifest),
        selection_coverage_complete=selected_seen == len(selection.selected_record_ids),
        generated_or_excluded_complete=(
            generated + construction_exclusions == len(selection.selected_record_ids)
        ),
        evaluation_splits_preserved=evaluation_preserved,
        train_caps_respected=train_caps_respected,
        split_assignments_preserved=split_mismatches == 0,
        sequence_limit_violations=sequence_violations,
        duplicate_example_ids=0,
        duplicate_record_ids=0,
        cross_split_leakage_findings=leakage_findings,
    )
    failed_gates = [
        field
        for field, value in integrity.model_dump().items()
        if (type(value) is bool and not value) or (type(value) is int and value != 0)
    ]
    if failed_gates:
        raise RuntimeError(f"Phase 4 integrity gates failed: {failed_gates}")

    manifest = Phase4Manifest(
        config=config,
        phase3_manifest_sha256=phase3_hash,
        candidate_records=selection.candidate_records,
        selected_records=len(selection.selected_record_ids),
        generated_examples=generated,
        sampling_dropped_records=(selection.candidate_records - len(selection.selected_record_ids)),
        phase3_long_target_exclusions=long_target_count,
        construction_exclusions=construction_exclusions,
        candidates_by_dataset=selection.candidates_by_dataset,
        candidates_by_split=selection.candidates_by_split,
        selected_by_dataset=selection.selected_by_dataset,
        selected_by_split=selection.selected_by_split,
        examples_by_dataset=dict(sorted(examples_by_dataset.items())),
        examples_by_split=dict(sorted(examples_by_split.items())),
        source_strategies=dict(sorted(source_strategies.items())),
        total_token_statistics=_length_statistics(total_lengths),
        integrity=integrity,
        output_artifacts=_output_digests(finalizing),
    )
    (finalizing / "phase4-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    finalizing.rename(output_directory)
    shutil.rmtree(staging)
    return Phase4Result(output_directory=output_directory, manifest=manifest)


def construct_sft_dataset(
    training_ready_directory: Path,
    output_directory: Path,
    tokenizer: ChatTokenizer,
    config: Phase4Config | None = None,
    *,
    progress_every: int = 25_000,
    progress_callback: ProgressCallback | None = None,
) -> Phase4Result:
    """Construct balanced prompt-completion shards with atomic resume checkpoints."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive.")
    active_config = config or Phase4Config()
    config_hash = _config_sha256(active_config)
    manifest_path = output_directory / "phase4-manifest.json"
    phase3_manifest_path = training_ready_directory / "phase3-manifest.json"
    if not phase3_manifest_path.is_file():
        raise FileNotFoundError("Phase 3 training-ready manifest is missing.")
    phase3_hash = _sha256(phase3_manifest_path)
    if manifest_path.is_file():
        manifest = Phase4Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.config != active_config or manifest.phase3_manifest_sha256 != phase3_hash:
            raise ValueError("Existing Phase 4 output used different inputs or configuration.")
        return Phase4Result(output_directory=output_directory, manifest=manifest)
    if output_directory.exists():
        raise FileExistsError(f"Phase 4 output is incomplete or non-empty: {output_directory}")
    staging = output_directory.with_name(f"{output_directory.name}.staging")
    selection_path = staging / "selection-checkpoint.json"
    if staging.exists() and not selection_path.is_file():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    phase3_manifest = Phase3Manifest.model_validate_json(
        phase3_manifest_path.read_text(encoding="utf-8")
    )
    if phase3_manifest.tokenizer_id != active_config.model_id:
        raise ValueError("Phase 3 tokenizer and Phase 4 model IDs do not match.")
    if not _phase3_passed(phase3_manifest):
        raise RuntimeError("Phase 3 integrity gates did not pass.")

    route_root = training_ready_directory / "routing"
    chunk_ids, chunk_route_count = _read_route_ids(route_root / "requires_chunking.jsonl")
    if chunk_route_count != phase3_manifest.by_action.get("requires_chunking", 0):
        raise RuntimeError("Phase 3 chunking route count does not match its manifest.")
    selection = _load_or_create_selection(
        staging,
        training_ready_directory,
        chunk_ids,
        active_config,
        config_hash,
        phase3_hash,
        progress_every,
        progress_callback,
    )
    long_target_count = sum(
        1
        for line in (route_root / "long_target.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if long_target_count != phase3_manifest.by_action.get("long_target", 0):
        raise RuntimeError("Phase 3 long-target route count does not match its manifest.")
    checkpoint = _load_or_create_build_checkpoint(staging, selection)
    source_token_lengths = _load_source_token_lengths(
        training_ready_directory,
        set(selection.selected_record_ids),
    )
    checkpoint = _build_resumable_parts(
        staging,
        training_ready_directory,
        chunk_ids,
        selection,
        checkpoint,
        tokenizer,
        active_config,
        source_token_lengths,
        progress_every,
        progress_callback,
    )
    if progress_callback is not None:
        progress_callback("finalize", checkpoint.completed_records)
    return _assemble_parts(
        staging,
        output_directory,
        training_ready_directory,
        selection,
        checkpoint,
        phase3_manifest,
        phase3_hash,
        active_config,
        long_target_count,
    )
