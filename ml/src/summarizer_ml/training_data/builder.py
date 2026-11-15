"""Build deterministic Qwen3 prompt-completion shards from Phase 3 artifacts."""

import hashlib
import heapq
import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from math import ceil
from numbers import Integral
from pathlib import Path
from typing import Protocol, TextIO

from summarizer_ml.preprocessing.models import ProcessedRecord, Split
from summarizer_ml.preprocessing.policy import (
    ArtifactDigest,
    Phase3Manifest,
    PolicyDecision,
)
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


class ChatTokenizer(Protocol):
    """Tokenizer surface required by the Phase 4 builder."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str: ...

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


def _select_training_records(
    training_ready_directory: Path,
    routed_record_ids: set[str],
    config: Phase4Config,
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> tuple[
    set[str],
    int,
    Counter[str],
    Counter[str],
    Counter[str],
    Counter[str],
]:
    heaps: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen_ids: set[str] = set()
    candidates_by_dataset: Counter[str] = Counter()
    candidates_by_split: Counter[str] = Counter()
    train_candidates_by_dataset: Counter[str] = Counter()
    non_train_selected_by_dataset: Counter[str] = Counter()
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
            rank = _sampling_rank(config.sampling_seed, record.record_id)
            candidate = (-rank, record.record_id)
            heap = heaps[dataset]
            if len(heap) < config.max_train_records_per_dataset:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
        else:
            non_train_selected_by_dataset[dataset] += 1
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
    selected_train_ids = {record_id for heap in heaps.values() for _, record_id in heap}
    selected_train_by_dataset = Counter(dataset for dataset, heap in heaps.items() for _ in heap)
    return (
        selected_train_ids,
        candidate_count,
        candidates_by_dataset,
        candidates_by_split,
        train_candidates_by_dataset,
        selected_train_by_dataset + non_train_selected_by_dataset,
    )


def _message_dicts(messages: tuple[ChatMessage, ...]) -> list[dict[str, str]]:
    return [{"role": message.role.value, "content": message.content} for message in messages]


def _chat_token_ids(
    tokenizer: ChatTokenizer,
    messages: tuple[ChatMessage, ...],
    *,
    add_generation_prompt: bool,
) -> tuple[int, ...]:
    raw_ids: object = tokenizer.apply_chat_template(
        _message_dicts(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(raw_ids, Mapping):
        raw_ids = raw_ids.get("input_ids")
    tolist = getattr(raw_ids, "tolist", None)
    if callable(tolist):
        raw_ids = tolist()
    if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
        raise ValueError("Tokenizer returned invalid chat-template token IDs.")
    if (
        len(raw_ids) == 1
        and isinstance(raw_ids[0], Sequence)
        and not isinstance(raw_ids[0], (str, bytes))
    ):
        raw_ids = raw_ids[0]
    token_ids: list[int] = []
    for value in raw_ids:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise ValueError("Tokenizer returned a non-integer chat-template token ID.")
        token_ids.append(int(value))
    return tuple(token_ids)


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


def _render_lengths(
    record: ProcessedRecord,
    source_text: str,
    tokenizer: ChatTokenizer,
) -> tuple[tuple[ChatMessage, ...], tuple[ChatMessage, ...], int, int, int]:
    prompt = build_prompt(record, source_text)
    completion = (ChatMessage(role=ChatRole.ASSISTANT, content=record.summary_text),)
    prompt_ids = _chat_token_ids(tokenizer, prompt, add_generation_prompt=True)
    full_ids = _chat_token_ids(
        tokenizer,
        (*prompt, *completion),
        add_generation_prompt=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "Chat template generation prompt is not a prefix of the completed conversation."
        )
    completion_tokens = len(full_ids) - len(prompt_ids)
    if completion_tokens < 1:
        raise ValueError("Chat template produced an empty completion span.")
    return prompt, completion, len(prompt_ids), completion_tokens, len(full_ids)


def _example_id(record: ProcessedRecord, config: Phase4Config) -> str:
    identity = f"{config.prompt_version}:{config.max_sequence_tokens}:{record.record_id}"
    return hashlib.sha256(identity.encode()).hexdigest()


def build_sft_example(
    record: ProcessedRecord,
    tokenizer: ChatTokenizer,
    config: Phase4Config,
) -> SftExample | ConstructionExclusion:
    """Fit one record without truncating its reference summary."""

    source_ids = tuple(tokenizer.encode(record.source_text, add_special_tokens=False))
    if not source_ids:
        raise ValueError("Processed record produced an empty source token sequence.")
    prompt, completion, prompt_tokens, completion_tokens, total_tokens = _render_lengths(
        record,
        record.source_text,
        tokenizer,
    )
    strategy = SourceStrategy.COMPLETE
    used_source_text = record.source_text

    if total_tokens > config.max_sequence_tokens:
        minimum_budget = min(config.minimum_source_tokens, len(source_ids))
        minimum_text = _uniform_windows(
            source_ids,
            minimum_budget,
            config.source_windows,
            tokenizer,
        )
        minimum_render = _render_lengths(record, minimum_text, tokenizer)
        if minimum_render[4] > config.max_sequence_tokens:
            reason = ConstructionExclusionReason.INSUFFICIENT_SOURCE_BUDGET
            if minimum_render[3] >= config.max_sequence_tokens:
                reason = ConstructionExclusionReason.TARGET_EXCEEDS_CONTEXT
            return ConstructionExclusion(
                record_id=record.record_id,
                dataset=record.dataset.value,
                split=record.split.value,
                reason=reason,
                source_tokens=len(source_ids),
                completion_tokens=minimum_render[3],
                minimum_total_tokens=minimum_render[4],
            )

        low = minimum_budget
        high = len(source_ids) - 1
        best = (minimum_budget, minimum_text, minimum_render)
        while low <= high:
            budget = (low + high) // 2
            candidate_text = _uniform_windows(
                source_ids,
                budget,
                config.source_windows,
                tokenizer,
            )
            candidate_render = _render_lengths(record, candidate_text, tokenizer)
            if candidate_render[4] <= config.max_sequence_tokens:
                best = (budget, candidate_text, candidate_render)
                low = budget + 1
            else:
                high = budget - 1
        _, used_source_text, rendered = best
        prompt, completion, prompt_tokens, completion_tokens, total_tokens = rendered
        strategy = SourceStrategy.UNIFORM_WINDOWS

    used_source_tokens = len(tokenizer.encode(used_source_text, add_special_tokens=False))
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
        original_source_tokens=len(source_ids),
        used_source_tokens=used_source_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


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


def construct_sft_dataset(
    training_ready_directory: Path,
    output_directory: Path,
    tokenizer: ChatTokenizer,
    config: Phase4Config | None = None,
    *,
    progress_every: int = 25_000,
    progress_callback: ProgressCallback | None = None,
) -> Phase4Result:
    """Construct balanced prompt-completion shards and validate every example."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive.")
    active_config = config or Phase4Config()
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
    if staging.exists():
        raise FileExistsError(f"A previous Phase 4 staging directory exists: {staging}")

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
    (
        selected_train_ids,
        candidate_count,
        candidates_by_dataset,
        candidates_by_split,
        train_candidates_by_dataset,
        selected_by_dataset,
    ) = _select_training_records(
        training_ready_directory,
        chunk_ids,
        active_config,
        progress_every,
        progress_callback,
    )
    selected_train_count = len(selected_train_ids)
    selected_count = selected_train_count + (
        candidate_count - candidates_by_split.get(Split.TRAIN.value, 0)
    )
    selected_by_split: Counter[str] = Counter(
        {split: count for split, count in candidates_by_split.items() if split != Split.TRAIN.value}
    )
    selected_by_split[Split.TRAIN.value] = selected_train_count

    staging.mkdir(parents=True)
    exclusions_root = staging / "exclusions"
    exclusions_root.mkdir()
    long_target_count = _copy_long_target_routes(
        route_root / "long_target.jsonl",
        exclusions_root / "long-target.jsonl",
    )
    if long_target_count != phase3_manifest.by_action.get("long_target", 0):
        raise RuntimeError("Phase 3 long-target route count does not match its manifest.")
    construction_exclusions_path = exclusions_root / "construction.jsonl"
    construction_exclusions_file = construction_exclusions_path.open(
        "x", encoding="utf-8", newline="\n"
    )
    writer = _ShardWriter(staging / "shards", active_config.records_per_shard)
    index_path = staging / "phase4-integrity.sqlite3"
    connection = sqlite3.connect(index_path)
    _create_integrity_index(connection)
    example_rows: list[tuple[str, str, str, str, str]] = []
    generated = 0
    selected_seen = 0
    construction_exclusions = 0
    sequence_violations = 0
    split_mismatches = 0
    examples_by_dataset: Counter[str] = Counter()
    examples_by_split: Counter[str] = Counter()
    excluded_by_split: Counter[str] = Counter()
    source_strategies: Counter[str] = Counter()
    total_lengths: list[int] = []
    try:
        for record in _iter_candidates(training_ready_directory, chunk_ids):
            if record.split is Split.TRAIN and record.record_id not in selected_train_ids:
                continue
            selected_seen += 1
            outcome = build_sft_example(record, tokenizer, active_config)
            if isinstance(outcome, ConstructionExclusion):
                construction_exclusions_file.write(
                    json.dumps(outcome.model_dump(mode="json"), sort_keys=True)
                )
                construction_exclusions_file.write("\n")
                construction_exclusions += 1
                excluded_by_split[outcome.split] += 1
            else:
                if outcome.total_tokens > active_config.max_sequence_tokens:
                    sequence_violations += 1
                if outcome.split != record.split.value:
                    split_mismatches += 1
                writer.write(outcome)
                generated += 1
                examples_by_dataset[outcome.dataset] += 1
                examples_by_split[outcome.split] += 1
                source_strategies[outcome.source_strategy.value] += 1
                total_lengths.append(outcome.total_tokens)
                example_rows.append(
                    (
                        outcome.example_id,
                        outcome.record_id,
                        record.source_sha256,
                        outcome.group_id,
                        outcome.split,
                    )
                )
                if len(example_rows) >= 1_000:
                    _flush_examples(connection, example_rows)
            _report_progress(
                progress_callback,
                "example_build",
                selected_seen,
                progress_every,
            )
        if example_rows:
            _flush_examples(connection, example_rows)
        connection.commit()
        leakage_findings = _count_leakage(connection)
    finally:
        writer.close()
        construction_exclusions_file.close()
        connection.close()
    index_path.unlink(missing_ok=True)
    if selected_seen % progress_every:
        _report_progress(
            progress_callback,
            "example_build",
            selected_seen,
            progress_every,
            final=True,
        )

    train_caps_respected = all(
        selected_by_dataset.get(dataset, 0)
        - (candidates_by_dataset[dataset] - train_candidates_by_dataset.get(dataset, 0))
        <= active_config.max_train_records_per_dataset
        for dataset in candidates_by_dataset
    )
    evaluation_preserved = all(
        examples_by_split.get(split, 0) + excluded_by_split.get(split, 0) == count
        for split, count in selected_by_split.items()
        if split != Split.TRAIN.value
    )
    integrity = Phase4IntegrityGates(
        phase3_integrity_passed=True,
        selection_coverage_complete=selected_seen == selected_count,
        generated_or_excluded_complete=(generated + construction_exclusions == selected_count),
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
        config=active_config,
        phase3_manifest_sha256=phase3_hash,
        candidate_records=candidate_count,
        selected_records=selected_count,
        generated_examples=generated,
        sampling_dropped_records=candidate_count - selected_count,
        phase3_long_target_exclusions=long_target_count,
        construction_exclusions=construction_exclusions,
        candidates_by_dataset=dict(sorted(candidates_by_dataset.items())),
        candidates_by_split=dict(sorted(candidates_by_split.items())),
        selected_by_dataset=dict(sorted(selected_by_dataset.items())),
        selected_by_split=dict(sorted(selected_by_split.items())),
        examples_by_dataset=dict(sorted(examples_by_dataset.items())),
        examples_by_split=dict(sorted(examples_by_split.items())),
        source_strategies=dict(sorted(source_strategies.items())),
        total_token_statistics=_length_statistics(total_lengths),
        integrity=integrity,
        output_artifacts=_output_digests(staging),
    )
    (staging / "phase4-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staging.rename(output_directory)
    return Phase4Result(output_directory=output_directory, manifest=manifest)
