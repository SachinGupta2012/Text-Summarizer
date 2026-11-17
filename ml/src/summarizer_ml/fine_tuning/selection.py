"""Leakage-safe deterministic Phase 6 data selection."""

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from summarizer_ml.fine_tuning.models import (
    Phase6Config,
    Phase6RunMode,
    Phase6SelectionIntegrity,
    Phase6SelectionManifest,
    Phase6SelectionResult,
)
from summarizer_ml.training_data.models import Phase4Manifest, SftExample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_sha256(config: Phase6Config, run_mode: Phase6RunMode) -> str:
    payload = {"config": config.model_dump(mode="json"), "run_mode": run_mode.value}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sampling_rank(seed: str, run_mode: Phase6RunMode, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{run_mode.value}:{record_id}".encode()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _iter_examples(phase4_directory: Path) -> Iterator[SftExample]:
    shard_paths = sorted((phase4_directory / "shards").glob("*/*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError("Phase 4 example shards are missing.")
    for path in shard_paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    yield SftExample.model_validate_json(line)
                except ValueError as error:
                    raise ValueError(f"Invalid Phase 4 example at {path}:{line_number}") from error


def _phase4_integrity_passed(manifest: Phase4Manifest) -> bool:
    values = manifest.integrity.model_dump()
    return all(
        value if type(value) is bool else value == 0
        for value in values.values()
    )


def _balanced_sample(
    examples: Sequence[SftExample],
    target: int,
    seed: str,
    run_mode: Phase6RunMode,
) -> tuple[SftExample, ...]:
    if target >= len(examples):
        return tuple(sorted(examples, key=lambda item: (item.dataset, item.record_id)))
    grouped: dict[str, list[SftExample]] = defaultdict(list)
    for example in examples:
        grouped[example.dataset].append(example)
    for dataset, records in grouped.items():
        grouped[dataset] = sorted(
            records,
            key=lambda item: (_sampling_rank(seed, run_mode, item.record_id), item.record_id),
        )
    selected: list[SftExample] = []
    offsets = {dataset: 0 for dataset in grouped}
    while len(selected) < target:
        progressed = False
        for dataset in sorted(grouped):
            offset = offsets[dataset]
            if offset >= len(grouped[dataset]):
                continue
            selected.append(grouped[dataset][offset])
            offsets[dataset] += 1
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            break
    if len(selected) != target:
        raise RuntimeError("Balanced selection could not reach its requested size.")
    return tuple(sorted(selected, key=lambda item: (item.dataset, item.record_id)))


def _validation_sample(
    examples: Sequence[SftExample],
    config: Phase6Config,
    run_mode: Phase6RunMode,
) -> tuple[SftExample, ...]:
    if run_mode is Phase6RunMode.SMOKE:
        return _balanced_sample(
            examples,
            min(config.smoke_validation_records, len(examples)),
            config.sampling_seed,
            run_mode,
        )
    if run_mode is Phase6RunMode.CALIBRATION:
        return _balanced_sample(
            examples,
            min(config.calibration_validation_records, len(examples)),
            config.sampling_seed,
            run_mode,
        )
    grouped: dict[str, list[SftExample]] = defaultdict(list)
    for example in examples:
        grouped[example.dataset].append(example)
    selected: list[SftExample] = []
    for dataset in sorted(grouped):
        ranked = sorted(
            grouped[dataset],
            key=lambda item: (
                _sampling_rank(config.sampling_seed, run_mode, item.record_id),
                item.record_id,
            ),
        )
        selected.extend(ranked[: config.validation_records_per_dataset])
    return tuple(sorted(selected, key=lambda item: (item.dataset, item.record_id)))


def _training_sample(
    examples: Sequence[SftExample],
    config: Phase6Config,
    run_mode: Phase6RunMode,
) -> tuple[SftExample, ...]:
    targets = {
        Phase6RunMode.SMOKE: config.smoke_train_records,
        Phase6RunMode.CALIBRATION: config.calibration_train_records,
        Phase6RunMode.PILOT: config.pilot_train_records,
    }
    if run_mode is Phase6RunMode.FULL:
        return tuple(sorted(examples, key=lambda item: (item.dataset, item.record_id)))
    return _balanced_sample(
        examples,
        min(targets[run_mode], len(examples)),
        config.sampling_seed,
        run_mode,
    )


def _benchmark_ids(path: Path | None, required: bool) -> tuple[set[str], str | None]:
    if path is None:
        if required:
            raise FileNotFoundError("The frozen Phase 5 benchmark selection is required.")
        return set(), None
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark selection is missing: {path}")
    record_ids: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"Invalid benchmark record at line {line_number}.")
            if record_id in record_ids:
                raise ValueError("Benchmark selection contains duplicate record IDs.")
            record_ids.add(record_id)
    return record_ids, _sha256(path)


def _write_examples(path: Path, examples: Iterable[SftExample]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for example in examples:
            output.write(
                json.dumps(example.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            )
            output.write("\n")
    temporary.replace(path)


def prepare_phase6_selection(
    phase4_directory: Path,
    output_directory: Path,
    config: Phase6Config,
    run_mode: Phase6RunMode,
    *,
    benchmark_selection_path: Path | None,
) -> Phase6SelectionResult:
    """Freeze one bounded, balanced, held-out-safe training selection."""

    phase4_manifest_path = phase4_directory / "phase4-manifest.json"
    if not phase4_manifest_path.is_file():
        raise FileNotFoundError("Phase 4 manifest is missing.")
    phase4_manifest = Phase4Manifest.model_validate_json(
        phase4_manifest_path.read_text(encoding="utf-8")
    )
    if phase4_manifest.config.model_id != config.model_id:
        raise ValueError("Phase 4 and Phase 6 model IDs do not match.")
    if phase4_manifest.config.max_sequence_tokens != config.max_sequence_tokens:
        raise ValueError("Phase 4 and Phase 6 sequence limits do not match.")
    phase4_hash = _sha256(phase4_manifest_path)
    benchmark_record_ids, benchmark_hash = _benchmark_ids(
        benchmark_selection_path,
        config.require_benchmark_overlap_check,
    )
    config_hash = _config_sha256(config, run_mode)
    manifest_path = output_directory / "selection-manifest.json"
    train_path = output_directory / "train.jsonl"
    validation_path = output_directory / "validation.jsonl"
    if manifest_path.is_file() and train_path.is_file() and validation_path.is_file():
        manifest = Phase6SelectionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            manifest.config_sha256 != config_hash
            or manifest.phase4_manifest_sha256 != phase4_hash
            or manifest.benchmark_selection_sha256 != benchmark_hash
        ):
            raise ValueError("Existing Phase 6 selection used different inputs or configuration.")
        return Phase6SelectionResult(output_directory=output_directory, manifest=manifest)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError("Phase 6 selection output is incomplete or non-empty.")
    output_directory.mkdir(parents=True, exist_ok=True)

    train_candidates: list[SftExample] = []
    validation_candidates: list[SftExample] = []
    seen_records: set[str] = set()
    seen_examples: set[str] = set()
    records_unique = True
    examples_unique = True
    for example in _iter_examples(phase4_directory):
        if example.record_id in seen_records:
            records_unique = False
        if example.example_id in seen_examples:
            examples_unique = False
        seen_records.add(example.record_id)
        seen_examples.add(example.example_id)
        if example.split == "train":
            train_candidates.append(example)
        elif example.split == "validation":
            validation_candidates.append(example)
    if not train_candidates or not validation_candidates:
        raise RuntimeError("Phase 4 must contain train and validation examples.")

    selected_train = _training_sample(train_candidates, config, run_mode)
    selected_validation = _validation_sample(validation_candidates, config, run_mode)
    train_ids = {example.record_id for example in selected_train}
    validation_ids = {example.record_id for example in selected_validation}
    benchmark_overlap = train_ids.intersection(benchmark_record_ids)
    integrity = Phase6SelectionIntegrity(
        phase4_integrity_passed=_phase4_integrity_passed(phase4_manifest),
        training_uses_only_train_split=all(item.split == "train" for item in selected_train),
        validation_uses_only_validation_split=all(
            item.split == "validation" for item in selected_validation
        ),
        record_ids_unique=records_unique,
        example_ids_unique=examples_unique,
        train_validation_overlap=len(train_ids.intersection(validation_ids)),
        benchmark_overlap_checked=benchmark_selection_path is not None,
        benchmark_overlap=len(benchmark_overlap),
    )
    failed = [
        field
        for field, value in integrity.model_dump().items()
        if not (field == "benchmark_overlap_checked" and not config.require_benchmark_overlap_check)
        if (type(value) is bool and not value) or (type(value) is int and value != 0)
    ]
    if failed:
        raise RuntimeError(f"Phase 6 selection integrity failed: {failed}")

    _write_examples(train_path, selected_train)
    _write_examples(validation_path, selected_validation)
    manifest = Phase6SelectionManifest(
        run_mode=run_mode,
        config_sha256=config_hash,
        phase4_manifest_sha256=phase4_hash,
        benchmark_selection_sha256=benchmark_hash,
        train_records=len(selected_train),
        validation_records=len(selected_validation),
        train_by_dataset=dict(sorted(Counter(item.dataset for item in selected_train).items())),
        validation_by_dataset=dict(
            sorted(Counter(item.dataset for item in selected_validation).items())
        ),
        train_tokens=sum(item.total_tokens for item in selected_train),
        validation_tokens=sum(item.total_tokens for item in selected_validation),
        train_path=train_path,
        validation_path=validation_path,
        integrity=integrity,
    )
    _atomic_write(manifest_path, manifest.model_dump_json(indent=2) + "\n")
    return Phase6SelectionResult(output_directory=output_directory, manifest=manifest)

