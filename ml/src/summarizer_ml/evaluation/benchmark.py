"""Deterministic Phase 5 selection and resumable generation checkpoints."""

import hashlib
import heapq
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from summarizer_ml.evaluation.models import (
    BaselinePrediction,
    BenchmarkExample,
    BenchmarkSelectionManifest,
    GeneratedSummary,
    Phase5Config,
)
from summarizer_ml.training_data.models import Phase4Manifest, SftExample

ProgressCallback = Callable[[str, int, int], None]


class SummaryGenerator(Protocol):
    """Batch generation surface implemented by the Kaggle GPU runtime."""

    @property
    def model_revision(self) -> str: ...

    def generate(self, examples: Sequence[BenchmarkExample]) -> Sequence[GeneratedSummary]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_sha256(config: Phase5Config) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _sampling_rank(seed: str, record_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _iter_phase4_examples(phase4_directory: Path) -> Iterator[SftExample]:
    for path in sorted((phase4_directory / "shards").glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                if not line.strip():
                    continue
                try:
                    yield SftExample.model_validate_json(line)
                except ValueError as error:
                    raise ValueError(f"Invalid Phase 4 example at {path}:{line_number}") from error


def _benchmark_example(example: SftExample) -> BenchmarkExample:
    if len(example.completion) != 1:
        raise ValueError("Phase 4 benchmark example must have exactly one completion.")
    return BenchmarkExample(
        example_id=example.example_id,
        record_id=example.record_id,
        dataset=example.dataset,
        domain=example.domain,
        split=example.split,
        prompt=example.prompt,
        reference_summary=example.completion[0].content,
        prompt_tokens=example.prompt_tokens,
        reference_tokens=example.completion_tokens,
    )


def prepare_benchmark_selection(
    phase4_directory: Path,
    output_directory: Path,
    config: Phase5Config | None = None,
) -> BenchmarkSelectionManifest:
    """Freeze a balanced test-only sample without reading training references."""

    active_config = config or Phase5Config()
    config_hash = _config_sha256(active_config)
    phase4_manifest_path = phase4_directory / "phase4-manifest.json"
    if not phase4_manifest_path.is_file():
        raise FileNotFoundError("Phase 4 manifest is missing.")
    phase4_manifest = Phase4Manifest.model_validate_json(
        phase4_manifest_path.read_text(encoding="utf-8")
    )
    if phase4_manifest.config.model_id != active_config.model_id:
        raise ValueError("Phase 4 and Phase 5 model IDs do not match.")
    phase4_hash = _sha256(phase4_manifest_path)
    manifest_path = output_directory / "selection-manifest.json"
    selection_path = output_directory / "benchmark-selection.jsonl"
    if manifest_path.is_file() and selection_path.is_file():
        manifest = BenchmarkSelectionManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.config_sha256 != config_hash:
            raise ValueError("Existing benchmark selection used a different configuration.")
        if manifest.phase4_manifest_sha256 != phase4_hash:
            raise ValueError("Existing benchmark selection used different Phase 4 inputs.")
        return manifest
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError("Phase 5 selection output is incomplete or non-empty.")
    output_directory.mkdir(parents=True, exist_ok=True)

    heaps: dict[str, list[tuple[int, str, BenchmarkExample]]] = defaultdict(list)
    eligible_by_dataset: Counter[str] = Counter()
    seen_ids: set[str] = set()
    eligible_records = 0
    for example in _iter_phase4_examples(phase4_directory):
        if example.split not in active_config.eligible_splits:
            continue
        if example.record_id in seen_ids:
            raise RuntimeError("Phase 4 evaluation candidates contain duplicate record IDs.")
        seen_ids.add(example.record_id)
        eligible_records += 1
        eligible_by_dataset[example.dataset] += 1
        rank = _sampling_rank(active_config.sampling_seed, example.record_id)
        candidate = (-rank, example.record_id, _benchmark_example(example))
        heap = heaps[example.dataset]
        if len(heap) < active_config.benchmark_records_per_dataset:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)

    if not heaps:
        raise RuntimeError("Phase 4 contains no eligible held-out examples.")
    selected = sorted(
        (item[2] for heap in heaps.values() for item in heap),
        key=lambda item: (item.dataset, item.split, item.record_id),
    )
    selected_by_dataset = Counter(item.dataset for item in selected)
    selected_by_split = Counter(item.split for item in selected)
    if set(selected_by_dataset) != set(eligible_by_dataset):
        raise RuntimeError("Benchmark selection omitted an eligible dataset.")
    with selection_path.open("x", encoding="utf-8", newline="\n") as output:
        for selected_example in selected:
            output.write(
                json.dumps(
                    selected_example.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            output.write("\n")
    manifest = BenchmarkSelectionManifest(
        config_sha256=config_hash,
        phase4_manifest_sha256=phase4_hash,
        eligible_records=eligible_records,
        selected_records=len(selected),
        eligible_by_dataset=dict(sorted(eligible_by_dataset.items())),
        selected_by_dataset=dict(sorted(selected_by_dataset.items())),
        selected_by_split=dict(sorted(selected_by_split.items())),
        selection_path=selection_path,
    )
    _atomic_write(
        manifest_path,
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    return manifest


def load_benchmark_selection(path: Path) -> tuple[BenchmarkExample, ...]:
    """Load and validate a frozen benchmark selection."""

    examples: list[BenchmarkExample] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                example = BenchmarkExample.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid benchmark selection at line {line_number}.") from error
            if example.record_id in seen_ids:
                raise ValueError("Benchmark selection contains duplicate record IDs.")
            seen_ids.add(example.record_id)
            examples.append(example)
    if not examples:
        raise ValueError("Benchmark selection is empty.")
    return tuple(examples)


def _load_predictions(path: Path) -> dict[str, BaselinePrediction]:
    predictions: dict[str, BaselinePrediction] = {}
    if not path.is_file():
        return predictions
    with path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            try:
                prediction = BaselinePrediction.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid prediction checkpoint at line {line_number}.") from error
            if prediction.record_id in predictions:
                raise ValueError("Prediction checkpoint contains duplicate record IDs.")
            predictions[prediction.record_id] = prediction
    return predictions


def generate_baseline_predictions(
    selection: Sequence[BenchmarkExample],
    predictions_path: Path,
    generator: SummaryGenerator,
    config: Phase5Config,
    *,
    progress_every: int = 25,
    progress_callback: ProgressCallback | None = None,
) -> tuple[BaselinePrediction, ...]:
    """Generate missing predictions and resume safely from an append-only checkpoint."""

    if progress_every < 1:
        raise ValueError("progress_every must be positive.")
    selected_by_id = {example.record_id: example for example in selection}
    if len(selected_by_id) != len(selection):
        raise ValueError("Benchmark selection contains duplicate record IDs.")
    existing = _load_predictions(predictions_path)
    unexpected = set(existing).difference(selected_by_id)
    if unexpected:
        raise ValueError("Prediction checkpoint contains records outside the selection.")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    completed = len(existing)
    total = len(selection)
    if progress_callback is not None and completed:
        progress_callback("generation_resume", completed, total)
    missing = [example for example in selection if example.record_id not in existing]
    mode = "a" if predictions_path.exists() else "x"
    with predictions_path.open(mode, encoding="utf-8", newline="\n") as output:
        for start in range(0, len(missing), config.inference_batch_size):
            batch = missing[start : start + config.inference_batch_size]
            generated = tuple(generator.generate(batch))
            if len(generated) != len(batch):
                raise RuntimeError("Generation backend returned the wrong batch size.")
            for example, result in zip(batch, generated, strict=True):
                prediction = BaselinePrediction(
                    example_id=example.example_id,
                    record_id=example.record_id,
                    dataset=example.dataset,
                    domain=example.domain,
                    split=example.split,
                    prediction=result.text.strip(),
                    generated_tokens=result.generated_tokens,
                    latency_seconds=result.latency_seconds,
                )
                output.write(
                    json.dumps(
                        prediction.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                output.write("\n")
                existing[prediction.record_id] = prediction
            output.flush()
            completed = len(existing)
            if progress_callback is not None and (
                completed == total or completed % progress_every < len(batch)
            ):
                progress_callback("generation", completed, total)
    if set(existing) != set(selected_by_id):
        raise RuntimeError("Baseline prediction coverage is incomplete.")
    return tuple(existing[example.record_id] for example in selection)


def prediction_map(
    predictions: Sequence[BaselinePrediction],
) -> Mapping[str, BaselinePrediction]:
    """Return a validated record-keyed prediction view."""

    result = {prediction.record_id: prediction for prediction in predictions}
    if len(result) != len(predictions):
        raise ValueError("Predictions contain duplicate record IDs.")
    return result
