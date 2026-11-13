"""End-to-end Phase 3 diagnostic preprocessing runner for Kaggle."""

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from itertools import chain
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from summarizer_ml.preprocessing.adapters import (
    adapt_billsum,
    adapt_cnn_dailymail,
    adapt_govreport,
)
from summarizer_ml.preprocessing.models import RawRecord
from summarizer_ml.preprocessing.pipeline import PreprocessingConfig
from summarizer_ml.preprocessing.sources import (
    iter_huggingface_records,
    iter_qmsum_records,
    iter_tweetsumm_records,
)
from summarizer_ml.preprocessing.streaming import (
    ProcessingReport,
    StreamingPreprocessor,
    iter_processed_records,
)
from summarizer_ml.preprocessing.tokenization import (
    TokenAnalysisReport,
    Tokenizer,
    analyze_token_lengths_stream,
)

DatasetSplits = Mapping[str, Iterable[Mapping[str, object]]]
ProgressCallback = Callable[[int], None]


class Phase3RunResult(BaseModel):
    """Reports and artifact location from one diagnostic preprocessing run."""

    model_config = ConfigDict(frozen=True)

    output_directory: Path
    processing: ProcessingReport
    token_analysis: TokenAnalysisReport


def _all_raw_records(
    cnn_dailymail: DatasetSplits,
    govreport: DatasetSplits,
    billsum: DatasetSplits,
    qmsum_root: Path,
    tweetsumm_root: Path,
    twcs_csv_path: Path,
) -> Iterable[RawRecord]:
    return chain(
        iter_huggingface_records(cnn_dailymail, adapt_cnn_dailymail),
        iter_huggingface_records(govreport, adapt_govreport),
        iter_huggingface_records(billsum, adapt_billsum),
        iter_qmsum_records(qmsum_root),
        iter_tweetsumm_records(tweetsumm_root, twcs_csv_path),
    )


def _prepare_records(
    records: Iterable[RawRecord],
    *,
    max_records_per_dataset_split: int | None,
    progress_every: int,
    progress_callback: ProgressCallback | None,
) -> Iterator[RawRecord]:
    counts: Counter[tuple[str, str]] = Counter()
    emitted = 0
    for record in records:
        key = (record.dataset.value, record.split.value)
        if (
            max_records_per_dataset_split is not None
            and counts[key] >= max_records_per_dataset_split
        ):
            continue
        counts[key] += 1
        emitted += 1
        yield record
        if progress_callback is not None and emitted % progress_every == 0:
            progress_callback(emitted)

    if progress_callback is not None:
        progress_callback(emitted)


def run_phase3_diagnostic(
    *,
    cnn_dailymail: DatasetSplits,
    govreport: DatasetSplits,
    billsum: DatasetSplits,
    qmsum_root: Path,
    tweetsumm_root: Path,
    twcs_csv_path: Path,
    tokenizer: Tokenizer,
    tokenizer_id: str,
    output_directory: Path,
    config: PreprocessingConfig | None = None,
    max_records_per_dataset_split: int | None = None,
    progress_every: int = 10_000,
    progress_callback: ProgressCallback | None = None,
) -> Phase3RunResult:
    """Normalize all five sources, audit exact leakage, and measure Qwen lengths."""

    if max_records_per_dataset_split is not None and max_records_per_dataset_split < 1:
        raise ValueError("max_records_per_dataset_split must be positive.")
    if progress_every < 1:
        raise ValueError("progress_every must be positive.")

    processor = StreamingPreprocessor(output_directory, config)
    try:
        processor.process(
            _prepare_records(
                _all_raw_records(
                    cnn_dailymail,
                    govreport,
                    billsum,
                    qmsum_root,
                    tweetsumm_root,
                    twcs_csv_path,
                ),
                max_records_per_dataset_split=max_records_per_dataset_split,
                progress_every=progress_every,
                progress_callback=progress_callback,
            )
        )
        processing_report = processor.finalize()
    finally:
        processor.close()

    token_report = analyze_token_lengths_stream(
        iter_processed_records(output_directory),
        tokenizer,
        tokenizer_id,
        output_directory / "analysis",
    )
    return Phase3RunResult(
        output_directory=output_directory,
        processing=processing_report,
        token_analysis=token_report,
    )
