"""Reproducible held-out evaluation for summarization models."""

from summarizer_ml.evaluation.benchmark import (
    SummaryGenerator,
    generate_baseline_predictions,
    load_benchmark_selection,
    prepare_benchmark_selection,
    regenerate_capped_predictions,
)
from summarizer_ml.evaluation.metrics import evaluate_baseline
from summarizer_ml.evaluation.models import (
    BaselinePrediction,
    BenchmarkExample,
    BenchmarkSelectionManifest,
    GeneratedSummary,
    MetricSummary,
    Phase5Config,
    Phase5IntegrityGates,
    Phase5Report,
    Phase5Result,
)

__all__ = [
    "BaselinePrediction",
    "BenchmarkExample",
    "BenchmarkSelectionManifest",
    "GeneratedSummary",
    "MetricSummary",
    "Phase5Config",
    "Phase5IntegrityGates",
    "Phase5Report",
    "Phase5Result",
    "SummaryGenerator",
    "evaluate_baseline",
    "generate_baseline_predictions",
    "load_benchmark_selection",
    "prepare_benchmark_selection",
    "regenerate_capped_predictions",
]
