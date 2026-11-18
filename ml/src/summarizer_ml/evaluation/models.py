"""Contracts for reproducible base and fine-tuned summarization benchmarks."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from summarizer_ml.training_data.models import ChatMessage


class Phase5Config(BaseModel):
    """Frozen selection, generation, and metric policy for the baseline."""

    model_config = ConfigDict(frozen=True)

    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    benchmark_records_per_dataset: int = Field(default=100, ge=10)
    eligible_splits: tuple[str, ...] = ("test", "ca_test")
    sampling_seed: str = "qwen3-summarization-baseline-v1"
    inference_batch_size: int = Field(default=4, ge=1, le=16)
    max_new_tokens: int = Field(default=1_024, ge=64, le=2_048)
    do_sample: bool = False
    load_in_4bit: bool = True
    quantization_type: str = "nf4"
    use_double_quantization: bool = True
    compute_dtype: str = "float16"
    bertscore_model_id: str = "roberta-large"
    metric_version: str = "phase5-metrics-v1"

    @model_validator(mode="after")
    def validate_evaluation_policy(self) -> "Phase5Config":
        if "train" in self.eligible_splits or "validation" in self.eligible_splits:
            raise ValueError("The frozen benchmark must use only final held-out splits.")
        if not self.eligible_splits:
            raise ValueError("At least one held-out split is required.")
        return self


class BenchmarkExample(BaseModel):
    """One frozen held-out example used by every compared model."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    record_id: str
    dataset: str
    domain: str
    split: str
    prompt: tuple[ChatMessage, ...]
    reference_summary: str
    prompt_tokens: int
    reference_tokens: int


class BenchmarkSelectionManifest(BaseModel):
    """Identity and coverage of the deterministic held-out selection."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    config_sha256: str
    phase4_manifest_sha256: str
    eligible_records: int
    selected_records: int
    eligible_by_dataset: dict[str, int]
    selected_by_dataset: dict[str, int]
    selected_by_split: dict[str, int]
    selection_path: Path


class GeneratedSummary(BaseModel):
    """Raw output returned by a generation backend."""

    model_config = ConfigDict(frozen=True)

    text: str
    generated_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)


class BaselinePrediction(BaseModel):
    """Checkpointed prediction with reference-free runtime metadata."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    record_id: str
    dataset: str
    domain: str
    split: str
    prediction: str
    generated_tokens: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    generation_max_new_tokens: int | None = Field(default=None, ge=64)


class MetricSummary(BaseModel):
    """Macro-averaged quality and generation measurements."""

    model_config = ConfigDict(frozen=True)

    records: int
    rouge1_f1: float = Field(ge=0, le=1)
    rouge2_f1: float = Field(ge=0, le=1)
    rouge_l_f1: float = Field(ge=0, le=1)
    bertscore_f1: float | None = Field(default=None, ge=-1, le=1)
    mean_prediction_tokens: float = Field(ge=0)
    mean_reference_tokens: float = Field(ge=0)
    mean_latency_seconds: float = Field(ge=0)
    empty_predictions: int = Field(ge=0)
    max_token_predictions: int = Field(default=0, ge=0)


class Phase5IntegrityGates(BaseModel):
    """Acceptance gates required before the baseline can be frozen."""

    model_config = ConfigDict(frozen=True)

    selection_uses_only_held_out_splits: bool
    prediction_coverage_complete: bool
    prediction_ids_unique: bool
    prediction_identity_matches_selection: bool
    empty_predictions: int = Field(ge=0)


class Phase5Report(BaseModel):
    """Machine-readable baseline used for the later paired comparison."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    config: Phase5Config
    phase4_manifest_sha256: str
    model_revision: str
    selected_records: int
    overall: MetricSummary
    by_dataset: dict[str, MetricSummary]
    by_domain: dict[str, MetricSummary]
    by_split: dict[str, MetricSummary]
    integrity: Phase5IntegrityGates
    selection_path: Path
    predictions_path: Path


class Phase5Result(BaseModel):
    """Paths and final report returned by the baseline pipeline."""

    model_config = ConfigDict(frozen=True)

    output_directory: Path
    report: Phase5Report


class Phase7Stage(StrEnum):
    """Bounded evaluation stages for the pilot adapter."""

    SCREEN = "screen"
    FULL = "full"


class Phase7Decision(StrEnum):
    """Action permitted by a Phase 7 comparison gate."""

    STOP = "stop"
    CONTINUE = "continue"
    ACCEPT_PILOT = "accept_pilot"
    REJECT_PILOT = "reject_pilot"


class Phase7Config(BaseModel):
    """Frozen staged-evaluation and acceptance policy."""

    model_config = ConfigDict(frozen=True)

    screen_records_per_dataset: int = Field(default=20, ge=5)
    screening_seed: str = "qwen3-summarization-phase7-screen-v1"
    inference_batch_size: int = Field(default=4, ge=1, le=16)
    bootstrap_samples: int = Field(default=2_000, ge=200)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1)
    minimum_rouge_l_gain: float = Field(default=0.005, ge=0, le=1)
    minimum_rouge2_gain: float = Field(default=0.003, ge=0, le=1)
    minimum_bertscore_gain: float = Field(default=0.005, ge=0, le=1)
    maximum_dataset_regression: float = Field(default=0.03, ge=0, le=1)


class MetricDelta(BaseModel):
    """Fine-tuned metric minus the paired base-model metric."""

    model_config = ConfigDict(frozen=True)

    rouge1_f1: float
    rouge2_f1: float
    rouge_l_f1: float
    bertscore_f1: float | None
    mean_prediction_tokens: float
    mean_latency_seconds: float
    empty_predictions: int
    max_token_predictions: int


class ConfidenceInterval(BaseModel):
    """Deterministic paired-bootstrap interval for one metric delta."""

    model_config = ConfigDict(frozen=True)

    lower: float
    point: float
    upper: float


class Phase7IntegrityGates(BaseModel):
    """Identity and coverage checks for a paired comparison."""

    model_config = ConfigDict(frozen=True)

    frozen_selection_unchanged: bool
    base_prediction_coverage_complete: bool
    adapter_prediction_coverage_complete: bool
    prediction_identities_match: bool
    adapter_prediction_ids_unique: bool
    evaluation_uses_only_held_out_splits: bool
    adapter_empty_predictions: int = Field(ge=0)


class Phase7Report(BaseModel):
    """Paired base-versus-adapter report and automatic decision gate."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    stage: Phase7Stage
    config: Phase7Config
    benchmark_selection_sha256: str
    base_predictions_sha256: str
    adapter_sha256: str
    records: int
    records_by_dataset: dict[str, int]
    base_overall: MetricSummary
    adapter_overall: MetricSummary
    overall_delta: MetricDelta
    by_dataset_delta: dict[str, MetricDelta]
    paired_intervals: dict[str, ConfidenceInterval]
    decision: Phase7Decision
    decision_reasons: tuple[str, ...]
    integrity: Phase7IntegrityGates
    adapter_predictions_path: Path


class Phase7Result(BaseModel):
    """Paths and final report returned by Phase 7 evaluation."""

    model_config = ConfigDict(frozen=True)

    output_directory: Path
    report: Phase7Report
