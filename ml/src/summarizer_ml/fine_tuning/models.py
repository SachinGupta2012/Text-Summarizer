"""Contracts for optimized, reproducible QLoRA fine-tuning."""

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Phase6RunMode(StrEnum):
    """Bounded Phase 6 execution modes."""

    SMOKE = "smoke"
    CALIBRATION = "calibration"
    PILOT = "pilot"
    FULL = "full"


class Phase6Config(BaseModel):
    """Frozen QLoRA, sampling, checkpoint, and optimization policy."""

    model_config = ConfigDict(frozen=True)

    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_sequence_tokens: int = Field(default=4_096, ge=512)
    sampling_seed: str = "qwen3-summarization-phase6-v1"
    random_seed: int = Field(default=42, ge=0)
    smoke_train_records: int = Field(default=32, ge=4)
    smoke_validation_records: int = Field(default=16, ge=1)
    smoke_steps: int = Field(default=4, ge=1)
    calibration_train_records: int = Field(default=160, ge=4)
    calibration_validation_records: int = Field(default=32, ge=1)
    calibration_steps: int = Field(default=10, ge=2)
    pilot_train_records: int = Field(default=4_000, ge=4)
    validation_records_per_dataset: int = Field(default=25, ge=1)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=1, ge=1)
    target_effective_batch_size: int = Field(default=16, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_target_modules: str = "all-linear"
    quantization_type: str = "nf4"
    use_double_quantization: bool = True
    compute_dtype: str = "float16"
    optimizer: str = "paged_adamw_8bit"
    scheduler: str = "cosine"
    logging_steps: int = Field(default=5, ge=1)
    evaluation_steps: int = Field(default=125, ge=1)
    checkpoint_steps: int = Field(default=125, ge=1)
    checkpoint_limit: int = Field(default=2, ge=1)
    tokenization_batch_size: int = Field(default=64, ge=1, le=512)
    tokenization_processes: int = Field(default=2, ge=1, le=8)
    dataloader_workers: int = Field(default=2, ge=0, le=8)
    dataloader_prefetch_factor: int = Field(default=2, ge=1, le=8)
    require_benchmark_overlap_check: bool = True

    @model_validator(mode="after")
    def validate_training_policy(self) -> Self:
        if self.smoke_train_records > self.calibration_train_records:
            raise ValueError("Smoke selection cannot exceed calibration selection.")
        if self.calibration_train_records > self.pilot_train_records:
            raise ValueError("Calibration selection cannot exceed pilot selection.")
        if self.checkpoint_steps % self.evaluation_steps:
            raise ValueError("Checkpoint steps must be divisible by evaluation steps.")
        if self.target_effective_batch_size < self.per_device_train_batch_size:
            raise ValueError("Effective batch size cannot be smaller than the micro-batch.")
        return self


class Phase6SelectionIntegrity(BaseModel):
    """Leakage and identity gates for one training selection."""

    model_config = ConfigDict(frozen=True)

    phase4_integrity_passed: bool
    training_uses_only_train_split: bool
    validation_uses_only_validation_split: bool
    record_ids_unique: bool
    example_ids_unique: bool
    train_validation_overlap: int = Field(ge=0)
    benchmark_overlap_checked: bool
    benchmark_overlap: int = Field(ge=0)


class Phase6SelectionManifest(BaseModel):
    """Frozen identity of records admitted to one Phase 6 run."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    run_mode: Phase6RunMode
    config_sha256: str
    phase4_manifest_sha256: str
    benchmark_selection_sha256: str | None
    train_records: int
    validation_records: int
    train_by_dataset: dict[str, int]
    validation_by_dataset: dict[str, int]
    train_tokens: int
    validation_tokens: int
    train_path: Path
    validation_path: Path
    integrity: Phase6SelectionIntegrity


class Phase6SelectionResult(BaseModel):
    """Paths returned by Phase 6 selection construction."""

    model_config = ConfigDict(frozen=True)

    output_directory: Path
    manifest: Phase6SelectionManifest


class RuntimeProjection(BaseModel):
    """Measured throughput projected onto a larger optimizer-step budget."""

    model_config = ConfigDict(frozen=True)

    measured_steps: int = Field(ge=1)
    measured_seconds: float = Field(gt=0)
    seconds_per_step: float = Field(gt=0)
    projected_steps: int = Field(ge=1)
    projected_seconds: float = Field(gt=0)
    projected_hours: float = Field(gt=0)


class TokenizedTrainingExample(BaseModel):
    """Completion-masked causal-language-model input."""

    model_config = ConfigDict(frozen=True)

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]
    length: int = Field(ge=1)

