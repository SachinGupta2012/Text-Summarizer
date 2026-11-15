"""Models for deterministic Qwen3 supervised-fine-tuning datasets."""

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from summarizer_ml.preprocessing.policy import ArtifactDigest


class ChatRole(StrEnum):
    """Roles supported by the Qwen chat template."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """One serializable chat-template message."""

    model_config = ConfigDict(frozen=True)

    role: ChatRole
    content: str = Field(min_length=1)


class SourceStrategy(StrEnum):
    """How a source document was fitted into the training context."""

    COMPLETE = "complete"
    UNIFORM_WINDOWS = "uniform_windows"


class ConstructionExclusionReason(StrEnum):
    """Reason a selected record could not become a valid SFT example."""

    INSUFFICIENT_SOURCE_BUDGET = "insufficient_source_budget"
    TARGET_EXCEEDS_CONTEXT = "target_exceeds_context"


class Phase4Config(BaseModel):
    """Model-specific construction and deterministic sampling policy."""

    model_config = ConfigDict(frozen=True)

    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    max_sequence_tokens: int = Field(default=4_096, ge=512)
    minimum_source_tokens: int = Field(default=128, ge=32)
    source_windows: int = Field(default=3, ge=1, le=5)
    max_train_records_per_dataset: int = Field(default=50_000, ge=1)
    records_per_shard: int = Field(default=10_000, ge=100)
    sampling_seed: str = Field(default="qwen3-summarization-phase4-v1", min_length=1)
    prompt_version: str = Field(default="qwen3-summarization-v1", min_length=1)

    @model_validator(mode="after")
    def validate_token_budgets(self) -> Self:
        """Keep a meaningful margin for the prompt and completion."""

        if self.minimum_source_tokens >= self.max_sequence_tokens:
            raise ValueError("Minimum source tokens must be smaller than the sequence limit.")
        return self


class SftExample(BaseModel):
    """Conversational prompt-completion example accepted by TRL SFTTrainer."""

    model_config = ConfigDict(frozen=True)

    example_id: str = Field(min_length=64, max_length=64)
    record_id: str = Field(min_length=64, max_length=64)
    source_id: str
    group_id: str
    dataset: str
    domain: str
    split: str
    query_type: str
    source_strategy: SourceStrategy
    prompt: tuple[ChatMessage, ...]
    completion: tuple[ChatMessage, ...]
    original_source_tokens: int
    used_source_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ConstructionExclusion(BaseModel):
    """Metadata-only record that cannot fit the configured SFT sequence."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    dataset: str
    split: str
    reason: ConstructionExclusionReason
    source_tokens: int
    completion_tokens: int
    minimum_total_tokens: int


class LengthStatistics(BaseModel):
    """Distribution summary for generated SFT sequence lengths."""

    model_config = ConfigDict(frozen=True)

    count: int
    minimum: int
    maximum: int
    mean: float
    p50: int
    p95: int
    p99: int


class Phase4IntegrityGates(BaseModel):
    """Machine-readable dataset-construction acceptance checks."""

    model_config = ConfigDict(frozen=True)

    phase3_integrity_passed: bool
    selection_coverage_complete: bool
    generated_or_excluded_complete: bool
    evaluation_splits_preserved: bool
    train_caps_respected: bool
    split_assignments_preserved: bool
    sequence_limit_violations: int
    duplicate_example_ids: int
    duplicate_record_ids: int
    cross_split_leakage_findings: int


class Phase4Manifest(BaseModel):
    """Frozen contract for the Qwen3 SFT dataset."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "1.0"
    config: Phase4Config
    phase3_manifest_sha256: str
    candidate_records: int
    selected_records: int
    generated_examples: int
    sampling_dropped_records: int
    phase3_long_target_exclusions: int
    construction_exclusions: int
    candidates_by_dataset: dict[str, int]
    candidates_by_split: dict[str, int]
    selected_by_dataset: dict[str, int]
    selected_by_split: dict[str, int]
    examples_by_dataset: dict[str, int]
    examples_by_split: dict[str, int]
    source_strategies: dict[str, int]
    total_token_statistics: LengthStatistics
    integrity: Phase4IntegrityGates
    output_artifacts: tuple[ArtifactDigest, ...]


class Phase4Result(BaseModel):
    """Location and manifest produced by Phase 4 construction."""

    model_config = ConfigDict(frozen=True)

    output_directory: Path
    manifest: Phase4Manifest
