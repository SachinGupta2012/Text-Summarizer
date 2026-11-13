"""Canonical records shared by all summarization datasets."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DatasetName(StrEnum):
    """Datasets selected during Phase 2."""

    CNN_DAILYMAIL = "cnn_dailymail"
    GOVREPORT = "govreport"
    BILLSUM = "billsum"
    QMSUM = "qmsum"
    TWEETSUMM = "tweetsumm"


class Domain(StrEnum):
    """Product-facing summarization domains."""

    GENERAL_ARTICLE = "general_article"
    LONG_REPORT = "long_report"
    LEGAL_DOCUMENT = "legal_document"
    MEETING = "meeting"
    CUSTOMER_SUPPORT = "customer_support"


class Split(StrEnum):
    """Supported source split names."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    CA_TEST = "ca_test"


class QueryType(StrEnum):
    """How a summary request relates to its source."""

    DOCUMENT = "document"
    GENERAL_MEETING = "general_meeting"
    SPECIFIC_MEETING = "specific_meeting"
    DIALOGUE = "dialogue"


class QualityFlag(StrEnum):
    """Non-fatal conditions retained for analysis."""

    LONG_SOURCE = "long_source"
    SUMMARY_LONGER_THAN_SOURCE = "summary_longer_than_source"
    PII_MASKED = "pii_masked"
    MULTIPLE_REFERENCES = "multiple_references"
    QUERY_FOCUSED = "query_focused"


class RejectionReason(StrEnum):
    """Fatal preprocessing failures."""

    EMPTY_SOURCE = "empty_source"
    EMPTY_SUMMARY = "empty_summary"
    SOURCE_TOO_SHORT = "source_too_short"
    SUMMARY_TOO_SHORT = "summary_too_short"
    SOURCE_EQUALS_SUMMARY = "source_equals_summary"


class RawRecord(BaseModel):
    """Dataset adapter output before text normalization and validation."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetName
    domain: Domain
    split: Split
    source_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    source_text: str
    summary_text: str
    query: str | None = None
    query_type: QueryType = QueryType.DOCUMENT
    metadata: dict[str, object] = Field(default_factory=dict)


class ProcessedRecord(RawRecord):
    """Validated canonical record written to processed datasets."""

    record_id: str = Field(min_length=64, max_length=64)
    source_sha256: str = Field(min_length=64, max_length=64)
    summary_sha256: str = Field(min_length=64, max_length=64)
    content_sha256: str = Field(min_length=64, max_length=64)
    quality_flags: tuple[QualityFlag, ...] = ()


class RejectedRecord(BaseModel):
    """Record rejected by a deterministic validation rule."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetName
    split: Split
    source_id: str
    group_id: str
    reason: RejectionReason
    message: str
