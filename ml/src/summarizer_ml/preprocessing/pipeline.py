"""Deterministic record normalization, validation, and hashing."""

from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field

from summarizer_ml.preprocessing.models import (
    Domain,
    ProcessedRecord,
    QualityFlag,
    RawRecord,
    RejectedRecord,
    RejectionReason,
)
from summarizer_ml.preprocessing.text import mask_customer_support_pii, normalize_text


class PreprocessingConfig(BaseModel):
    """Quality thresholds applied consistently to every source."""

    model_config = ConfigDict(frozen=True)

    minimum_source_characters: int = Field(default=40, ge=1)
    minimum_summary_characters: int = Field(default=10, ge=1)
    long_source_characters: int = Field(default=30_000, ge=1)
    mask_customer_support_pii: bool = True


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _reject(record: RawRecord, reason: RejectionReason, message: str) -> RejectedRecord:
    return RejectedRecord(
        dataset=record.dataset,
        split=record.split,
        source_id=record.source_id,
        group_id=record.group_id,
        reason=reason,
        message=message,
    )


def preprocess_record(
    record: RawRecord,
    config: PreprocessingConfig | None = None,
) -> ProcessedRecord | RejectedRecord:
    """Normalize one adapter record and apply deterministic quality rules."""

    active_config = config or PreprocessingConfig()
    source_text = normalize_text(record.source_text)
    summary_text = normalize_text(record.summary_text)
    query = normalize_text(record.query) if record.query is not None else None
    flags: set[QualityFlag] = set()

    if active_config.mask_customer_support_pii and record.domain is Domain.CUSTOMER_SUPPORT:
        source_masking = mask_customer_support_pii(source_text)
        summary_masking = mask_customer_support_pii(summary_text)
        source_text = source_masking.text
        summary_text = summary_masking.text
        if source_masking.masked_types or summary_masking.masked_types:
            flags.add(QualityFlag.PII_MASKED)

    if not source_text:
        return _reject(record, RejectionReason.EMPTY_SOURCE, "Source is empty after normalization.")
    if not summary_text:
        return _reject(
            record, RejectionReason.EMPTY_SUMMARY, "Summary is empty after normalization."
        )
    if len(source_text) < active_config.minimum_source_characters:
        return _reject(
            record,
            RejectionReason.SOURCE_TOO_SHORT,
            f"Source has {len(source_text)} characters.",
        )
    if len(summary_text) < active_config.minimum_summary_characters:
        return _reject(
            record,
            RejectionReason.SUMMARY_TOO_SHORT,
            f"Summary has {len(summary_text)} characters.",
        )
    if source_text.casefold() == summary_text.casefold():
        return _reject(
            record,
            RejectionReason.SOURCE_EQUALS_SUMMARY,
            "Source and summary are identical after normalization.",
        )

    if len(source_text) >= active_config.long_source_characters:
        flags.add(QualityFlag.LONG_SOURCE)
    if len(summary_text) > len(source_text):
        flags.add(QualityFlag.SUMMARY_LONGER_THAN_SOURCE)
    if query:
        flags.add(QualityFlag.QUERY_FOCUSED)
    reference_count = record.metadata.get("reference_count", 1)
    if isinstance(reference_count, int) and reference_count > 1:
        flags.add(QualityFlag.MULTIPLE_REFERENCES)

    source_hash = _digest(source_text)
    summary_hash = _digest(summary_text)
    content_hash = _digest(f"{source_hash}\x1f{summary_hash}\x1f{query or ''}")
    record_id = _digest(
        f"{record.dataset.value}\x1f{record.split.value}\x1f{record.source_id}\x1f{content_hash}"
    )

    return ProcessedRecord(
        **record.model_dump(exclude={"source_text", "summary_text", "query"}),
        source_text=source_text,
        summary_text=summary_text,
        query=query,
        record_id=record_id,
        source_sha256=source_hash,
        summary_sha256=summary_hash,
        content_sha256=content_hash,
        quality_flags=tuple(sorted(flags, key=str)),
    )
