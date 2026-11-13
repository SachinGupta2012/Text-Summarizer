"""Exact deduplication and cross-split leakage detection."""

from collections import defaultdict
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from summarizer_ml.preprocessing.models import ProcessedRecord, Split


class LeakageKind(StrEnum):
    """Keys that must remain inside one split."""

    SOURCE_HASH = "source_hash"
    GROUP_ID = "group_id"


class ExactDuplicate(BaseModel):
    """An exact duplicate removed from the same split."""

    model_config = ConfigDict(frozen=True)

    kept_record_id: str
    duplicate_record_id: str
    split: Split


class DeduplicationResult(BaseModel):
    """Exact-deduplication output and audit trail."""

    model_config = ConfigDict(frozen=True)

    kept_records: tuple[ProcessedRecord, ...]
    duplicates: tuple[ExactDuplicate, ...]


class LeakageFinding(BaseModel):
    """Source or group observed in more than one official split."""

    model_config = ConfigDict(frozen=True)

    kind: LeakageKind
    key: str
    splits: tuple[Split, ...]
    record_ids: tuple[str, ...]


def deduplicate_exact(records: list[ProcessedRecord]) -> DeduplicationResult:
    """Remove identical source, summary, and query records within each split."""

    seen: dict[tuple[Split, str], ProcessedRecord] = {}
    kept: list[ProcessedRecord] = []
    duplicates: list[ExactDuplicate] = []
    for record in records:
        key = (record.split, record.content_sha256)
        existing = seen.get(key)
        if existing is None:
            seen[key] = record
            kept.append(record)
            continue
        duplicates.append(
            ExactDuplicate(
                kept_record_id=existing.record_id,
                duplicate_record_id=record.record_id,
                split=record.split,
            )
        )
    return DeduplicationResult(kept_records=tuple(kept), duplicates=tuple(duplicates))


def find_split_leakage(records: list[ProcessedRecord]) -> tuple[LeakageFinding, ...]:
    """Find source hashes or group IDs that cross official split boundaries."""

    findings: list[LeakageFinding] = []
    grouped_by_kind: dict[LeakageKind, dict[str, list[ProcessedRecord]]] = {
        LeakageKind.SOURCE_HASH: defaultdict(list),
        LeakageKind.GROUP_ID: defaultdict(list),
    }
    for record in records:
        grouped_by_kind[LeakageKind.SOURCE_HASH][record.source_sha256].append(record)
        grouped_by_kind[LeakageKind.GROUP_ID][record.group_id].append(record)

    for kind, grouped in grouped_by_kind.items():
        for key, grouped_records in grouped.items():
            splits = {record.split for record in grouped_records}
            if len(splits) < 2:
                continue
            findings.append(
                LeakageFinding(
                    kind=kind,
                    key=key,
                    splits=tuple(sorted(splits, key=str)),
                    record_ids=tuple(sorted(record.record_id for record in grouped_records)),
                )
            )
    return tuple(sorted(findings, key=lambda finding: (finding.kind, finding.key)))
