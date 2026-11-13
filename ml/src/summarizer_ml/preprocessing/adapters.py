"""Adapters from source-specific schemas to canonical raw records."""

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from summarizer_ml.preprocessing.models import (
    DatasetName,
    Domain,
    QueryType,
    RawRecord,
    Split,
)


class AdapterError(ValueError):
    """Raised when a source row does not match its documented schema."""


def _required_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise AdapterError(f"Expected string field {key!r}.")
    return value


def _content_id(dataset: DatasetName, text: str) -> str:
    digest = sha256(text.encode("utf-8")).hexdigest()
    return f"{dataset.value}:{digest}"


def adapt_cnn_dailymail(row: Mapping[str, object], split: Split) -> RawRecord:
    """Adapt one CNN/DailyMail article and highlight summary."""

    source_id = _required_text(row, "id")
    return RawRecord(
        dataset=DatasetName.CNN_DAILYMAIL,
        domain=Domain.GENERAL_ARTICLE,
        split=split,
        source_id=source_id,
        group_id=f"cnn_dailymail:{source_id}",
        source_text=_required_text(row, "article"),
        summary_text=_required_text(row, "highlights"),
    )


def adapt_govreport(row: Mapping[str, object], split: Split) -> RawRecord:
    """Adapt one GovReport document."""

    source_text = _required_text(row, "report")
    source_id = _content_id(DatasetName.GOVREPORT, source_text)
    return RawRecord(
        dataset=DatasetName.GOVREPORT,
        domain=Domain.LONG_REPORT,
        split=split,
        source_id=source_id,
        group_id=source_id,
        source_text=source_text,
        summary_text=_required_text(row, "summary"),
    )


def adapt_billsum(row: Mapping[str, object], split: Split) -> RawRecord:
    """Adapt one BillSum bill and its human-authored summary."""

    source_text = _required_text(row, "text")
    source_id = _content_id(DatasetName.BILLSUM, source_text)
    title = row.get("title")
    metadata: dict[str, object] = {}
    if isinstance(title, str):
        metadata["title"] = title
    return RawRecord(
        dataset=DatasetName.BILLSUM,
        domain=Domain.LEGAL_DOCUMENT,
        split=split,
        source_id=source_id,
        group_id=source_id,
        source_text=source_text,
        summary_text=_required_text(row, "summary"),
        metadata=metadata,
    )


def _mapping_sequence(value: object, field: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise AdapterError(f"Expected a list of objects in {field!r}.")
    return list(value)


def adapt_qmsum(meeting: Mapping[str, object], split: Split, meeting_id: str) -> list[RawRecord]:
    """Expand one QMSum meeting into one record per general or specific query."""

    transcript_items = _mapping_sequence(meeting.get("meeting_transcripts"), "meeting_transcripts")
    transcript_lines = [
        f"[{_required_text(item, 'speaker')}] {_required_text(item, 'content')}"
        for item in transcript_items
    ]
    source_text = "\n".join(transcript_lines)
    group_id = f"qmsum:{meeting_id}"
    records: list[RawRecord] = []

    query_groups = (
        ("general_query_list", QueryType.GENERAL_MEETING),
        ("specific_query_list", QueryType.SPECIFIC_MEETING),
    )
    for field, query_type in query_groups:
        for index, query_item in enumerate(_mapping_sequence(meeting.get(field), field)):
            metadata: dict[str, object] = {
                "meeting_id": meeting_id,
                "query_index": index,
                "query_list": field,
            }
            relevant_spans = query_item.get("relevant_text_span")
            if isinstance(relevant_spans, list):
                metadata["relevant_text_span"] = relevant_spans
            records.append(
                RawRecord(
                    dataset=DatasetName.QMSUM,
                    domain=Domain.MEETING,
                    split=split,
                    source_id=f"{group_id}:{field}:{index}",
                    group_id=group_id,
                    source_text=source_text,
                    summary_text=_required_text(query_item, "answer"),
                    query=_required_text(query_item, "query"),
                    query_type=query_type,
                    metadata=metadata,
                )
            )

    if not records:
        raise AdapterError(f"QMSum meeting {meeting_id!r} contains no queries.")
    return records


@dataclass(frozen=True, slots=True)
class TweetRow:
    """Minimal Customer Support on Twitter row needed by TweetSumm."""

    tweet_id: str
    inbound: bool
    text: str
    author_id: str | None = None


def collect_tweetsumm_tweet_ids(entries: Iterable[Mapping[str, object]]) -> set[str]:
    """Collect only the tweet IDs needed to reconstruct TweetSumm dialogues."""

    tweet_ids: set[str] = set()
    for entry in entries:
        offsets = _mapping_sequence(
            entry.get("tweet_ids_sentence_offset"), "tweet_ids_sentence_offset"
        )
        for offset in offsets:
            tweet_ids.add(str(offset.get("tweet_id")))
    return tweet_ids


def load_selected_tweets(csv_path: Path, required_ids: set[str]) -> dict[str, TweetRow]:
    """Stream the large TWCS CSV and retain only IDs referenced by TweetSumm."""

    selected: dict[str, TweetRow] = {}
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"tweet_id", "author_id", "inbound", "text"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise AdapterError("Customer Support on Twitter CSV has unexpected columns.")
        for row in reader:
            tweet_id = row["tweet_id"]
            if tweet_id not in required_ids:
                continue
            selected[tweet_id] = TweetRow(
                tweet_id=tweet_id,
                inbound=row["inbound"].strip().casefold() == "true",
                text=row["text"],
                author_id=row["author_id"] or None,
            )
            if len(selected) == len(required_ids):
                break
    return selected


def adapt_tweetsumm(
    entry: Mapping[str, object],
    split: Split,
    tweet_lookup: Mapping[str, TweetRow],
) -> list[RawRecord]:
    """Expand one reconstructed TweetSumm dialogue into its reference summaries."""

    conversation_id = _required_text(entry, "conversation_id")
    offsets = _mapping_sequence(entry.get("tweet_ids_sentence_offset"), "tweet_ids_sentence_offset")
    tweet_ids = [str(offset.get("tweet_id")) for offset in offsets]
    missing_ids = [tweet_id for tweet_id in tweet_ids if tweet_id not in tweet_lookup]
    if missing_ids:
        raise AdapterError(
            f"Conversation {conversation_id!r} is missing {len(missing_ids)} tweet rows."
        )

    dialogue_lines: list[str] = []
    for tweet_id in tweet_ids:
        tweet = tweet_lookup[tweet_id]
        role = "CUSTOMER" if tweet.inbound else "AGENT"
        dialogue_lines.append(f"[{role}] {tweet.text}")
    source_text = "\n".join(dialogue_lines)

    annotations = _mapping_sequence(entry.get("annotations"), "annotations")
    records: list[RawRecord] = []
    group_id = f"tweetsumm:{conversation_id}"
    for index, annotation in enumerate(annotations):
        abstractive = annotation.get("abstractive")
        if abstractive is None:
            summary_text = ""
        elif not isinstance(abstractive, Sequence) or isinstance(abstractive, str):
            raise AdapterError("TweetSumm abstractive summary must be a list of strings.")
        else:
            if not all(isinstance(sentence, str) for sentence in abstractive):
                raise AdapterError("TweetSumm abstractive summary contains a non-string value.")
            summary_text = " ".join(abstractive)
        records.append(
            RawRecord(
                dataset=DatasetName.TWEETSUMM,
                domain=Domain.CUSTOMER_SUPPORT,
                split=split,
                source_id=f"{group_id}:reference:{index}",
                group_id=group_id,
                source_text=source_text,
                summary_text=summary_text,
                query_type=QueryType.DIALOGUE,
                metadata={
                    "conversation_id": conversation_id,
                    "reference_index": index,
                    "reference_count": len(annotations),
                    "summary_missing": abstractive is None,
                    "tweet_ids": tweet_ids,
                },
            )
        )
    if not records:
        raise AdapterError(f"TweetSumm conversation {conversation_id!r} has no annotations.")
    return records
