"""File and dataset iterators used by the Kaggle preprocessing launcher."""

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path

from summarizer_ml.preprocessing.adapters import (
    adapt_qmsum,
    adapt_tweetsumm,
    collect_tweetsumm_tweet_ids,
    load_selected_tweets,
)
from summarizer_ml.preprocessing.models import RawRecord, Split

HuggingFaceAdapter = Callable[[Mapping[str, object], Split], RawRecord]

_SPLIT_NAMES = {
    "train": Split.TRAIN,
    "validation": Split.VALIDATION,
    "val": Split.VALIDATION,
    "test": Split.TEST,
    "ca_test": Split.CA_TEST,
}


def iter_huggingface_records(
    dataset_splits: Mapping[str, Iterable[Mapping[str, object]]],
    adapter: HuggingFaceAdapter,
) -> Iterator[RawRecord]:
    """Adapt every supported split in a Hugging Face DatasetDict-like object."""

    for split_name, rows in dataset_splits.items():
        split = _SPLIT_NAMES.get(split_name)
        if split is None:
            continue
        for row in rows:
            yield adapter(row, split)


def _load_json_object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _load_jsonl_objects(path: Path) -> list[Mapping[str, object]]:
    values: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as jsonl_file:
        for line_number, line in enumerate(jsonl_file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            values.append(value)
    return values


def iter_qmsum_records(qmsum_root: Path) -> Iterator[RawRecord]:
    """Read QMSum meeting files and emit one record per query."""

    data_root = qmsum_root / "data" / "ALL"
    split_directories = {
        "train": Split.TRAIN,
        "val": Split.VALIDATION,
        "test": Split.TEST,
    }
    for directory_name, split in split_directories.items():
        split_root = data_root / directory_name
        for path in sorted(split_root.glob("*.json")):
            yield from adapt_qmsum(_load_json_object(path), split, path.stem)


def iter_tweetsumm_records(tweetsumm_root: Path, twcs_csv_path: Path) -> Iterator[RawRecord]:
    """Reconstruct TweetSumm dialogues while retaining only referenced TWCS rows."""

    data_root = tweetsumm_root / "tweet_sum_data_files"
    split_files = {
        "final_train_tweetsum.jsonl": Split.TRAIN,
        "final_valid_tweetsum.jsonl": Split.VALIDATION,
        "final_test_tweetsum.jsonl": Split.TEST,
    }
    entries_by_split = {
        split: _load_jsonl_objects(data_root / file_name)
        for file_name, split in split_files.items()
    }
    required_ids = collect_tweetsumm_tweet_ids(
        entry for entries in entries_by_split.values() for entry in entries
    )
    tweet_lookup = load_selected_tweets(twcs_csv_path, required_ids)
    missing_count = len(required_ids.difference(tweet_lookup))
    if missing_count:
        raise ValueError(f"Customer Support on Twitter is missing {missing_count} required tweets.")

    for split, entries in entries_by_split.items():
        for entry in entries:
            yield from adapt_tweetsumm(entry, split, tweet_lookup)
