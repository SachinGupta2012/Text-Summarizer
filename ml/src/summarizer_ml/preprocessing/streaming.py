"""Memory-bounded preprocessing with partitioned outputs and a SQLite hash index."""

import json
import sqlite3
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import TracebackType
from typing import Self, TextIO

from pydantic import BaseModel, ConfigDict

from summarizer_ml.preprocessing.deduplication import ExactDuplicate, LeakageFinding, LeakageKind
from summarizer_ml.preprocessing.models import ProcessedRecord, RawRecord, RejectedRecord, Split
from summarizer_ml.preprocessing.pipeline import PreprocessingConfig, preprocess_record


class ProcessingReport(BaseModel):
    """Compact statistics produced by one complete preprocessing run."""

    model_config = ConfigDict(frozen=True)

    input_records: int
    accepted_records: int
    rejected_records: int
    exact_duplicates: int
    leakage_findings: int
    accepted_by_dataset: dict[str, int]
    accepted_by_split: dict[str, int]
    rejected_by_reason: dict[str, int]
    quality_flags: dict[str, int]


class StreamingPreprocessor:
    """Process arbitrary record streams without retaining document text in memory."""

    def __init__(
        self,
        output_directory: Path,
        config: PreprocessingConfig | None = None,
        *,
        retain_index: bool = False,
    ) -> None:
        if output_directory.exists() and any(output_directory.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {output_directory}")
        output_directory.mkdir(parents=True, exist_ok=True)

        self.output_directory = output_directory
        self.config = config or PreprocessingConfig()
        self.retain_index = retain_index
        self.index_path = output_directory / "preprocessing-index.sqlite3"
        self._connection = sqlite3.connect(self.index_path)
        self._writers: dict[Path, TextIO] = {}
        self._closed = False
        self._input_records = 0
        self._accepted_records = 0
        self._rejected_records = 0
        self._exact_duplicates = 0
        self._accepted_by_dataset: Counter[str] = Counter()
        self._accepted_by_split: Counter[str] = Counter()
        self._rejected_by_reason: Counter[str] = Counter()
        self._quality_flags: Counter[str] = Counter()
        self._create_index()

    def _create_index(self) -> None:
        self._connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE exact_seen (
                split TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                record_id TEXT NOT NULL,
                PRIMARY KEY (split, content_sha256)
            );
            CREATE TABLE source_occurrence (
                source_sha256 TEXT NOT NULL,
                split TEXT NOT NULL,
                record_id TEXT NOT NULL
            );
            CREATE INDEX source_occurrence_hash
                ON source_occurrence (source_sha256, split);
            CREATE TABLE group_occurrence (
                group_id TEXT NOT NULL,
                split TEXT NOT NULL,
                record_id TEXT NOT NULL
            );
            CREATE INDEX group_occurrence_id
                ON group_occurrence (group_id, split);
            """
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _writer(self, relative_path: Path) -> TextIO:
        path = self.output_directory / relative_path
        writer = self._writers.get(path)
        if writer is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = path.open("a", encoding="utf-8", newline="\n")
            self._writers[path] = writer
        return writer

    def _write_jsonl(self, relative_path: Path, payload: Mapping[str, object]) -> None:
        writer = self._writer(relative_path)
        writer.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        writer.write("\n")

    def _write_rejected(self, record: RejectedRecord) -> None:
        path = Path("rejected") / record.dataset.value / f"{record.split.value}.jsonl"
        self._write_jsonl(path, record.model_dump(mode="json"))

    def _write_processed(self, record: ProcessedRecord) -> None:
        path = Path("processed") / record.dataset.value / f"{record.split.value}.jsonl"
        self._write_jsonl(path, record.model_dump(mode="json"))

    def _write_duplicate(self, duplicate: ExactDuplicate) -> None:
        self._write_jsonl(
            Path("audit") / "exact-duplicates.jsonl",
            duplicate.model_dump(mode="json"),
        )

    def _index_record(self, record: ProcessedRecord) -> str | None:
        existing = self._connection.execute(
            "SELECT record_id FROM exact_seen WHERE split = ? AND content_sha256 = ?",
            (record.split.value, record.content_sha256),
        ).fetchone()
        if existing is not None:
            return str(existing[0])

        self._connection.execute(
            "INSERT INTO exact_seen (split, content_sha256, record_id) VALUES (?, ?, ?)",
            (record.split.value, record.content_sha256, record.record_id),
        )
        self._connection.execute(
            "INSERT INTO source_occurrence (source_sha256, split, record_id) VALUES (?, ?, ?)",
            (record.source_sha256, record.split.value, record.record_id),
        )
        self._connection.execute(
            "INSERT INTO group_occurrence (group_id, split, record_id) VALUES (?, ?, ?)",
            (record.group_id, record.split.value, record.record_id),
        )
        return None

    def process(self, records: Iterable[RawRecord]) -> None:
        """Consume records and write accepted, rejected, and duplicate partitions."""

        if self._closed:
            raise RuntimeError("Cannot process records after the preprocessor is closed.")
        for raw_record in records:
            self._input_records += 1
            outcome = preprocess_record(raw_record, self.config)
            if isinstance(outcome, RejectedRecord):
                self._rejected_records += 1
                self._rejected_by_reason[outcome.reason.value] += 1
                self._write_rejected(outcome)
                continue

            kept_record_id = self._index_record(outcome)
            if kept_record_id is not None:
                self._exact_duplicates += 1
                self._write_duplicate(
                    ExactDuplicate(
                        kept_record_id=kept_record_id,
                        duplicate_record_id=outcome.record_id,
                        split=outcome.split,
                    )
                )
                continue

            self._accepted_records += 1
            self._accepted_by_dataset[outcome.dataset.value] += 1
            self._accepted_by_split[outcome.split.value] += 1
            self._quality_flags.update(flag.value for flag in outcome.quality_flags)
            self._write_processed(outcome)

        self._connection.commit()

    def _write_leakage_findings(
        self,
        kind: LeakageKind,
        table: str,
        key_column: str,
    ) -> int:
        finding_count = 0
        keys = self._connection.execute(
            f"""
            SELECT {key_column}
            FROM {table}
            GROUP BY {key_column}
            HAVING COUNT(DISTINCT split) > 1
            """
        ).fetchall()
        for (key_value,) in keys:
            occurrences = self._connection.execute(
                f"SELECT split, record_id FROM {table} WHERE {key_column} = ?"
                " ORDER BY split, record_id",
                (key_value,),
            ).fetchall()
            splits = tuple(sorted({Split(split) for split, _ in occurrences}, key=str))
            finding = LeakageFinding(
                kind=kind,
                key=str(key_value),
                splits=splits,
                record_ids=tuple(str(record_id) for _, record_id in occurrences),
            )
            self._write_jsonl(
                Path("audit") / "split-leakage.jsonl",
                finding.model_dump(mode="json"),
            )
            finding_count += 1
        return finding_count

    def finalize(self) -> ProcessingReport:
        """Write leakage findings and the final machine-readable report."""

        if self._closed:
            raise RuntimeError("Preprocessor has already been closed.")
        leakage_findings = self._write_leakage_findings(
            LeakageKind.SOURCE_HASH,
            "source_occurrence",
            "source_sha256",
        )
        leakage_findings += self._write_leakage_findings(
            LeakageKind.GROUP_ID,
            "group_occurrence",
            "group_id",
        )
        report = ProcessingReport(
            input_records=self._input_records,
            accepted_records=self._accepted_records,
            rejected_records=self._rejected_records,
            exact_duplicates=self._exact_duplicates,
            leakage_findings=leakage_findings,
            accepted_by_dataset=dict(sorted(self._accepted_by_dataset.items())),
            accepted_by_split=dict(sorted(self._accepted_by_split.items())),
            rejected_by_reason=dict(sorted(self._rejected_by_reason.items())),
            quality_flags=dict(sorted(self._quality_flags.items())),
        )
        report_path = self.output_directory / "processing-report.json"
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.close()
        return report

    def close(self) -> None:
        """Close all resources and optionally remove the temporary SQLite index."""

        if self._closed:
            return
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()
        self._connection.close()
        self._closed = True
        if not self.retain_index and self.index_path.exists():
            self.index_path.unlink()


def iter_processed_records(output_directory: Path) -> Iterator[ProcessedRecord]:
    """Read canonical processed partitions in deterministic path order."""

    processed_root = output_directory / "processed"
    for path in sorted(processed_root.glob("*/*.jsonl")):
        with path.open(encoding="utf-8") as jsonl_file:
            for line_number, line in enumerate(jsonl_file, start=1):
                if not line.strip():
                    continue
                try:
                    yield ProcessedRecord.model_validate_json(line)
                except ValueError as error:
                    raise ValueError(f"Invalid processed record at {path}:{line_number}") from error
