"""Tests for canonical adapters, cleaning, validation, and leakage checks."""

import json
from pathlib import Path

from summarizer_ml.preprocessing.adapters import (
    AdapterError,
    TweetRow,
    adapt_billsum,
    adapt_cnn_dailymail,
    adapt_govreport,
    adapt_qmsum,
    adapt_tweetsumm,
    collect_tweetsumm_tweet_ids,
    load_selected_tweets,
)
from summarizer_ml.preprocessing.deduplication import (
    LeakageKind,
    deduplicate_exact,
    find_split_leakage,
)
from summarizer_ml.preprocessing.finalization import materialize_leakage_safe_partitions
from summarizer_ml.preprocessing.models import (
    DatasetName,
    Domain,
    ProcessedRecord,
    QualityFlag,
    RawRecord,
    RejectedRecord,
    RejectionReason,
    Split,
)
from summarizer_ml.preprocessing.pipeline import PreprocessingConfig, preprocess_record
from summarizer_ml.preprocessing.runner import run_phase3_diagnostic
from summarizer_ml.preprocessing.sources import (
    iter_huggingface_records,
    iter_qmsum_records,
    iter_tweetsumm_records,
)
from summarizer_ml.preprocessing.streaming import StreamingPreprocessor, iter_processed_records
from summarizer_ml.preprocessing.text import mask_customer_support_pii, normalize_text
from summarizer_ml.preprocessing.tokenization import (
    analyze_token_lengths,
    analyze_token_lengths_stream,
    measure_token_lengths,
)


class WhitespaceTokenizer:
    """Deterministic tokenizer double used for local tests."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))

    def __call__(
        self,
        text: list[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
        return_length: bool = False,
    ) -> dict[str, object]:
        del add_special_tokens, padding, truncation
        if not return_length:
            raise AssertionError("The preprocessing pipeline must request batch lengths.")
        return {"length": [len(value.split()) for value in text]}


class CountingWhitespaceTokenizer(WhitespaceTokenizer):
    """Tokenizer double that records single and batched encoding calls."""

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []
        self.batch_sizes: list[int] = []

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        self.encoded_texts.append(text)
        return super().encode(text, add_special_tokens=add_special_tokens)

    def __call__(
        self,
        text: list[str],
        *,
        add_special_tokens: bool = False,
        padding: bool = False,
        truncation: bool = False,
        return_length: bool = False,
    ) -> dict[str, object]:
        self.batch_sizes.append(len(text))
        return super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            return_length=return_length,
        )


def _accepted(record: RawRecord) -> ProcessedRecord:
    result = preprocess_record(record)
    assert isinstance(result, ProcessedRecord)
    return result


def _article_record(*, split: Split = Split.TRAIN, source_id: str = "article-1") -> RawRecord:
    return RawRecord(
        dataset=DatasetName.CNN_DAILYMAIL,
        domain=Domain.GENERAL_ARTICLE,
        split=split,
        source_id=source_id,
        group_id=f"cnn:{source_id}",
        source_text="A sufficiently long source article with useful factual information.",
        summary_text="A concise factual summary.",
    )


def test_normalize_text_preserves_paragraphs_and_removes_controls() -> None:
    value = " First\t paragraph. \r\n\r\n\r\n Second\x00 paragraph.  "

    assert normalize_text(value) == "First paragraph.\n\nSecond paragraph."


def test_customer_support_pii_masking_reports_types() -> None:
    result = mask_customer_support_pii("Email a.user@example.com or call +1 (212) 555-0199.")

    assert result.text == "Email [EMAIL] or call [PHONE]."
    assert result.masked_types == ("email", "phone")


def test_preprocess_record_normalizes_hashes_and_is_deterministic() -> None:
    raw = _article_record()

    first = _accepted(raw)
    second = _accepted(raw)

    assert first == second
    assert len(first.record_id) == 64
    assert len(first.source_sha256) == 64
    assert first.quality_flags == ()


def test_preprocess_record_rejects_identical_source_and_summary() -> None:
    raw = _article_record().model_copy(
        update={
            "source_text": "The same sufficiently long content appears in both fields.",
            "summary_text": "The same sufficiently long content appears in both fields.",
        }
    )

    result = preprocess_record(raw)

    assert isinstance(result, RejectedRecord)
    assert result.reason is RejectionReason.SOURCE_EQUALS_SUMMARY


def test_preprocess_record_rejects_short_source() -> None:
    result = preprocess_record(
        _article_record().model_copy(update={"source_text": "Too short."}),
        PreprocessingConfig(minimum_source_characters=20),
    )

    assert isinstance(result, RejectedRecord)
    assert result.reason is RejectionReason.SOURCE_TOO_SHORT


def test_customer_support_processing_masks_pii_and_marks_multiple_references() -> None:
    raw = RawRecord(
        dataset=DatasetName.TWEETSUMM,
        domain=Domain.CUSTOMER_SUPPORT,
        split=Split.TRAIN,
        source_id="conversation-1:reference:0",
        group_id="conversation-1",
        source_text="[CUSTOMER] Contact me at customer@example.com about the delayed order.",
        summary_text="Customer provided customer@example.com regarding a delayed order.",
        metadata={"reference_count": 3},
    )

    result = _accepted(raw)

    assert "customer@example.com" not in result.source_text
    assert QualityFlag.PII_MASKED in result.quality_flags
    assert QualityFlag.MULTIPLE_REFERENCES in result.quality_flags


def test_hugging_face_adapters_map_source_schemas() -> None:
    cnn = adapt_cnn_dailymail(
        {"id": "cnn-1", "article": "article", "highlights": "summary"},
        Split.TRAIN,
    )
    gov = adapt_govreport({"report": "government report", "summary": "summary"}, Split.TEST)
    bill = adapt_billsum(
        {"text": "bill text", "summary": "bill summary", "title": "Bill title"},
        Split.CA_TEST,
    )

    assert cnn.source_id == "cnn-1"
    assert cnn.domain is Domain.GENERAL_ARTICLE
    assert gov.source_id.startswith("govreport:")
    assert gov.domain is Domain.LONG_REPORT
    assert bill.metadata["title"] == "Bill title"
    assert bill.domain is Domain.LEGAL_DOCUMENT


def test_qmsum_adapter_preserves_speakers_queries_and_grouping() -> None:
    meeting = {
        "meeting_transcripts": [
            {"speaker": "Alice", "content": "We reviewed the release plan."},
            {"speaker": "Bob", "content": "The launch remains scheduled for Friday."},
        ],
        "general_query_list": [
            {"query": "Summarize the meeting.", "answer": "The team reviewed its release plan."}
        ],
        "specific_query_list": [
            {
                "query": "When is launch?",
                "answer": "The launch is scheduled for Friday.",
                "relevant_text_span": [["1", "1"]],
            }
        ],
    }

    records = adapt_qmsum(meeting, Split.VALIDATION, "meeting-7")

    assert len(records) == 2
    assert {record.group_id for record in records} == {"qmsum:meeting-7"}
    assert records[0].source_text.startswith("[Alice]")
    assert records[1].query == "When is launch?"
    assert records[1].metadata["relevant_text_span"] == [["1", "1"]]


def test_tweetsumm_id_collection_and_adapter_create_reference_records() -> None:
    entry = {
        "conversation_id": "conversation-7",
        "tweet_ids_sentence_offset": [
            {"tweet_id": 10, "sentence_offsets": ["[0, 10]"]},
            {"tweet_id": 11, "sentence_offsets": ["[0, 10]"]},
        ],
        "annotations": [
            {"abstractive": ["Customer reported a problem.", "Agent resolved it."]},
            {"abstractive": ["The support issue was resolved."]},
        ],
    }
    tweet_lookup = {
        "10": TweetRow(tweet_id="10", inbound=True, text="My order has not arrived."),
        "11": TweetRow(tweet_id="11", inbound=False, text="I have issued a replacement."),
    }

    assert collect_tweetsumm_tweet_ids([entry]) == {"10", "11"}
    records = adapt_tweetsumm(entry, Split.TEST, tweet_lookup)

    assert len(records) == 2
    assert records[0].source_text == (
        "[CUSTOMER] My order has not arrived.\n[AGENT] I have issued a replacement."
    )
    assert records[0].summary_text == "Customer reported a problem. Agent resolved it."
    assert records[0].metadata["reference_count"] == 2
    assert records[0].group_id == records[1].group_id


def test_tweetsumm_adapter_rejects_missing_tweets() -> None:
    entry = {
        "conversation_id": "conversation-7",
        "tweet_ids_sentence_offset": [{"tweet_id": 10}],
        "annotations": [{"abstractive": ["Summary"]}],
    }

    try:
        adapt_tweetsumm(entry, Split.TEST, {})
    except AdapterError as error:
        assert "missing 1 tweet rows" in str(error)
    else:
        raise AssertionError("Expected missing TweetSumm rows to be rejected.")


def test_tweetsumm_null_summary_is_audited_without_losing_valid_references() -> None:
    entry = {
        "conversation_id": "conversation-with-missing-summary",
        "tweet_ids_sentence_offset": [{"tweet_id": 10}, {"tweet_id": 11}],
        "annotations": [
            {"abstractive": None},
            {"abstractive": ["The customer received help from the support agent."]},
        ],
    }
    tweet_lookup = {
        "10": TweetRow(
            tweet_id="10",
            inbound=True,
            text="I need detailed assistance with my delayed order.",
        ),
        "11": TweetRow(
            tweet_id="11",
            inbound=False,
            text="I can investigate the order and provide an update.",
        ),
    }

    records = adapt_tweetsumm(entry, Split.TRAIN, tweet_lookup)
    missing_outcome = preprocess_record(records[0])
    valid_outcome = preprocess_record(records[1])

    assert len(records) == 2
    assert records[0].metadata["summary_missing"] is True
    assert isinstance(missing_outcome, RejectedRecord)
    assert missing_outcome.reason is RejectionReason.EMPTY_SUMMARY
    assert records[1].metadata["summary_missing"] is False
    assert isinstance(valid_outcome, ProcessedRecord)


def test_load_selected_tweets_streams_only_requested_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "twcs.csv"
    csv_path.write_text(
        "tweet_id,author_id,inbound,text\n"
        "10,user-a,True,Customer message\n"
        "11,brand,False,Agent message\n"
        "12,user-b,True,Unused message\n",
        encoding="utf-8",
    )

    selected = load_selected_tweets(csv_path, {"10", "11"})

    assert set(selected) == {"10", "11"}
    assert selected["10"].inbound is True
    assert selected["11"].inbound is False


def test_exact_deduplication_keeps_first_record_per_split() -> None:
    first = _accepted(_article_record(source_id="first"))
    duplicate = first.model_copy(update={"record_id": "f" * 64, "source_id": "duplicate"})

    result = deduplicate_exact([first, duplicate])

    assert result.kept_records == (first,)
    assert result.duplicates[0].duplicate_record_id == "f" * 64


def test_split_leakage_detects_source_and_group_crossing_splits() -> None:
    train = _accepted(_article_record(split=Split.TRAIN))
    test = _accepted(_article_record(split=Split.TEST))

    findings = find_split_leakage([train, test])

    assert {finding.kind for finding in findings} == {
        LeakageKind.SOURCE_HASH,
        LeakageKind.GROUP_ID,
    }
    assert all(finding.splits == (Split.TEST, Split.TRAIN) for finding in findings)


def test_streaming_preprocessor_writes_partitions_and_audit_files(tmp_path: Path) -> None:
    output_directory = tmp_path / "processed-output"
    train = _article_record(split=Split.TRAIN, source_id="shared")
    duplicate = train.model_copy(update={"source_id": "duplicate", "group_id": "duplicate"})
    test = _article_record(split=Split.TEST, source_id="shared")
    rejected = _article_record(source_id="short").model_copy(update={"source_text": "short"})

    processor = StreamingPreprocessor(output_directory)
    processor.process([train, duplicate, test, rejected])
    report = processor.finalize()

    assert report.input_records == 4
    assert report.accepted_records == 2
    assert report.rejected_records == 1
    assert report.exact_duplicates == 1
    assert report.leakage_findings == 2
    assert report.accepted_by_dataset == {"cnn_dailymail": 2}
    assert report.rejected_by_reason == {"source_too_short": 1}
    assert (output_directory / "processed" / "cnn_dailymail" / "train.jsonl").is_file()
    assert (output_directory / "processed" / "cnn_dailymail" / "test.jsonl").is_file()
    assert (output_directory / "rejected" / "cnn_dailymail" / "train.jsonl").is_file()
    assert (output_directory / "audit" / "exact-duplicates.jsonl").is_file()
    assert (output_directory / "audit" / "split-leakage.jsonl").is_file()
    assert (output_directory / "processing-report.json").is_file()
    assert not (output_directory / "preprocessing-index.sqlite3").exists()
    persisted = list(iter_processed_records(output_directory))
    assert len(persisted) == 2
    assert {record.split for record in persisted} == {Split.TRAIN, Split.TEST}


def test_streaming_preprocessor_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    output_directory = tmp_path / "existing"
    output_directory.mkdir()
    (output_directory / "result.json").write_text("{}", encoding="utf-8")

    try:
        StreamingPreprocessor(output_directory)
    except FileExistsError as error:
        assert "not empty" in str(error)
    else:
        raise AssertionError("Expected non-empty output directory to be rejected.")


def test_leakage_safe_finalization_preserves_validation_and_removes_train(
    tmp_path: Path,
) -> None:
    diagnostic_directory = tmp_path / "diagnostic"
    shared_source = "A shared article contains enough detailed text to pass validation safely."
    raw_records = [
        _article_record(source_id="safe-train").model_copy(
            update={
                "source_text": (
                    "A unique training article contains detailed information for testing."
                )
            }
        ),
        _article_record(source_id="leaked-train").model_copy(update={"source_text": shared_source}),
        _article_record(split=Split.VALIDATION, source_id="leaked-validation").model_copy(
            update={"source_text": shared_source}
        ),
        _article_record(split=Split.TEST, source_id="safe-test").model_copy(
            update={"source_text": "A unique test article contains separate factual information."}
        ),
    ]
    with StreamingPreprocessor(diagnostic_directory) as processor:
        processor.process(raw_records)
        diagnostic_report = processor.finalize()
    analyze_token_lengths_stream(
        iter_processed_records(diagnostic_directory),
        WhitespaceTokenizer(),
        tokenizer_id="test/whitespace",
        output_directory=diagnostic_directory / "analysis",
    )

    result = materialize_leakage_safe_partitions(diagnostic_directory)
    final_records = list(iter_processed_records(result.final_directory))

    assert diagnostic_report.accepted_records == 4
    assert diagnostic_report.leakage_findings == 1
    assert result.report.leakage_findings_before == 1
    assert result.report.leakage_removals == 1
    assert result.report.leakage_findings_after == 0
    assert result.report.final_records == 3
    assert result.report.final_token_records == 3
    assert result.report.final_by_split == {"test": 1, "train": 1, "validation": 1}
    assert {record.source_id for record in final_records} == {
        "safe-train",
        "leaked-validation",
        "safe-test",
    }
    audit_lines = (result.final_directory / "audit" / "leakage-removals.jsonl").read_text(
        encoding="utf-8"
    )
    audit_record = json.loads(audit_lines)
    assert audit_record["finding_kind"] == "source_hash"
    assert audit_record["removed_split"] == "train"
    assert audit_record["retained_split"] == "validation"
    repeated = materialize_leakage_safe_partitions(diagnostic_directory)
    assert repeated == result


def test_token_measurement_includes_query_in_model_input() -> None:
    record = _accepted(
        _article_record().model_copy(
            update={
                "query": "What happened?",
                "source_text": "One two three four five extraordinarilylongword.",
            }
        )
    )

    measurement = measure_token_lengths(record, WhitespaceTokenizer())

    assert measurement.source_tokens == 6
    assert measurement.query_tokens == 2
    assert measurement.input_tokens == 8
    assert measurement.summary_tokens == 4


def test_token_measurement_reuses_source_length_without_query() -> None:
    tokenizer = CountingWhitespaceTokenizer()

    measurement = measure_token_lengths(_accepted(_article_record()), tokenizer)

    assert measurement.input_tokens == measurement.source_tokens
    assert len(tokenizer.encoded_texts) == 2


def test_token_analysis_reports_percentiles_and_threshold_counts() -> None:
    records = [
        _accepted(_article_record(source_id="one")),
        _accepted(
            _article_record(source_id="two").model_copy(
                update={
                    "source_text": (
                        "A second article contains substantially more words for length analysis."
                    )
                }
            )
        ),
    ]

    measurements, report = analyze_token_lengths(
        records,
        WhitespaceTokenizer(),
        tokenizer_id="test/whitespace",
        thresholds=(5, 10),
    )

    assert len(measurements) == 2
    assert report.tokenizer_id == "test/whitespace"
    assert report.overall.input_tokens.count == 2
    assert report.overall.input_tokens.maximum == 10
    assert report.overall.input_tokens.over_threshold == {"5": 2, "10": 0}
    assert set(report.by_dataset) == {"cnn_dailymail"}


def test_streaming_token_analysis_writes_lengths_and_report(tmp_path: Path) -> None:
    output_directory = tmp_path / "token-analysis"
    tokenizer = CountingWhitespaceTokenizer()
    progress: list[int] = []

    report = analyze_token_lengths_stream(
        [_accepted(_article_record())],
        tokenizer,
        tokenizer_id="test/whitespace",
        output_directory=output_directory,
        thresholds=(5,),
        batch_size=8,
        progress_every=1,
        progress_callback=progress.append,
    )

    assert report.overall.input_tokens.count == 1
    assert tokenizer.batch_sizes == [1, 1]
    assert progress == [1]
    assert (output_directory / "token-lengths.jsonl").is_file()
    saved_report = json.loads(
        (output_directory / "token-analysis-report.json").read_text(encoding="utf-8")
    )
    assert saved_report["tokenizer_id"] == "test/whitespace"


def test_streaming_token_analysis_resumes_checkpoint_in_batches(tmp_path: Path) -> None:
    output_directory = tmp_path / "token-analysis"
    first = _accepted(_article_record(source_id="first"))
    second = _accepted(_article_record(source_id="second"))

    analyze_token_lengths_stream(
        [first],
        WhitespaceTokenizer(),
        tokenizer_id="test/whitespace",
        output_directory=output_directory,
        batch_size=2,
    )
    (output_directory / "token-analysis-report.json").unlink()

    progress: list[int] = []
    report = analyze_token_lengths_stream(
        [first, second],
        WhitespaceTokenizer(),
        tokenizer_id="test/whitespace",
        output_directory=output_directory,
        batch_size=2,
        progress_every=1,
        progress_callback=progress.append,
    )

    lines = (output_directory / "token-lengths.jsonl").read_text(encoding="utf-8").splitlines()
    assert report.overall.input_tokens.count == 2
    assert len(lines) == 2
    assert progress == [1, 2]


def test_streaming_token_analysis_batches_only_query_specific_extra_work(
    tmp_path: Path,
) -> None:
    tokenizer = CountingWhitespaceTokenizer()
    records = [
        _accepted(_article_record(source_id="document")),
        _accepted(
            _article_record(source_id="query").model_copy(
                update={"query": "What happened in this article?"}
            )
        ),
    ]

    analyze_token_lengths_stream(
        records,
        tokenizer,
        tokenizer_id="test/whitespace",
        output_directory=tmp_path / "token-analysis",
        batch_size=8,
    )

    assert tokenizer.batch_sizes == [2, 2, 1, 1]


def test_hugging_face_source_iterator_maps_supported_splits() -> None:
    rows = {
        "train": [{"id": "train-1", "article": "article", "highlights": "summary"}],
        "unsupported": [{"id": "skip", "article": "article", "highlights": "summary"}],
    }

    records = list(iter_huggingface_records(rows, adapt_cnn_dailymail))

    assert len(records) == 1
    assert records[0].split is Split.TRAIN


def test_qmsum_source_iterator_reads_split_directories(tmp_path: Path) -> None:
    qmsum_root = tmp_path / "qmsum"
    meeting_root = qmsum_root / "data" / "ALL" / "train"
    meeting_root.mkdir(parents=True)
    meeting = {
        "meeting_transcripts": [{"speaker": "Alice", "content": "A detailed discussion."}],
        "general_query_list": [
            {"query": "Summarize this meeting.", "answer": "Alice led a detailed discussion."}
        ],
        "specific_query_list": [],
    }
    (meeting_root / "meeting-1.json").write_text(json.dumps(meeting), encoding="utf-8")

    records = list(iter_qmsum_records(qmsum_root))

    assert len(records) == 1
    assert records[0].group_id == "qmsum:meeting-1"
    assert records[0].split is Split.TRAIN


def test_tweetsumm_source_iterator_reconstructs_all_splits(tmp_path: Path) -> None:
    tweetsumm_root = tmp_path / "tweetsumm"
    data_root = tweetsumm_root / "tweet_sum_data_files"
    data_root.mkdir(parents=True)
    entries = {
        "final_train_tweetsum.jsonl": ("train-conversation", 10, 11),
        "final_valid_tweetsum.jsonl": ("valid-conversation", 12, 13),
        "final_test_tweetsum.jsonl": ("test-conversation", 14, 15),
    }
    for file_name, (conversation_id, customer_id, agent_id) in entries.items():
        entry = {
            "conversation_id": conversation_id,
            "tweet_ids_sentence_offset": [
                {"tweet_id": customer_id},
                {"tweet_id": agent_id},
            ],
            "annotations": [{"abstractive": ["The support issue was resolved correctly."]}],
        }
        (data_root / file_name).write_text(json.dumps(entry) + "\n", encoding="utf-8")

    twcs_path = tmp_path / "twcs.csv"
    twcs_path.write_text(
        "tweet_id,author_id,inbound,text\n"
        "10,user,True,Train customer has a detailed problem.\n"
        "11,brand,False,Train agent provides a detailed answer.\n"
        "12,user,True,Valid customer has a detailed problem.\n"
        "13,brand,False,Valid agent provides a detailed answer.\n"
        "14,user,True,Test customer has a detailed problem.\n"
        "15,brand,False,Test agent provides a detailed answer.\n",
        encoding="utf-8",
    )

    records = list(iter_tweetsumm_records(tweetsumm_root, twcs_path))

    assert len(records) == 3
    assert {record.split for record in records} == {
        Split.TRAIN,
        Split.VALIDATION,
        Split.TEST,
    }


def test_phase3_runner_processes_all_five_sources(tmp_path: Path) -> None:
    qmsum_root = tmp_path / "qmsum"
    qmsum_train = qmsum_root / "data" / "ALL" / "train"
    qmsum_train.mkdir(parents=True)
    qmsum_meeting = {
        "meeting_transcripts": [
            {
                "speaker": "Alice",
                "content": "The team discussed a detailed release plan and agreed on next steps.",
            }
        ],
        "general_query_list": [
            {"query": "Summarize the meeting.", "answer": "The team agreed on release steps."}
        ],
        "specific_query_list": [],
    }
    (qmsum_train / "meeting-1.json").write_text(json.dumps(qmsum_meeting), encoding="utf-8")

    tweetsumm_root = tmp_path / "tweetsumm"
    tweetsumm_data = tweetsumm_root / "tweet_sum_data_files"
    tweetsumm_data.mkdir(parents=True)
    for file_name in (
        "final_train_tweetsum.jsonl",
        "final_valid_tweetsum.jsonl",
        "final_test_tweetsum.jsonl",
    ):
        content = ""
        if file_name == "final_train_tweetsum.jsonl":
            content = json.dumps(
                {
                    "conversation_id": "conversation-1",
                    "tweet_ids_sentence_offset": [{"tweet_id": 10}, {"tweet_id": 11}],
                    "annotations": [
                        {"abstractive": None},
                        {"abstractive": ["The customer received a replacement order."]},
                    ],
                }
            )
        (tweetsumm_data / file_name).write_text(f"{content}\n" if content else "", encoding="utf-8")
    twcs_path = tmp_path / "twcs.csv"
    twcs_path.write_text(
        "tweet_id,author_id,inbound,text\n"
        "10,user,True,My delayed order still has not arrived after several days.\n"
        "11,brand,False,I have created a replacement order that ships today.\n",
        encoding="utf-8",
    )

    progress_updates: list[int] = []
    token_progress_updates: list[int] = []
    result = run_phase3_diagnostic(
        cnn_dailymail={
            "train": [
                {
                    "id": "cnn-1",
                    "article": (
                        "A detailed news article contains enough information for processing."
                    ),
                    "highlights": "The article contains useful information.",
                }
            ]
        },
        govreport={
            "train": [
                {
                    "report": "A detailed government report contains findings and recommendations.",
                    "summary": "The report presents findings and recommendations.",
                }
            ]
        },
        billsum={
            "train": [
                {
                    "text": "A proposed bill establishes a detailed public assistance program.",
                    "summary": "The bill establishes a public assistance program.",
                    "title": "Public Assistance Act",
                }
            ]
        },
        qmsum_root=qmsum_root,
        tweetsumm_root=tweetsumm_root,
        twcs_csv_path=twcs_path,
        tokenizer=WhitespaceTokenizer(),
        tokenizer_id="test/whitespace",
        output_directory=tmp_path / "phase3-output",
        max_records_per_dataset_split=2,
        progress_every=2,
        progress_callback=progress_updates.append,
        token_batch_size=2,
        token_progress_every=2,
        token_progress_callback=token_progress_updates.append,
    )

    assert result.processing.input_records == 6
    assert result.processing.accepted_records == 5
    assert result.processing.rejected_records == 1
    assert result.processing.rejected_by_reason == {"empty_summary": 1}
    assert progress_updates == [2, 4, 6]
    assert token_progress_updates == [2, 4, 5]
    assert set(result.processing.accepted_by_dataset) == {
        "billsum",
        "cnn_dailymail",
        "govreport",
        "qmsum",
        "tweetsumm",
    }
    assert set(result.token_analysis.by_dataset) == set(result.processing.accepted_by_dataset)


def test_phase3_runner_validates_sampling_and_progress_arguments(tmp_path: Path) -> None:
    arguments = {
        "cnn_dailymail": {},
        "govreport": {},
        "billsum": {},
        "qmsum_root": tmp_path,
        "tweetsumm_root": tmp_path,
        "twcs_csv_path": tmp_path / "twcs.csv",
        "tokenizer": WhitespaceTokenizer(),
        "tokenizer_id": "test-tokenizer",
        "output_directory": tmp_path / "output",
    }

    try:
        run_phase3_diagnostic(**arguments, max_records_per_dataset_split=0)
    except ValueError as error:
        assert str(error) == "max_records_per_dataset_split must be positive."
    else:
        raise AssertionError("Expected invalid sample limit to be rejected.")

    try:
        run_phase3_diagnostic(**arguments, progress_every=0)
    except ValueError as error:
        assert str(error) == "progress_every must be positive."
    else:
        raise AssertionError("Expected invalid progress interval to be rejected.")
