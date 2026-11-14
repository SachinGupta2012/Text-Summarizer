"""Reusable Phase 3 preprocessing primitives."""

from summarizer_ml.preprocessing.finalization import (
    FinalizationReport,
    FinalizationResult,
    LeakageRemoval,
    materialize_leakage_safe_partitions,
)
from summarizer_ml.preprocessing.models import (
    DatasetName,
    Domain,
    ProcessedRecord,
    QualityFlag,
    QueryType,
    RawRecord,
    RejectedRecord,
    RejectionReason,
    Split,
)
from summarizer_ml.preprocessing.outliers import (
    OutlierAuditConfig,
    OutlierAuditReport,
    OutlierAuditResult,
    OutlierFlag,
    OutlierRecord,
    audit_outliers,
)
from summarizer_ml.preprocessing.pipeline import PreprocessingConfig, preprocess_record
from summarizer_ml.preprocessing.policy import (
    ArtifactDigest,
    Phase3IntegrityGates,
    Phase3Manifest,
    Phase3PolicyConfig,
    Phase3PolicyResult,
    PolicyAction,
    PolicyDecision,
    PolicyReason,
    classify_outlier,
    materialize_training_ready_partitions,
)
from summarizer_ml.preprocessing.runner import (
    Phase3RunResult,
    resume_phase3_token_analysis,
    run_phase3_diagnostic,
)
from summarizer_ml.preprocessing.tokenization import (
    TokenAnalysisReport,
    TokenLengthRecord,
    analyze_token_lengths,
    analyze_token_lengths_stream,
    measure_token_lengths,
)

__all__ = [
    "ArtifactDigest",
    "DatasetName",
    "Domain",
    "FinalizationReport",
    "FinalizationResult",
    "LeakageRemoval",
    "OutlierAuditConfig",
    "OutlierAuditReport",
    "OutlierAuditResult",
    "OutlierFlag",
    "OutlierRecord",
    "Phase3IntegrityGates",
    "Phase3Manifest",
    "Phase3PolicyConfig",
    "Phase3PolicyResult",
    "Phase3RunResult",
    "PolicyAction",
    "PolicyDecision",
    "PolicyReason",
    "PreprocessingConfig",
    "ProcessedRecord",
    "QualityFlag",
    "QueryType",
    "RawRecord",
    "RejectedRecord",
    "RejectionReason",
    "Split",
    "TokenAnalysisReport",
    "TokenLengthRecord",
    "analyze_token_lengths",
    "analyze_token_lengths_stream",
    "audit_outliers",
    "classify_outlier",
    "materialize_leakage_safe_partitions",
    "materialize_training_ready_partitions",
    "measure_token_lengths",
    "preprocess_record",
    "resume_phase3_token_analysis",
    "run_phase3_diagnostic",
]
