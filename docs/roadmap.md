# Phased roadmap

Each phase has an exit gate. Later work should not be pulled forward merely
because a directory or interface already exists.

| Phase                 | Outcome                                                                                       |
| --------------------- | --------------------------------------------------------------------------------------------- |
| 0 — Analysis          | Audit, scope, model, infrastructure, and storage decisions                                    |
| 1 — Architecture      | Preserved POC, documented architecture, runnable shells, tooling, and CI                      |
| 2 — Dataset strategy  | Domain taxonomy, dataset candidates, licenses, quality rules, and evaluation-set design       |
| 3 — Preprocessing     | Reproducible cleaning, deduplication, leakage prevention, token analysis, splits, and reports |
| 4 — Base evaluation   | Untouched Qwen3 benchmark on a fixed test set using Kaggle GPU                                |
| 5 — QLoRA training    | Smoke, pilot, full Kaggle training, adapter export, and fair comparison                       |
| 6 — Document pipeline | PDF/DOCX/TXT/article extraction, chunking, hierarchical summarization, and citations          |
| 7 — Core backend      | PostgreSQL, Redis, FastAPI feature routes, asynchronous worker, jobs, and cleanup             |
| 8 — Hardening         | Authentication, sessions, authorization, rate limits, audit events, logs, and metrics         |
| 9 — Frontend          | Upload, job progress, result, history, feedback, and session interfaces                       |
| 10 — Validation       | End-to-end, failure, security, load, storage, CI, and portfolio validation                    |
| 11 — Document chat    | Embeddings, retrieval, reranking, grounded answers, citations, and RAG evaluation             |

## Phase 1 exit gate

The completed delivery sequence and validation evidence are recorded in the
[Phase 1 implementation record](phase-1-implementation.md).

- The original POC is recoverable through both Git history and a legacy folder.
- Architecture and decisions are understandable without inspecting source code.
- The API exposes only liveness and readiness probes.
- The web shell shows API reachability and the current phase boundary.
- The ML package imports without installing a model framework.
- Docker runs the API and web shells without persistent storage.
- Formatting, linting, typing, tests, builds, and CI configuration pass.
