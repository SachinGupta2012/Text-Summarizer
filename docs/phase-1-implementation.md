# Phase 1 implementation record

## Objective

Create a clean, reproducible architecture foundation without mixing dataset,
training, document-processing, persistence, or product features into the first
phase.

## Delivery order

| Step | Deliverable                                                          | Status   |
| ---- | -------------------------------------------------------------------- | -------- |
| 1    | Tag the pre-refactor commit and create the refactor branch           | Complete |
| 2    | Preserve the original Streamlit POC under `legacy/streamlit-poc`     | Complete |
| 3    | Record product scope, constraints, target architecture, and ADRs     | Complete |
| 4    | Establish the Python, Node.js, and repository workspaces             | Complete |
| 5    | Add the minimal FastAPI application and health contracts             | Complete |
| 6    | Add the minimal Next.js application and runtime API status           | Complete |
| 7    | Add a dependency-light ML package containing agreed identifiers only | Complete |
| 8    | Add formatting, linting, typing, testing, lockfiles, and CI          | Complete |
| 9    | Add non-root, resource-limited API and web containers                | Complete |
| 10   | Validate local builds and real HTTP behavior through Docker Compose  | Complete |

## Implemented outputs

- Monorepo boundaries for web, API, worker, ML, contracts, infrastructure,
  documentation, and legacy code.
- A FastAPI application factory with liveness and readiness endpoints.
- A strict TypeScript Next.js shell that reports runtime API reachability.
- A lightweight ML configuration package that does not install PyTorch,
  Transformers, download a model, or download a dataset.
- Reproducible Python and JavaScript lockfiles.
- Ruff, mypy, pytest, coverage, Prettier, ESLint, TypeScript, and Vitest gates.
- GitHub Actions checks for Python, web, and container configuration.
- Docker Compose shells with non-root processes, health checks, memory limits,
  bounded logs, and no persistent volumes.
- Architecture documentation and decision records for the long-term system.

## Explicit phase boundary

| Capability                                                                 | Scheduled phase |
| -------------------------------------------------------------------------- | --------------- |
| Dataset selection, licensing, taxonomy, and evaluation-set design          | Phase 2         |
| Cleaning, normalization, deduplication, leakage checks, and token analysis | Phase 3         |
| Untouched Qwen3 baseline evaluation                                        | Phase 4         |
| Kaggle QLoRA training and adapter comparison                               | Phase 5         |
| PDF, DOCX, article, and long-document processing                           | Phase 6         |
| PostgreSQL, Redis, jobs, worker runtime, and cleanup policies              | Phase 7         |
| Authentication, sessions, rate limits, logs, metrics, and audit events     | Phase 8         |
| Upload, progress, results, history, and feedback user interfaces           | Phase 9         |
| End-to-end, security, failure, load, and portfolio validation              | Phase 10        |
| Document-grounded conversational question answering                        | Phase 11        |

Directories reserved for later components are architectural boundaries, not
partial implementations of those future phases.

## Acceptance evidence

- Python formatting, linting, strict typing, tests, and coverage pass.
- Frontend formatting, linting, strict typing, component tests, and production
  build pass.
- Dependency peer checks and frozen lockfile installation pass.
- Docker Compose validates and builds both images.
- Both containers become healthy.
- `/health/live` and `/health/ready` return `healthy` with version `0.1.0`.
- The web route returns HTTP 200 and dynamically displays API readiness.
- The smoke test creates no database, cache, model, dataset, or Docker volume.

## Next phase

Phase 2 begins with a dataset and evaluation strategy. No training notebook or
preprocessing implementation should be started until its domain taxonomy,
licenses, quality criteria, and fixed evaluation-set policy are approved.
