# Transformer Summarizer

A production-style learning project for multi-domain document summarization. The
project will compare the untouched `Qwen/Qwen3-4B-Instruct-2507` model with a
QLoRA adapter trained and evaluated on Kaggle GPU infrastructure.

## Current status

**Phase 1: architecture and repository foundation.**

The repository currently contains runnable API and web service shells plus a
dependency-light ML package. It does not yet download a dataset, load a model,
train an adapter, accept documents, or generate summaries. Those capabilities
have explicit later phases in [the roadmap](docs/roadmap.md).

## Repository layout

```text
apps/web/              Next.js application
services/api/          FastAPI service
services/worker/       Reserved asynchronous worker boundary
ml/                    Data, evaluation, training, and inference package
packages/contracts/    Reserved generated API contracts
infrastructure/        Container and optional observability configuration
docs/                  Scope, architecture, roadmap, and decisions
legacy/streamlit-poc/  Preserved original proof of concept
```

## Architecture boundary

```text
Next.js -> FastAPI -> asynchronous job -> worker -> document pipeline -> model
```

PostgreSQL, Redis, document extraction, Qwen inference, and the worker are shown
in the target architecture but are not implemented during Phase 1.

## Prerequisites

- Docker Desktop with Compose
- Python 3.11 and `uv` for host-side Python development
- Node.js 24 and pnpm 11 for host-side frontend development

On the development workstation, Docker Desktop's virtual disk is stored on the
D: drive. The repository never assumes a machine-specific Docker path.

## Run with Docker

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Open:

- Web: <http://localhost:3000>
- API documentation: <http://localhost:8000/docs>
- API readiness: <http://localhost:8000/health/ready>

Stop the service shells with:

```powershell
docker compose down
```

Phase 1 creates no database or persistent Docker volume.

## Host-side development

```powershell
uv sync --all-packages --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy services/api/src ml/src

pnpm install --frozen-lockfile
pnpm lint
pnpm typecheck
pnpm test
pnpm build
```

## Documentation

- [Phase 1 implementation record](docs/phase-1-implementation.md)
- [Product scope](docs/product-scope.md)
- [Phased roadmap](docs/roadmap.md)
- [System overview](docs/architecture/system-overview.md)
- [Component architecture](docs/architecture/component-architecture.md)
- [ML lifecycle](docs/architecture/ml-lifecycle.md)
- [Storage and retention](docs/architecture/storage-and-retention.md)
- [Security boundaries](docs/architecture/security-boundaries.md)

## Legacy proof of concept

The original Streamlit/SAMSum implementation is preserved under
[`legacy/streamlit-poc`](legacy/streamlit-poc). The Git tag
`legacy-streamlit-poc` points to the exact pre-refactor commit. Legacy code and
datasets are excluded from the new dependency graph, Docker context, tests, and
CI.
