$ErrorActionPreference = 'Stop'

uv run ruff format --check .
uv run ruff check .
uv run mypy services/api/src ml/src
uv run pytest

pnpm format:check
pnpm lint
pnpm typecheck
pnpm test
pnpm build

docker compose config --quiet
