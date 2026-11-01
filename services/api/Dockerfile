# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.10.0 AS uv

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:${PATH}"

COPY --from=uv /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY services/api/pyproject.toml services/api/README.md services/api/
COPY ml/pyproject.toml ml/README.md ml/

RUN uv sync --frozen --no-dev --package summarizer-api --no-install-workspace

COPY services/api/src services/api/src
RUN uv sync --frozen --no-dev --package summarizer-api --no-editable

RUN addgroup --system app && adduser --system --ingroup app app
USER app

EXPOSE 8000

CMD ["uvicorn", "summarizer_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
