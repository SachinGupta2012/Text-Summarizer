"""FastAPI application factory."""

from fastapi import FastAPI

from summarizer_api.config import Settings, get_settings
from summarizer_api.routes.health import router as health_router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application with explicit, testable configuration."""

    resolved_settings = settings or get_settings()
    application = FastAPI(
        title="Transformer Summarizer API",
        summary="Architecture foundation for the document summarization platform.",
        version=resolved_settings.app_version,
    )
    application.state.settings = resolved_settings
    application.include_router(health_router)
    return application


app = create_app()
