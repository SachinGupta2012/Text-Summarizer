"""Liveness and readiness routes for the Phase 1 service shell."""

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from summarizer_api.config import Settings

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Stable response shared by health probes."""

    status: Literal["healthy"] = "healthy"
    service: Literal["summarizer-api"] = "summarizer-api"
    version: str


def _response(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    return HealthResponse(version=settings.app_version)


@router.get("/live", response_model=HealthResponse)
async def live(request: Request) -> HealthResponse:
    """Report whether the API process can serve requests."""

    return _response(request)


@router.get("/ready", response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    """Report readiness; external dependencies are added in later phases."""

    return _response(request)
