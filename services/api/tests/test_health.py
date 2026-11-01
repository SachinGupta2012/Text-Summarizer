"""Health route contract tests."""

import pytest
from fastapi import Request
from summarizer_api.config import Settings
from summarizer_api.main import create_app
from summarizer_api.routes.health import _response


@pytest.fixture
def api_request() -> Request:
    app = create_app(Settings(app_env="test", app_version="0.1.0-test"))
    return Request({"type": "http", "app": app})


def test_health_contract(api_request: Request) -> None:
    response = _response(api_request)

    assert response.model_dump() == {
        "status": "healthy",
        "service": "summarizer-api",
        "version": "0.1.0-test",
    }


def test_health_routes_are_registered() -> None:
    app = create_app(Settings(app_env="test"))
    paths = {route.path for route in app.routes}

    assert "/health/live" in paths
    assert "/health/ready" in paths
