"""Environment-backed API configuration."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings shared by the API application."""

    app_env: Literal["development", "test", "production"] = "development"
    app_version: str = "0.1.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
