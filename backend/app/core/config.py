from functools import lru_cache
import os
from typing import Literal
from uuid import UUID

from pydantic import Field, SecretStr

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "WorkFlow AI API"
    app_environment: Literal[
        "development", "testing", "production"
    ] = "development"
    debug: bool = False

    postgres_user: str = "workflow_ai"
    postgres_password: SecretStr
    postgres_db: str = "workflow_ai"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432

    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    llm_max_retries: int = Field(default=1, ge=0, le=2)
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "workflow-ai"
    workflow_api_token: SecretStr | None = None
    workflow_executor_token: SecretStr | None = None
    workflow_user_id: UUID = UUID("00000000-0000-4000-8000-000000000001")
    workflow_user_email: str = "workflow-ai@example.local"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_langsmith_tracing(settings: Settings | None = None) -> None:
    """Pass opt-in LangSmith settings to LangChain's tracing integration."""
    active_settings = settings or get_settings()
    if not active_settings.langsmith_tracing:
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        return

    api_key = active_settings.langsmith_api_key
    if api_key is None or not api_key.get_secret_value().strip():
        raise ValueError(
            "LANGSMITH_API_KEY is required when LANGSMITH_TRACING is enabled"
        )

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key.get_secret_value()
    os.environ["LANGSMITH_PROJECT"] = active_settings.langsmith_project
