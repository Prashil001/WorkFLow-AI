from functools import lru_cache
import os
from typing import  Literal
from pydantic import SecretStr

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
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "workflow-ai"

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
