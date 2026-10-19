from functools import lru_cache
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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()