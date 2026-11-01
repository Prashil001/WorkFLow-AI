import os

import pytest
from pydantic import SecretStr

from app.core.config import configure_langsmith_tracing


def test_langsmith_tracing_is_off_by_default(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    settings = type("SettingsStub", (), {
        "langsmith_tracing": False,
        "langsmith_api_key": None,
    })()

    configure_langsmith_tracing(settings)

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ


def test_langsmith_tracing_requires_an_api_key(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    settings = type("SettingsStub", (), {
        "langsmith_tracing": True,
        "langsmith_api_key": None,
    })()

    with pytest.raises(ValueError, match="LANGSMITH_API_KEY is required"):
        configure_langsmith_tracing(settings)


def test_langsmith_tracing_uses_configured_project_without_logging_secret(
    monkeypatch,
):
    monkeypatch.setattr(os, "environ", {})
    settings = type("SettingsStub", (), {
        "langsmith_tracing": True,
        "langsmith_api_key": SecretStr("test-key-do-not-log"),
        "langsmith_project": "workflow-ai-tests",
    })()

    configure_langsmith_tracing(settings)

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == (
        "workflow-ai-tests"
    )
    assert os.environ["LANGSMITH_API_KEY"] == (
        "test-key-do-not-log"
    )
