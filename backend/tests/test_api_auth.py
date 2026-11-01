from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import HTTPException
from pydantic import SecretStr

import app.api.dependencies as dependencies


def test_bearer_token_maps_to_one_configured_identity(monkeypatch):
    user_id = UUID("88d2c400-5308-41cb-9a9d-e40cdd80e1b7")
    settings = SimpleNamespace(
        workflow_api_token=SecretStr("api-test-token"),
        workflow_user_id=user_id,
    )
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)

    assert dependencies.get_current_user_id(
        "Bearer api-test-token"
    ) == user_id


@pytest.mark.parametrize(
    "authorization",
    [None, "Basic api-test-token", "Bearer wrong-token"],
)
def test_invalid_bearer_credentials_are_rejected(monkeypatch, authorization):
    settings = SimpleNamespace(
        workflow_api_token=SecretStr("api-test-token"),
        workflow_user_id=UUID("88d2c400-5308-41cb-9a9d-e40cdd80e1b7"),
    )
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)

    with pytest.raises(HTTPException) as error:
        dependencies.get_current_user_id(authorization)

    assert error.value.status_code == 401
    assert "WWW-Authenticate" in error.value.headers


def test_workflow_executor_requires_its_separate_token(monkeypatch):
    settings = SimpleNamespace(
        workflow_executor_token=SecretStr("executor-test-token"),
    )
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)

    with pytest.raises(HTTPException) as error:
        dependencies.require_workflow_executor("api-test-token")

    assert error.value.status_code == 403

    assert dependencies.require_workflow_executor(
        "executor-test-token"
    ) is None
