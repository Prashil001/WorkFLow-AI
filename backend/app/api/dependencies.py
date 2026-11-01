import hmac
from typing import Annotated
from uuid import UUID
from collections.abc import Callable

from fastapi import Header, HTTPException, status

from app.core.config import get_settings
from app.workflow_engine.executor import TaskRunner, execute_task
from app.workflow_engine.planner import generate_plan
from app.workflow_engine.schemas import WorkflowPlan

def get_current_user_id(
    authorization: Annotated[str | None, Header()] = None,
) -> UUID:
    settings = get_settings()
    expected_token = settings.workflow_api_token
    if expected_token is None or not expected_token.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API authentication is not configured",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, separator, supplied_token = (
        authorization.partition(" ")
        if authorization is not None
        else ("", "", "")
    )
    if (
        scheme.casefold() != "bearer"
        or not separator
        or not supplied_token
        or not hmac.compare_digest(
            supplied_token,
            expected_token.get_secret_value(),
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return settings.workflow_user_id


def require_workflow_executor(
    executor_token: Annotated[
        str | None,
        Header(alias="X-Workflow-Executor-Token"),
    ] = None,
) -> None:
    expected_token = get_settings().workflow_executor_token
    if expected_token is None or not expected_token.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workflow execution is not configured",
        )

    if (
        executor_token is None
        or not hmac.compare_digest(
            executor_token,
            expected_token.get_secret_value(),
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workflow executor token is invalid",
        )


def get_workflow_plan_generator() -> Callable[[str], WorkflowPlan]:
    return generate_plan


def get_workflow_task_runner() -> TaskRunner:
    return execute_task
