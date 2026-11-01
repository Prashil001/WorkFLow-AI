from uuid import UUID
from collections.abc import Callable

from fastapi import HTTPException, status

from app.workflow_engine.executor import TaskRunner, execute_task
from app.workflow_engine.planner import generate_plan
from app.workflow_engine.schemas import WorkflowPlan

def get_current_user_id() -> UUID:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication is not configured"
    )


def require_workflow_executor() -> None:
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Workflow execution is not configured",
    )


def get_workflow_plan_generator() -> Callable[[str], WorkflowPlan]:
    return generate_plan


def get_workflow_task_runner() -> TaskRunner:
    return execute_task
