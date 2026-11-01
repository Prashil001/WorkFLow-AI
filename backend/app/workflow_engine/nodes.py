from app.workflow_engine.state import WorkflowGraphState
from app.workflow_engine.schemas import WorkflowPlan
from app.workflow_engine.executor import (
    TaskRunner,
    execute_plan,
    execute_task,
)
from app.workflow_engine.validation import (
    PlanValidationError,
    get_execution_order,
)

from pydantic import ValidationError

from typing import Any, Literal
from collections.abc import Callable

def normalize_request(
        state: WorkflowGraphState,
) -> dict[str,str]:
    original_request = state['original_request']

    cleaned_request = original_request.strip()

    return{
        "normalized_request": cleaned_request
    }


def validate_request(
    state: WorkflowGraphState,
) -> dict[str, bool]:

    request = state['normalized_request']

    if not 1 <= len(request) <= 5000:
        raise ValueError(
            "workflow request must contain "
            "between 1 and 5000 characters"
        )

    return {
        "is_valid": True,
    }


def make_plan_node(
        plan_generator: Callable[[str], WorkflowPlan],
):
    def plan_tasks(
        state: WorkflowGraphState,
    ) -> dict[str, dict[str, Any]]:

        request = state["normalized_request"]
        plan = plan_generator(request)

        return {
            "plan": plan.model_dump(mode="json")
        }

    return plan_tasks


def validate_ai_plan(
    state: WorkflowGraphState,
) -> dict[str, list[str]]:

    # Convert the stored dictionary
    # back into a Pydantic model.
    try:
        plan = WorkflowPlan.model_validate(
            state["plan"]
        )

    except (ValidationError, KeyError, TypeError) as exc:
        raise PlanValidationError(
            "Plan is missing or has invalid structure"
        ) from exc

    # Run basic validation and cycle detection.
    execution_order = get_execution_order(plan)

    # Save the result in LangGraph state.
    return {
        "execution_order": execution_order
    }



def route_after_validation(
    state: WorkflowGraphState,
) -> Literal["needs_input", "ready"]:

    missing_information = (
        state["plan"]["missing_information"]
    )

    if missing_information:
        return "needs_input"

    return "ready"



def mark_needs_input(
    state: WorkflowGraphState,
) -> dict[str, str]:

    return {
        "planning_outcome": "NEEDS_INPUT"
    }


def mark_ready(
    state: WorkflowGraphState,
) -> dict[str, str]:

    return {
        "planning_outcome": "READY"
    }


def make_execution_node(
    task_runner: TaskRunner = execute_task,
):
    def execute_validated_plan(
        state: WorkflowGraphState,
    ) -> dict[str, list[dict[str, Any]]]:
        plan = WorkflowPlan.model_validate(state["plan"])
        results = execute_plan(
            plan,
            state["execution_order"],
            task_runner=task_runner,
        )

        return {
            "execution_results": [
                result.model_dump(mode="json")
                for result in results
            ]
        }

    return execute_validated_plan
