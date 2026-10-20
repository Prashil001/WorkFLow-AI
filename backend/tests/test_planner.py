
from unittest.mock import Mock

import pytest

from app.workflow_engine.planner import (
    PlannerServiceError,
    generate_plan,
)

from app.workflow_engine.schemas import (
    PlannedTask,
    WorkflowPlan,
)


def test_generate_plan_returns_structured_plan():
    expected = WorkflowPlan(
        summary="Prepare client proposal",
        missing_information=[],
        tasks=[
            PlannedTask(
                task_id="T1",
                task_type="retrieve_documents",
                description="Find previous pricing",
                depends_on=[],
            )
        ],
    )

    fake_planner = Mock()
    fake_planner.invoke.return_value = expected

    result = generate_plan(
        "Prepare client proposal",
        planner=fake_planner,
    )

    assert isinstance(result, WorkflowPlan)
    assert result.tasks[0].task_id == "T1"
    fake_planner.invoke.assert_called_once()


def test_rejects_empty_request():
    fake_planner = Mock()

    with pytest.raises(ValueError):
        generate_plan(
            "   ",
            planner=fake_planner,
        )

    fake_planner.invoke.assert_not_called()


def test_provider_error_is_handled():
    fake_planner = Mock()
    fake_planner.invoke.side_effect = TimeoutError(
        "Simulated timeout"
    )

    with pytest.raises(
        PlannerServiceError,
        match="AI planner request failed",
    ):
        generate_plan(
            "Prepare a proposal",
            planner=fake_planner,
        )


def test_rejects_unexpected_response():
    fake_planner = Mock()
    fake_planner.invoke.return_value = {
        "unexpected": "response"
    }

    with pytest.raises(
        PlannerServiceError,
        match="invalid response",
    ):
        generate_plan(
            "Prepare a proposal",
            planner=fake_planner,
        )
