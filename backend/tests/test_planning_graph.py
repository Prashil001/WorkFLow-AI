
import pytest

from app.workflow_engine.graph import build_workflow_graph
from app.workflow_engine.planner import PlannerServiceError

from app.workflow_engine.schemas import (
    PlannedTask,
    WorkflowPlan,
)


def test_graph_generates_plan():
    received_requests = []

    def fake_planner(request: str) -> WorkflowPlan:
        received_requests.append(request)

        return WorkflowPlan(
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

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    result = graph.invoke({
        "original_request":
            "  Prepare client proposal  "
    })

    assert received_requests == [
        "Prepare client proposal"
    ]

    assert result["is_valid"] is True

    assert result["plan"]["summary"] == (
        "Prepare client proposal"
    )

    assert result["plan"]["tasks"][0]["task_id"] == "T1"

    assert isinstance(result["plan"], dict)


def test_invalid_request_never_calls_ai():
    def fake_planner(request: str) -> WorkflowPlan:
        raise AssertionError(
            "AI should not be called"
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    with pytest.raises(ValueError):
        graph.invoke({
            "original_request": "   "
        })


def test_planner_failure_stops_graph():
    def failing_planner(request: str) -> WorkflowPlan:
        raise PlannerServiceError(
            "Simulated planner failure"
        )

    graph = build_workflow_graph(
        plan_generator=failing_planner
    )

    with pytest.raises(PlannerServiceError):
        graph.invoke({
            "original_request": "Prepare client proposal"
        })
