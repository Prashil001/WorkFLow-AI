
import pytest

from app.workflow_engine.graph import build_workflow_graph
from app.workflow_engine.nodes import validate_ai_plan

from app.workflow_engine.schemas import (
    PlannedTask,
    WorkflowPlan,
)

from app.workflow_engine.validation import PlanValidationError


def test_graph_orders_tasks_correctly():

    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Prepare client proposal",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T3",
                    task_type="draft_email",
                    description="Draft the email",
                    depends_on=["T2"],
                ),
                PlannedTask(
                    task_id="T2",
                    task_type="draft_proposal",
                    description="Draft the proposal",
                    depends_on=["T1"],
                ),
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Find previous pricing",
                    depends_on=[],
                ),
            ],
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    result = graph.invoke({
        "original_request": "Prepare client proposal"
    })

    assert result["execution_order"] == [
        "T1", "T2", "T3"
    ]

    assert len(result["plan"]["tasks"]) == 3



def test_graph_rejects_circular_plan():

    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Invalid circular plan",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="First task",
                    depends_on=["T2"],
                ),
                PlannedTask(
                    task_id="T2",
                    task_type="draft_proposal",
                    description="Second task",
                    depends_on=["T1"],
                ),
            ],
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    with pytest.raises(
        PlanValidationError,
        match="Circular dependency",
    ):
        graph.invoke({
            "original_request": "Prepare client proposal"
        })


def test_graph_rejects_unknown_dependency():

    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Invalid dependency",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="draft_proposal",
                    description="Prepare proposal",
                    depends_on=["T99"],
                ),
            ],
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    with pytest.raises(
        PlanValidationError,
        match="unknown task",
    ):
        graph.invoke({
            "original_request": "Prepare proposal"
        })



def test_graph_accepts_clarification_plan():

    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Client details required",
            missing_information=[
                "Which client is this for?"
            ],
            tasks=[],
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    result = graph.invoke({
        "original_request": "Prepare a proposal"
    })

    assert result["execution_order"] == []

    assert result["plan"]["missing_information"] == [
        "Which client is this for?"
    ]


def test_validation_node_rejects_wrong_structure():

    with pytest.raises(
        PlanValidationError,
        match="invalid structure",
    ):
        validate_ai_plan({
            "plan": {
                "summary": "Prepare proposal",
                "missing_information": [],
                "tasks": {
                    "items": []
                },
            }
        })


def test_graph_rejects_partial_cycle():

    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Plan containing a cycle",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Retrieve pricing",
                    depends_on=[],
                ),
                PlannedTask(
                    task_id="T2",
                    task_type="draft_proposal",
                    description="Draft proposal",
                    depends_on=["T3"],
                ),
                PlannedTask(
                    task_id="T3",
                    task_type="draft_email",
                    description="Draft email",
                    depends_on=["T2"],
                ),
            ],
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    with pytest.raises(
        PlanValidationError,
        match="Circular dependency",
    ):
        graph.invoke({
            "original_request": "Prepare a proposal"
        })



def test_graph_rejects_duplicate_task_ids():

    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Plan with duplicates",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Find pricing",
                    depends_on=[],
                ),
                PlannedTask(
                    task_id="T1",
                    task_type="draft_proposal",
                    description="Draft proposal",
                    depends_on=[],
                ),
            ],
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    with pytest.raises(
        PlanValidationError,
        match="Duplicate task IDs",
    ):
        graph.invoke({
            "original_request": "Prepare proposal"
        })



def test_graph_orders_multiple_dependencies():

    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Prepare proposal and email",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T4",
                    task_type="draft_email",
                    description="Draft follow-up email",
                    depends_on=["T3"],
                ),
                PlannedTask(
                    task_id="T3",
                    task_type="draft_proposal",
                    description="Draft proposal",
                    depends_on=["T1", "T2"],
                ),
                PlannedTask(
                    task_id="T2",
                    task_type="check_calendar",
                    description="Check available times",
                    depends_on=[],
                ),
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Retrieve pricing",
                    depends_on=[],
                ),
            ],
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    result = graph.invoke({
        "original_request": "Prepare client proposal"
    })

    order = result["execution_order"]

    # Every task must appear exactly once.
    assert len(order) == 4
    assert set(order) == {"T1", "T2", "T3", "T4"}

    # Dependencies must run before dependents.
    assert order.index("T1") < order.index("T3")
    assert order.index("T2") < order.index("T3")
    assert order.index("T3") < order.index("T4")

    # The AI's original plan is preserved.
    assert result["plan"]["tasks"][0]["task_id"] == "T4"
