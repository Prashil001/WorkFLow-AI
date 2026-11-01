
from app.workflow_engine.graph import build_workflow_graph
from app.workflow_engine.schemas import (
    PlannedTask,
    WorkflowPlan,
)


def test_ready_route():

    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Prepare client proposal",
            missing_information=[],
            tasks=[
                PlannedTask(
                    task_id="T1",
                    task_type="retrieve_documents",
                    description="Find client pricing",
                    depends_on=[],
                ),
            ],
        )

    graph = build_workflow_graph(
        plan_generator=fake_planner
    )

    result = graph.invoke({
        "original_request": "Prepare proposal for Client ABC"
    })

    assert result["planning_outcome"] == "READY"
    assert result["execution_order"] == ["T1"]


def test_missing_information_route():

    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Need client information",
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

    assert result["planning_outcome"] == "NEEDS_INPUT"
    assert result["execution_order"] == []
    assert result["plan"]["missing_information"] == [
        "Which client is this for?"
    ]


def test_missing_information_takes_priority():

    def fake_planner(request: str) -> WorkflowPlan:
        return WorkflowPlan(
            summary="Prepare proposal",
            missing_information=[
                "What is the client's budget?"
            ],
            tasks=[
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

    assert result["planning_outcome"] == "NEEDS_INPUT"
    assert result["execution_order"] == ["T1"]
