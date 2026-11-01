from langgraph.graph import StateGraph, START, END

from collections.abc import Callable

from app.workflow_engine.state import WorkflowGraphState
from app.workflow_engine.schemas import WorkflowPlan
from app.workflow_engine.planner import generate_plan
from app.workflow_engine.nodes import (
    normalize_request,
    validate_request,
    make_plan_node,
    validate_ai_plan,
    route_after_validation,
    mark_needs_input,
    mark_ready,
    make_execution_node,
)
from app.workflow_engine.executor import TaskRunner, execute_task

def build_workflow_graph(
        plan_generator: Callable[[str], WorkflowPlan] = generate_plan,
        task_runner: TaskRunner | None = None,
):
    if task_runner is None:
        task_runner = execute_task
    
    builder = StateGraph(WorkflowGraphState)

    builder.add_node(
        "normalize_request",
        normalize_request
    )

    builder.add_node(
        "validate_request",
        validate_request
    )

    builder.add_node(
        "validate_ai_plan",
        validate_ai_plan,
    )

    builder.add_node(
        "plan_tasks",
        make_plan_node(plan_generator)
    )

    builder.add_node(
        "mark_needs_input",
        mark_needs_input,
    )

    builder.add_node(
        "mark_ready",
        mark_ready,
    )

    builder.add_node(
        "execute_tasks",
        make_execution_node(task_runner),
    )

    builder.add_edge(START, "normalize_request")
    builder.add_edge("normalize_request", "validate_request")
    builder.add_edge("validate_request","plan_tasks")
    builder.add_edge(
        "plan_tasks",
        "validate_ai_plan",
    )

    builder.add_conditional_edges(
        "validate_ai_plan",
        route_after_validation,
        {
            "needs_input": "mark_needs_input",
            "ready": "mark_ready",
        },
    )

    builder.add_edge(
        "mark_needs_input",
        END,
    )

    builder.add_edge(
        "mark_ready",
        "execute_tasks",
    )

    builder.add_edge(
        "execute_tasks",
        END,
    )


    graph = builder.compile()

    return graph
