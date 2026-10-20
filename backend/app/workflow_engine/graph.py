from langgraph.graph import StateGraph, START, END

from collections.abc import Callable

from app.workflow_engine.state import WorkflowGraphState
from app.workflow_engine.schemas import WorkflowPlan
from app.workflow_engine.planner import generate_plan
from app.workflow_engine.nodes import (
    normalize_request,
    validate_request,
    make_plan_node,
)

def build_workflow_graph(
        plan_generator: Callable[[str], WorkflowPlan] = generate_plan,
):
    
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
        "plan_tasks",
        make_plan_node(plan_generator)
    )

    builder.add_edge(START, "normalize_request")
    builder.add_edge("normalize_request", "validate_request")
    builder.add_edge("validate_request","plan_tasks")
    builder.add_edge("plan_tasks", END)

    graph = builder.compile()

    return graph