from langgraph.graph import StateGraph, START, END

from app.workflow_engine.state import WorkflowGraphState
from app.workflow_engine.nodes import (
    normalize_request,
    validate_request,
)

def build_workflow_graph():
    builder = StateGraph(WorkflowGraphState)

    builder.add_node(
        "normalize_request",
        normalize_request
    )

    builder.add_node(
        "validate_request",
        validate_request
    )

    builder.add_edge(START, "normalize_request")
    builder.add_edge("normalize_request", "validate_request")
    builder.add_edge("validate_request", END)

    graph = builder.compile()

    return graph