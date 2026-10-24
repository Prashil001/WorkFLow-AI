from typing import TypedDict, NotRequired, Any

class WorkflowGraphState(TypedDict):
    original_request: str
    normalized_request: NotRequired[str]
    is_valid: NotRequired[bool]
    plan: NotRequired[dict[str, Any]]
    execution_order: NotRequired[list[str]]

