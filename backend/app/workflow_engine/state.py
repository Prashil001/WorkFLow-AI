from typing import TypedDict, NotRequired

class WorkflowGraphState(TypedDict):
    original_request: str
    normalized_request: NotRequired[str]
    is_valid: NotRequired[bool]

    