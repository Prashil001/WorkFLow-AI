from app.workflow_engine.state import WorkflowGraphState

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