from uuid import UUID

from fastapi import HTTPException, status

def get_current_user_id() -> UUID:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication is not configured"
    )


def require_workflow_executor() -> None:
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Workflow execution is not configured",
    )