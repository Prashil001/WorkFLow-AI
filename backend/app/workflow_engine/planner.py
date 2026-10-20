from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import(
    HumanMessage,
    SystemMessage,
)

from app.core.config import get_settings
from app.workflow_engine.schemas import WorkflowPlan

class PlannerServiceError(Exception):
    pass

PLANNER_INSTRUCTIONS = """
JSON output rules:
- "tasks" MUST be a JSON array, even when empty.
- Never wrap tasks in an object containing "items".
- "missing_information" MUST be an array of strings.
- Each task must contain task_id, task_type,
  description, and depends_on.
- "depends_on" MUST be an array of task ID strings.
- If the user has not identified the client,
  include that in missing_information.
- Do not invent missing client details.

Correct structure example:
{
  "summary": "Prepare a client proposal",
  "missing_information": ["Which client?"],
  "tasks": [
    {
      "task_id": "T1",
      "task_type": "retrieve_documents",
      "description": "Find relevant proposal templates",
      "depends_on": []
    }
  ]
}
"""


def create_planner() -> Any:
    settings = get_settings()

    if (
        settings.groq_api_key is None
        or not settings.groq_api_key.get_secret_value()
    ):
        raise PlannerServiceError(
            "GROQ_API_KEY is not configured"
        )

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key.get_secret_value(),
        temperature=0,
        timeout=30,
        max_retries=0,
    )

    return llm.with_structured_output(
        WorkflowPlan,
        method="json_schema",
        strict=True,
    )


def generate_plan(
    request: str,
    *,
    planner: Any | None = None,
) -> WorkflowPlan:

    if(
        not isinstance(request, str)
        or not 1 <= len(request.strip()) <= 5000
    ):
        raise ValueError("Invalid workflow request")

    active_planner = (
        planner if planner is not None
        else create_planner()
    )

    try:
        result = active_planner.invoke([
            SystemMessage(content=PLANNER_INSTRUCTIONS),
            HumanMessage(content=request.strip()),
        ])
    except Exception as exc:
        raise PlannerServiceError(
            "AI planner request failed"
        ) from exc

    if not isinstance(result, WorkflowPlan):
        raise PlannerServiceError(
            "AI planner returned an invalid response"
        )

    return result