from collections.abc import Mapping

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.workflow_engine.schemas import (
    PlannedTask,
    TaskExecutionResult,
)


def generate_grounded_draft(
    task: PlannedTask,
    source_documents: list[dict[str, str]],
    dependency_results: Mapping[str, TaskExecutionResult],
) -> str:
    settings = get_settings()
    if (
        settings.groq_api_key is None
        or not settings.groq_api_key.get_secret_value()
    ):
        raise ValueError("GROQ_API_KEY is not configured")

    source_text = "\n\n".join(
        f"[source:{document['document_id']}] "
        f"{document['title']} ({document['client']}): "
        f"{document['content']}"
        for document in source_documents
    )
    upstream_outputs = "\n\n".join(
        f"Task {task_id}: {result.result or ''}"
        for task_id, result in dependency_results.items()
        if result.status == "SIMULATED" and result.result
    )

    model = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key.get_secret_value(),
        temperature=0,
        timeout=30,
        max_retries=0,
    )
    response = model.invoke([
        SystemMessage(content=(
            "Write the requested business draft using only the supplied "
            "source documents and upstream task outputs. Cite factual claims "
            "with the exact source ID in brackets, such as "
            "[source:mock-pricing-001]. Do not invent names, prices, dates, "
            "or commitments. If the sources do not answer something, say it "
            "needs confirmation. Produce a draft only; do not send or publish it."
        )),
        HumanMessage(content=(
            f"Task: {task.description}\n\n"
            f"Retrieved source documents:\n{source_text}\n\n"
            f"Upstream task outputs:\n{upstream_outputs}"
        )),
    ])

    content = response.content
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, str) and block.strip():
                text_blocks.append(block.strip())
            elif isinstance(block, dict) and block.get("text"):
                text_blocks.append(str(block["text"]).strip())
        if text_blocks:
            return "\n".join(text_blocks).strip()

    raise ValueError("The drafting model returned no text")
