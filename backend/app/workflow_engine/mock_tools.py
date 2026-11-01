import json
import re

from app.workflow_engine.schemas import PlannedTask
from app.workflow_engine.mock_data import load_mock_business_data
from app.workflow_engine.rag import (
    InMemoryRetriever,
    retrieve_pricing_documents,
)
_STOP_WORDS = {
    "a",
    "about",
    "and",
    "for",
    "find",
    "from",
    "in",
    "of",
    "the",
    "to",
    "with",
}


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if term not in _STOP_WORDS
    }


def search_pricing_documents(
    query: str,
    limit: int = 3,
    *,
    retriever: InMemoryRetriever | None = None,
) -> str:
    """Retrieve relevant passages from the fictional pricing dataset."""
    matches = retrieve_pricing_documents(
        query,
        limit=limit,
        retriever=retriever,
    ) if query.strip() else []
    return json.dumps({
        "source": "mock_local_dataset",
        "retrieval_method": "fastembed_cosine",
        "query": query,
        "notice": "All pricing data is fictional sample data.",
        "matches": matches,
    })


def check_calendar_availability(query: str) -> str:
    """Return matching sample slots from the fixed mock calendar."""
    data = load_mock_business_data()
    slots = data["calendar_slots"]
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", query)
    requested_date = date_match.group(0) if date_match else None
    query_terms = _terms(query)

    if requested_date:
        slots = [slot for slot in slots if slot["date"] == requested_date]

    if "morning" in query_terms:
        slots = [slot for slot in slots if int(slot["time"][:2]) < 12]
    elif "afternoon" in query_terms:
        slots = [slot for slot in slots if int(slot["time"][:2]) >= 12]

    return json.dumps({
        "source": "mock_local_dataset",
        "query": query,
        "notice": "Availability is fictional sample data, not a live calendar.",
        "slots": slots,
    })


def run_read_only_tool(
    task: PlannedTask,
    *,
    retriever: InMemoryRetriever | None = None,
) -> str | None:
    """Dispatch only explicitly allowed read-only task types."""
    if task.task_type == "retrieve_documents":
        return search_pricing_documents(
            task.description,
            retriever=retriever,
        )
    if task.task_type == "check_calendar":
        return check_calendar_availability(task.description)
    return None
