import json
import re
from pathlib import Path
from typing import Any

from app.workflow_engine.schemas import PlannedTask


_DATA_PATH = Path(__file__).parent / "data" / "mock_business_data.json"
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


def _load_mock_data() -> dict[str, Any]:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if term not in _STOP_WORDS
    }


def search_pricing_documents(
    query: str,
    limit: int = 3,
) -> str:
    """Search the fixed fictional pricing dataset by matching terms."""
    if not query.strip():
        return json.dumps({
            "source": "mock_local_dataset",
            "query": query,
            "matches": [],
        })

    data = _load_mock_data()
    query_terms = _terms(query)
    ranked_documents = []

    for document in data["pricing_documents"]:
        searchable_text = " ".join(
            str(document[field])
            for field in ("client", "title", "content")
        )
        score = len(query_terms & _terms(searchable_text))
        if score:
            ranked_documents.append((score, document))

    ranked_documents.sort(
        key=lambda item: (-item[0], item[1]["document_id"])
    )
    matches = [
        document
        for _, document in ranked_documents[: max(1, min(limit, 5))]
    ]

    return json.dumps({
        "source": "mock_local_dataset",
        "query": query,
        "notice": "All pricing data is fictional sample data.",
        "matches": matches,
    })


def check_calendar_availability(query: str) -> str:
    """Return matching sample slots from the fixed mock calendar."""
    data = _load_mock_data()
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


def run_read_only_tool(task: PlannedTask) -> str | None:
    """Dispatch only explicitly allowed read-only task types."""
    if task.task_type == "retrieve_documents":
        return search_pricing_documents(task.description)
    if task.task_type == "check_calendar":
        return check_calendar_availability(task.description)
    return None
