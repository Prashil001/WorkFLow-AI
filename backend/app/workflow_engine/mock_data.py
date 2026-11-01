import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_DATA_PATH = Path(__file__).parent / "data" / "mock_business_data.json"


@lru_cache(maxsize=1)
def load_mock_business_data() -> dict[str, Any]:
    return json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def load_pricing_documents_by_ids(
    source_ids: set[str],
) -> list[dict[str, str]]:
    return [
        document
        for document in load_mock_business_data()["pricing_documents"]
        if document["document_id"] in source_ids
    ]
