from app.evaluation.dataset_checks import load_dataset_v2
from app.evaluation.smoke import (
    DISTRACTOR_STRIDE,
    QUERIES_PER_CATEGORY,
    check_floors,
    select_documents,
    select_queries,
)
from app.rag.corpus import load_docs_documents


def test_smoke_selection_is_fixed_and_covers_every_category():
    queries = select_queries(load_dataset_v2()["queries"])

    assert len(queries) == sum(QUERIES_PER_CATEGORY.values())
    assert [q["id"] for q in queries][:2] == ["easy-01", "easy-02"]
    assert {q["category"] for q in queries} == set(QUERIES_PER_CATEGORY)


def test_smoke_slice_contains_all_relevant_pages_plus_distractors():
    queries = select_queries(load_dataset_v2()["queries"])
    documents = load_docs_documents()

    selected = select_documents(documents, queries)
    selected_ids = {d["document_id"] for d in selected}
    relevant = {s for q in queries for s in q["relevant_source_ids"]}

    assert relevant <= selected_ids
    assert len(selected_ids - relevant) >= len(documents) // DISTRACTOR_STRIDE - len(relevant)
    assert len(selected) < len(documents)


def test_smoke_floors_report_each_failure():
    failures = check_floors({
        "vector": {"recall@5": 0.9, "mrr@10": 0.1},
        "hybrid": {"recall@5": 0.1, "mrr@10": 0.9},
    })

    assert failures == [
        "vector mrr@10=0.1 is below the floor 0.6",
        "hybrid recall@5=0.1 is below the floor 0.7",
    ]
    assert check_floors({
        "vector": {"recall@5": 1.0, "mrr@10": 1.0},
        "hybrid": {"recall@5": 1.0, "mrr@10": 1.0},
    }) == []
