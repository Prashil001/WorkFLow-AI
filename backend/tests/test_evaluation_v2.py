import copy
import itertools

import pytest

from app.evaluation.dataset_checks import find_dataset_problems, load_dataset_v2
from app.evaluation.generation import (
    ABSTAIN_TEXT,
    ClaimVerdict,
    JudgeVerdict,
    check_citations,
    cited_source_ids,
    is_abstention,
    validate_verdict,
    with_retries,
)
from app.evaluation.metrics import (
    latency_summary,
    ndcg_at_k,
    rank_documents,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.run_v2 import (
    RetrieverConfig,
    evaluate_generation,
    evaluate_retrieval,
    run_evaluation,
)
from app.workflow_engine.rag import RetrievedChunk, retrieve_pricing_documents


def _chunk(source_id, score=0.9, index=0):
    return RetrievedChunk(source_id, "Docs", "Title", index, f"text of {source_id}", score)


# --- metrics -------------------------------------------------------------------


def test_rank_documents_collapses_chunks_to_first_occurrence():
    chunks = [_chunk("a"), _chunk("b"), _chunk("a", index=1), _chunk("c")]

    assert rank_documents(chunks) == ["a", "b", "c"]


def test_recall_mrr_and_ndcg_match_hand_computed_values():
    ranked = ["x", "a", "y", "b", "z"]
    relevant = {"a", "b"}

    assert recall_at_k(ranked, relevant, 1) == 0.0
    assert recall_at_k(ranked, relevant, 3) == 0.5
    assert recall_at_k(ranked, relevant, 5) == 1.0
    assert reciprocal_rank(ranked, relevant) == 0.5
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1 + 1/log2(3)
    assert ndcg_at_k(ranked, relevant, 5) == pytest.approx(
        (1 / 1.5849625 + 1 / 2.3219281) / (1 + 1 / 1.5849625), rel=1e-6
    )
    assert ndcg_at_k(["a", "b"], relevant, 5) == pytest.approx(1.0)
    assert reciprocal_rank(["x"], relevant) == 0.0


def test_metrics_reject_queries_without_relevant_documents():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), 5)


def test_latency_summary_percentiles():
    summary = latency_summary([float(value) for value in range(1, 101)])

    assert summary["count"] == 100
    assert summary["p50_ms"] == pytest.approx(50.5)
    assert summary["p95_ms"] == pytest.approx(95.05)
    assert latency_summary([]) == {"count": 0}


# --- generation helpers --------------------------------------------------------


def test_citation_checks_flag_unknown_and_missing_citations():
    answer = "Use CORSMiddleware [source:a]. Also see [source:zzz]."

    check = check_citations(answer, context_ids={"a", "b"}, relevant_ids={"a"})

    assert cited_source_ids(answer) == ["a", "zzz"]
    assert check.unknown == ["zzz"] and check.valid is False
    assert check.cites_relevant is True
    assert check_citations("No citations.", {"a"}, {"a"}).has_citation is False
    assert check_citations("x [source:a]", {"a"}, set()).cites_relevant is None


def test_abstention_detection_is_tolerant_of_case_and_punctuation():
    assert is_abstention(ABSTAIN_TEXT)
    assert is_abstention("i don't know based on the provided documents")
    assert not is_abstention("Use CORSMiddleware [source:a].")


def test_judge_scores_must_be_in_range():
    with pytest.raises(ValueError):
        validate_verdict(JudgeVerdict(claims=[], answer_relevance=6, answer_correctness=3, rationale=""))


def test_with_retries_backs_off_then_succeeds_or_raises():
    calls = itertools.count()
    delays = []

    def flaky():
        if next(calls) < 2:
            raise RuntimeError("429")
        return "ok"

    assert with_retries(flaky, sleep=delays.append) == "ok"
    assert delays == [2.0, 4.0]

    with pytest.raises(RuntimeError):
        with_retries(lambda: (_ for _ in ()).throw(RuntimeError("down")), attempts=2, sleep=lambda _: None)


# --- retrieval and generation evaluation ----------------------------------------


QUERIES = [
    {"id": "q1", "category": "easy", "question": "one", "relevant_source_ids": ["a"],
     "reference_answer": "A."},
    {"id": "q2", "category": "multi_hop", "question": "two", "relevant_source_ids": ["b", "c"],
     "reference_answer": "B and C."},
    {"id": "q3", "category": "unanswerable", "question": "three", "relevant_source_ids": [],
     "reference_answer": "Not covered."},
]


class MappingRetriever:
    def __init__(self, rankings):
        self.rankings = rankings
        self.calls = []

    def retrieve(self, query, *, limit=3):
        self.calls.append((query, limit))
        return [_chunk(source_id, 0.9 - 0.1 * i) for i, source_id in enumerate(self.rankings[query])]


def test_evaluate_retrieval_reports_per_query_and_aggregate_metrics():
    retriever = MappingRetriever({
        "one": ["a", "x"],
        "two": ["x", "b", "y", "z", "q", "c"],
        "three": ["x"],
    })
    ticks = itertools.count(step=0.004)

    report, chunks = evaluate_retrieval(QUERIES, retriever, clock=lambda: next(ticks), warmup=1)

    metrics = report["metrics"]
    assert report["answerable_query_count"] == 2
    assert metrics["recall@1"] == 0.5          # q1: 1, q2: 0
    assert metrics["recall@5"] == 0.75         # q1: 1, q2: 0.5 (c is 6th)
    assert metrics["mrr@10"] == 0.75           # 1 and 1/2
    assert report["metrics_by_category"]["multi_hop"]["recall@5"] == 0.5
    assert report["latency"]["count"] == 3
    assert report["latency"]["p50_ms"] == pytest.approx(4.0)
    assert len(retriever.calls) == 4           # 1 warm-up + 3 measured
    assert [c.source_id for c in chunks["q1"]] == ["a", "x"]


def test_evaluate_generation_scores_answers_abstentions_and_citations():
    chunks = {"q1": [_chunk("a")], "q2": [_chunk("b"), _chunk("x")], "q3": [_chunk("x")]}
    answers = {
        "one": "A is the answer [source:a].",
        "two": "B is true [source:b]. C is invented [source:c].",
        "three": ABSTAIN_TEXT,
    }

    def generator(question, context):
        return answers[question]

    def judge(question, reference, context, answer):
        if question == "one":
            return JudgeVerdict(claims=[ClaimVerdict(claim="A", supported=True)],
                                answer_relevance=5, answer_correctness=5, rationale="ok")
        return JudgeVerdict(claims=[ClaimVerdict(claim="B", supported=True),
                                    ClaimVerdict(claim="C", supported=False)],
                            answer_relevance=4, answer_correctness=3, rationale="partial")

    report = evaluate_generation(QUERIES, chunks, generator, judge)
    metrics = report["metrics"]

    assert metrics["faithfulness"]["mean"] == 0.75
    assert metrics["faithfulness"]["fully_grounded_rate"] == 0.5
    assert metrics["answer_relevance"]["mean"] == 4.5
    assert metrics["answer_correctness"]["mean"] == 4.0
    assert metrics["abstention"]["unanswerable_abstention_rate"] == 1.0
    assert metrics["abstention"]["answerable_false_abstention_rate"] == 0.0
    # q2 cites "c", which was not in its context.
    assert metrics["citation_integrity"]["valid_citation_rate"] == 0.5
    assert metrics["citation_integrity"]["cites_relevant_source_rate"] == 1.0
    assert "judge" not in report["cases"][2]


def test_false_abstention_counts_as_lowest_correctness():
    report = evaluate_generation(
        QUERIES[:1],
        {"q1": [_chunk("a")]},
        lambda question, context: ABSTAIN_TEXT,
        lambda *args: pytest.fail("judge must not run for abstentions"),
    )

    assert report["metrics"]["answer_correctness"]["mean"] == 1
    assert report["metrics"]["abstention"]["answerable_false_abstention_rate"] == 1.0


def test_generation_and_judge_errors_are_recorded_not_fatal():
    def generator(question, context):
        if question == "one":
            raise ValueError("The generator returned no text")
        return "B [source:b]." if question == "two" else ABSTAIN_TEXT

    def judge(*args):
        raise RuntimeError("judge down")

    report = evaluate_generation(
        QUERIES,
        {"q1": [_chunk("a")], "q2": [_chunk("b")], "q3": [_chunk("x")]},
        generator,
        judge,
    )
    metrics = report["metrics"]

    assert metrics["generation_error_count"] == 1
    assert metrics["judge_error_count"] == 1
    assert report["cases"][0]["generation_error"].startswith("ValueError")
    assert metrics["answer_correctness"]["mean"] == 1   # both answerable cases failed
    assert metrics["answer_relevance"]["mean"] is None
    assert metrics["answered_count"] == 1
    assert metrics["abstention"]["unanswerable_abstention_rate"] == 1.0


def test_resume_reuses_only_clean_cases_with_identical_context():
    chunks = {"q1": [_chunk("a")], "q2": [_chunk("b")], "q3": [_chunk("x")]}
    previous = {
        "q1": {"id": "q1", "category": "easy", "answerable": True, "context_source_ids": ["a"],
               "answer": "Old A [source:a].", "abstained": False,
               "citations": {"valid": True, "has_citation": True, "cites_relevant": True},
               "judge": {"claims": [], "answer_relevance": 5, "answer_correctness": 5,
                         "rationale": ""}, "faithfulness": 1.0},
        "q2": {"id": "q2", "context_source_ids": ["b"], "generation_error": "RateLimitError"},
        "q3": {"id": "q3", "context_source_ids": ["DIFFERENT"], "answer": ABSTAIN_TEXT,
               "abstained": True},
    }
    asked = []

    def generator(question, context):
        asked.append(question)
        return ABSTAIN_TEXT

    report = evaluate_generation(QUERIES, chunks, generator, lambda *a: None,
                                 previous_cases=previous)

    assert report["reused_case_count"] == 1
    assert asked == ["two", "three"]          # errored and changed-context cases rerun
    assert report["cases"][0]["answer"] == "Old A [source:a]."


def test_resume_ignores_reports_from_other_models_or_datasets():
    from app.evaluation.run_v2 import _resumable_cases

    details = {"generator_model": "g", "judge_model": "j", "prompt_sha256": "p",
               "reasoning_effort": "low"}
    report = {"dataset": {"sha256": "d"},
              "generation": {**details, "cases": [{"id": "q1"}]}}

    assert set(_resumable_cases(report, "d", details)) == {"q1"}
    assert _resumable_cases(report, "other", details) == {}
    assert _resumable_cases(report, "d", {**details, "judge_model": "x"}) == {}
    assert _resumable_cases(None, "d", details) == {}


def test_empty_generator_output_is_retried():
    from app.evaluation import generation as generation_module

    responses = iter(["", "", "Answer [source:a]."])

    class FakeModel:
        def invoke(self, messages):
            return type("Response", (), {"content": next(responses)})()

    generator = object.__new__(generation_module.GroqAnswerGenerator)
    generator._model = FakeModel()
    original = generation_module.with_retries
    generation_module.with_retries = lambda call: original(call, sleep=lambda _: None)
    try:
        assert generator("q", [_chunk("a")]) == "Answer [source:a]."
    finally:
        generation_module.with_retries = original


def test_run_evaluation_end_to_end_with_gold_retriever():
    dataset = load_dataset_v2()
    rankings = {
        query["question"]: query["relevant_source_ids"] or ["fastapi__index"]
        for query in dataset["queries"]
    }
    config = RetrieverConfig(name="gold", retriever=MappingRetriever(rankings))

    report = run_evaluation(config)

    retrieval = report["retrieval"]["metrics"]
    assert retrieval["query_count"] == 65
    assert retrieval["recall@5"] == 1.0 and retrieval["mrr@10"] == 1.0
    assert retrieval["ndcg@5"] == 1.0
    assert report["generation"] is None
    assert report["dataset"]["version"] == 2
    assert len(report["dataset"]["sha256"]) == 64
    assert report["corpus"]["document_count"] == 227


# --- dataset -----------------------------------------------------------------------


def test_v2_dataset_is_verified_against_the_corpus():
    dataset = load_dataset_v2()
    categories = {query["category"] for query in dataset["queries"]}

    assert 60 <= len(dataset["queries"]) <= 100
    assert categories == {"easy", "paraphrased", "multi_hop", "unanswerable"}
    assert find_dataset_problems(dataset) == []


def test_dataset_checks_catch_bad_quotes_absence_hits_and_thin_multi_hop():
    dataset = copy.deepcopy(load_dataset_v2())
    queries = {query["id"]: query for query in dataset["queries"]}
    queries["easy-01"]["evidence"][0]["quote"] = "This sentence is not in the docs."
    queries["unans-01"]["absence_pattern"] = "FastAPI"
    queries["multi-01"]["evidence"] = queries["multi-01"]["evidence"][:1]

    problems = find_dataset_problems(dataset)

    assert any(p.startswith("easy-01: quote not found") for p in problems)
    assert any(p.startswith("unans-01: absence_pattern found") for p in problems)
    assert any(p.startswith("multi-01: multi-hop evidence") for p in problems)


def test_tool_results_stay_capped_at_ten():
    retriever = MappingRetriever({"q": ["a"]})

    retrieve_pricing_documents("q", limit=50, retriever=retriever)

    assert retriever.calls == [("q", 10)]
