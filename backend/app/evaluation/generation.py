"""Grounded answer generation, LLM-as-judge scoring, and citation checks."""

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.workflow_engine.rag import RetrievedChunk


PROMPT_VERSION = 1
ABSTAIN_TEXT = "I don't know based on the provided documents."
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
_CITATION = re.compile(r"\[source:([A-Za-z0-9_-]+)\]")

GENERATOR_INSTRUCTIONS = f"""You answer questions about FastAPI, Pydantic, and SQLModel
using only the numbered context passages provided.
- Cite every factual sentence with the passage's source ID in brackets, e.g. [source:fastapi__tutorial__cors].
- Use only source IDs that appear in the context.
- If the context does not contain the answer, reply with exactly: {ABSTAIN_TEXT}
- Be concise: at most 120 words."""

JUDGE_INSTRUCTIONS = """You are a strict evaluator of answers produced by a retrieval-augmented system.
You receive the question, a reference answer, the context passages the system saw, and its answer.

1. claims: split the system answer into its atomic factual claims (ignore citations and
   filler). For each claim set supported=true only if the context passages state or
   directly imply it. Judge support against the context, not against your own knowledge.
2. answer_relevance (1-5): how directly the answer addresses the question asked,
   regardless of correctness. 5 = fully on point, 1 = off-topic.
3. answer_correctness (1-5): agreement with the reference answer. 5 = all key facts of the
   reference are present and nothing contradicts it, 3 = partially correct, 1 = wrong or
   missing the key facts.
4. rationale: one short sentence."""


class EmptyGenerationError(ValueError):
    pass


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool


class JudgeVerdict(BaseModel):
    claims: list[ClaimVerdict]
    answer_relevance: int
    answer_correctness: int
    rationale: str


class AnswerGenerator(Protocol):
    def __call__(self, question: str, context: Sequence[RetrievedChunk]) -> str: ...


class AnswerJudge(Protocol):
    def __call__(
        self,
        question: str,
        reference_answer: str,
        context: Sequence[RetrievedChunk],
        answer: str,
    ) -> JudgeVerdict: ...


def format_context(context: Sequence[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{index}] source_id={chunk.source_id}\n{chunk.text}"
        for index, chunk in enumerate(context, start=1)
    )


def is_abstention(answer: str) -> bool:
    return ABSTAIN_TEXT.lower().rstrip(".") in answer.lower()


def cited_source_ids(answer: str) -> list[str]:
    return sorted(set(_CITATION.findall(answer)))


@dataclass(frozen=True)
class CitationCheck:
    cited: list[str]
    unknown: list[str]  # cited IDs that were not in the provided context
    valid: bool  # every cited ID was in the context
    has_citation: bool
    cites_relevant: bool | None  # at least one gold-relevant ID cited (answerable only)


def check_citations(
    answer: str,
    context_ids: set[str],
    relevant_ids: set[str],
) -> CitationCheck:
    cited = cited_source_ids(answer)
    unknown = [source_id for source_id in cited if source_id not in context_ids]
    return CitationCheck(
        cited=cited,
        unknown=unknown,
        valid=not unknown,
        has_citation=bool(cited),
        cites_relevant=bool(relevant_ids.intersection(cited)) if relevant_ids else None,
    )


def validate_verdict(verdict: JudgeVerdict) -> JudgeVerdict:
    for name in ("answer_relevance", "answer_correctness"):
        value = getattr(verdict, name)
        if not 1 <= value <= 5:
            raise ValueError(f"Judge returned {name}={value}, expected 1-5")
    return verdict


def with_retries(
    call: Callable[[], Any],
    *,
    attempts: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Retry provider calls with exponential backoff (rate limits, timeouts)."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception:
            if attempt == attempts - 1:
                raise
            sleep(min(max_delay, base_delay * 2**attempt))
    raise RuntimeError("unreachable")


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, (str, dict))
        ]
        return "\n".join(part.strip() for part in parts if part.strip())
    return ""


REASONING_EFFORT = "low"  # gpt-oss reasoning tokens are billed; keep them small


def _chat_model(model_name: str) -> Any:
    from langchain_groq import ChatGroq

    from app.core.config import get_settings

    settings = get_settings()
    if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value():
        raise ValueError("GROQ_API_KEY is required for generation evaluation")
    return ChatGroq(
        model=model_name,
        api_key=settings.groq_api_key.get_secret_value(),
        temperature=0,
        timeout=max(settings.llm_timeout_seconds, 60),
        max_retries=0,  # with_retries owns backoff
        reasoning_effort=REASONING_EFFORT,
    )


class GroqAnswerGenerator:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = _chat_model(model_name)

    def __call__(self, question: str, context: Sequence[RetrievedChunk]) -> str:
        messages = [
            SystemMessage(content=GENERATOR_INSTRUCTIONS),
            HumanMessage(content=(
                f"Context passages:\n{format_context(context)}\n\nQuestion: {question}"
            )),
        ]

        def invoke() -> str:
            answer = _text_content(self._model.invoke(messages).content)
            if not answer:
                # Reasoning models occasionally return reasoning but no final
                # content; treat it like any other transient provider failure.
                raise EmptyGenerationError("The generator returned no text")
            return answer

        return with_retries(invoke)


class GroqAnswerJudge:
    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL) -> None:
        self.model_name = model_name
        self._model = _chat_model(model_name).with_structured_output(
            JudgeVerdict,
            method="json_schema",
            strict=True,
        )

    def __call__(
        self,
        question: str,
        reference_answer: str,
        context: Sequence[RetrievedChunk],
        answer: str,
    ) -> JudgeVerdict:
        verdict = with_retries(lambda: self._model.invoke([
            SystemMessage(content=JUDGE_INSTRUCTIONS),
            HumanMessage(content=(
                f"Question: {question}\n\n"
                f"Reference answer: {reference_answer}\n\n"
                f"Context passages:\n{format_context(context)}\n\n"
                f"System answer:\n{answer}"
            )),
        ]))
        if not isinstance(verdict, JudgeVerdict):
            raise ValueError("The judge returned an invalid verdict")
        return validate_verdict(verdict)
