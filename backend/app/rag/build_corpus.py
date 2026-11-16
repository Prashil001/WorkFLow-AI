"""Build the documentation retrieval corpus from pinned upstream checkouts.

The corpus combines three MIT-licensed documentation sets from this project's
own stack: FastAPI, Pydantic, and SQLModel. Each source is pinned to a commit
so the generated file is reproducible. Usage (paths are local checkouts):

    uv run python -m app.rag.build_corpus \\
        --fastapi ../fastapi --pydantic ../pydantic --sqlmodel ../sqlmodel

The markdown is cleaned to plain prose: build macros, code-include directives,
heading anchors, HTML markup, and API-reference stubs are removed; fenced code
blocks that are written inline in the page are kept.
"""

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


CORPUS_NAME = "stack-docs"
CORPUS_VERSION = 1
CORPUS_PATH = Path(__file__).parent / "data" / "stack_docs_corpus.v1.json"
MIN_DOCUMENT_WORDS = 80


@dataclass(frozen=True)
class DocSource:
    key: str
    collection: str
    repository: str
    docs_dir: str
    base_url: str
    copyright: str
    excluded_prefixes: tuple[str, ...]


SOURCES = (
    DocSource(
        key="fastapi",
        collection="FastAPI",
        repository="https://github.com/fastapi/fastapi",
        docs_dir="docs/en/docs",
        base_url="https://fastapi.tiangolo.com/",
        copyright="Copyright (c) 2018 Sebastián Ramírez",
        # reference/ pages are mkdocstrings stubs; release notes are a changelog.
        excluded_prefixes=("reference/", "release-notes", "management"),
    ),
    DocSource(
        key="pydantic",
        collection="Pydantic",
        repository="https://github.com/pydantic/pydantic",
        docs_dir="docs",
        base_url="https://docs.pydantic.dev/latest/",
        copyright=(
            "Copyright (c) 2017 to present Pydantic Services Inc. "
            "and individual contributors."
        ),
        excluded_prefixes=("api/",),
    ),
    DocSource(
        key="sqlmodel",
        collection="SQLModel",
        repository="https://github.com/fastapi/sqlmodel",
        docs_dir="docs",
        base_url="https://sqlmodel.tiangolo.com/",
        copyright="Copyright (c) 2021 Sebastián Ramírez",
        excluded_prefixes=("release-notes", "management"),
    ),
)


_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SVG = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)
_CODE_INCLUDE = re.compile(r"\{\*.*?\*\}", re.DOTALL)
_MACRO_INCLUDE = re.compile(r"\{!.*?!\}", re.DOTALL)
_SNIPPET_INCLUDE = re.compile(r"^\s*--8<--.*$", re.MULTILINE)
_HEADING_ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}")
_ATTR_LIST = re.compile(r"\{\s*[.:][^}]*\}")
_API_BOX = re.compile(r"^\?\?\? api\b.*?\n(?:(?:[ \t]+.*)?\n)*", re.MULTILINE)
_ADMONITION_BANG = re.compile(r"^([ \t]*)[!?]{3}\+?[ \t]*([\w-]+)(?:[ \t]+(.*?))?[ \t]*$", re.MULTILINE)
_ADMONITION_SLASH = re.compile(r"^/{3,}[ \t]*([\w-]+)?(?:[ \t]*\|[ \t]*(.*?))?[ \t]*$", re.MULTILINE)
_CONTENT_TAB = re.compile(r'^\s*===\s*"[^"]*"\s*$', re.MULTILINE)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_EMPTY_LINK = re.compile(r"\[\]\([^)]*\)")
_CODE_FENCE = re.compile(r"^\s*(```|~~~)")
_INLINE_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_REFERENCE_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}.*$", re.MULTILINE)
_BLANK_RUNS = re.compile(r"\n{3,}")


def clean_markdown(markdown: str) -> tuple[str | None, str]:
    """Return (H1 title or None, cleaned body text)."""
    text = markdown.replace("\r\n", "\n")
    text = _FRONT_MATTER.sub("", text)
    text = _HTML_COMMENT.sub("", text)
    text = _SVG.sub("", text)
    text = _CODE_INCLUDE.sub("", text)
    text = _MACRO_INCLUDE.sub("", text)
    text = _SNIPPET_INCLUDE.sub("", text)
    text = _API_BOX.sub("", text)
    text = _HEADING_ANCHOR.sub("", text)
    text = _ATTR_LIST.sub("", text)
    text = _ADMONITION_BANG.sub(_render_admonition_bang, text)
    text = _ADMONITION_SLASH.sub(_render_admonition_slash, text)
    text = _CONTENT_TAB.sub("", text)
    text = _IMAGE.sub("", text)
    text = _EMPTY_LINK.sub("", text)
    text = _INLINE_LINK.sub(r"\1", text)
    text = _REFERENCE_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub("", text)
    text = _TABLE_RULE.sub("", text)

    title = None
    lines = []
    in_code = False
    for line in text.split("\n"):
        stripped = line.rstrip()
        if _CODE_FENCE.match(stripped):
            in_code = not in_code
        if title is None and not in_code and stripped.startswith("# "):
            title = _strip_inline_markup(stripped[2:])
            continue
        lines.append(stripped)

    body = _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip()
    return title, body


def _admonition_heading(kind: str, label: str | None) -> str:
    label = (label or "").strip().strip('"').strip()
    if kind.startswith("version"):
        # "/// version-added | v2.10" -> "Version added: v2.10"
        return f"{kind.replace('-', ' ').capitalize()}: {label}".rstrip(": ")
    return f"{label or kind}:"


def _render_admonition_bang(match: re.Match[str]) -> str:
    indent, kind, label = match.group(1), match.group(2), match.group(3)
    return f"{indent}{_admonition_heading(kind, label)}"


def _render_admonition_slash(match: re.Match[str]) -> str:
    kind, label = match.group(1), match.group(2)
    if not kind:
        return ""
    return _admonition_heading(kind, label)


def _strip_inline_markup(text: str) -> str:
    return re.sub(r"[`*_]", "", text).strip()


def _document_id(source: DocSource, relative_path: str) -> str:
    stem = relative_path.removesuffix(".md")
    slug = "__".join(part for part in stem.split("/") if part)
    slug = re.sub(r"[^A-Za-z0-9_-]", "-", slug)
    return f"{source.key}__{slug}"


def _page_url(source: DocSource, relative_path: str) -> str:
    stem = relative_path.removesuffix(".md")
    if stem == "index":
        return source.base_url
    stem = stem.removesuffix("/index")
    return f"{source.base_url}{stem}/"


def _fallback_title(relative_path: str) -> str:
    stem = Path(relative_path).stem
    if stem == "index":
        stem = Path(relative_path).parent.name or "Overview"
    return stem.replace("_", " ").replace("-", " ").strip().capitalize()


def _git_commit(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_source_documents(
    source: DocSource,
    checkout: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    docs_root = checkout / source.docs_dir
    documents = []
    skipped = []
    for path in sorted(docs_root.rglob("*.md")):
        relative_path = path.relative_to(docs_root).as_posix()
        if relative_path.startswith(source.excluded_prefixes):
            skipped.append(f"{relative_path} (excluded section)")
            continue

        title, body = clean_markdown(path.read_text(encoding="utf-8"))
        if len(body.split()) < MIN_DOCUMENT_WORDS:
            skipped.append(f"{relative_path} (under {MIN_DOCUMENT_WORDS} words)")
            continue

        documents.append({
            "document_id": _document_id(source, relative_path),
            "collection": source.collection,
            "title": title or _fallback_title(relative_path),
            "url": _page_url(source, relative_path),
            "path": relative_path,
            "content": body,
        })
    return documents, skipped


def build_corpus(checkouts: dict[str, Path]) -> tuple[dict, list[str]]:
    all_documents = []
    all_skipped = []
    source_records = []
    for source in SOURCES:
        checkout = checkouts[source.key]
        documents, skipped = build_source_documents(source, checkout)
        all_documents.extend(documents)
        all_skipped.extend(f"{source.key}: {item}" for item in skipped)
        source_records.append({
            "key": source.key,
            "collection": source.collection,
            "repository": source.repository,
            "commit": _git_commit(checkout),
            "docs_dir": source.docs_dir,
            "license": "MIT",
            "copyright": source.copyright,
            "document_count": len(documents),
        })

    document_ids = [document["document_id"] for document in all_documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Generated document IDs are not unique")

    corpus = {
        "corpus": CORPUS_NAME,
        "version": CORPUS_VERSION,
        "description": (
            "Cleaned English documentation pages from FastAPI, Pydantic, and "
            "SQLModel, used as a realistic retrieval corpus."
        ),
        "sources": source_records,
        "document_count": len(all_documents),
        "documents": all_documents,
    }
    return corpus, all_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for source in SOURCES:
        parser.add_argument(f"--{source.key}", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CORPUS_PATH)
    args = parser.parse_args()

    corpus, skipped = build_corpus({
        source.key: getattr(args, source.key) for source in SOURCES
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(corpus, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for record in corpus["sources"]:
        print(f"{record['collection']}: {record['document_count']} documents @ {record['commit'][:12]}")
    print(f"Total: {corpus['document_count']} documents; skipped {len(skipped)}:")
    for item in skipped:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
