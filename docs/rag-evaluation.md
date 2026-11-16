# RAG evaluation: method and full results

This is the detailed write-up behind the results table in the [README](../README.md). Every number here comes from a saved results file in [`backend/evaluation/`](../backend/evaluation/).

- [Corpus](#corpus)
- [Ingestion and chunking](#ingestion-and-chunking)
- [Evaluation dataset (v2)](#evaluation-dataset-v2)
- [Metrics](#metrics)
- [Results](#results)
- [Generation status](#generation-status)
- [Reproducing the results](#reproducing-the-results)
- [CI smoke test](#ci-smoke-test)
- [Workflow regression set (v1)](#workflow-regression-set-v1)
- [Limitations](#limitations)

## Corpus

`stack-docs` holds **227 documentation pages** (1.7 MB of cleaned text, 1,364 chunks at the default chunk size) from three MIT-licensed projects, each pinned to a commit:

| Collection | Pages | Source | License |
| --- | ---: | --- | --- |
| FastAPI | 120 | [fastapi/fastapi](https://github.com/fastapi/fastapi) `docs/en/docs` | MIT |
| Pydantic | 51 | [pydantic/pydantic](https://github.com/pydantic/pydantic) `docs` | MIT |
| SQLModel | 56 | [fastapi/sqlmodel](https://github.com/fastapi/sqlmodel) `docs` | MIT |

- Corpus file: [`stack_docs_corpus.v1.json`](../backend/app/rag/data/stack_docs_corpus.v1.json)
- Commit SHAs, copyright lines, and full license texts: [`THIRD_PARTY_NOTICES.md`](../backend/app/rag/data/THIRD_PARTY_NOTICES.md)
- Builder: [`build_corpus.py`](../backend/app/rag/build_corpus.py) regenerates the file from local checkouts.

Cleaning removes:

- build macros and code-include directives, so code examples kept in separate files are not in the corpus;
- heading anchors and HTML;
- API-reference stubs, changelogs, and internal `_`-prefixed files;
- pages under 80 words.

Inline fenced code is kept.

## Ingestion and chunking

**Storage.** Chunks live in the PostgreSQL `document_chunks` table: a `vector(384)` embedding with an HNSW cosine index, a generated English `tsvector` with a GIN index, and a `corpus` column so several corpora share one table. The database image is `pgvector/pgvector:pg17-trixie`.

**Idempotent ingestion** (`app.rag.ingest`) syncs one corpus per run, in one transaction under a per-corpus advisory lock:

- unchanged chunks are not re-embedded;
- changed chunks are updated in place;
- chunks removed from the corpus are deleted.

The stored embedding-model identity includes the FastEmbed snapshot revision (for example `BAAI/bge-small-en-v1.5@aa8f8b060edb`). FastEmbed does not pin model revisions, so if a cache refresh pulls a different snapshot, the next ingestion re-embeds instead of mixing vectors.

**Chunking.**

- **Splitting:** pages are split into headings, sentences, and whole fenced code blocks.
- **Budget:** units are packed up to `RAG_CHUNK_TOKENS` (default 400), counted with the embedding model's own WordPiece tokenizer, because bge-small-en-v1.5 truncates at 512 tokens.
- **Overlap:** each chunk starts with whole trailing sentences from the previous chunk, up to `RAG_CHUNK_OVERLAP_TOKENS` (default 60, i.e. 15%). If the boundary falls after a sentence or code block longer than that, the chunk gets no overlap rather than a mid-sentence cut.
- **Header:** each chunk is prefixed with `collection | page title > section`.
- **Pricing corpus:** the fictional `pricing-mock` corpus used by the workflow tools keeps the original word chunker, so the v1 evaluation stays comparable.

## Evaluation dataset (v2)

[`dataset.v2.json`](../backend/app/evaluation/dataset.v2.json) has 80 queries, each with a question, relevant source IDs, and a reference answer.

| Category | Queries | What it tests |
| --- | ---: | --- |
| Easy | 25 | Wording close to the source page |
| Paraphrased | 25 | The need described without the page's key terms |
| Multi-hop | 15 | Facts from two pages, often across FastAPI, Pydantic, and SQLModel |
| Unanswerable | 15 | Plausible for this stack but absent from the corpus |

Questions were drafted with an LLM assistant from passages found by searching the corpus, then made checkable:

- every answerable item carries verbatim evidence quotes from its relevant pages;
- every unanswerable item carries an `absence_pattern`;
- a test fails if any quote is missing from its cited page, or if any absence pattern matches a corpus page.

Relevance labels were audited against the pooled top-5 pages of dense and full-text retrieval before any metric was recorded. 11 labels were added where a pooled page explicitly stated a required fact (99 labels in total).

## Metrics

**Retrieval** is scored per page: chunk results are collapsed to their source page, and recall@1/3/5, MRR@10, and nDCG@5 (binary relevance) are averaged over the 65 answerable queries.

**Latency** is the wall time of one `retrieve()` call, including CPU query embedding, after 3 warm-up queries.

**Generation** uses the top 5 chunks as context for `openai/gpt-oss-20b` (the app's model); `openai/gpt-oss-120b` judges each answer at temperature 0:

- **Faithfulness:** share of the answer's atomic claims supported by the context.
- **Answer relevance:** 1-5.
- **Answer correctness:** 1-5 against the reference answer; an abstention on an answerable question scores 1.

**Citation integrity** is checked without an LLM: cited IDs must be in the given context, answers must cite something, and the rate of citing a gold page is reported. **Abstention** is measured on both unanswerable and answerable queries.

**Significance.** [`compare.py`](../backend/app/evaluation/compare.py) computes paired bootstrap 95% confidence intervals of each config's difference from the baseline (10,000 resamples, fixed seed).

## Results

All four configs ran on the same dataset, corpus, chunks, and machine, back to back at commit `145620a`, with `BAAI/bge-small-en-v1.5@aa8f8b060edb`, PostgreSQL 17.12, and pgvector 0.8.7.

| Config | What it does |
| --- | --- |
| **A: vector** | Dense cosine similarity over the pgvector HNSW index |
| **B: hybrid** | Dense top-50 and Postgres full-text top-50, fused with Reciprocal Rank Fusion (k=60) |
| **C-lite** | B, then `Xenova/ms-marco-MiniLM-L-6-v2` (80 MB) reranks the top 20 chunks |
| **C** | B, then `BAAI/bge-reranker-base` (1.04 GB) reranks the top 20 chunks |

Both reranker models and every setting were fixed before any config was measured; nothing was tuned on this dataset. The full-text side uses Postgres `ts_rank_cd` (term frequency and proximity, no IDF), so it is lexical retrieval but not BM25.

### Comparison

| Config | recall@1 | recall@3 | recall@5 | MRR@10 | nDCG@5 | p50 latency | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: vector | 0.554 | 0.756 | 0.810 | 0.805 | 0.764 | 26 ms | 46 ms |
| B: hybrid (RRF) | 0.469 | 0.803 | 0.877 | 0.752 | 0.763 | 53 ms | 75 ms |
| C-lite: hybrid + MiniLM-L6 rerank | 0.559 | **0.856** | 0.874 | 0.834 | 0.808 | 1.56 s | 1.74 s |
| C: hybrid + bge-reranker-base | **0.580** | 0.815 | **0.892** | **0.839** | **0.820** | 8.18 s | 9.92 s |

Paraphrased-query recall@5 rose from 0.700 (A) to 0.740 (B), 0.800 (C-lite), and 0.813 (C).

### Confidence intervals vs. A

Source: [`comparison-v2.json`](../backend/evaluation/comparison-v2.json).

| Difference vs. A | Interval excludes 0 (resolved) | Interval includes 0 (not resolved) |
| --- | --- | --- |
| B: hybrid | recall@5 +0.067 [+0.008, +0.133]; recall@1 −0.085 [−0.169, −0.008] | recall@3, MRR@10, nDCG@5 |
| C-lite | recall@3 +0.100 [+0.026, +0.182]; recall@5 +0.064 [+0.003, +0.133] | recall@1, MRR@10, nDCG@5 |
| C | recall@5 +0.082 [+0.028, +0.146]; nDCG@5 +0.056 [+0.009, +0.108] | recall@1, recall@3, MRR@10 |

### Baseline by category (config A)

Source: [`results-v2-vector.json`](../backend/evaluation/results-v2-vector.json). The metrics are identical to the first baseline run at `d0427cf`, since retrieval is deterministic.

| Queries | recall@1 | recall@3 | recall@5 | MRR@10 | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| All answerable (65) | 0.554 | 0.756 | 0.810 | 0.805 | 0.764 |
| Easy (25) | 0.740 | 0.900 | 0.900 | 0.913 | 0.892 |
| Paraphrased (25) | 0.480 | 0.633 | 0.700 | 0.655 | 0.621 |
| Multi-hop (15) | 0.367 | 0.722 | 0.844 | 0.872 | 0.788 |

Three runs of config A on the same machine measured p95 latency between 32 and 67 ms, so treat latencies as single-machine figures that vary by tens of percent between runs.

### Findings

- **Paraphrased queries are the baseline's weak spot.** Dense recall@5 is 0.70, against 0.90 when the wording matches the page.
- **Hybrid alone is a trade-off.** Full-text search adds relevant pages the dense retriever missed (recall@5 up), but RRF also promotes chunks that only share common terms, which significantly lowers recall@1. nDCG@5 does not change.
- **Reranking turns the extra recall into better ranking.** C is the only config with a resolved nDCG@5 gain over A. Its recall@1 and MRR@10 gains are not resolved by this dataset.
- **Reranking is expensive on CPU.** Each query cross-encodes 20 chunks of about 400 tokens. On a laptop CPU (no GPU), bge-reranker-base adds about 8 s per query and MiniLM-L6 about 1.5 s. MiniLM keeps most of the gain at about a fifth of the cost, so for an interactive service C-lite or a GPU-backed reranker is the practical choice.
- **Multi-hop recall@1 is capped at 0.5** by definition, since one result can hold only one of two required pages. recall@5 is the more useful number there.
- **A score cutoff cannot catch unanswerable questions.** Their mean top dense score (0.754) sits close to the answerable mean (0.820), so abstention has to come from the generator.

The B, C-lite, and C results files record `git_dirty: true`. The only uncommitted changes during those runs were the results files written by the runs before them; the code was `145620a`.

## Generation status

Generation results are **not recorded yet**. The Groq free tier allows 200,000 tokens per model per day, and one full pass needs more than that for the judge alone. A first attempt hit the limit for both models after 34 answers and 18 judgments; that partial run was discarded rather than reported.

The runner now uses low reasoning effort and `--resume`, which reuses finished cases when the dataset, models, prompts, and retrieved context are unchanged, so a pass can span the daily limit:

```powershell
uv run python -m app.evaluation.run_v2 --config vector --pace 1 --resume
uv run python -m app.evaluation.run_v2 --config hybrid-rerank --pace 1 --resume
```

## Reproducing the results

From `backend`, with the database running and migrated:

```powershell
uv run python -m app.rag.ingest --corpus stack-docs
uv run python -m app.evaluation.run_v2 --config vector --skip-generation
uv run python -m app.evaluation.run_v2 --config hybrid --skip-generation
uv run python -m app.evaluation.run_v2 --config hybrid-rerank-minilm --skip-generation
uv run python -m app.evaluation.run_v2 --config hybrid-rerank --skip-generation
uv run python -m app.evaluation.compare vector hybrid hybrid-rerank-minilm hybrid-rerank
```

Each run writes `evaluation/results-v2-<config>.json` with:

- dataset and corpus SHA-256;
- the embedding-model snapshot;
- git commit, PostgreSQL, and pgvector versions;
- per-query results and judge rationales.

Drop `--skip-generation` to include generation; it needs `GROQ_API_KEY`.

Other ingestion options:

```powershell
uv run python -m app.rag.ingest --corpus pricing-mock                      # fictional data for workflow tools
uv run python -m app.rag.ingest --corpus stack-docs --chunk-size 300 --overlap 45
uv run python -m app.rag.ingest --corpus stack-docs --dry-run              # report changes only
```

## CI smoke test

The `eval-smoke` job in [`backend-ci.yml`](../.github/workflows/backend-ci.yml) runs after the test suite:

- it ingests a fixed 71-page slice of the corpus into an isolated `stack-docs-smoke` corpus, using the real embedding model (cached between runs);
- it runs the vector and hybrid retrievers on 16 fixed v2 queries;
- it fails if recall@5 or MRR@10 falls below a regression floor.

The floors sit well below what the slice scored when they were set; they guard against breakage and are not reported results. The report is uploaded as a build artifact. Run it locally with:

```powershell
uv run python -m app.evaluation.smoke
```

## Workflow regression set (v1)

```powershell
uv run python -m app.evaluation.run --output evaluation/results-v1.json
```

Dataset: [`dataset.v1.json`](../backend/app/evaluation/dataset.v1.json). Results: [`results-v1.json`](../backend/evaluation/results-v1.json).

The recorded run passed 11/11 checks: plan validation 3/3, dependency execution 2/2, citation integrity 2/2, and retrieval 4/4. On the three positive retrieval queries, Hit@1 and precision@3 were both 3/3; the unrelated query returned no matches. These are small synthetic regression checks, not benchmark scores. Planning cases validate known plan fixtures rather than live Groq output, and citation checks verify source IDs rather than the truth of every sentence.

## Limitations

- **Evaluation size.** 80 queries (65 answerable) over one 227-page English corpus. Several differences between configs are not resolved by the bootstrap intervals, and results may not transfer to other domains.
- **How the labels were made.** The dataset was drafted with an LLM assistant and verified mechanically against the corpus (quotes and absence patterns), not reviewed by independent annotators. Relevance is binary and per page. The label audit pooled only dense and full-text results, so a relevant page that neither retrieved may be unlabeled.
- **Generation quality is not yet measured.** The generation metrics and the LLM-as-judge pipeline are implemented, but no full run fits the Groq free-tier daily limit yet. When recorded, they will be judged by a model, not by people.
- **Lexical ranking is not BM25.** Postgres `ts_rank_cd` has no IDF, which is the likely reason hybrid fusion lowers recall@1.
- **Latency.** Measured on one laptop CPU with PostgreSQL in Docker; it varies between runs. The bge reranker's 8 s p50 is not interactive without a GPU.
- **No retrieval-side abstention.** Dense similarity scores do not separate unanswerable questions from answerable ones.
- **Corpus coverage.** Code examples that the docs include from separate files are not in the corpus, and the corpus is a pinned snapshot of each project's docs.
- **Scope of the hybrid pipeline.** Hybrid retrieval and reranking are evaluated on the docs corpus. The workflow API still serves the fictional pricing data through the in-memory or dense pgvector retriever (`RAG_RETRIEVER`).
