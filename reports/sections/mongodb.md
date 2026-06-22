# MongoDB Vector Search and Grounded RAG Track

## Scope and research method

This section implements Ossama's independent data-engineering track. It evaluates MongoDB as a
combined document store, metadata-filter engine, vector-retrieval engine, lexical-search engine,
and evidence source for grounded RAG. The live experiment uses the frozen corpus, queries,
relevance judgments, 384-dimensional cosine vectors, and `all-MiniLM-L6-v2` model shared by both
contributors.

The committed offline-validation run is a functional and reproducibility check. It does **not**
measure MongoDB: its manifest, summary rows, and tuning rows carry the
`offline_validation` label. Atlas latency, ANN recall, index-build duration, explain output, and
cost conclusions require a live Atlas deployment and must be reported only from an `atlas`
manifest.

## Architecture

```text
JSONL corpus -> validated Chunk contract -> embedding adapter
       -> idempotent batched MongoDB upsert -> Vector + Search indexes
       -> exact / ANN / filtered ANN / lexical search
       -> client RRF (or MongoDB 8+ $rankFusion pipeline)
       -> bounded untrusted context -> cited answer or refusal
       -> immutable JSONL/CSV run directory
```

The MongoDB collection is injected into ingestion and retrieval classes. Unit tests therefore
exercise batching, retry, index, aggregation-pipeline, ranking, RAG, and artifact behavior without
network access. The Atlas integration test is enabled only when `MONGODB_URI` is present and drops
only its uniquely named smoke-test collection.

## Document model and lineage

Each source chunk becomes one BSON document keyed by `_id = chunk_id`. It contains the original
document identifier, chunk order, title, text, source, category, language, optional publication
date, content SHA-256, embedding-model identifier, 384 finite floats, `schema_version`, and an
ingestion object with `run_id` and UTC `indexed_at`. The existing content hash is read before each
batch. Unchanged chunks are skipped; new and changed chunks use unordered `UpdateOne(...,
upsert=True)` operations.

Transient selection, reconnect, and network-timeout failures use bounded exponential retries.
An exhausted batch is recorded as JSON Lines in `results/mongodb/dead_letters/<run_id>.jsonl` with
its chunk identifier and sanitized error type. Summaries separately record accepted, upserted,
updated, skipped, failed, elapsed time, and the optional dead-letter path.

## Indexes and retrieval modes

`configs/mongodb/vector-index.json` defines cosine vector search over `embedding` and filter fields
for category, language, source, and publication date. `configs/mongodb/text-index.json` maps title
and text for lexical search. The index manager lists names first, creates only missing definitions,
and polls their `status` and `queryable` values.

Five standard modes are exported as `RetrievalResult` records:

1. `exact` uses `$vectorSearch` with `exact: true` and no `numCandidates`.
2. `ann` uses `$vectorSearch` with an explicit candidate budget.
3. `filtered_ann` adds indexed metadata predicates inside `$vectorSearch`.
4. `text` uses `$search` over title and text.
5. `hybrid_rrf` combines vector and lexical ranks with `1 / (60 + rank)`. A MongoDB 8+
   `$rankFusion` builder is also included; client-side fusion remains the portable fallback.

All ranks start at one and retain backend, query, mode, filter, latency, and run identifiers.

## Grounded RAG controls

Retrieved chunks are HTML-escaped, character-budgeted, and wrapped in an explicit
`retrieved_context untrusted="true"` boundary. The generator contract is provider-neutral. The
default extractive generator is deterministic and credential-free; the optional Ollama adapter
calls only a configurable local endpoint.

Generated citations must match retrieved chunk identifiers. Empty retrieval, missing citations,
unknown citations, timeouts, malformed responses, and insufficient evidence produce the stable
refusal `Insufficient evidence in the retrieved context.` Retrieval, generation, and total latency
remain separately schema-valid.

## Evaluation and artifacts

Per query, the benchmark calculates Recall@5, MRR@10, graded nDCG@10, and client-observed p50/p95
latency. Live Atlas runs also sweep `numCandidates` values 10, 25, 50, 100, and 200 against exact
search. Each immutable run stores:

- `run_manifest.json` with execution environment, hashes, model, dimensions, UTC time, and run ID;
- `retrieval_results.jsonl` and `rag_results.jsonl` using frozen shared contracts;
- `benchmark_summary.csv` and `tuning_matrix.csv`;
- `ingestion_summary.json` for live runs.

The small project-authored corpus is appropriate for smoke testing, not general conclusions.
Meaningful performance analysis needs a larger representative corpus, repeated warm/cold runs,
documented Atlas tier and region, index-build timing, concurrent load, and cost measurements.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,mongodb,embeddings]'
ruff check .
mypy src
pytest -q
```

Credential-free full-flow validation:

```bash
python -m vector_rag.mongodb_track benchmark \
  --offline-validation \
  --run-id offline-validation-v1
```

Measured Atlas run (after exporting `MONGODB_URI`):

```bash
python -m vector_rag.mongodb_track indexes --wait
python -m vector_rag.mongodb_track benchmark --run-id atlas-v1
```

The second command creates an `atlas` run only after the indexes are queryable, ingests the frozen
corpus idempotently, executes all five modes and the candidate sweep, and writes an immutable run
directory. Never compare an `offline_validation` artifact with Hamza's measured Qdrant results.
