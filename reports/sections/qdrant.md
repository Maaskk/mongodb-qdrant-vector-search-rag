# Qdrant Dense Retrieval Track

## Scope

This section integrates Hamza El Haddaji's Qdrant contribution into the shared project structure.
The branch originally contained a working local backend and a smoke benchmark result. During
integration, the backend was moved under `src/vector_rag/qdrant_track/`, optional dependencies were
made lazy, point identifiers were made deterministic, and metadata filters were wired into the
shared `RetrievalResult` contract.

## Method

Qdrant stores vectors in a collection and attaches chunk metadata as payload. The project uses one
collection named `rag_corpus`, cosine distance, 384-dimensional vectors, and the frozen
`sentence-transformers/all-MiniLM-L6-v2` embedding model. Payload indexes are declared for
`category`, `language`, and `source` so filtered semantic search can use the same fields as the
MongoDB track.

The backend accepts injected fake clients and embedders for unit tests. Live runs require the
optional Qdrant and embedding extras:

```bash
pip install -e '.[dev,qdrant,embeddings]'
docker compose up -d qdrant
```

## Integration fixes

- The old top-level `vector_rag.qdrant_backend` import path remains as a compatibility wrapper.
- The real implementation now lives in `vector_rag.qdrant_track.backend`.
- Optional packages are imported only when a live Qdrant backend is constructed.
- Point ids use a SHA-256-derived stable integer instead of Python's process-randomized `hash()`.
- Search forwards equality filters to Qdrant and exports readable filter names.
- Unit tests no longer download transformer models or require Docker.

## Current evidence

Hamza's committed smoke result reports six ingested chunks, five tested queries, p50 latency around
1.00 ms, and p95 latency around 2.00 ms on a local run. That result is useful as an implementation
smoke check. It is not yet a full fair benchmark because the branch did not export per-query
retrieval rows, Recall@5, MRR@10, or nDCG@10.

The final comparison therefore treats Qdrant as `local_smoke` until a full qrels-backed run is
exported. This avoids overstating the evidence while still preserving Hamza's working contribution.

## Reproduction

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,qdrant,embeddings]'
ruff check .
mypy src
pytest -q
```

## Remaining Qdrant work before a strict final benchmark

1. Export standard `retrieval_results.jsonl` rows for every evaluation query.
2. Compute Recall@5, MRR@10, and nDCG@10 against `data/evaluation/qrels.tsv`.
3. Export `rag_results.jsonl` using the shared citation/refusal behavior.
4. Repeat the run after documenting machine, Docker, and Qdrant versions.
