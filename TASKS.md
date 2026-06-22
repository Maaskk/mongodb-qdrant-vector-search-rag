# Independent Task Division

The work is divided by database engine. Each contributor owns a complete pipeline and can finish it without waiting for the other contributor's code or results.

## Shared output contract

Both tracks read the same versioned fixtures and implement `VectorBackend` from `src/vector_rag/backend.py`. Both export `RetrievalResult` and `RAGResult` records from `src/vector_rag/contracts.py`.

Do not change these shared files on a personal branch:

```text
configs/embeddings.yaml
configs/benchmark.yaml
data/demo/corpus.jsonl
data/evaluation/queries.jsonl
data/evaluation/qrels.tsv
src/vector_rag/contracts.py
src/vector_rag/backend.py
```

If a shared contract must change, open a separate proposal pull request into `main` first.

## Ossama - Advanced MongoDB Track

**Branch:** `ossama`

**Owned paths:**

```text
configs/mongodb/
src/vector_rag/mongodb_track/
tests/unit/mongodb/
tests/integration/mongodb/
results/mongodb/
reports/sections/mongodb.md
```

### O1 - MongoDB Document Model and Resilient Ingestion

- Model chunks, float embeddings, source lineage, metadata, schema version, and ingestion-run information as BSON documents.
- Implement idempotent bulk upserts keyed by `chunk_id` and `content_hash`.
- Add configurable batches, bounded retries for transient errors, resumable runs, and a dead-letter JSONL file containing failed identifiers.
- Validate the embedding dimension before writes.
- Record accepted, updated, skipped, and failed counts plus elapsed time.
- Add a safe Atlas setup guide and index-readiness check.

**Outputs:**

```text
configs/mongodb/vector-index.json
configs/mongodb/text-index.json
src/vector_rag/mongodb_track/ingest.py
src/vector_rag/mongodb_track/schema.py
results/mongodb/ingestion_summary.json
```

**Acceptance:** Re-running ingestion does not duplicate chunks; malformed vectors are rejected; tests cover retry exhaustion and content-hash updates; no credential appears in logs.

### O2 - Five MongoDB Retrieval Modes

- Implement exact nearest-neighbor retrieval as a quality baseline.
- Implement approximate nearest-neighbor retrieval with `$vectorSearch`.
- Implement metadata pre-filters for language, category, source, and date.
- Implement MongoDB lexical full-text search.
- Implement hybrid lexical plus vector retrieval using reciprocal rank fusion.
- Preserve the standard result schema for every mode.

**Outputs:**

```text
src/vector_rag/mongodb_track/search.py
src/vector_rag/mongodb_track/hybrid.py
results/mongodb/retrieval_results.jsonl
```

**Acceptance:** `exact`, `vector`, `filtered_vector`, `text`, and `hybrid_rrf` modes all run from one CLI; ranks start at one; query and run identifiers are stable.

### O3 - Grounded MongoDB RAG

- Assemble retrieved chunks within a configurable context budget.
- Delimit retrieved content as untrusted input.
- Require inline chunk citations in generated answers.
- Refuse to invent an answer when the retrieved evidence is insufficient.
- Handle empty retrieval, provider timeouts, malformed provider responses, and partial failures.
- Keep generation provider-neutral and document one runnable provider.

**Outputs:**

```text
src/vector_rag/mongodb_track/rag.py
results/mongodb/rag_results.jsonl
```

**Acceptance:** Every supported answer cites known chunk identifiers; refusal records remain schema-valid; retrieval, generation, and total latency are recorded separately.

### O4 - Advanced MongoDB Benchmark and Analysis

- Sweep multiple `numCandidates` values and compare recall against exact search.
- Compare unfiltered and selective metadata-filter workloads.
- Compare vector, lexical, and hybrid retrieval quality.
- Report ingestion throughput, index-build time, Recall@5/10, MRR@10, nDCG@10, and p50/p95 latency.
- Capture explain output where MongoDB supports it.
- Discuss document-modeling benefits, Atlas operational constraints, cost caveats, and threats to validity.

**Outputs:**

```text
results/mongodb/benchmark_summary.csv
results/mongodb/tuning_matrix.csv
results/mongodb/figures/
reports/sections/mongodb.md
```

**Acceptance:** Every result records corpus hash, query-set hash, embedding model, configuration, timestamp, and run ID; the report separates measured results from interpretation.

## Hamza El Haddaji - Guided Qdrant Track

**Branch:** `HamzaElhaddaji`

**Owned paths:**

```text
configs/qdrant/
src/vector_rag/qdrant_track/
tests/unit/qdrant/
tests/integration/qdrant/
results/qdrant/
reports/sections/qdrant.md
```

Hamza uses the supplied Docker service, embedding configuration, fixtures, data loader, contracts, and benchmark settings. Custom HNSW tuning, quantization experiments, hybrid sparse/dense fusion, and evaluation-framework design are outside this track.

### H1 - Start Qdrant and Ingest the Frozen Corpus

- Start the provided Qdrant Docker Compose service.
- Create one collection with 384 dimensions and cosine distance.
- Create keyword payload indexes for category, language, and source before ingestion.
- Upsert chunks and payload metadata through `VectorBackend`.
- Return the number of accepted chunks and make repeated ingestion safe.

**Outputs:**

```text
configs/qdrant/collection.yaml
src/vector_rag/qdrant_track/backend.py
results/qdrant/ingestion_summary.json
```

**Acceptance:** The health check passes; the collection contains the expected number of unique chunk IDs; a second ingestion run does not create duplicates.

### H2 - Dense Search and Metadata Filters

- Embed evaluation questions with the frozen model.
- Run dense semantic search with Qdrant's supplied default HNSW settings.
- Support equality filters for category, language, and source.
- Convert hits into standard `RetrievalResult` records.

**Outputs:**

```text
src/vector_rag/qdrant_track/search.py
results/qdrant/retrieval_results.jsonl
```

**Acceptance:** Dense and filtered search run from one documented CLI; every result has a positive rank, score, latency, query ID, and run ID.

### H3 - Qdrant RAG with Shared Helpers

- Pass retrieved Qdrant chunks to the shared context builder.
- Use the shared generator interface and citation format.
- Export valid `RAGResult` records.

**Outputs:**

```text
src/vector_rag/qdrant_track/rag.py
results/qdrant/rag_results.jsonl
```

**Acceptance:** Answers cite only retrieved chunks; empty evidence produces a refusal; latency fields are present.

### H4 - Run the Standard Qdrant Benchmark

- Run the supplied benchmark configuration without changing the corpus, embedding model, metric definitions, or HNSW matrix.
- Export the standard retrieval metrics and latency summary.
- Write the Qdrant setup, method, results, and limitation sections.

**Outputs:**

```text
results/qdrant/benchmark_summary.csv
results/qdrant/figures/
reports/sections/qdrant.md
```

**Acceptance:** Results record the same corpus, queries, embedding model, and schema version used by MongoDB; all Qdrant unit and Docker-backed integration tests pass.

## Final Integration - Together After Both Tracks

This work starts only after both personal pull requests are ready. It is not a prerequisite for either personal track.

1. Validate both result directories against the shared schemas.
2. Reject comparisons with different corpus hashes, query hashes, embedding models, or metric definitions.
3. Build side-by-side quality, latency, throughput, and operational-complexity tables.
4. Explain where the experiment is fair and where platform differences limit direct comparison.
5. Merge the two report sections into the final report and presentation.

**Outputs:**

```text
results/comparison/final_comparison.csv
results/comparison/figures/
reports/final_report.md
```

## Definition of Done for Every Task

- Code is limited to the contributor's owned paths.
- Automated tests cover success and failure behavior.
- Commands and environment variables are documented.
- Generated artifacts match the frozen schema.
- Secrets and personal data are absent from commits and logs.
- `ruff check .`, `mypy src`, and `pytest -q` pass.
- A pull request explains the method, evidence, limitations, and exact reproduction command.

