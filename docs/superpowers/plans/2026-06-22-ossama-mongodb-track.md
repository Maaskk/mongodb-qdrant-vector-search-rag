# Ossama MongoDB Vector Search and RAG Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Ossama's complete MongoDB track: production-shaped ingestion, five retrieval modes, grounded RAG, reproducible evaluation, artifacts, and report documentation.

**Architecture:** Backend-specific code lives under `src/vector_rag/mongodb_track/` and consumes the frozen shared contracts without changing Hamza's Qdrant paths. MongoDB access is injected behind small collection protocols so unit tests exercise real track logic without cloud credentials, while marked integration tests run against Atlas when `MONGODB_URI` is available. Benchmark outputs include a run manifest that identifies `atlas` versus `offline_validation` execution.

**Tech Stack:** Python 3.11+, Pydantic 2, PyMongo 4, sentence-transformers, pytest, Ruff, mypy, MongoDB Atlas Vector Search/Search, local Ollama-compatible generation, CSV/JSONL artifacts.

---

## File Map

- `configs/mongodb/vector-index.json`: vector and filter index definition.
- `configs/mongodb/text-index.json`: full-text Search index definition.
- `configs/mongodb/benchmark.yaml`: MongoDB modes and `numCandidates` sweep.
- `src/vector_rag/mongodb_track/config.py`: validated environment configuration and redacted diagnostics.
- `src/vector_rag/mongodb_track/schema.py`: BSON conversion, vector validation, and summaries.
- `src/vector_rag/mongodb_track/indexes.py`: create/wait index orchestration.
- `src/vector_rag/mongodb_track/ingest.py`: idempotent batched upserts, retries, and dead letters.
- `src/vector_rag/mongodb_track/search.py`: exact, ANN, filtered ANN, and text pipeline execution.
- `src/vector_rag/mongodb_track/hybrid.py`: reciprocal rank fusion and MongoDB `$rankFusion` pipeline builder.
- `src/vector_rag/mongodb_track/rag.py`: context budgeting, untrusted-content delimiters, generation, citations, and refusal.
- `src/vector_rag/mongodb_track/embeddings.py`: frozen sentence-transformers embedder and deterministic test embedder.
- `src/vector_rag/mongodb_track/metrics.py`: Recall@k, MRR@10, nDCG@10, and percentiles.
- `src/vector_rag/mongodb_track/benchmark.py`: run orchestration and immutable artifact export.
- `src/vector_rag/mongodb_track/__main__.py`: `indexes`, `ingest`, `search`, `rag`, and `benchmark` CLI commands.
- `tests/unit/mongodb/`: credential-free behavior and pipeline tests.
- `tests/integration/mongodb/`: environment-gated Atlas smoke tests.
- `results/mongodb/`: run manifest, retrieval/RAG records, summaries, tuning matrix, and figures/data.
- `reports/sections/mongodb.md`: method, evidence, limitations, and reproduction guide.

### Task 1: Configuration, Schema, and Index Definitions

**Files:**
- Create: `tests/unit/mongodb/test_config.py`
- Create: `tests/unit/mongodb/test_schema.py`
- Create: `tests/unit/mongodb/test_indexes.py`
- Create: `src/vector_rag/mongodb_track/config.py`
- Create: `src/vector_rag/mongodb_track/schema.py`
- Create: `src/vector_rag/mongodb_track/indexes.py`
- Create: `configs/mongodb/vector-index.json`
- Create: `configs/mongodb/text-index.json`

- [ ] **Step 1: Write failing configuration and schema tests**

```python
def test_config_repr_redacts_uri(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://secret@localhost")
    config = MongoConfig.from_env()
    assert "secret" not in repr(config)


def test_chunk_document_rejects_wrong_vector_dimension(sample_chunk):
    with pytest.raises(ValueError, match="expected 384"):
        chunk_document(sample_chunk, [0.1, 0.2], run_id="run-1")
```

- [ ] **Step 2: Verify red**

Run: `pytest tests/unit/mongodb/test_config.py tests/unit/mongodb/test_schema.py -q`
Expected: collection fails because `vector_rag.mongodb_track` does not exist.

- [ ] **Step 3: Implement immutable config and BSON conversion**

Implement `MongoConfig.from_env()`, `__repr__()` with `<redacted>`, `chunk_document()`, `IngestionSummary`, `IndexStatus`, and UTC timestamp helpers. Validate 384 finite floats and keep `chunk_id` as `_id` for idempotency.

- [ ] **Step 4: Write and verify failing index-manager tests**

```python
def test_index_manager_creates_missing_indexes(fake_collection):
    manager = IndexManager(fake_collection, vector_definition, text_definition)
    assert manager.ensure_indexes() == ["vector_index", "text_index"]
```

Run: `pytest tests/unit/mongodb/test_indexes.py -q`
Expected: FAIL because `IndexManager` is missing.

- [ ] **Step 5: Implement index creation and readiness polling**

Use `SearchIndexModel`, avoid duplicate creation by name, poll `list_search_indexes()` until READY or timeout, and return typed status records. Keep definitions in committed JSON files.

- [ ] **Step 6: Verify and commit**

Run: `pytest tests/unit/mongodb/test_config.py tests/unit/mongodb/test_schema.py tests/unit/mongodb/test_indexes.py -q`
Expected: all pass.

```bash
git add configs/mongodb src/vector_rag/mongodb_track tests/unit/mongodb
git commit -m "feat(mongodb): add schema and index management"
```

### Task 2: Idempotent Resilient Ingestion

**Files:**
- Create: `tests/unit/mongodb/test_ingest.py`
- Create: `src/vector_rag/mongodb_track/ingest.py`

- [ ] **Step 1: Write failing ingestion behavior tests**

```python
def test_ingestion_upserts_by_chunk_id(fake_collection, sample_chunks):
    summary = MongoIngestor(fake_collection, dimensions=384).ingest(
        sample_chunks, embeddings, run_id="run-1"
    )
    assert summary.accepted == 2
    assert all(op._filter == {"_id": chunk.chunk_id} for op in fake_collection.operations)


def test_transient_failure_is_retried_then_succeeds(flaky_collection, sample_chunks):
    summary = MongoIngestor(flaky_collection, max_attempts=3, sleep=lambda _: None).ingest(
        sample_chunks, embeddings, run_id="run-1"
    )
    assert summary.failed == 0
    assert flaky_collection.calls == 2
```

- [ ] **Step 2: Verify red**

Run: `pytest tests/unit/mongodb/test_ingest.py -q`
Expected: FAIL because `MongoIngestor` is missing.

- [ ] **Step 3: Implement minimal passing ingestion**

Batch `UpdateOne(..., upsert=True)` operations, count inserted/matched/modified/upserted outcomes, retry only `AutoReconnect`, `NetworkTimeout`, and `ConnectionFailure`, validate vector counts before any write, and write schema-valid dead-letter JSONL on exhausted batches.

- [ ] **Step 4: Add content-hash and failure tests**

Test unchanged chunks, changed hashes, invalid vector counts, exhausted retries, dead-letter identifiers, and that exceptions/log strings never include the URI.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/unit/mongodb/test_ingest.py -q`
Expected: all ingestion tests pass.

```bash
git add src/vector_rag/mongodb_track/ingest.py tests/unit/mongodb/test_ingest.py
git commit -m "feat(mongodb): add resilient idempotent ingestion"
```

### Task 3: Exact, ANN, Filtered, Text, and Hybrid Retrieval

**Files:**
- Create: `tests/unit/mongodb/test_search.py`
- Create: `tests/unit/mongodb/test_hybrid.py`
- Create: `src/vector_rag/mongodb_track/search.py`
- Create: `src/vector_rag/mongodb_track/hybrid.py`

- [ ] **Step 1: Write failing pipeline tests**

```python
def test_exact_pipeline_uses_enn(retriever):
    pipeline = retriever.vector_pipeline(query_vector, top_k=5, exact=True)
    assert pipeline[0]["$vectorSearch"]["exact"] is True
    assert "numCandidates" not in pipeline[0]["$vectorSearch"]


def test_filtered_ann_indexes_metadata(retriever):
    pipeline = retriever.vector_pipeline(
        query_vector, top_k=5, num_candidates=100, filters={"category": "rag"}
    )
    assert pipeline[0]["$vectorSearch"]["filter"] == {"category": "rag"}
```

- [ ] **Step 2: Verify red, implement vector/text pipeline builders, verify green**

Run: `pytest tests/unit/mongodb/test_search.py -q`
Expected: fail before implementation, then pass after `MongoRetriever` maps aggregation documents to ranked `RetrievalResult` records and records wall-clock latency.

- [ ] **Step 3: Write failing RRF tests**

```python
def test_rrf_deduplicates_and_combines_rank_contributions():
    fused = reciprocal_rank_fusion(vector_hits, text_hits, rank_constant=60)
    assert [hit.chunk_id for hit in fused] == ["shared", "vector-only", "text-only"]
```

- [ ] **Step 4: Implement pure RRF and `$rankFusion` builder**

Use `1 / (60 + rank)` with configurable vector/text weights. Provide a MongoDB 8+ `$rankFusion` pipeline and a client-side fallback returning the same standard schema.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/unit/mongodb/test_search.py tests/unit/mongodb/test_hybrid.py -q`
Expected: all retrieval tests pass.

```bash
git add src/vector_rag/mongodb_track/search.py src/vector_rag/mongodb_track/hybrid.py tests/unit/mongodb
git commit -m "feat(mongodb): add five retrieval modes"
```

### Task 4: Grounded RAG and Local Generation

**Files:**
- Create: `tests/unit/mongodb/test_rag.py`
- Create: `src/vector_rag/mongodb_track/rag.py`

- [ ] **Step 1: Write failing context, refusal, and citation tests**

```python
def test_context_wraps_untrusted_chunks():
    context = ContextAssembler(max_characters=2000).assemble(chunks)
    assert "<retrieved_chunk" in context
    assert "untrusted" in context


def test_empty_evidence_returns_refusal(rag_pipeline):
    result = rag_pipeline.answer(query, [])
    assert result.answer == INSUFFICIENT_EVIDENCE
    assert result.cited_chunk_ids == []
```

- [ ] **Step 2: Verify red**

Run: `pytest tests/unit/mongodb/test_rag.py -q`
Expected: FAIL because RAG classes are missing.

- [ ] **Step 3: Implement RAG interfaces and providers**

Implement `Generator` protocol, credential-free `ExtractiveGenerator`, optional local `OllamaGenerator`, `ContextAssembler`, citation parsing/validation, timeout mapping, refusal handling, and `MongoRAGPipeline`. Never send provider output directly to logs.

- [ ] **Step 4: Add malformed-citation and timeout tests**

Invalid citations must be removed or converted to refusal; provider timeout must produce a typed failure without inventing evidence; all three latency fields must be non-negative.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/unit/mongodb/test_rag.py -q`
Expected: all RAG tests pass.

```bash
git add src/vector_rag/mongodb_track/rag.py tests/unit/mongodb/test_rag.py
git commit -m "feat(mongodb): add grounded RAG pipeline"
```

### Task 5: Embeddings, Metrics, Benchmark, and Artifacts

**Files:**
- Create: `tests/unit/mongodb/test_embeddings.py`
- Create: `tests/unit/mongodb/test_metrics.py`
- Create: `tests/unit/mongodb/test_benchmark.py`
- Create: `src/vector_rag/mongodb_track/embeddings.py`
- Create: `src/vector_rag/mongodb_track/metrics.py`
- Create: `src/vector_rag/mongodb_track/benchmark.py`
- Create: `configs/mongodb/benchmark.yaml`

- [ ] **Step 1: Write failing metric tests with hand-calculated values**

```python
def test_recall_mrr_and_ndcg_known_ranking():
    qrels = {"a": 2, "b": 1}
    ranking = ["x", "a", "b"]
    assert recall_at_k(ranking, qrels, 3) == 1.0
    assert mrr_at_k(ranking, qrels, 10) == 0.5
    assert ndcg_at_k(ranking, qrels, 3) == pytest.approx(expected_ndcg)
```

- [ ] **Step 2: Verify red, implement metrics, verify green**

Implement deterministic DCG/IDCG, percentiles with interpolation, per-query rows, and aggregate summary.

- [ ] **Step 3: Write failing artifact-writer tests**

Ensure a run writes `run_manifest.json`, `retrieval_results.jsonl`, `rag_results.jsonl`, `benchmark_summary.csv`, and `tuning_matrix.csv` without overwriting an existing run directory.

- [ ] **Step 4: Implement embedding adapters and benchmark orchestration**

Use the frozen sentence-transformers model in live runs and a deterministic hashing embedder only for tests/offline validation. Sweep `numCandidates`, preserve corpus/query SHA-256 values, and label execution environment as `atlas` or `offline_validation`.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/unit/mongodb/test_embeddings.py tests/unit/mongodb/test_metrics.py tests/unit/mongodb/test_benchmark.py -q`
Expected: all pass.

```bash
git add configs/mongodb/benchmark.yaml src/vector_rag/mongodb_track tests/unit/mongodb
git commit -m "feat(mongodb): add reproducible benchmark harness"
```

### Task 6: CLI, Integration Test, Results, and Report

**Files:**
- Create: `tests/unit/mongodb/test_cli.py`
- Create: `tests/integration/mongodb/test_atlas_smoke.py`
- Create: `src/vector_rag/mongodb_track/__main__.py`
- Create: `reports/sections/mongodb.md`
- Create: `results/mongodb/offline_validation/*`

- [ ] **Step 1: Write failing CLI parser tests**

Assert that `indexes`, `ingest`, `search`, `rag`, and `benchmark` parse without contacting MongoDB, and that missing live credentials fail with a concise error.

- [ ] **Step 2: Implement argparse CLI and gated integration test**

Lazy-load MongoDB, embeddings, and Ollama. Skip the live test only when `MONGODB_URI` is absent; otherwise create a test collection, ingest fixtures, run exact/ANN/text/hybrid smoke queries, and clean up only that test collection.

- [ ] **Step 3: Attempt local Atlas and run benchmark**

Start Docker Desktop if available, install Atlas CLI if required, create a local Atlas deployment, then run the live benchmark. If infrastructure cannot start, run `benchmark --offline-validation` and keep the manifest's environment label explicit.

- [ ] **Step 4: Write evidence-backed report**

Document architecture, document schema, index definitions, ingestion guarantees, retrieval modes, metrics, artifact paths, limitations, and exact reproduction commands. Never interpret offline validation values as MongoDB performance.

- [ ] **Step 5: Run complete verification and commit**

Run: `ruff check . && mypy src && pytest -q`
Expected: zero lint/type errors and all credential-free tests pass; live tests pass or skip with the explicit missing-URI reason.

```bash
git add src/vector_rag/mongodb_track tests configs/mongodb results/mongodb reports/sections/mongodb.md
git commit -m "docs(mongodb): add benchmark evidence and report"
git push origin ossama
```
