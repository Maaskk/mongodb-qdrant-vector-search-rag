# MongoDB Vector Search vs Qdrant RAG Project Design

## 1. Purpose

Build a reproducible data-engineering project that compares MongoDB Vector Search and Qdrant for semantic retrieval and retrieval-augmented generation (RAG). The project must demonstrate data ingestion, document and metadata modeling, vector indexing, filtered search, retrieval evaluation, RAG generation, and operational benchmarking.

The repository will be public at GitHub under the `Maaskk` account. Its planned name is:

```text
mongodb-qdrant-vector-search-rag
```

The two contributor branches are exactly:

```text
ossama
HamzaElhaddaji
```

Hamza's GitHub account, `HamzaElhaddaji`, will be invited with push access.

## 2. Collaboration Model

The work is divided by database engine, not by pipeline stage. Each contributor owns a complete standalone pipeline and can ingest the frozen corpus, build an index, run retrieval, generate RAG answers, execute tests, and export results without code or output from the other contributor.

Both tracks consume shared, versioned input fixtures and implement a stable adapter interface. Both export the same result schemas. No contributor owns a shared prerequisite that the other must wait for.

Integration happens only after both tracks are complete. It consists of comparing already-exported result files, producing common plots, and assembling the final report.

## 3. System Architecture

```text
Frozen corpus + queries + relevance labels
                    |
          Shared benchmark contract
             /                 \
 MongoDB standalone track   Qdrant standalone track
             \                 /
        Standard result JSON/CSV files
                    |
        Final comparison and report
```

### Shared Foundation

The initial `main` branch provides these finished prerequisites:

- A small, legally redistributable demonstration corpus.
- A versioned corpus schema using JSON Lines.
- Evaluation queries and relevance labels.
- Deterministic chunk identifiers and metadata fields.
- A fixed local embedding model configuration.
- Backend adapter protocols for ingestion and retrieval.
- A benchmark runner and result schemas.
- Unit-test fixtures, linting, and continuous integration.
- `.env.example` files without credentials.

The shared foundation is frozen before the personal branches diverge. Changes to a shared contract require a pull request into `main` and must preserve compatibility.

### Data Contract

Every chunk contains:

```text
chunk_id, document_id, title, text, source, category, language,
published_at, chunk_index, content_hash, embedding_model
```

Every retrieval result contains:

```text
backend, query_id, rank, chunk_id, score, latency_ms, search_mode,
filter_name, run_id
```

Every RAG result contains:

```text
backend, query_id, answer, cited_chunk_ids, retrieval_latency_ms,
generation_latency_ms, total_latency_ms, run_id
```

The benchmark exports Recall@5, Recall@10, MRR@10, nDCG@10, ingestion throughput, index-build time, query latency p50/p95, filter selectivity, citation coverage, and answer-grounding checks. Optional LLM-judge metrics must be clearly separated from deterministic metrics.

## 4. Ossama Track - Advanced MongoDB and Hybrid RAG

Ossama receives the larger and more technically demanding track.

### MongoDB Data Engineering

- Design the MongoDB document model for chunks, embeddings, metadata, source lineage, and ingestion-run metadata.
- Implement idempotent bulk ingestion with content hashes, retries, resumability, duplicate prevention, and batch-size tuning.
- Create and document Vector Search and Search index definitions.
- Implement typed pre-filters for category, language, date, and source.
- Record index readiness and failure diagnostics.

### Retrieval Engineering

- Implement approximate nearest-neighbor search and exact nearest-neighbor baselines.
- Tune `numCandidates` against recall and latency.
- Implement MongoDB full-text retrieval.
- Implement hybrid full-text plus vector retrieval using reciprocal rank fusion.
- Compare semantic-only, keyword-only, hybrid, filtered, and exact-search modes.
- Capture aggregation-pipeline explain output where supported.

### RAG Engineering

- Build retrieval-to-prompt context assembly with token budgeting.
- Add source citations and reject answers that cannot be grounded in retrieved context.
- Handle empty retrieval, malformed documents, provider timeouts, and partial failures.
- Provide a provider-neutral generator interface and one documented runnable provider.

### Ossama Deliverables

- MongoDB ingestion package and CLI commands.
- MongoDB index definitions and setup guide.
- Semantic, exact, keyword, hybrid, and filtered retrieval implementations.
- MongoDB RAG pipeline with citations.
- Advanced MongoDB benchmark matrix and exported results.
- MongoDB architecture and analysis section for the report.
- Unit, contract, and opt-in Atlas integration tests.

## 5. Hamza Track - Guided Qdrant RAG

Hamza receives a narrower path with fixed defaults and ready-made shared tooling.

### Qdrant Data Engineering

- Start Qdrant from the supplied Docker Compose configuration.
- Create one collection using the supplied vector dimension and distance metric.
- Create payload indexes for the documented metadata fields.
- Ingest chunks through the prepared adapter interface.

### Retrieval and RAG

- Implement dense semantic search using the default HNSW configuration.
- Implement category, language, and source payload filters.
- Connect retrieved chunks to the shared context builder and generator interface.
- Run the supplied benchmark command without designing new metrics or tuning matrices.

### Hamza Deliverables

- Qdrant collection setup and ingestion adapter.
- Dense and metadata-filtered search.
- Qdrant RAG pipeline using shared prompt and citation helpers.
- Standard Qdrant benchmark results.
- Qdrant setup, method, and results section for the report.
- Unit, contract, and Docker-backed integration tests.

Hamza is not responsible for hybrid sparse/dense fusion, quantization studies, custom HNSW tuning, evaluation-framework design, or final cross-engine aggregation.

## 6. Repository Layout

```text
.
├── .github/
│   ├── ISSUE_TEMPLATE/
│   ├── pull_request_template.md
│   └── workflows/ci.yml
├── configs/
│   ├── benchmark.yaml
│   ├── embeddings.yaml
│   ├── mongodb/
│   └── qdrant/
├── data/
│   ├── demo/
│   └── evaluation/
├── docs/
│   ├── architecture.md
│   └── superpowers/specs/
├── reports/
│   └── sections/
├── results/
│   ├── mongodb/
│   ├── qdrant/
│   └── comparison/
├── src/vector_rag/
│   ├── common/
│   ├── mongodb_track/
│   ├── qdrant_track/
│   └── comparison/
├── tests/
│   ├── contract/
│   ├── integration/
│   └── unit/
├── COLLABORATION.md
├── TASKS.md
├── README.md
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## 7. GitHub Workflow

- `main` is the integration branch.
- `ossama` is Ossama's long-lived contributor branch.
- `HamzaElhaddaji` is Hamza's long-lived contributor branch.
- Each task is represented by a GitHub issue carrying an `ossama`, `hamza`, or `shared` label.
- Personal branches merge through pull requests into `main`.
- Pull requests run formatting, linting, unit tests, and contract tests.
- Integration tests requiring Atlas credentials are opt-in and never expose secrets.
- Commits and pull requests should reference their task issue.

## 8. Error Handling and Security

- Credentials are read only from environment variables and are never committed.
- Logs redact connection strings, tokens, document contents, and generated prompts where appropriate.
- Ingestion is idempotent and records failed chunk identifiers for retry.
- Backend adapters use bounded timeouts and retry only transient failures.
- Result writers use run identifiers and avoid silently overwriting previous runs.
- Public documentation explains Atlas network access and least-privilege database roles.
- Retrieved content is treated as untrusted input, and prompt-injection-resistant delimiters and instructions are included in the RAG context builder.

## 9. Testing and Quality Gates

- Unit tests cover chunk validation, identifiers, metrics, context assembly, and result serialization.
- Shared contract tests run against both adapters.
- Qdrant integration tests run against Docker.
- MongoDB unit and contract tests run without cloud credentials; Atlas integration tests run only when explicitly enabled.
- A smoke benchmark verifies that both tracks can export schema-valid results.
- Ruff performs formatting and linting; pytest runs the test suite; mypy checks the stable adapter boundary.
- The final comparison rejects incompatible embedding models, corpus hashes, query sets, or result-schema versions.

## 10. Completion Criteria

The project is complete when:

1. Each personal branch contains a runnable, documented standalone backend track.
2. Both tracks pass the shared contract tests.
3. Both tracks evaluate the same corpus, embeddings, queries, and relevance labels.
4. Results contain retrieval-quality and operational metrics.
5. RAG answers cite retrieved chunks and pass deterministic grounding checks.
6. The final report explains architecture, experimental protocol, results, trade-offs, limitations, and recommendations.
7. A fresh user can reproduce the Qdrant track locally and can run the MongoDB track after supplying documented Atlas credentials.

## 11. Explicit Non-Goals

- Building a production multi-user web application.
- Training a custom embedding or generative model.
- Requiring a paid LLM API for the core benchmark.
- Making one contributor responsible for a prerequisite needed by the other.
- Claiming a universally superior database from a single small benchmark.
