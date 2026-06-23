# MongoDB Vector Search vs Qdrant for Semantic Search and RAG

## Executive summary

The project now contains both contributor tracks in one integrated branch. Ossama's MongoDB track is
the most complete part: it includes document modeling, idempotent ingestion, five retrieval modes,
grounded RAG controls, offline validation artifacts, and a detailed report section. Hamza's Qdrant
track exists and was repaired during integration so it is testable without heavy optional
dependencies. It currently has a local smoke result rather than a full qrels-backed benchmark.

## What is ready

| Area | Status | Evidence |
| --- | --- | --- |
| MongoDB backend | Complete offline validation | `results/mongodb/offline_validation/offline-validation-v1/` |
| Qdrant backend | Integrated and unit-tested | `src/vector_rag/qdrant_track/backend.py` |
| Qdrant benchmark | Smoke result only | `results/qdrant/benchmark_results.json` |
| Frontend | Static GitHub Pages site | `docs/index.html` |
| Final comparison | Honest preliminary comparison | `results/comparison/final_comparison.csv` |

## Main technical takeaway

MongoDB is strongest when the application already needs a document model, metadata filters, lexical
search, and vector search in the same operational database. Qdrant is simpler as a dedicated vector
engine and gives a focused collection/payload model, but the current project still needs a complete
Qdrant benchmark export before strict quality comparisons are made.

## Threats to validity

- MongoDB committed results are offline validation, not Atlas latency measurements.
- Qdrant committed results are smoke measurements, not full relevance metrics.
- The corpus is intentionally small and project-authored, so it proves workflow correctness more
  than real-world retrieval performance.
- Any final claim about speed or quality must use the same corpus hash, query hash, embedding model,
  hardware/cluster environment, and metric definitions.

## Recommended next step

Run one live Atlas benchmark and one full Qdrant qrels-backed benchmark, then regenerate the final
comparison table. The code is structured so those runs can be added without changing the frontend or
the shared contracts.
