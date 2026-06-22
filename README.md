# MongoDB Vector Search vs Qdrant for Semantic Search and RAG

**Sujet proposé :** Bases de données vectorielles et RAG : comment connecter NoSQL, IA générative et recherche sémantique.

**Étude :** Indexation vectorielle dans les bases NoSQL : MongoDB Vector Search vs Qdrant pour la recherche sémantique et le RAG.

This public Master S2 data-engineering project compares two complete retrieval-augmented generation pipelines over the same corpus, embedding model, queries, relevance labels, and output contracts.

## Research Question

How do MongoDB Vector Search and Qdrant differ in retrieval quality, filtered-search behavior, ingestion performance, latency, operational complexity, and suitability for grounded RAG?

This is a comparative experiment, not a claim that one database is universally superior.

## Architecture

```text
Versioned corpus + queries + qrels
                |
     Frozen shared contracts
        /               \
 MongoDB track       Qdrant track
 (Ossama)            (Hamza)
        \               /
 Standard JSONL/CSV benchmark results
                |
        Final comparison report
```

Each track is standalone. Both contributors can ingest, retrieve, run RAG, test, and export results without waiting for the other track.

## Contributor Tracks

| Contributor | Branch | Independent track |
| --- | --- | --- |
| Ossama | `ossama` | MongoDB modeling, resilient ingestion, ANN/ENN, filtering, lexical search, reciprocal-rank fusion, grounded RAG, and advanced tuning |
| Hamza El Haddaji | `HamzaElhaddaji` | Docker Qdrant, collection and payload indexes, dense retrieval with fixed defaults, filters, shared RAG helpers, and standard benchmark |

The complete acceptance criteria and output paths are in [TASKS.md](TASKS.md). The working rules are in [COLLABORATION.md](COLLABORATION.md).

## Frozen Fairness Contract

Both tracks must use:

- `sentence-transformers/all-MiniLM-L6-v2`
- 384 dimensions and cosine similarity
- `data/demo/corpus.jsonl`
- `data/evaluation/queries.jsonl`
- `data/evaluation/qrels.tsv`
- the validated models in `src/vector_rag/contracts.py`
- the settings in `configs/benchmark.yaml`

Core metrics are Recall@5, Recall@10, MRR@10, nDCG@10, ingestion throughput, index-build time, and query latency p50/p95. RAG outputs also record citations and deterministic grounding checks. Optional LLM-judge results must be reported separately.

## Starter Setup

```bash
git clone https://github.com/Maaskk/mongodb-qdrant-vector-search-rag.git
cd mongodb-qdrant-vector-search-rag
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Hamza can start Qdrant locally with:

```bash
docker compose up -d qdrant
```

Ossama copies `.env.example` to `.env` and supplies a MongoDB Atlas URI. Secrets must never be committed.

## Repository Map

| Path | Purpose |
| --- | --- |
| `configs/` | Frozen embedding and benchmark configuration |
| `data/` | Project-authored smoke corpus, queries, and relevance labels |
| `src/vector_rag/` | Shared contracts now; independent backend packages during implementation |
| `tests/` | Offline contract and fixture tests; backend integration tests later |
| `results/mongodb/` | Ossama's versioned benchmark artifacts |
| `results/qdrant/` | Hamza's versioned benchmark artifacts |
| `reports/sections/` | Independently authored report sections |
| `docs/` | Architecture, approved design, and implementation plan |

## Quality Rules

- Never commit `.env`, API keys, database passwords, or connection strings.
- Do not change the frozen corpus or embedding configuration on a personal branch.
- Keep generated results inside the contributor's own result directory.
- Run `ruff check .`, `mypy src`, and `pytest -q` before opening a pull request.
- Merge personal work into `main` through reviewed pull requests.

## License

The starter code and project-authored demonstration data are released under the MIT License.

