# Vector Databases RAG Collaboration Starter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a collaboration-ready public repository with frozen data contracts, a tested Python starter, clearly independent MongoDB and Qdrant work tracks, personal branches, GitHub issues, and collaborator access.

**Architecture:** The `main` branch owns immutable corpus and result contracts plus a backend protocol. Personal backend packages implement that protocol independently and export schema-compatible benchmark artifacts. Documentation and GitHub issues assign the advanced MongoDB/hybrid-RAG track to Ossama and the guided Qdrant track to Hamza.

**Tech Stack:** Python 3.11+, Pydantic 2, pytest, Ruff, mypy, PyYAML, Docker Compose, MongoDB Atlas Vector Search, Qdrant, GitHub Actions, GitHub REST API.

---

## File Map

- `pyproject.toml`: package metadata, dependencies, and quality-tool settings.
- `.gitignore`: secrets, environments, caches, and generated benchmark data.
- `.env.example`: safe backend and generation environment-variable names.
- `docker-compose.yml`: local Qdrant service for Hamza's track.
- `configs/*.yaml`: frozen embedding and benchmark settings.
- `data/demo/corpus.jsonl`: redistributable smoke-test corpus.
- `data/evaluation/queries.jsonl`: deterministic evaluation queries.
- `data/evaluation/qrels.tsv`: query-to-relevant-chunk labels.
- `src/vector_rag/contracts.py`: validated chunk, query, retrieval, and RAG models.
- `src/vector_rag/backend.py`: database-adapter protocol.
- `src/vector_rag/io.py`: JSONL loading and validation.
- `tests/`: contract and fixture tests that need no cloud credentials.
- `README.md`: public project landing page and quick start.
- `TASKS.md`: detailed independent contributor checklists and deliverables.
- `COLLABORATION.md`: branch and pull-request workflow.
- `docs/architecture.md`: system boundaries and data flow.
- `.github/workflows/ci.yml`: lint, type-check, and test workflow.
- `.github/ISSUE_TEMPLATE/*.yml`: structured work-item templates.
- `.github/pull_request_template.md`: contributor quality checklist.

### Task 1: Python Project and Runtime Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `docker-compose.yml`
- Create: `configs/embeddings.yaml`
- Create: `configs/benchmark.yaml`
- Test: `tests/test_project_files.py`

- [ ] **Step 1: Write the failing project-file test**

```python
from pathlib import Path


def test_required_project_files_exist() -> None:
    for path in (
        "pyproject.toml",
        ".env.example",
        "docker-compose.yml",
        "configs/embeddings.yaml",
        "configs/benchmark.yaml",
    ):
        assert Path(path).is_file(), path
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `pytest tests/test_project_files.py -q`
Expected: FAIL because the required project files do not exist.

- [ ] **Step 3: Add the runtime configuration**

Define a Python 3.11 package with Pydantic, PyYAML, pytest, Ruff, and mypy. Add optional `mongodb`, `qdrant`, `embeddings`, and `generation` dependency groups. Configure Qdrant 6333/6334 ports and a named volume. Freeze `sentence-transformers/all-MiniLM-L6-v2`, cosine similarity, dimension 384, top-k values 5 and 10, and benchmark warmup/repetition counts.

- [ ] **Step 4: Run the project-file test**

Run: `pytest tests/test_project_files.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore .env.example docker-compose.yml configs tests/test_project_files.py
git commit -m "chore: configure vector RAG starter"
```

### Task 2: Frozen Data and Result Contracts

**Files:**
- Create: `src/vector_rag/__init__.py`
- Create: `src/vector_rag/contracts.py`
- Create: `src/vector_rag/backend.py`
- Create: `src/vector_rag/io.py`
- Create: `tests/test_contracts.py`
- Create: `tests/test_backend_protocol.py`

- [ ] **Step 1: Write failing validation tests**

```python
import pytest
from pydantic import ValidationError
from vector_rag.contracts import Chunk, RetrievalResult


def test_chunk_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            chunk_id="doc-1:0",
            document_id="doc-1",
            title="Example",
            text="",
            source="demo",
            category="database",
            language="en",
            chunk_index=0,
            content_hash="a" * 64,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        )


def test_retrieval_rank_is_positive() -> None:
    with pytest.raises(ValidationError):
        RetrievalResult(
            backend="qdrant",
            query_id="q1",
            rank=0,
            chunk_id="doc-1:0",
            score=0.9,
            latency_ms=2.0,
            search_mode="dense",
            filter_name="none",
            run_id="run-1",
        )
```

- [ ] **Step 2: Confirm tests fail before models exist**

Run: `pytest tests/test_contracts.py -q`
Expected: collection error for missing `vector_rag.contracts`.

- [ ] **Step 3: Implement strict Pydantic models and backend protocol**

Create models named `Chunk`, `EvaluationQuery`, `RetrievalResult`, and `RAGResult`. Enforce non-empty identifiers and text, SHA-256-length content hashes, non-negative latencies, positive result ranks, and cited chunk identifiers. Define a runtime-checkable `VectorBackend` protocol with `ingest`, `search`, and `healthcheck` methods.

- [ ] **Step 4: Add validated JSONL loading**

Implement `load_jsonl(path: Path, model: type[BaseModel]) -> list[BaseModel]`. Raise an error that includes the filename and one-based line number when JSON or model validation fails.

- [ ] **Step 5: Run contract tests**

Run: `pytest tests/test_contracts.py tests/test_backend_protocol.py -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src tests/test_contracts.py tests/test_backend_protocol.py
git commit -m "feat: define shared backend contracts"
```

### Task 3: Demonstration Corpus and Evaluation Fixtures

**Files:**
- Create: `data/demo/corpus.jsonl`
- Create: `data/evaluation/queries.jsonl`
- Create: `data/evaluation/qrels.tsv`
- Create: `tests/test_data_fixtures.py`

- [ ] **Step 1: Write a failing fixture-integrity test**

```python
from pathlib import Path
from vector_rag.contracts import Chunk, EvaluationQuery
from vector_rag.io import load_jsonl


def test_demo_fixtures_are_valid_and_linked() -> None:
    chunks = load_jsonl(Path("data/demo/corpus.jsonl"), Chunk)
    queries = load_jsonl(Path("data/evaluation/queries.jsonl"), EvaluationQuery)
    chunk_ids = {item.chunk_id for item in chunks}
    qrel_lines = Path("data/evaluation/qrels.tsv").read_text().splitlines()[1:]
    assert len(chunks) >= 6
    assert len(queries) >= 4
    assert all(line.split("\t")[1] in chunk_ids for line in qrel_lines)
```

- [ ] **Step 2: Confirm the test fails because fixtures are absent**

Run: `pytest tests/test_data_fixtures.py -q`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Add original educational fixtures**

Write six short original documents covering vector databases, HNSW, MongoDB document modeling, metadata filtering, hybrid search, and RAG grounding. Add four questions with expected-answer text and relevance labels. Use deterministic chunk identifiers and SHA-256 hashes matching the stored text.

- [ ] **Step 4: Run fixture and contract tests**

Run: `pytest tests/test_data_fixtures.py tests/test_contracts.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add data tests/test_data_fixtures.py
git commit -m "test: add frozen evaluation fixtures"
```

### Task 4: Public Documentation and Independent Task Split

**Files:**
- Create: `README.md`
- Create: `TASKS.md`
- Create: `COLLABORATION.md`
- Create: `docs/architecture.md`
- Create: `reports/sections/README.md`
- Test: `tests/test_documentation.py`

- [ ] **Step 1: Write a failing documentation test**

```python
from pathlib import Path


def test_task_split_names_real_branches_and_backends() -> None:
    tasks = Path("TASKS.md").read_text()
    assert "`ossama`" in tasks
    assert "`HamzaElhaddaji`" in tasks
    assert "MongoDB" in tasks
    assert "Qdrant" in tasks
    assert "Shared output contract" in tasks
```

- [ ] **Step 2: Confirm the test fails because documentation is absent**

Run: `pytest tests/test_documentation.py -q`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Write the public project guide**

The README must explain the research question, architecture, reproducibility rules, quick start, branch ownership, metrics, and secret handling. `TASKS.md` must give Ossama the advanced MongoDB data modeling, idempotent ingestion, ANN/ENN, `numCandidates` tuning, full-text search, reciprocal-rank fusion, filtered retrieval, grounded RAG, and deep benchmark tasks. It must give Hamza Docker Qdrant, one collection, payload indexes, dense retrieval with fixed defaults, filters, shared RAG helpers, and the supplied benchmark command. Each deliverable must have an acceptance checklist and output path.

- [ ] **Step 4: Document workflow and architecture**

Explain that contributors branch from the frozen starter, work only inside their backend package and result directory, and merge with pull requests. Define integration as a final comparison of already-compatible result files.

- [ ] **Step 5: Run documentation tests**

Run: `pytest tests/test_documentation.py -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add README.md TASKS.md COLLABORATION.md docs/architecture.md reports tests/test_documentation.py
git commit -m "docs: assign independent contributor tracks"
```

### Task 5: GitHub Quality Automation

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/ISSUE_TEMPLATE/task.yml`
- Create: `.github/pull_request_template.md`
- Test: `tests/test_github_configuration.py`

- [ ] **Step 1: Write a failing GitHub-configuration test**

```python
from pathlib import Path
import yaml


def test_ci_runs_quality_commands() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text())
    rendered = str(workflow)
    assert "ruff check" in rendered
    assert "mypy" in rendered
    assert "pytest" in rendered
```

- [ ] **Step 2: Confirm the test fails because workflow is absent**

Run: `pytest tests/test_github_configuration.py -q`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Add CI and contribution templates**

Configure Actions for Python 3.11, editable installation, Ruff, mypy, and pytest. Add an issue form that captures owner, backend, acceptance criteria, output files, and verification command. Add a pull-request checklist for tests, secrets, schema versions, reproducibility, and documentation.

- [ ] **Step 4: Run the complete local quality suite**

Run: `python -m ruff check . && python -m mypy src && python -m pytest -q`
Expected: all commands exit zero.

- [ ] **Step 5: Commit**

```bash
git add .github tests/test_github_configuration.py
git commit -m "ci: add collaboration quality gates"
```

### Task 6: Publish and Configure GitHub Collaboration

**External state:**
- Create: `Maaskk/mongodb-qdrant-vector-search-rag`
- Create branches: `ossama`, `HamzaElhaddaji`
- Invite collaborator: `HamzaElhaddaji`
- Create labels and task issues from `TASKS.md`

- [ ] **Step 1: Verify GitHub authentication without printing credentials**

Run a GitHub API request to `/user` using the access token returned by the macOS Git credential helper. Print only the authenticated login.
Expected: `Maaskk`.

- [ ] **Step 2: Create the public repository and push main**

Create the repository with description `Comparative data-engineering study of MongoDB Vector Search and Qdrant for semantic search and RAG.` Enable issues and disable wiki. Set the local `origin` and push `main`.

- [ ] **Step 3: Create and push personal branches**

Create both branch refs from the verified `main` commit and confirm the GitHub API lists `main`, `ossama`, and `HamzaElhaddaji`.

- [ ] **Step 4: Invite Hamza**

Send a repository collaborator invitation to `HamzaElhaddaji` with `push` permission. Accept HTTP 201 for a new invitation or HTTP 204 if access already exists.

- [ ] **Step 5: Add labels and independent issues**

Create `ossama`, `hamza`, `shared`, `mongodb`, `qdrant`, `data-engineering`, `rag`, and `benchmark` labels. Create separate issues for Ossama's MongoDB model/ingestion, retrieval modes, RAG, and advanced benchmark; Hamza's Qdrant setup/ingestion, dense search/filters, RAG, and standard benchmark; and one final shared integration issue.

- [ ] **Step 6: Verify remote state**

Query the GitHub API and confirm the repository is public, the three branches exist, the collaborator invitation or access exists, and every planned issue is present. Run `git status --short --branch` and confirm `main` tracks `origin/main` with only ignored local scratch directories.

- [ ] **Step 7: Record the repository URL**

Expected URL: `https://github.com/Maaskk/mongodb-qdrant-vector-search-rag`
