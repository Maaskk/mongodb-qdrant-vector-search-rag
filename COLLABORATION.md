# Collaboration Guide

## Repository

```text
https://github.com/Maaskk/mongodb-qdrant-vector-search-rag
```

Hamza must accept the GitHub collaborator invitation before pushing directly to his branch.

## Clone and Select Your Branch

Ossama:

```bash
git clone https://github.com/Maaskk/mongodb-qdrant-vector-search-rag.git
cd mongodb-qdrant-vector-search-rag
git switch ossama
```

Hamza:

```bash
git clone https://github.com/Maaskk/mongodb-qdrant-vector-search-rag.git
cd mongodb-qdrant-vector-search-rag
git switch HamzaElhaddaji
```

## Independence Rule

Work only in the owned paths listed in `TASKS.md`. The frozen starter already contains the corpus, query set, contracts, and benchmark settings, so both people can begin and finish without waiting for the other person's code.

If shared code truly needs to change, open a small dedicated pull request into `main`. Do not quietly change shared contracts on a personal branch.

## Daily Workflow

```bash
git switch YOUR_BRANCH
git pull --ff-only
git status
# make and test one focused change
git add PATHS_YOU_CHANGED
git commit -m "short, specific message"
git push
```

Commit generated benchmark outputs separately from source-code changes so reviews remain readable.

## Pull Request Workflow

1. Complete one GitHub issue and its acceptance criteria.
2. Run `ruff check .`, `mypy src`, and `pytest -q`.
3. Push the personal branch.
4. Open a pull request from the personal branch into `main`.
5. Link the issue and include the exact reproduction command.
6. Ask the other contributor to review the evidence and changed contracts.
7. Merge only when checks pass and no secrets are present.

## Conflict Prevention

- Ossama edits only the MongoDB-owned paths.
- Hamza edits only the Qdrant-owned paths.
- Do not reformat or reorganize the other track.
- Keep reports in separate section files until final integration.
- Keep result artifacts in separate backend directories.
- Never use force-push on `main`.

## Fair Comparison Rules

- Same corpus and queries.
- Same embedding model and vector dimensions.
- Same relevance labels and metric definitions.
- Same hardware/environment information recorded with each run.
- Warmup queries are excluded from measured latency.
- Failed queries are reported, not deleted from the experiment.

## Secrets

Copy `.env.example` to `.env`. The real `.env` is ignored by Git. Before every push, inspect `git diff --cached` and confirm that no URI, password, API key, prompt containing private content, or raw credential is present.

