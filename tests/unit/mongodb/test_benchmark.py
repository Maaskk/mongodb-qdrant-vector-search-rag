import json
from pathlib import Path

import pytest

from vector_rag.contracts import RAGResult, RetrievalResult
from vector_rag.mongodb_track.benchmark import ArtifactWriter, RunManifest


def retrieval() -> RetrievalResult:
    return RetrievalResult(
        backend="mongodb",
        query_id="q1",
        rank=1,
        chunk_id="mongodb-overview:0",
        score=0.9,
        latency_ms=1.0,
        search_mode="exact",
        filter_name="none",
        run_id="test-run",
    )


def rag() -> RAGResult:
    return RAGResult(
        backend="mongodb",
        query_id="q1",
        answer="MongoDB stores vectors [mongodb-overview:0].",
        cited_chunk_ids=["mongodb-overview:0"],
        retrieval_latency_ms=1.0,
        generation_latency_ms=0.1,
        total_latency_ms=1.1,
        run_id="test-run",
    )


def test_artifact_writer_creates_complete_immutable_run(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path, run_id="test-run")
    manifest = RunManifest(
        run_id="test-run",
        environment="offline_validation",
        corpus_sha256="a" * 64,
        queries_sha256="b" * 64,
        embedding_model="hashing-v1",
        dimensions=384,
    )

    run_dir = writer.write(
        manifest,
        retrieval_results=[retrieval()],
        rag_results=[rag()],
        summary_rows=[{"mode": "exact", "recall_at_5": 1.0}],
        tuning_rows=[{"num_candidates": 100, "recall_at_5": 1.0}],
    )

    assert {path.name for path in run_dir.iterdir()} == {
        "run_manifest.json",
        "retrieval_results.jsonl",
        "rag_results.jsonl",
        "benchmark_summary.csv",
        "tuning_matrix.csv",
    }
    assert json.loads((run_dir / "run_manifest.json").read_text())["environment"] == (
        "offline_validation"
    )

    with pytest.raises(FileExistsError):
        writer.write(manifest, [], [], [], [])
