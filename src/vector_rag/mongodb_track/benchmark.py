"""Reproducible benchmark artifacts and credential-free functional validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import Any

from vector_rag.contracts import Chunk, EvaluationQuery, RAGResult, RetrievalResult
from vector_rag.io import load_jsonl
from vector_rag.mongodb_track.embeddings import HashingEmbedder
from vector_rag.mongodb_track.hybrid import reciprocal_rank_fusion
from vector_rag.mongodb_track.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k
from vector_rag.mongodb_track.rag import ExtractiveGenerator, MongoRAGPipeline


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    environment: str
    corpus_sha256: str
    queries_sha256: str
    embedding_model: str
    dimensions: int
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    backend: str = "mongodb"
    warning: str = (
        "offline_validation checks functionality only; its latency and ranking values are not "
        "MongoDB Atlas performance measurements"
    )


class ArtifactWriter:
    """Write one immutable benchmark run directory."""

    def __init__(self, root: Path, *, run_id: str) -> None:
        self.root = root
        self.run_id = run_id

    def write(
        self,
        manifest: RunManifest,
        retrieval_results: Sequence[RetrievalResult],
        rag_results: Sequence[RAGResult],
        summary_rows: Sequence[Mapping[str, Any]],
        tuning_rows: Sequence[Mapping[str, Any]],
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        run_dir = self.root / self.run_id
        run_dir.mkdir(exist_ok=False)
        self._write_json(run_dir / "run_manifest.json", asdict(manifest))
        self._write_jsonl(run_dir / "retrieval_results.jsonl", retrieval_results)
        self._write_jsonl(run_dir / "rag_results.jsonl", rag_results)
        self._write_csv(run_dir / "benchmark_summary.csv", summary_rows)
        self._write_csv(run_dir / "tuning_matrix.csv", tuning_rows)
        return run_dir

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, records: Sequence[Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                payload = record.model_dump(mode="json")
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8", newline="") as handle:
            if not fieldnames:
                return
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels[row["query_id"]][row["chunk_id"]] = int(row["relevance"])
    return dict(qrels)


def run_offline_validation(
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    output_root: Path,
    run_id: str,
    dimensions: int = 384,
    top_k: int = 5,
) -> Path:
    """Exercise the full evaluation flow without claiming a MongoDB benchmark."""

    chunks = load_jsonl(corpus_path, Chunk)
    queries = load_jsonl(queries_path, EvaluationQuery)
    qrels = load_qrels(qrels_path)
    embedder = HashingEmbedder(dimensions=dimensions)
    chunk_vectors = embedder.embed([f"{chunk.title} {chunk.text}" for chunk in chunks])
    query_vectors = embedder.embed([query.text for query in queries])
    retrieval_results: list[RetrievalResult] = []
    rag_results: list[RAGResult] = []
    results_by_mode: dict[str, list[RetrievalResult]] = defaultdict(list)

    for query, query_vector in zip(queries, query_vectors, strict=True):
        vector_hits = _offline_vector_hits(
            query,
            query_vector,
            chunks,
            chunk_vectors,
            run_id=run_id,
            top_k=top_k,
            mode="exact",
        )
        ann_hits = [hit.model_copy(update={"search_mode": "ann"}) for hit in vector_hits]
        filtered_hits = _offline_vector_hits(
            query,
            query_vector,
            chunks,
            chunk_vectors,
            run_id=run_id,
            top_k=top_k,
            mode="filtered_ann",
            filters=query.filters,
        )
        text_hits = _offline_text_hits(query, chunks, run_id=run_id, top_k=top_k)
        hybrid_hits = reciprocal_rank_fusion(vector_hits, text_hits, top_k=top_k)
        modes = {
            "exact": vector_hits,
            "ann": ann_hits,
            "filtered_ann": filtered_hits,
            "text": text_hits,
            "hybrid_rrf": hybrid_hits,
        }
        for mode, hits in modes.items():
            retrieval_results.extend(hits)
            results_by_mode[mode].extend(hits)

        hybrid_chunk_ids = [hit.chunk_id for hit in hybrid_hits]
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        evidence = [by_id[item] for item in hybrid_chunk_ids if item in by_id]
        retrieval_latency = hybrid_hits[0].latency_ms if hybrid_hits else 0.0
        rag_results.append(
            MongoRAGPipeline(ExtractiveGenerator(), run_id=run_id).answer(
                query, evidence, retrieval_latency_ms=retrieval_latency
            )
        )

    summary_rows = _summaries(results_by_mode, qrels)
    tuning_rows = [
        {
            "environment": "offline_validation",
            "num_candidates": candidates,
            "recall_at_5": next(
                row["recall_at_5"] for row in summary_rows if row["mode"] == "ann"
            ),
            "note": "functional sweep only; MongoDB ANN was not executed",
        }
        for candidates in (10, 25, 50, 100, 200)
    ]
    manifest = RunManifest(
        run_id=run_id,
        environment="offline_validation",
        corpus_sha256=_file_sha256(corpus_path),
        queries_sha256=_file_sha256(queries_path),
        embedding_model=embedder.model_name,
        dimensions=dimensions,
    )
    return ArtifactWriter(output_root, run_id=run_id).write(
        manifest, retrieval_results, rag_results, summary_rows, tuning_rows
    )


def _offline_vector_hits(
    query: EvaluationQuery,
    query_vector: Sequence[float],
    chunks: Sequence[Chunk],
    chunk_vectors: Sequence[Sequence[float]],
    *,
    run_id: str,
    top_k: int,
    mode: str,
    filters: Mapping[str, Any] | None = None,
) -> list[RetrievalResult]:
    started = time.perf_counter()
    scored: list[tuple[float, Chunk]] = []
    for chunk, vector in zip(chunks, chunk_vectors, strict=True):
        if filters and any(getattr(chunk, key, None) != value for key, value in filters.items()):
            continue
        scored.append((HashingEmbedder.similarity(query_vector, vector), chunk))
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return [
        RetrievalResult(
            backend="mongodb",
            query_id=query.query_id,
            rank=rank,
            chunk_id=chunk.chunk_id,
            score=score,
            latency_ms=elapsed_ms,
            search_mode=mode,
            filter_name="query_filters" if filters else "none",
            run_id=run_id,
        )
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
    ]


def _offline_text_hits(
    query: EvaluationQuery,
    chunks: Sequence[Chunk],
    *,
    run_id: str,
    top_k: int,
) -> list[RetrievalResult]:
    token_pattern = re.compile(r"[\w-]+")
    query_tokens = set(token_pattern.findall(query.text.lower()))
    started = time.perf_counter()
    scored = []
    for chunk in chunks:
        tokens = set(token_pattern.findall(f"{chunk.title} {chunk.text}".lower()))
        union = query_tokens | tokens
        score = len(query_tokens & tokens) / len(union) if union else 0.0
        scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
    elapsed_ms = (time.perf_counter() - started) * 1000
    return [
        RetrievalResult(
            backend="mongodb",
            query_id=query.query_id,
            rank=rank,
            chunk_id=chunk.chunk_id,
            score=score,
            latency_ms=elapsed_ms,
            search_mode="text",
            filter_name="none",
            run_id=run_id,
        )
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
    ]


def _summaries(
    results_by_mode: Mapping[str, Sequence[RetrievalResult]],
    qrels: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, results in results_by_mode.items():
        by_query: dict[str, list[RetrievalResult]] = defaultdict(list)
        for result in results:
            by_query[result.query_id].append(result)
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []
        for query_id, query_results in by_query.items():
            ordered = sorted(query_results, key=lambda item: item.rank)
            ranking = [item.chunk_id for item in ordered]
            judgments = qrels.get(query_id, {})
            recalls.append(recall_at_k(ranking, judgments, 5))
            mrrs.append(mrr_at_k(ranking, judgments, 10))
            ndcgs.append(ndcg_at_k(ranking, judgments, 10))
            latencies.append(ordered[0].latency_ms)
        rows.append(
            {
                "environment": "offline_validation",
                "mode": mode,
                "queries": len(by_query),
                "recall_at_5": sum(recalls) / len(recalls),
                "mrr_at_10": sum(mrrs) / len(mrrs),
                "ndcg_at_10": sum(ndcgs) / len(ndcgs),
                "latency_p50_ms": percentile(latencies, 50),
                "latency_p95_ms": percentile(latencies, 95),
                "latency_warning": "Python functional timing; not MongoDB latency",
            }
        )
    return rows


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
