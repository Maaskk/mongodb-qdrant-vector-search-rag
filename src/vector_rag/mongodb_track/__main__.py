"""Command-line entry point for Ossama's independent MongoDB track."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vector_rag.contracts import Chunk, EvaluationQuery
from vector_rag.io import load_jsonl
from vector_rag.mongodb_track.benchmark import run_live_benchmark, run_offline_validation
from vector_rag.mongodb_track.config import MongoConfig, MongoConfigError
from vector_rag.mongodb_track.embeddings import HashingEmbedder, SentenceTransformerEmbedder
from vector_rag.mongodb_track.hybrid import reciprocal_rank_fusion
from vector_rag.mongodb_track.indexes import IndexManager
from vector_rag.mongodb_track.ingestion import MongoIngestor
from vector_rag.mongodb_track.rag import ExtractiveGenerator, MongoRAGPipeline, OllamaGenerator
from vector_rag.mongodb_track.search import MongoRetriever

DEFAULT_CORPUS = Path("data/demo/corpus.jsonl")
DEFAULT_QUERIES = Path("data/evaluation/queries.jsonl")
DEFAULT_QRELS = Path("data/evaluation/qrels.tsv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vector_rag.mongodb_track",
        description="MongoDB Vector Search and grounded RAG track",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    indexes = subparsers.add_parser("indexes", help="create missing search indexes")
    indexes.add_argument("--wait", action="store_true", help="wait until both are queryable")

    ingest = subparsers.add_parser("ingest", help="embed and idempotently ingest the corpus")
    ingest.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ingest.add_argument("--run-id", default=None)
    ingest.add_argument("--batch-size", type=int, default=100)
    ingest.add_argument(
        "--embedding-provider",
        choices=("sentence-transformers", "hashing"),
        default="sentence-transformers",
    )

    search = subparsers.add_parser("search", help="run one of five retrieval modes")
    search.add_argument("query")
    search.add_argument(
        "--mode",
        choices=("exact", "ann", "filtered_ann", "text", "hybrid_rrf"),
        default="ann",
    )
    search.add_argument("--query-id", default="interactive")
    search.add_argument("--run-id", default="interactive")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--num-candidates", type=int, default=100)
    search.add_argument("--filters", default="{}", help="JSON object of metadata filters")
    search.add_argument(
        "--embedding-provider",
        choices=("sentence-transformers", "hashing"),
        default="sentence-transformers",
    )

    rag = subparsers.add_parser("rag", help="retrieve evidence and generate a cited answer")
    rag.add_argument("query")
    rag.add_argument("--query-id", default="interactive")
    rag.add_argument("--run-id", default="interactive")
    rag.add_argument("--top-k", type=int, default=5)
    rag.add_argument("--num-candidates", type=int, default=100)
    rag.add_argument("--generator", choices=("extractive", "ollama"), default="extractive")
    rag.add_argument("--ollama-model", default="llama3.2:3b")
    rag.add_argument(
        "--embedding-provider",
        choices=("sentence-transformers", "hashing"),
        default="sentence-transformers",
    )

    benchmark = subparsers.add_parser("benchmark", help="run and export the benchmark")
    benchmark.add_argument("--offline-validation", action="store_true")
    benchmark.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    benchmark.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    benchmark.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    benchmark.add_argument("--output-root", type=Path, default=Path("results/mongodb"))
    benchmark.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "benchmark" and args.offline_validation:
            run_id = args.run_id or _run_id("offline")
            run_dir = run_offline_validation(
                corpus_path=args.corpus,
                queries_path=args.queries,
                qrels_path=args.qrels,
                output_root=args.output_root / "offline_validation",
                run_id=run_id,
            )
            print(run_dir)
            return 0

        config = MongoConfig.from_env()
        client, collection = _connect(config)
        try:
            if args.command == "indexes":
                return _indexes(collection, args.wait)
            if args.command == "ingest":
                return _ingest(collection, config, args)
            if args.command == "search":
                return _search(collection, config, args)
            if args.command == "rag":
                return _rag(collection, config, args)
            if args.command == "benchmark":
                return _live_benchmark(collection, config, args)
        finally:
            client.close()
    except (MongoConfigError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 2


def _connect(config: MongoConfig) -> tuple[Any, Any]:
    from pymongo import MongoClient

    client: Any = MongoClient(config.uri, serverSelectionTimeoutMS=config.timeout_ms)
    client.admin.command("ping")
    return client, client[config.database][config.collection]


def _indexes(collection: Any, wait: bool) -> int:
    manager = IndexManager.from_files(
        collection,
        Path("configs/mongodb/vector-index.json"),
        Path("configs/mongodb/text-index.json"),
    )
    created = manager.ensure_indexes()
    payload: dict[str, Any] = {"created": created}
    if wait:
        payload["statuses"] = [status.__dict__ for status in manager.wait_until_ready()]
    print(json.dumps(payload, default=str, sort_keys=True))
    return 0


def _embedder(provider: str, dimensions: int) -> Any:
    if provider == "hashing":
        return HashingEmbedder(dimensions=dimensions)
    return SentenceTransformerEmbedder()


def _ingest(collection: Any, config: MongoConfig, args: argparse.Namespace) -> int:
    chunks = load_jsonl(args.corpus, Chunk)
    embedder = _embedder(args.embedding_provider, config.dimensions)
    embeddings = embedder.embed([f"{chunk.title} {chunk.text}" for chunk in chunks])
    run_id = args.run_id or _run_id("ingest")
    summary = MongoIngestor(
        collection, dimensions=config.dimensions, batch_size=args.batch_size
    ).ingest(chunks, embeddings, run_id=run_id)
    output = Path("results/mongodb/ingestion_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary.to_dict(), sort_keys=True))
    return 0 if summary.failed == 0 else 1


def _search(collection: Any, config: MongoConfig, args: argparse.Namespace) -> int:
    filters = json.loads(args.filters)
    if not isinstance(filters, dict):
        raise ValueError("--filters must be a JSON object")
    retriever = MongoRetriever(
        collection,
        vector_index=config.vector_index,
        text_index=config.text_index,
        dimensions=config.dimensions,
        run_id=args.run_id,
    )
    if args.mode == "text":
        results = retriever.search_text(args.query, query_id=args.query_id, top_k=args.top_k)
    else:
        query_vector = _embedder(args.embedding_provider, config.dimensions).embed([args.query])[0]
        vector_results = retriever.search_vector(
            query_vector,
            query_id=args.query_id,
            top_k=args.top_k,
            exact=args.mode == "exact",
            num_candidates=args.num_candidates,
            filters=filters if args.mode == "filtered_ann" else None,
            filter_name="cli_filters" if filters else "none",
        )
        if args.mode == "hybrid_rrf":
            text_results = retriever.search_text(
                args.query, query_id=args.query_id, top_k=args.top_k
            )
            results = reciprocal_rank_fusion(vector_results, text_results, top_k=args.top_k)
        else:
            results = vector_results
    for result in results:
        print(result.model_dump_json())
    return 0


def _rag(collection: Any, config: MongoConfig, args: argparse.Namespace) -> int:
    query_vector = _embedder(args.embedding_provider, config.dimensions).embed([args.query])[0]
    retriever = MongoRetriever(collection, dimensions=config.dimensions, run_id=args.run_id)
    vector_hits = retriever.search_vector(
        query_vector,
        query_id=args.query_id,
        top_k=args.top_k,
        num_candidates=args.num_candidates,
    )
    text_hits = retriever.search_text(args.query, query_id=args.query_id, top_k=args.top_k)
    hits = reciprocal_rank_fusion(vector_hits, text_hits, top_k=args.top_k)
    ids = [hit.chunk_id for hit in hits]
    documents = {str(item["_id"]): item for item in collection.find({"_id": {"$in": ids}})}
    chunks = [_document_to_chunk(documents[item]) for item in ids if item in documents]
    generator = (
        OllamaGenerator(model=args.ollama_model)
        if args.generator == "ollama"
        else ExtractiveGenerator()
    )
    query = EvaluationQuery(
        query_id=args.query_id,
        text=args.query,
        expected_answer="Interactive query; no gold answer supplied.",
    )
    retrieval_latency = max((hit.latency_ms for hit in hits), default=0.0)
    result = MongoRAGPipeline(generator, run_id=args.run_id).answer(
        query, chunks, retrieval_latency_ms=retrieval_latency
    )
    print(result.model_dump_json())
    return 0


def _document_to_chunk(document: dict[str, Any]) -> Chunk:
    fields = Chunk.model_fields
    return Chunk.model_validate({key: document[key] for key in fields})


def _live_benchmark(collection: Any, config: MongoConfig, args: argparse.Namespace) -> int:
    manager = IndexManager.from_files(
        collection,
        Path("configs/mongodb/vector-index.json"),
        Path("configs/mongodb/text-index.json"),
    )
    manager.ensure_indexes()
    manager.wait_until_ready()
    run_id = args.run_id or _run_id("atlas")
    run_dir = run_live_benchmark(
        collection,
        SentenceTransformerEmbedder(),
        corpus_path=args.corpus,
        queries_path=args.queries,
        qrels_path=args.qrels,
        output_root=args.output_root / "atlas",
        run_id=run_id,
        vector_index=config.vector_index,
        text_index=config.text_index,
        dimensions=config.dimensions,
    )
    print(run_dir)
    return 0


def _run_id(prefix: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
