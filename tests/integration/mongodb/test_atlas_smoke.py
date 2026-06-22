import os
from pathlib import Path
from uuid import uuid4

import pytest

from vector_rag.contracts import Chunk
from vector_rag.mongodb_track.config import MongoConfig
from vector_rag.mongodb_track.embeddings import HashingEmbedder
from vector_rag.mongodb_track.hybrid import reciprocal_rank_fusion
from vector_rag.mongodb_track.indexes import IndexManager
from vector_rag.mongodb_track.ingestion import MongoIngestor
from vector_rag.mongodb_track.search import MongoRetriever

pymongo = pytest.importorskip("pymongo")


@pytest.mark.skipif(
    not os.getenv("MONGODB_URI"), reason="MONGODB_URI not set; Atlas smoke test skipped"
)
def test_atlas_ingestion_and_all_retrieval_families() -> None:
    config = MongoConfig.from_env()
    client = pymongo.MongoClient(config.uri, serverSelectionTimeoutMS=config.timeout_ms)
    collection_name = f"codex_smoke_{uuid4().hex}"
    collection = client[config.database][collection_name]
    chunk = Chunk(
        chunk_id="smoke:0",
        document_id="smoke",
        title="Smoke test",
        text="MongoDB Atlas accepts an embedding document.",
        source="integration-test",
        category="mongodb",
        language="en",
        chunk_index=0,
        content_hash="a" * 64,
        embedding_model="deterministic-feature-hashing-v1",
    )
    vector = HashingEmbedder().embed([chunk.text])[0]
    try:
        client.admin.command("ping")
        first = MongoIngestor(collection).ingest([chunk], [vector], run_id="smoke-1")
        second = MongoIngestor(collection).ingest([chunk], [vector], run_id="smoke-2")
        assert first.upserted == 1
        assert second.skipped == 1
        assert collection.count_documents({"_id": "smoke:0"}) == 1

        manager = IndexManager.from_files(
            collection,
            Path("configs/mongodb/vector-index.json"),
            Path("configs/mongodb/text-index.json"),
        )
        manager.ensure_indexes()
        manager.wait_until_ready(timeout_seconds=600, poll_interval_seconds=5)
        retriever = MongoRetriever(collection, run_id="atlas-smoke")
        exact = retriever.search_vector(
            vector, query_id="smoke", top_k=1, exact=True
        )
        ann = retriever.search_vector(
            vector, query_id="smoke", top_k=1, num_candidates=10
        )
        text = retriever.search_text(
            "MongoDB Atlas embedding", query_id="smoke", top_k=1
        )
        hybrid = reciprocal_rank_fusion(ann, text, top_k=1)
        assert exact[0].chunk_id == chunk.chunk_id
        assert ann[0].chunk_id == chunk.chunk_id
        assert text[0].chunk_id == chunk.chunk_id
        assert hybrid[0].chunk_id == chunk.chunk_id
    finally:
        collection.drop()
        client.close()
