import os
from uuid import uuid4

import pytest

from vector_rag.contracts import Chunk
from vector_rag.mongodb_track.config import MongoConfig
from vector_rag.mongodb_track.embeddings import HashingEmbedder
from vector_rag.mongodb_track.ingestion import MongoIngestor

pymongo = pytest.importorskip("pymongo")


@pytest.mark.skipif(
    not os.getenv("MONGODB_URI"), reason="MONGODB_URI not set; Atlas smoke test skipped"
)
def test_atlas_connection_and_idempotent_upsert() -> None:
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
    finally:
        collection.drop()
        client.close()
