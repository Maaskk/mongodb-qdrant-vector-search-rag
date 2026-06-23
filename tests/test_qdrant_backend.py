import pytest
from vector_rag.qdrant_backend import QdrantVectorBackend
from vector_rag.contracts import Chunk
from datetime import datetime


@pytest.fixture
def backend():
    """Create fresh backend for each test."""
    return QdrantVectorBackend(storage_path="./qdrant_test")


def test_healthcheck(backend):
    """Test backend healthcheck."""
    assert backend.healthcheck() is True


def test_ingest_chunks(backend):
    """Test chunk ingestion."""
    chunks = [
        Chunk(
            chunk_id="test_1",
            document_id="doc_1",
            title="Test",
            text="This is test content",
            source="test",
            category="test",
            language="en",
            chunk_index=0,
            content_hash="a" * 64,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
    ]
    
    accepted = backend.ingest(chunks)
    assert accepted == 1


def test_search(backend):
    """Test search functionality."""
    from sentence_transformers import SentenceTransformer
    
    chunks = [
        Chunk(
            chunk_id="test_1",
            document_id="doc_1",
            title="Python",
            text="Python is a programming language",
            source="test",
            category="tech",
            language="en",
            chunk_index=0,
            content_hash="b" * 64,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
    ]
    backend.ingest(chunks)
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vector = model.encode("programming").tolist()
    
    results = backend.search(
        query_vector=query_vector,
        query_id="q_1",
        top_k=5
    )
    
    assert len(results) > 0
    assert results[0].score > 0
