import pytest
from pydantic import ValidationError

from vector_rag.contracts import Chunk, RetrievalResult


def valid_chunk_data() -> dict[str, object]:
    return {
        "chunk_id": "doc-1:0",
        "document_id": "doc-1",
        "title": "Example",
        "text": "Vector search compares embedding similarity.",
        "source": "demo",
        "category": "database",
        "language": "en",
        "chunk_index": 0,
        "content_hash": "a" * 64,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


def test_chunk_accepts_valid_data() -> None:
    chunk = Chunk(**valid_chunk_data())
    assert chunk.chunk_id == "doc-1:0"


def test_chunk_rejects_empty_text() -> None:
    data = valid_chunk_data()
    data["text"] = ""
    with pytest.raises(ValidationError):
        Chunk(**data)


def test_chunk_rejects_invalid_content_hash() -> None:
    data = valid_chunk_data()
    data["content_hash"] = "short"
    with pytest.raises(ValidationError):
        Chunk(**data)


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

