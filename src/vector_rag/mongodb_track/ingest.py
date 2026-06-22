"""Compatibility import for the ingestion module named in the task contract."""

from vector_rag.mongodb_track.ingestion import MongoIngestor

__all__ = ["MongoIngestor"]
