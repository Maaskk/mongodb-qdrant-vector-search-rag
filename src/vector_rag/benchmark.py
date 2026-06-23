"""Compatibility wrapper for the Qdrant benchmark backend.

The real implementation lives in `vector_rag.qdrant_track.backend` so Hamza's
track has the same clear structure as Ossama's MongoDB track.
"""

from vector_rag.qdrant_track.backend import QdrantVectorBackend

__all__ = ["QdrantVectorBackend"]
