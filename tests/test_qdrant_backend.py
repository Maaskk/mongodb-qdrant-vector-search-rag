from vector_rag.qdrant_backend import QdrantVectorBackend
from vector_rag.qdrant_track.backend import QdrantVectorBackend as TrackBackend


def test_legacy_qdrant_import_path_points_to_track_backend() -> None:
    assert QdrantVectorBackend is TrackBackend
