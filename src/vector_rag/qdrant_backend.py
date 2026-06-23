import contextlib
import time
from collections.abc import Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from vector_rag.backend import VectorBackend
from vector_rag.contracts import Chunk, RetrievalResult


class QdrantVectorBackend(VectorBackend):
    """Qdrant implementation of the VectorBackend protocol."""

    name = "qdrant"

    def __init__(self, collection_name: str = "rag_corpus", storage_path: str = "./qdrant_storage"):
        self.client = QdrantClient(path=storage_path)
        self.collection_name = collection_name
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = 384
        self._init_collection()

    def _init_collection(self) -> None:
        """Create or reset the collection."""
        with contextlib.suppress(Exception):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            ),
        )

    def healthcheck(self) -> bool:
        """Check if Qdrant is ready."""
        try:
            self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def ingest(self, chunks: Sequence[Chunk]) -> int:
        """Insert chunks into Qdrant and return count accepted."""
        accepted = 0
        
        for chunk in chunks:
            try:
                vector = self.model.encode(chunk.text).tolist()
                point = PointStruct(
                    id=hash(chunk.chunk_id) % (2**31),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "title": chunk.title,
                        "text": chunk.text,
                        "source": chunk.source,
                        "category": chunk.category,
                    }
                )
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )
                accepted += 1
            except Exception as e:
                print(f"Error ingesting chunk {chunk.chunk_id}: {e}")
                continue
        
        return accepted

    def search(
        self,
        query_vector: Sequence[float],
        *,
        query_id: str,
        top_k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Search and return ranked RetrievalResult objects."""
        start_time = time.time()
        
        # Use query_points method for search
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            limit=top_k,
            with_payload=True,
        ).points
        
        latency_ms = (time.time() - start_time) * 1000
        
        results = []
        for rank, result in enumerate(search_results, start=1):
            results.append(
                RetrievalResult(
                    backend=self.name,
                    query_id=query_id,
                    rank=rank,
                    chunk_id=result.payload.get("chunk_id", "unknown"),
                    score=float(result.score),
                    latency_ms=latency_ms,
                    search_mode="dense",
                    filter_name="none",
                    run_id="benchmark_run_001"
                )
            )
        
        return results
