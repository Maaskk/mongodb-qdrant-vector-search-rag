from pathlib import Path
import hashlib

from vector_rag.contracts import Chunk, EvaluationQuery
from vector_rag.io import load_jsonl


def test_demo_fixtures_are_valid_and_linked() -> None:
    chunks = load_jsonl(Path("data/demo/corpus.jsonl"), Chunk)
    queries = load_jsonl(Path("data/evaluation/queries.jsonl"), EvaluationQuery)
    chunk_ids = {item.chunk_id for item in chunks}
    query_ids = {item.query_id for item in queries}
    qrel_lines = Path("data/evaluation/qrels.tsv").read_text(encoding="utf-8").splitlines()[1:]

    assert len(chunks) >= 6
    assert len(queries) >= 4
    assert all(hashlib.sha256(item.text.encode()).hexdigest() == item.content_hash for item in chunks)
    assert all(line.split("\t")[0] in query_ids for line in qrel_lines)
    assert all(line.split("\t")[1] in chunk_ids for line in qrel_lines)
