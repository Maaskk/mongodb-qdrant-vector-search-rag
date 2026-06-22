from pathlib import Path

import pytest

from vector_rag.contracts import Chunk
from vector_rag.io import load_jsonl


def test_invalid_jsonl_reports_filename_and_line(tmp_path: Path) -> None:
    fixture = tmp_path / "broken.jsonl"
    fixture.write_text("{}\nnot-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"broken\.jsonl:1: invalid record"):
        load_jsonl(fixture, Chunk)
