from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def test_frontend_entrypoint_and_local_assets_exist() -> None:
    index = DOCS / "index.html"

    assert index.exists()
    html = index.read_text(encoding="utf-8")

    for asset in re.findall(r'(?:href|src)="([^":]+)"', html):
        if asset.startswith("#") or asset.startswith("mailto:"):
            continue
        assert (DOCS / asset).exists(), asset


def test_frontend_copy_names_both_tracks_and_avoids_placeholders() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    css = (DOCS / "assets" / "styles.css").read_text(encoding="utf-8")
    js = (DOCS / "assets" / "app.js").read_text(encoding="utf-8")
    combined = f"{html}\n{css}\n{js}".lower()

    assert "mongodb vector search" in combined
    assert "qdrant" in combined
    assert "rag" in combined
    assert "todo" not in combined
    assert "lorem ipsum" not in combined
    assert "placeholder" not in combined


def test_frontend_does_not_depend_on_external_cdns() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")

    assert "https://cdn" not in html
    assert "unpkg.com" not in html
    assert "tailwind" not in html.lower()
