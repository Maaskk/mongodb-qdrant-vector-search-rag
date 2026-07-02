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


def test_frontend_uses_light_gnn_reference_style_not_dark_pdf_cover() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    css = (DOCS / "assets" / "styles.css").read_text(encoding="utf-8")

    assert 'lang="fr"' in html
    assert "site-header" in html
    assert "brand-mark" in html
    assert "heroCanvas" in html
    assert "spaceFieldCanvas" in html
    assert "stat-band" in html
    assert "dashboard-grid" in html
    assert "segmented-control" in html
    assert "color-scheme: light" in css
    assert "--accent-red" in css
    assert "--accent-teal" in css
    assert "#ffffff" in css.lower()
    assert "color-scheme: dark" not in css
    assert "#11100e" not in css.lower()
    assert "Georgia" not in css


def test_frontend_has_interactive_canvas_and_chart_code() -> None:
    js = (DOCS / "assets" / "app.js").read_text(encoding="utf-8")

    assert "startGlobalSpaceField" in js
    assert "drawHeroNetwork" in js
    assert "drawQualityChart" in js
    assert "drawLatencyChart" in js
    assert "pointermove" in js
    assert "--mouse-x" in js
    assert "--mouse-y" in js


def test_frontend_contains_real_demo_functionalities() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    js = (DOCS / "assets" / "app.js").read_text(encoding="utf-8")

    assert "semanticSearchSection" in html
    assert "semanticQueryInput" in html
    assert "categoryFilter" in html
    assert "runSemanticSearch" in html
    assert "mongodbResults" in html
    assert "qdrantResults" in html
    assert "ragGeneratedAnswer" in html
    assert "decisionComparator" in html
    assert "semanticSearch" in js
    assert "searchCorpus" in js
    assert "renderSearchResults" in js
    assert "generateGroundedAnswer" in js
    assert "updateDecisionComparator" in js


def test_frontend_does_not_depend_on_external_cdns() -> None:
    html = (DOCS / "index.html").read_text(encoding="utf-8")

    assert "https://cdn" not in html
    assert "unpkg.com" not in html
    assert "tailwind" not in html.lower()
