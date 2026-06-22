from pathlib import Path

import yaml


def test_ci_runs_quality_commands() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    rendered = str(workflow)
    assert "ruff check" in rendered
    assert "mypy" in rendered
    assert "pytest" in rendered


def test_issue_form_collects_owner_and_acceptance_criteria() -> None:
    form = yaml.safe_load(Path(".github/ISSUE_TEMPLATE/task.yml").read_text(encoding="utf-8"))
    ids = {item.get("id") for item in form["body"]}
    assert {"owner", "acceptance", "verification"} <= ids
