from pathlib import Path


def test_task_split_names_real_branches_and_backends() -> None:
    tasks = Path("TASKS.md").read_text(encoding="utf-8")
    assert "`ossama`" in tasks
    assert "`HamzaElhaddaji`" in tasks
    assert "MongoDB" in tasks
    assert "Qdrant" in tasks
    assert "Shared output contract" in tasks


def test_collaboration_guide_enforces_independent_work() -> None:
    guide = Path("COLLABORATION.md").read_text(encoding="utf-8")
    assert "without waiting" in guide
    assert "pull request" in guide.lower()
