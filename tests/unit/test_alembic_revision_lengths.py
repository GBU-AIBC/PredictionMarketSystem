from __future__ import annotations

from pathlib import Path
import ast


def test_alembic_revision_ids_fit_version_column() -> None:
    versions_dir = Path(__file__).parents[2] / "alembic" / "versions"
    for path in versions_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        revision = next(
            node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "revision" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        assert len(revision) <= 32, (path.name, revision)
