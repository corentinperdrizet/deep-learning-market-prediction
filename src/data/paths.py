# -----------------------------
# File: src/data/paths.py
# -----------------------------
from __future__ import annotations
from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Return project root by searching upwards for a marker (e.g., .git or pyproject.toml).
    Falls back to current working directory if not found.
    """
    start = start or Path.cwd()
    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return start


def data_dir() -> Path:
    root = project_root()
    d = root / "data"
    d.mkdir(exist_ok=True)
    (d / "raw").mkdir(exist_ok=True)
    (d / "processed").mkdir(exist_ok=True)
    (d / "artifacts").mkdir(exist_ok=True)
    return d


