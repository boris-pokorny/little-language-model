"""Provenance: who/what produced an artifact, embedded in checkpoints and metrics.

Collects the pipeline version (pyproject.toml), the git commit and whether the
tree was dirty, content hashes of the inputs that went in, library versions and
a timestamp. Tolerant of running outside git. Everything is plain JSON-able so
it can live inside model.pt, model.json and data/metrics/eval.json alike.
"""

import hashlib
import platform
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PYPROJECT_PATH = Path("pyproject.toml")


def pipeline_version() -> str:
    if not PYPROJECT_PATH.exists():
        return "unknown"
    return tomllib.loads(PYPROJECT_PATH.read_text())["project"]["version"]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(inputs: dict[str, Path] | None = None) -> dict[str, Any]:
    """Provenance record; `inputs` maps a label to a file whose sha256 is recorded."""
    return {
        "pipeline_version": pipeline_version(),
        "git": git_state(),
        "inputs": {name: sha256(p) for name, p in (inputs or {}).items() if p.exists()},
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
