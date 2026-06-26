"""Small filesystem and JSON helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def reset_dir(path: str | Path) -> Path:
    """Remove and recreate a directory."""

    out = Path(path)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON with stable formatting."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
