from __future__ import annotations

from pathlib import Path


def load_text(path: Path, encoding: str = "utf-8") -> str:
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    return path.read_text(encoding=encoding, errors="replace")
