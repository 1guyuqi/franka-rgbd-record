"""Repository root paths (no hard-coded machine paths)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
MODEL_DIR = ROOT / "model"
ASSETS_DIR = ROOT / "assets"
DEFAULT_URDF = MODEL_DIR / "fr3_franka_hand.urdf"


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()
