from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = REPO_ROOT / "configs"
IMG_DIR = REPO_ROOT / "imgs"
FONT_DIR = REPO_ROOT / "fonts"
MODEL_DIR = REPO_ROOT / "models"


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()
