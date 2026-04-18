from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from library_escape.paths import CONFIG_DIR, resolve_path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data)!r}")
    return data


@lru_cache(maxsize=None)
def load_env_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_yaml(resolve_path(path or (CONFIG_DIR / "env.yaml")))


@lru_cache(maxsize=None)
def load_reward_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_yaml(resolve_path(path or (CONFIG_DIR / "rewards.yaml")))


@lru_cache(maxsize=None)
def load_training_config(path: str | Path | None = None) -> dict[str, Any]:
    return _load_yaml(resolve_path(path or (CONFIG_DIR / "training.yaml")))
