"""Project-wide config loaders."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return (REPO_ROOT / raw_path).resolve()


def read_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if data is not None else {}


def read_json(path: str | Path) -> dict[str, Any]:
    with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=8)
def _cached_env_config(path: str) -> dict[str, Any]:
    return read_yaml(path)


@lru_cache(maxsize=8)
def _cached_rewards_config(path: str) -> dict[str, Any]:
    return read_yaml(path)


@lru_cache(maxsize=8)
def _cached_training_config(path: str) -> dict[str, Any]:
    return read_yaml(path)


@lru_cache(maxsize=8)
def _cached_map_config(path: str) -> dict[str, Any]:
    return read_json(path)


def load_env_config(path: str | Path = "configs/env.yaml") -> dict[str, Any]:
    return copy.deepcopy(_cached_env_config(str(path)))


def load_rewards_config(path: str | Path = "configs/rewards.yaml") -> dict[str, Any]:
    return copy.deepcopy(_cached_rewards_config(str(path)))


def load_training_config(path: str | Path = "configs/training.yaml") -> dict[str, Any]:
    return copy.deepcopy(_cached_training_config(str(path)))


def load_map_config(path: str | Path | None = None, env_config: dict[str, Any] | None = None) -> dict[str, Any]:
    if path is None:
        env_config = env_config or load_env_config()
        path = env_config["map_path"]
    return copy.deepcopy(_cached_map_config(str(path)))
