from __future__ import annotations

from typing import Any

import numpy as np

from library_escape.core.constants import DISCRETE_ACTION_VECTORS
from library_escape.core.entities import vec2


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vec2()
    return (vector / norm).astype(np.float32)


def action_to_direction(action: Any, scheme: str) -> np.ndarray:
    if scheme == "continuous":
        array = np.asarray(action, dtype=np.float32).reshape(2)
        array = np.clip(array, -1.0, 1.0)
        return normalize_vector(array) if np.linalg.norm(array) > 1.0 else array

    index = int(action)
    if index < 0 or index >= len(DISCRETE_ACTION_VECTORS):
        raise ValueError(f"Action index {index} out of range for discrete action space")
    return np.asarray(DISCRETE_ACTION_VECTORS[index], dtype=np.float32)
