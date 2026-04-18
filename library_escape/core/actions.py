"""Action helpers shared by play mode and RL environments."""

from __future__ import annotations

from typing import Iterable


DISCRETE_ACTIONS: dict[int, tuple[float, float]] = {
    0: (0.0, 0.0),
    1: (0.0, -1.0),
    2: (1.0, -1.0),
    3: (1.0, 0.0),
    4: (1.0, 1.0),
    5: (0.0, 1.0),
    6: (-1.0, 1.0),
    7: (-1.0, 0.0),
    8: (-1.0, -1.0),
}


def vector_length(vector: Iterable[float]) -> float:
    x, y = vector
    return float((x * x + y * y) ** 0.5)


def normalize_vector(vector: Iterable[float]) -> tuple[float, float]:
    x, y = vector
    length = vector_length((x, y))
    if length <= 1e-8:
        return 0.0, 0.0
    return float(x / length), float(y / length)


def clamp_vector(vector: Iterable[float], max_length: float = 1.0) -> tuple[float, float]:
    x, y = vector
    length = vector_length((x, y))
    if length <= max_length or length <= 1e-8:
        return float(x), float(y)
    scale = max_length / length
    return float(x * scale), float(y * scale)


def discrete_to_vector(action: int) -> tuple[float, float]:
    try:
        direction = DISCRETE_ACTIONS[int(action)]
    except KeyError as exc:
        raise ValueError(f"Unsupported discrete action: {action}") from exc
    return normalize_vector(direction)


def action_to_vector(action: int | float | Iterable[float], action_config: dict[str, object]) -> tuple[float, float]:
    action_type = str(action_config["type"]).lower()
    if action_type == "discrete":
        return discrete_to_vector(int(action))
    if action_type == "continuous":
        try:
            x, y = action  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Continuous action must be iterable with 2 values, got {action!r}") from exc
        return clamp_vector((float(x), float(y)), max_length=float(action_config.get("continuous_scale", 1.0)))
    raise ValueError(f"Unsupported action type: {action_type}")
