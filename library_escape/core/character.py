"""Shared character state."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import clamp_vector, normalize_vector


@dataclass(slots=True)
class CharacterState:
    name: str
    x: float
    y: float
    radius: float
    base_speed: float
    facing_x: float = 0.0
    facing_y: float = 1.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def apply_action(self, action: tuple[float, float], speed_scale: float = 1.0) -> None:
        ax, ay = clamp_vector(action, max_length=1.0)
        self.velocity_x = ax * self.base_speed * speed_scale
        self.velocity_y = ay * self.base_speed * speed_scale
        if abs(ax) > 1e-6 or abs(ay) > 1e-6:
            self.facing_x, self.facing_y = normalize_vector((ax, ay))


@dataclass(slots=True)
class PlayerState(CharacterState):
    coffee_timer: float = 0.0
