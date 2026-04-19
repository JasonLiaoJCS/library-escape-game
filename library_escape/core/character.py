"""Shared character state."""

from __future__ import annotations

import math
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

    def apply_action(
        self,
        action: tuple[float, float],
        speed_scale: float = 1.0,
        dt: float | None = None,
        max_turn_rate_deg: float | None = None,
    ) -> None:
        ax, ay = clamp_vector(action, max_length=1.0)
        self.velocity_x = ax * self.base_speed * speed_scale
        self.velocity_y = ay * self.base_speed * speed_scale
        if abs(ax) <= 1e-6 and abs(ay) <= 1e-6:
            return
        desired_fx, desired_fy = normalize_vector((ax, ay))
        if max_turn_rate_deg is None or max_turn_rate_deg <= 0.0 or dt is None or dt <= 0.0:
            self.facing_x, self.facing_y = desired_fx, desired_fy
            return
        current_fx, current_fy = self.facing_x, self.facing_y
        if abs(current_fx) <= 1e-6 and abs(current_fy) <= 1e-6:
            self.facing_x, self.facing_y = desired_fx, desired_fy
            return
        current_angle = math.atan2(current_fy, current_fx)
        desired_angle = math.atan2(desired_fy, desired_fx)
        diff = (desired_angle - current_angle + math.pi) % (2.0 * math.pi) - math.pi
        max_step = math.radians(max_turn_rate_deg) * dt
        if abs(diff) <= max_step:
            new_angle = desired_angle
        else:
            new_angle = current_angle + math.copysign(max_step, diff)
        self.facing_x = math.cos(new_angle)
        self.facing_y = math.sin(new_angle)


@dataclass(slots=True)
class PlayerState(CharacterState):
    coffee_timer: float = 0.0
