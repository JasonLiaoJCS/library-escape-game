"""Enemy state and vision cone utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .character import CharacterState
from .obstacle import Obstacle
from .physics import has_line_of_sight


@dataclass(slots=True)
class VisionCone:
    range_cells: float
    angle_deg: float

    def sees(
        self,
        owner_position: tuple[float, float],
        facing: tuple[float, float],
        target_position: tuple[float, float],
        obstacles: list[Obstacle],
    ) -> bool:
        ox, oy = owner_position
        tx, ty = target_position
        dx = tx - ox
        dy = ty - oy
        distance = math.hypot(dx, dy)
        if distance > self.range_cells:
            return False

        if distance <= 1e-8:
            return True

        facing_x, facing_y = facing
        dot = (facing_x * dx + facing_y * dy) / distance
        dot = max(-1.0, min(1.0, dot))
        angle_to_target = math.degrees(math.acos(dot))
        if angle_to_target > (self.angle_deg / 2.0):
            return False

        return has_line_of_sight(owner_position, target_position, obstacles)


@dataclass(slots=True)
class EnemyState(CharacterState):
    vision: VisionCone = field(default_factory=lambda: VisionCone(range_cells=6.0, angle_deg=70.0))
    patrol_index: int = 0
    freeze_timer: float = 0.0
    last_seen_player: tuple[float, float] | None = None
