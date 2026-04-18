from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


def vec2(x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([x, y], dtype=np.float32)


@dataclass(slots=True)
class Rectangle:
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def top(self) -> float:
        return self.y

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> np.ndarray:
        return vec2(self.x + self.w / 2.0, self.y + self.h / 2.0)

    def contains_point(self, point: np.ndarray) -> bool:
        return (
            self.left <= float(point[0]) <= self.right
            and self.top <= float(point[1]) <= self.bottom
        )

    def contains_circle(self, point: np.ndarray, radius: float) -> bool:
        return (
            self.left + radius <= float(point[0]) <= self.right - radius
            and self.top + radius <= float(point[1]) <= self.bottom - radius
        )


@dataclass(slots=True)
class ObstacleState:
    kind: str
    rect: Rectangle
    blocks_movement: bool = True
    blocks_sight: bool = True


@dataclass(slots=True)
class NoteState:
    note_id: str
    position: np.ndarray
    radius: float
    collected: bool = False


@dataclass(slots=True)
class ActorState:
    name: str
    role: Literal["player", "enemy"]
    position: np.ndarray
    velocity: np.ndarray
    facing: float
    radius: float
    max_speed: float
    view_range: float
    view_angle: float
    catch_radius: float = 0.0
    patrol_index: int = 0
    last_seen_player: np.ndarray | None = None
    last_seen_timer: float = 0.0
    wall_hits: int = 0
    score: float = 0.0


@dataclass(slots=True)
class StepMetrics:
    note_collected: int = 0
    player_visible_steps: int = 0
    player_wall_hits: int = 0
    enemy_wall_hits: int = 0
    distance_delta: float = 0.0
    caught: bool = False
    escaped: bool = False
    timeout: bool = False
    stalemate: bool = False


@dataclass(slots=True)
class WorldSnapshot:
    player_position: np.ndarray
    enemy_position: np.ndarray
    player_velocity: np.ndarray
    enemy_velocity: np.ndarray
    player_facing: float
    enemy_facing: float
    remaining_time: float
    notes_remaining: int
    total_notes: int
    enemy_sees_player: bool
    player_can_see_enemy: bool
    distance: float


@dataclass(slots=True)
class WorldStatus:
    terminated: bool = False
    truncated: bool = False
    outcome: str | None = None


@dataclass(slots=True)
class InputState:
    movement: np.ndarray = field(default_factory=vec2)
