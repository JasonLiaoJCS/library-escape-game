from __future__ import annotations

import math
from typing import Any

import numpy as np

from library_escape.core.actions import normalize_vector
from library_escape.core.navigation import astar_path, cell_center, position_to_cell
from library_escape.core.physics import length
from library_escape.core.world import LibraryWorld


def _direction_to_action(direction: np.ndarray, scheme: str) -> Any:
    if scheme == "continuous":
        return direction.astype(np.float32)

    dx = int(np.sign(direction[0]))
    dy = int(np.sign(direction[1]))
    mapping = {
        (0, 0): 0,
        (0, -1): 1,
        (1, -1): 2,
        (1, 0): 3,
        (1, 1): 4,
        (0, 1): 5,
        (-1, 1): 6,
        (-1, 0): 7,
        (-1, -1): 8,
    }
    return mapping[(dx, dy)]


class BaseController:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.action_scheme = config["actions"]["scheme"]

    def reset(self) -> None:
        return None


class RandomController(BaseController):
    def __init__(self, config: dict[str, Any], hold_steps: int = 18, seed: int | None = None) -> None:
        super().__init__(config)
        self.hold_steps = hold_steps
        self.rng = np.random.default_rng(seed)
        self.current_action: Any = 0
        self.remaining = 0

    def reset(self) -> None:
        self.remaining = 0
        self.current_action = 0

    def act(self, world: LibraryWorld) -> Any:
        if self.remaining <= 0:
            if self.action_scheme == "continuous":
                direction = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
                self.current_action = normalize_vector(direction)
            else:
                self.current_action = int(self.rng.integers(0, 9))
            self.remaining = self.hold_steps
        self.remaining -= 1
        return self.current_action


class MixtureController(BaseController):
    def __init__(self, config: dict[str, Any], weighted_controllers: list[tuple[float, BaseController]], seed: int | None = None) -> None:
        super().__init__(config)
        self.weighted_controllers = weighted_controllers
        self.rng = np.random.default_rng(seed)
        self.active_controller: BaseController | None = None

    def reset(self) -> None:
        if not self.weighted_controllers:
            raise RuntimeError("MixtureController requires at least one child controller")
        weights = np.asarray([max(weight, 0.0) for weight, _ in self.weighted_controllers], dtype=np.float64)
        if weights.sum() <= 0.0:
            weights = np.ones_like(weights)
        probabilities = weights / weights.sum()
        index = int(self.rng.choice(len(self.weighted_controllers), p=probabilities))
        self.active_controller = self.weighted_controllers[index][1]
        if hasattr(self.active_controller, "reset"):
            self.active_controller.reset()

    def act(self, world: LibraryWorld) -> Any:
        if self.active_controller is None:
            self.reset()
        assert self.active_controller is not None
        return self.active_controller.act(world)


class _PathController(BaseController):
    def _path_direction(self, world: LibraryWorld, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        start = position_to_cell(source, world.grid_size)
        goal = position_to_cell(target, world.grid_size)
        path = astar_path(start, goal, world.walkable_grid)
        if len(path) >= 2:
            waypoint = cell_center(path[1], world.grid_size)
        else:
            waypoint = target
        return normalize_vector(waypoint - source)


class RuleBasedEnemyController(_PathController):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        patrol_cfg = config["world"]["enemy"].get("patrol_points", [])
        self.patrol_points = [np.asarray(point, dtype=np.float32) for point in patrol_cfg]
        self.patrol_index = 0

    def reset(self) -> None:
        self.patrol_index = 0

    def act(self, world: LibraryWorld) -> Any:
        enemy = world.enemy
        player = world.player
        target: np.ndarray

        if world.enemy_can_see_player():
            target = player.position
        elif enemy.last_seen_player is not None and enemy.last_seen_timer <= self.config["observations"]["memory_seconds"]:
            target = enemy.last_seen_player
        elif self.patrol_points:
            target = self.patrol_points[self.patrol_index]
            if length(target - enemy.position) <= world.grid_size * 0.35:
                self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
                target = self.patrol_points[self.patrol_index]
        else:
            target = player.position

        direction = self._path_direction(world, enemy.position, target)
        return _direction_to_action(direction, self.action_scheme)


class RuleBasedPlayerController(_PathController):
    def __init__(self, config: dict[str, Any], flee_weight: float = 1.35) -> None:
        super().__init__(config)
        self.flee_weight = flee_weight

    def act(self, world: LibraryWorld) -> Any:
        player = world.player
        enemy = world.enemy
        if world.notes_remaining > 0:
            notes = [note for note in world.notes if not note.collected]
            target = min(notes, key=lambda note: length(note.position - player.position)).position
        else:
            target = world.exit_zone.center

        direction = self._path_direction(world, player.position, target)
        enemy_vector = player.position - enemy.position
        if world.enemy_can_see_player() and length(enemy_vector) <= 220.0:
            direction = normalize_vector(direction + normalize_vector(enemy_vector) * self.flee_weight)
        return _direction_to_action(direction, self.action_scheme)
