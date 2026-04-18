"""Rule-based baseline enemy controller ported from the original patrol idea."""

from __future__ import annotations

import random

from ..core.actions import DISCRETE_ACTIONS
from ..core.actions import normalize_vector


class RuleBasedEnemyController:
    def reset(self) -> None:
        return None

    def act(self, world) -> tuple[float, float]:
        enemy = world.enemy
        if enemy.freeze_timer > 0.0:
            return 0.0, 0.0

        if world.player_visible_to_enemy():
            target = world.player.position
            enemy.last_seen_player = target
            return normalize_vector((target[0] - enemy.x, target[1] - enemy.y))

        if enemy.last_seen_player is not None:
            target = enemy.last_seen_player
            dx = target[0] - enemy.x
            dy = target[1] - enemy.y
            if abs(dx) < 0.25 and abs(dy) < 0.25:
                enemy.last_seen_player = None
            else:
                return normalize_vector((dx, dy))

        if not world.enemy_waypoints:
            return 0.0, 0.0

        waypoint = world.enemy_waypoints[enemy.patrol_index]
        target = (waypoint[0] + 0.5, waypoint[1] + 0.5)
        dx = target[0] - enemy.x
        dy = target[1] - enemy.y
        if abs(dx) < 0.30 and abs(dy) < 0.30:
            enemy.patrol_index = (enemy.patrol_index + 1) % len(world.enemy_waypoints)
            waypoint = world.enemy_waypoints[enemy.patrol_index]
            target = (waypoint[0] + 0.5, waypoint[1] + 0.5)
            dx = target[0] - enemy.x
            dy = target[1] - enemy.y
        return normalize_vector((dx, dy))


class RandomEnemyController:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        return None

    def act(self, world) -> tuple[float, float]:
        action_id = self._rng.choice(list(DISCRETE_ACTIONS))
        direction = DISCRETE_ACTIONS[action_id]
        return normalize_vector(direction)
