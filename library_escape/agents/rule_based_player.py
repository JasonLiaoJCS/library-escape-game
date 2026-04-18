"""Rule-based and random player controllers."""

from __future__ import annotations

import heapq
import math
import random
from typing import Iterable

from ..core.actions import DISCRETE_ACTIONS, normalize_vector


def _cell_center(cell: tuple[int, int]) -> tuple[float, float]:
    return cell[0] + 0.5, cell[1] + 0.5


class RandomPlayerController:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        return None

    def act(self, world) -> tuple[float, float]:
        action_id = self._rng.choice(list(DISCRETE_ACTIONS))
        action = DISCRETE_ACTIONS[action_id]
        return normalize_vector(action)

    def collect_pressed(self, world) -> bool:
        return False


class MixedPlayerController:
    def __init__(self, controllers: list[object], seed: int | None = None) -> None:
        self._controllers = controllers
        self._rng = random.Random(seed)
        self._current = controllers[0] if controllers else RandomPlayerController(seed=seed)

    def reset(self) -> None:
        if not self._controllers:
            return
        self._current = self._rng.choice(self._controllers)
        if hasattr(self._current, "reset"):
            self._current.reset()

    def act(self, world) -> tuple[float, float]:
        return self._current.act(world)

    def collect_pressed(self, world) -> bool:
        if hasattr(self._current, "collect_pressed"):
            return bool(self._current.collect_pressed(world))
        return False


class HeuristicPlayerController:
    def __init__(self) -> None:
        self._last_goal: tuple[int, int] | None = None

    def reset(self) -> None:
        self._last_goal = None

    def act(self, world) -> tuple[float, float]:
        if self.collect_pressed(world):
            return 0.0, 0.0
        if hasattr(world, "nearest_interactable_collectible") and world.nearest_interactable_collectible() is not None:
            return 0.0, 0.0

        start = self._grid_position(world.player.position)
        goal = self._choose_goal(world, start)
        if goal is None:
            return 0.0, 0.0
        evade_goal = self._evade_goal(world, start, fallback_goal=goal)
        if evade_goal is not None:
            goal = evade_goal

        path = self._a_star(world, start, goal)
        if len(path) >= 2:
            next_cell = path[1]
        else:
            next_cell = goal

        target_x, target_y = _cell_center(next_cell)
        return normalize_vector((target_x - world.player.x, target_y - world.player.y))

    def collect_pressed(self, world) -> bool:
        if not getattr(world, "manual_collect_required", False):
            return False
        if hasattr(world, "nearest_interactable_collectible"):
            return world.nearest_interactable_collectible() is not None
        nearest = world.nearest_collectible(world.player.position)
        if nearest is None:
            return False
        return nearest.distance_to(world.player.position) <= float(world.interaction_radius)

    def _evade_goal(self, world, start: tuple[int, int], fallback_goal: tuple[int, int]) -> tuple[int, int] | None:
        if getattr(world, "is_collection_mode", lambda: False)():
            return None
        visible_enemies = world.visible_enemies() if hasattr(world, "visible_enemies") else []
        if not visible_enemies:
            return None

        best_cell: tuple[int, int] | None = None
        best_score = float("-inf")
        radius = 5 if world.can_player_escape() else 4
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                candidate = (start[0] + dx, start[1] + dy)
                if not self._is_free(world, candidate):
                    continue
                score = self._cover_score(world, candidate, fallback_goal, visible_enemies)
                if score > best_score:
                    best_score = score
                    best_cell = candidate

        if best_cell is None or best_score < 0.5:
            return None
        return best_cell

    def _choose_goal(self, world, start: tuple[int, int]) -> tuple[int, int] | None:
        if world.can_player_escape():
            escape_cells = self._escape_cells(world)
            return min(escape_cells, key=lambda cell: self._manhattan(start, cell))

        note_goal = self._nearest_collectible_goal(world, start, kinds=("note",))
        if note_goal is not None:
            return note_goal

        exam_goal = self._nearest_collectible_goal(world, start, kinds=("exam",))
        if exam_goal is not None:
            return exam_goal

        power_goal = self._nearest_collectible_goal(world, start, kinds=("coffee", "freeze"))
        if power_goal is not None:
            return power_goal

        return None

    def _nearest_collectible_goal(self, world, start: tuple[int, int], kinds: Iterable[str]) -> tuple[int, int] | None:
        items = [item for item in world.active_collectibles() if item.kind in kinds]
        items.sort(key=lambda item: item.distance_to(world.player.position))
        for item in items:
            if hasattr(world, "interaction_cells_for_collectible"):
                candidates = list(world.interaction_cells_for_collectible(item))
            else:
                target_cell = (int(item.x), int(item.y))
                candidates = [cell for cell in self._adjacent_cells(target_cell) if self._is_free(world, cell)]
            if not candidates:
                continue
            return min(candidates, key=lambda cell: self._manhattan(start, cell))
        return None

    def _escape_cells(self, world) -> list[tuple[int, int]]:
        zone = world.escape_zone
        cells: list[tuple[int, int]] = []
        for gx in range(int(zone["x"]), int(zone["x"] + zone["w"])):
            for gy in range(int(zone["y"]), int(zone["y"] + zone["h"])):
                cell = (gx, gy)
                if self._is_free(world, cell):
                    cells.append(cell)
        return cells or [(int(zone["x"]), int(zone["y"]))]

    def _a_star(self, world, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        frontier: list[tuple[int, tuple[int, int]]] = []
        heapq.heappush(frontier, (0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for neighbor in self._neighbors(world, current):
                new_cost = cost_so_far[current] + 1.0 + self._cell_risk(world, neighbor, goal)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._manhattan(goal, neighbor)
                    heapq.heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal not in came_from:
            return [start]

        path = [goal]
        current = goal
        while current != start:
            current = came_from[current]
            if current is None:
                break
            path.append(current)
        path.reverse()
        return path

    def _neighbors(self, world, cell: tuple[int, int]) -> list[tuple[int, int]]:
        candidates = [
            (cell[0] + 1, cell[1]),
            (cell[0] - 1, cell[1]),
            (cell[0], cell[1] + 1),
            (cell[0], cell[1] - 1),
        ]
        return [candidate for candidate in candidates if self._is_free(world, candidate)]

    def _adjacent_cells(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        return [
            (cell[0] + 1, cell[1]),
            (cell[0] - 1, cell[1]),
            (cell[0], cell[1] + 1),
            (cell[0], cell[1] - 1),
        ]

    def _cell_risk(self, world, cell: tuple[int, int], goal: tuple[int, int]) -> float:
        cx, cy = _cell_center(cell)
        risk = 0.0
        goal_bias = 0.0 if cell == goal else 1.0
        for enemy in world.all_enemies():
            distance = math.hypot(cx - enemy.x, cy - enemy.y)
            if distance <= max(world.capture_radius * 1.8, 1.2):
                risk += 7.5 if enemy.is_primary else 5.0
                continue

            visible = enemy.vision.sees(
                owner_position=enemy.position,
                facing=(enemy.facing_x, enemy.facing_y),
                target_position=(cx, cy),
                obstacles=world.obstacles,
            )
            if visible:
                risk += (3.8 if enemy.is_primary else 2.4) * goal_bias

            vision_margin = enemy.vision.range_cells + 1.5
            if distance < vision_margin:
                normalized = 1.0 - (distance / vision_margin)
                risk += normalized * (1.25 if enemy.is_primary else 0.85)

            if enemy.last_seen_player is not None:
                sx, sy = enemy.last_seen_player
                support_bias = max(0.0, 1.0 - (math.hypot(cx - sx, cy - sy) / 3.5))
                risk += support_bias * (0.8 if enemy.is_primary else 0.5)

        if world.can_player_escape():
            escape_distance = world.distance_player_to_escape()
            if escape_distance > 0.0:
                risk *= 0.92
        return risk

    def _cover_score(
        self,
        world,
        cell: tuple[int, int],
        fallback_goal: tuple[int, int],
        visible_enemies: list,
    ) -> float:
        cx, cy = _cell_center(cell)
        safe_from_visible = 1.0
        min_distance = float("inf")
        for enemy in world.all_enemies():
            distance = math.hypot(cx - enemy.x, cy - enemy.y)
            min_distance = min(min_distance, distance)
            if enemy in visible_enemies and enemy.vision.sees(
                owner_position=enemy.position,
                facing=(enemy.facing_x, enemy.facing_y),
                target_position=(cx, cy),
                obstacles=world.obstacles,
            ):
                safe_from_visible = 0.0

        distance_term = min(1.0, min_distance / max(1.0, world.max_map_distance() * 0.35))
        progress_term = 1.0 - min(1.0, self._manhattan(cell, fallback_goal) / 8.0)
        return safe_from_visible * 4.5 + distance_term * 2.2 + progress_term * 0.9

    def _is_free(self, world, cell: tuple[int, int]) -> bool:
        gx, gy = cell
        if gx < 0 or gy < 0 or gx >= int(world.width) or gy >= int(world.height):
            return False
        cx, cy = _cell_center(cell)
        return not any(obstacle.contains_point(cx, cy) for obstacle in world.obstacles)

    def _grid_position(self, position: tuple[float, float]) -> tuple[int, int]:
        return int(position[0]), int(position[1])

    def _manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
