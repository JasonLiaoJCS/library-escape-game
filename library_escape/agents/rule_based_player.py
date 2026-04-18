"""Rule-based and random player controllers."""

from __future__ import annotations

import heapq
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


class HeuristicPlayerController:
    def __init__(self) -> None:
        self._last_goal: tuple[int, int] | None = None

    def reset(self) -> None:
        self._last_goal = None

    def act(self, world) -> tuple[float, float]:
        start = self._grid_position(world.player.position)
        goal = self._choose_goal(world, start)
        if goal is None:
            return 0.0, 0.0

        path = self._a_star(world, start, goal)
        if len(path) >= 2:
            next_cell = path[1]
        else:
            next_cell = goal

        target_x, target_y = _cell_center(next_cell)
        return normalize_vector((target_x - world.player.x, target_y - world.player.y))

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
        cost_so_far: dict[tuple[int, int], int] = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for neighbor in self._neighbors(world, current):
                new_cost = cost_so_far[current] + 1
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
