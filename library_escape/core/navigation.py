from __future__ import annotations

from collections import deque
import heapq

import numpy as np

from library_escape.core.entities import ObstacleState, Rectangle, vec2
from library_escape.core.physics import circle_rect_collision


GridCell = tuple[int, int]


def cell_center(cell: GridCell, grid_size: float) -> np.ndarray:
    return vec2((cell[0] + 0.5) * grid_size, (cell[1] + 0.5) * grid_size)


def position_to_cell(position: np.ndarray, grid_size: float) -> GridCell:
    return int(position[0] // grid_size), int(position[1] // grid_size)


def build_walkable_grid(
    width: float,
    height: float,
    grid_size: float,
    obstacles: list[ObstacleState],
    radius: float,
) -> list[list[bool]]:
    cols = int(width // grid_size)
    rows = int(height // grid_size)
    walkable = [[True for _ in range(rows)] for _ in range(cols)]
    for x in range(cols):
        for y in range(rows):
            center = cell_center((x, y), grid_size)
            if (
                center[0] - radius <= 0
                or center[0] + radius >= width
                or center[1] - radius <= 0
                or center[1] + radius >= height
            ):
                walkable[x][y] = False
                continue
            for obstacle in obstacles:
                if obstacle.blocks_movement and circle_rect_collision(center, radius, obstacle.rect):
                    walkable[x][y] = False
                    break
    return walkable


def nearest_walkable_cell(start: GridCell, walkable: list[list[bool]]) -> GridCell:
    cols = len(walkable)
    rows = len(walkable[0]) if cols else 0
    if 0 <= start[0] < cols and 0 <= start[1] < rows and walkable[start[0]][start[1]]:
        return start

    queue: deque[GridCell] = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            cell = (nx, ny)
            if cell in seen:
                continue
            if walkable[nx][ny]:
                return cell
            seen.add(cell)
            queue.append(cell)
    return start


def astar_path(
    start: GridCell,
    goal: GridCell,
    walkable: list[list[bool]],
) -> list[GridCell]:
    cols = len(walkable)
    rows = len(walkable[0]) if cols else 0
    start = nearest_walkable_cell(start, walkable)
    goal = nearest_walkable_cell(goal, walkable)
    if start == goal:
        return [start]

    def heuristic(cell: GridCell) -> int:
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    frontier: list[tuple[int, GridCell]] = [(heuristic(start), start)]
    came_from: dict[GridCell, GridCell | None] = {start: None}
    cost_so_far: dict[GridCell, int] = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        cx, cy = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < cols and 0 <= ny < rows) or not walkable[nx][ny]:
                continue
            next_cell = (nx, ny)
            new_cost = cost_so_far[current] + 1
            if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                cost_so_far[next_cell] = new_cost
                priority = new_cost + heuristic(next_cell)
                heapq.heappush(frontier, (priority, next_cell))
                came_from[next_cell] = current

    if goal not in came_from:
        return [start]

    path: list[GridCell] = []
    current: GridCell | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path
