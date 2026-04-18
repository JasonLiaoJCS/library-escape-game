"""Collision and ray utilities for the continuous-time world."""

from __future__ import annotations

import math

from .obstacle import Obstacle


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def circle_intersects_rect(
    cx: float,
    cy: float,
    radius: float,
    obstacle: Obstacle,
) -> bool:
    nearest_x = clamp(cx, obstacle.x, obstacle.x2)
    nearest_y = clamp(cy, obstacle.y, obstacle.y2)
    dx = cx - nearest_x
    dy = cy - nearest_y
    return dx * dx + dy * dy < radius * radius


def collides_with_any_obstacle(
    cx: float,
    cy: float,
    radius: float,
    obstacles: list[Obstacle],
) -> bool:
    return any(circle_intersects_rect(cx, cy, radius, obstacle) for obstacle in obstacles)


def move_circle(
    x: float,
    y: float,
    radius: float,
    velocity: tuple[float, float],
    dt: float,
    width: float,
    height: float,
    obstacles: list[Obstacle],
) -> tuple[float, float, bool]:
    vx, vy = velocity
    collided = False

    next_x = clamp(x + vx * dt, radius, width - radius)
    if collides_with_any_obstacle(next_x, y, radius, obstacles):
        next_x = x
        collided = True

    next_y = clamp(y + vy * dt, radius, height - radius)
    if collides_with_any_obstacle(next_x, next_y, radius, obstacles):
        next_y = y
        collided = True

    return next_x, next_y, collided


def has_line_of_sight(
    start: tuple[float, float],
    end: tuple[float, float],
    obstacles: list[Obstacle],
    samples_per_cell: int = 10,
) -> bool:
    sx, sy = start
    ex, ey = end
    distance = math.hypot(ex - sx, ey - sy)
    if distance <= 1e-8:
        return True

    total_samples = max(2, int(distance * samples_per_cell))
    for sample in range(1, total_samples):
        t = sample / total_samples
        px = sx + (ex - sx) * t
        py = sy + (ey - sy) * t
        if any(obstacle.contains_point(px, py) for obstacle in obstacles):
            return False
    return True


def raycast_distance(
    origin: tuple[float, float],
    angle_radians: float,
    max_distance: float,
    obstacles: list[Obstacle],
    width: float,
    height: float,
    step: float = 0.05,
) -> float:
    ox, oy = origin
    distance = 0.0
    while distance < max_distance:
        px = ox + math.cos(angle_radians) * distance
        py = oy + math.sin(angle_radians) * distance
        if px <= 0.0 or py <= 0.0 or px >= width or py >= height:
            return distance
        if any(obstacle.contains_point(px, py) for obstacle in obstacles):
            return distance
        distance += step
    return max_distance
