from __future__ import annotations

import math

import numpy as np

from library_escape.core.entities import ObstacleState, Rectangle, vec2


def length(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = length(vector)
    if norm <= 1e-8:
        return vec2()
    return (vector / norm).astype(np.float32)


def angle_to_vector(angle: float) -> np.ndarray:
    return vec2(math.cos(angle), math.sin(angle))


def circle_rect_collision(point: np.ndarray, radius: float, rect: Rectangle) -> bool:
    closest_x = min(max(float(point[0]), rect.left), rect.right)
    closest_y = min(max(float(point[1]), rect.top), rect.bottom)
    dx = float(point[0]) - closest_x
    dy = float(point[1]) - closest_y
    return dx * dx + dy * dy < radius * radius


def _clamp_to_bounds(
    point: np.ndarray,
    radius: float,
    width: float,
    height: float,
) -> tuple[np.ndarray, int]:
    wall_hits = 0
    point = point.astype(np.float32, copy=True)
    min_x = radius
    max_x = width - radius
    min_y = radius
    max_y = height - radius
    if point[0] < min_x:
        point[0] = min_x
        wall_hits += 1
    if point[0] > max_x:
        point[0] = max_x
        wall_hits += 1
    if point[1] < min_y:
        point[1] = min_y
        wall_hits += 1
    if point[1] > max_y:
        point[1] = max_y
        wall_hits += 1
    return point, wall_hits


def move_circle(
    position: np.ndarray,
    velocity: np.ndarray,
    dt: float,
    radius: float,
    obstacles: list[ObstacleState],
    width: float,
    height: float,
) -> tuple[np.ndarray, int]:
    delta = velocity.astype(np.float32) * float(dt)
    next_point = position.astype(np.float32, copy=True)
    wall_hits = 0

    candidate_x = next_point.copy()
    candidate_x[0] += delta[0]
    candidate_x, hits = _clamp_to_bounds(candidate_x, radius, width, height)
    wall_hits += hits
    for obstacle in obstacles:
        if not obstacle.blocks_movement:
            continue
        if circle_rect_collision(candidate_x, radius, obstacle.rect):
            wall_hits += 1
            if delta[0] > 0:
                candidate_x[0] = obstacle.rect.left - radius
            elif delta[0] < 0:
                candidate_x[0] = obstacle.rect.right + radius
    next_point[0] = candidate_x[0]

    candidate_y = next_point.copy()
    candidate_y[1] += delta[1]
    candidate_y, hits = _clamp_to_bounds(candidate_y, radius, width, height)
    wall_hits += hits
    for obstacle in obstacles:
        if not obstacle.blocks_movement:
            continue
        if circle_rect_collision(candidate_y, radius, obstacle.rect):
            wall_hits += 1
            if delta[1] > 0:
                candidate_y[1] = obstacle.rect.top - radius
            elif delta[1] < 0:
                candidate_y[1] = obstacle.rect.bottom + radius
    next_point[1] = candidate_y[1]

    return next_point, wall_hits


def _segment_intersects_rect(start: np.ndarray, end: np.ndarray, rect: Rectangle) -> bool:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    p = (-dx, dx, -dy, dy)
    q = (
        float(start[0] - rect.left),
        float(rect.right - start[0]),
        float(start[1] - rect.top),
        float(rect.bottom - start[1]),
    )
    u1 = 0.0
    u2 = 1.0
    for pk, qk in zip(p, q, strict=True):
        if abs(pk) <= 1e-12:
            if qk < 0.0:
                return False
            continue
        t = qk / pk
        if pk < 0.0:
            if t > u2:
                return False
            u1 = max(u1, t)
        else:
            if t < u1:
                return False
            u2 = min(u2, t)
    return True


def line_of_sight(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: list[ObstacleState],
) -> bool:
    for obstacle in obstacles:
        if obstacle.blocks_sight and _segment_intersects_rect(start, end, obstacle.rect):
            return False
    return True


def _ray_rect_distance(origin: np.ndarray, direction: np.ndarray, rect: Rectangle) -> float | None:
    direction = normalize(direction)
    if length(direction) <= 1e-8:
        return None

    inv_dx = math.inf if abs(direction[0]) <= 1e-12 else 1.0 / float(direction[0])
    inv_dy = math.inf if abs(direction[1]) <= 1e-12 else 1.0 / float(direction[1])

    tx1 = (rect.left - float(origin[0])) * inv_dx
    tx2 = (rect.right - float(origin[0])) * inv_dx
    ty1 = (rect.top - float(origin[1])) * inv_dy
    ty2 = (rect.bottom - float(origin[1])) * inv_dy

    tmin = max(min(tx1, tx2), min(ty1, ty2))
    tmax = min(max(tx1, tx2), max(ty1, ty2))
    if tmax < 0.0 or tmin > tmax:
        return None
    return max(tmin, 0.0)


def _ray_bounds_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    width: float,
    height: float,
) -> float:
    direction = normalize(direction)
    candidates: list[float] = []
    if abs(direction[0]) > 1e-8:
        if direction[0] > 0:
            candidates.append((width - float(origin[0])) / float(direction[0]))
        else:
            candidates.append((0.0 - float(origin[0])) / float(direction[0]))
    if abs(direction[1]) > 1e-8:
        if direction[1] > 0:
            candidates.append((height - float(origin[1])) / float(direction[1]))
        else:
            candidates.append((0.0 - float(origin[1])) / float(direction[1]))
    positive = [value for value in candidates if value >= 0.0]
    return min(positive) if positive else 0.0


def raycast_distance(
    origin: np.ndarray,
    direction: np.ndarray,
    max_distance: float,
    obstacles: list[ObstacleState],
    width: float,
    height: float,
) -> float:
    nearest = min(float(max_distance), _ray_bounds_distance(origin, direction, width, height))
    for obstacle in obstacles:
        if not obstacle.blocks_sight:
            continue
        hit = _ray_rect_distance(origin, direction, obstacle.rect)
        if hit is not None:
            nearest = min(nearest, hit)
    return max(0.0, nearest)
