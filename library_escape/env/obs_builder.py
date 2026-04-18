from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from library_escape.core.constants import RAY_ANGLES
from library_escape.core.entities import NoteState, vec2
from library_escape.core.physics import angle_to_vector, length, raycast_distance
from library_escape.core.world import LibraryWorld


class ObservationBuilder:
    def __init__(self, config: dict[str, Any], world: LibraryWorld) -> None:
        self.config = config
        self.world = world
        obs_cfg = config["observations"]
        self.normalize = bool(obs_cfg.get("normalize", True))
        self.include_velocity = bool(obs_cfg.get("include_velocity", True))
        self.include_heading = bool(obs_cfg.get("include_heading", True))
        self.include_last_seen_age = bool(obs_cfg.get("include_last_seen_age", True))
        self.wall_ray_count = int(obs_cfg.get("wall_ray_count", 8))
        self.wall_ray_range = float(obs_cfg.get("wall_ray_range", 220.0))
        self.memory_seconds = float(obs_cfg.get("memory_seconds", 1.5))

    def _normalize_position(self, position: np.ndarray) -> list[float]:
        return [
            (float(position[0]) / self.world.width) * 2.0 - 1.0,
            (float(position[1]) / self.world.height) * 2.0 - 1.0,
        ]

    def _normalize_velocity(self, velocity: np.ndarray, max_speed: float) -> list[float]:
        divisor = max(max_speed, 1.0)
        return [
            float(np.clip(velocity[0] / divisor, -1.0, 1.0)),
            float(np.clip(velocity[1] / divisor, -1.0, 1.0)),
        ]

    def _relative_target(self, source: np.ndarray, target: np.ndarray | None) -> list[float]:
        if target is None:
            return [0.0, 0.0]
        delta = target - source
        return [
            float(np.clip(delta[0] / self.world.width, -1.0, 1.0)),
            float(np.clip(delta[1] / self.world.height, -1.0, 1.0)),
        ]

    def _heading_features(self, angle: float) -> list[float]:
        return [math.cos(angle), math.sin(angle)]

    def _ray_features(self, origin: np.ndarray) -> list[float]:
        if self.wall_ray_count <= len(RAY_ANGLES):
            angles = RAY_ANGLES[: self.wall_ray_count]
        else:
            angles = tuple((math.tau / self.wall_ray_count) * index for index in range(self.wall_ray_count))
        values = []
        for angle in angles:
            direction = angle_to_vector(angle)
            distance = raycast_distance(
                origin,
                direction,
                self.wall_ray_range,
                self.world.obstacles,
                self.world.width,
                self.world.height,
            )
            values.append(float(np.clip(distance / self.wall_ray_range, 0.0, 1.0)))
        return values

    def _nearest_note(self) -> NoteState | None:
        remaining = [note for note in self.world.notes if not note.collected]
        if not remaining:
            return None
        return min(remaining, key=lambda note: length(note.position - self.world.player.position))

    def build(self, role: str) -> np.ndarray:
        if role == "enemy":
            return self._enemy_obs()
        return self._player_obs()

    def _player_obs(self) -> np.ndarray:
        player = self.world.player
        enemy = self.world.enemy
        enemy_visible = self.world.player_can_see_enemy()
        nearest_note = self._nearest_note()
        nearest_note_pos = None if nearest_note is None else nearest_note.position
        exit_pos = self.world.exit_zone.center if self.world.all_notes_collected() else None

        values: list[float] = []
        values.extend(self._normalize_position(player.position))
        if self.include_velocity:
            values.extend(self._normalize_velocity(player.velocity, player.max_speed))
        if self.include_heading:
            values.extend(self._heading_features(player.facing))
        values.extend(self._relative_target(player.position, enemy.position if enemy_visible else None))
        values.append(1.0 if enemy_visible else 0.0)
        values.extend(self._relative_target(player.position, nearest_note_pos))
        values.extend(self._relative_target(player.position, exit_pos))
        values.append(float(self.world.notes_remaining / max(self.world.total_notes, 1)))
        values.append(float(self.world.remaining_time / self.world.max_time))
        values.extend(self._ray_features(player.position))
        return np.asarray(values, dtype=np.float32)

    def _enemy_obs(self) -> np.ndarray:
        enemy = self.world.enemy
        player = self.world.player
        player_visible = self.world.enemy_can_see_player()

        values: list[float] = []
        values.extend(self._normalize_position(enemy.position))
        if self.include_velocity:
            values.extend(self._normalize_velocity(enemy.velocity, enemy.max_speed))
        if self.include_heading:
            values.extend(self._heading_features(enemy.facing))
        values.extend(self._relative_target(enemy.position, player.position if player_visible else None))
        values.append(1.0 if player_visible else 0.0)

        last_seen = enemy.last_seen_player if enemy.last_seen_timer <= self.memory_seconds else None
        values.extend(self._relative_target(enemy.position, last_seen))
        if self.include_last_seen_age:
            age = min(enemy.last_seen_timer / max(self.memory_seconds, 1e-6), 1.0)
            values.append(float(age))
        values.append(float(self.world.notes_remaining / max(self.world.total_notes, 1)))
        values.append(float(self.world.remaining_time / self.world.max_time))
        values.extend(self._ray_features(enemy.position))
        return np.asarray(values, dtype=np.float32)

    def observation_space(self, role: str) -> gym.spaces.Box:
        sample = self.build(role)
        return gym.spaces.Box(low=-1.0, high=1.0, shape=sample.shape, dtype=np.float32)

    def action_space(self, role: str) -> gym.Space[Any]:
        scheme = self.config["actions"]["scheme"]
        if scheme == "continuous":
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        return gym.spaces.Discrete(9)
