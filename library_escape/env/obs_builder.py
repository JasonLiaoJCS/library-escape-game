"""Observation builders for the Gymnasium and PettingZoo APIs."""

from __future__ import annotations

import math

import numpy as np

from ..core.physics import has_line_of_sight


class ObsBuilder:
    def __init__(self, env_config: dict) -> None:
        self.env_config = env_config
        self.obs_config = env_config["observation"]
        self.max_ray_distance = float(self.obs_config["max_ray_distance"])
        self.wall_rays = int(self.obs_config["wall_rays"])
        self.player_enemy_rays = int(self.obs_config["player_enemy_rays"])
        self.normalize = bool(self.obs_config["normalize"])
        self.partial_observability = bool(self.obs_config["partial_observability"])

    def build(self, world, role: str) -> np.ndarray:
        if role == "player":
            return self.build_player_obs(world)
        if role == "enemy":
            return self.build_enemy_obs(world)
        raise ValueError(f"Unsupported role: {role}")

    def build_enemy_obs(self, world) -> np.ndarray:
        enemy = world.enemy
        player = world.player
        visible = world.player_visible_to_enemy()
        rel_x = player.x - enemy.x if (visible or not self.partial_observability) else 0.0
        rel_y = player.y - enemy.y if (visible or not self.partial_observability) else 0.0

        features = [
            enemy.x / world.width,
            enemy.y / world.height,
            enemy.facing_x,
            enemy.facing_y,
            rel_x / world.width,
            rel_y / world.height,
            1.0 if visible else 0.0,
            world.remaining_count("note") / max(1, int(world.env_config["collectibles"]["notes"])),
            world.remaining_count("exam") / max(1, int(world.env_config["collectibles"]["exams"])),
            world.time_remaining / world.max_episode_seconds,
        ]
        features.extend(self._wall_rays(world, enemy.position, self.wall_rays))
        return np.asarray(features, dtype=np.float32)

    def build_player_obs(self, world) -> np.ndarray:
        player = world.player
        enemy = world.enemy
        enemy_visible = self._enemy_visible_to_player(world)
        rel_x = enemy.x - player.x if (enemy_visible or not self.partial_observability) else 0.0
        rel_y = enemy.y - player.y if (enemy_visible or not self.partial_observability) else 0.0

        nearest_note = world.nearest_collectible(player.position, kinds=("note", "exam"))
        note_dx = 0.0
        note_dy = 0.0
        if nearest_note is not None:
            note_dx = (nearest_note.x - player.x) / world.width
            note_dy = (nearest_note.y - player.y) / world.height

        features = [
            player.x / world.width,
            player.y / world.height,
            player.facing_x,
            player.facing_y,
            rel_x / world.width,
            rel_y / world.height,
            1.0 if enemy_visible else 0.0,
            note_dx,
            note_dy,
            world.time_remaining / world.max_episode_seconds,
            1.0 if world.can_player_escape() else 0.0,
            player.coffee_timer / max(1.0, float(world.env_config["world"]["coffee_duration_seconds"])),
            enemy.freeze_timer / max(1.0, float(world.env_config["world"]["freeze_duration_seconds"])),
        ]
        features.extend(self._fan_rays(world, player.position, (player.facing_x, player.facing_y), self.player_enemy_rays))
        features.extend(self._wall_rays(world, player.position, self.wall_rays))
        return np.asarray(features, dtype=np.float32)

    def enemy_obs_dim(self) -> int:
        return len(self.build_enemy_obs.__annotations__) if False else 10 + self.wall_rays

    def player_obs_dim(self) -> int:
        return 13 + self.player_enemy_rays + self.wall_rays

    def _enemy_visible_to_player(self, world) -> bool:
        player = world.player
        enemy = world.enemy
        distance = world.distance_between_agents()
        if distance > self.max_ray_distance:
            return False
        return has_line_of_sight(player.position, enemy.position, world.obstacles)

    def _wall_rays(self, world, origin: tuple[float, float], count: int) -> list[float]:
        values: list[float] = []
        for index in range(count):
            angle = (math.tau * index) / count
            distance = world.raycast_from(origin, angle_radians=angle, max_distance=self.max_ray_distance)
            values.append(distance / self.max_ray_distance)
        return values

    def _fan_rays(
        self,
        world,
        origin: tuple[float, float],
        facing: tuple[float, float],
        count: int,
    ) -> list[float]:
        facing_angle = math.atan2(facing[1], facing[0]) if abs(facing[0]) > 1e-6 or abs(facing[1]) > 1e-6 else 0.0
        if count == 1:
            angles = [facing_angle]
        else:
            offsets = np.linspace(-math.pi / 4.0, math.pi / 4.0, num=count)
            angles = [facing_angle + float(offset) for offset in offsets]
        return [
            world.raycast_from(origin, angle_radians=angle, max_distance=self.max_ray_distance) / self.max_ray_distance
            for angle in angles
        ]
