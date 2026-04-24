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
        collectible_cfg = env_config.get("collectibles", {})
        self.player_note_slots = int(self.obs_config.get("player_note_slots", collectible_cfg.get("notes", 0)))
        self.player_exam_slots = int(self.obs_config.get("player_exam_slots", collectible_cfg.get("exams", 0)))
        self.player_coffee_slots = int(self.obs_config.get("player_coffee_slots", collectible_cfg.get("coffee", 0)))
        self.player_freeze_slots = int(self.obs_config.get("player_freeze_slots", collectible_cfg.get("freeze", 0)))

    def build(self, world, role: str) -> np.ndarray:
        if role == "player":
            return self.build_player_obs(world)
        if role == "enemy":
            return self.build_enemy_obs(world)
        raise ValueError(f"Unsupported role: {role}")

    def build_enemy_obs(self, world) -> np.ndarray:
        enemy = world.enemy
        player = world.player
        metrics = world.transition_metrics()
        primary_visible = world.primary_enemy_sees_player() if hasattr(world, "primary_enemy_sees_player") else world.player_visible_to_enemy()
        team_visible = world.player_visible_to_enemy()
        # Escape mode can disable partial observability so the enemy sees the
        # player's relative position directly. Collection mode keeps the older
        # stealth-oriented masking behavior.
        rel_x = player.x - enemy.x if (team_visible or not self.partial_observability) else 0.0
        rel_y = player.y - enemy.y if (team_visible or not self.partial_observability) else 0.0
        nearest_ally_dx = 0.0
        nearest_ally_dy = 0.0
        allies = [ally for ally in getattr(world, "support_enemies", [])]
        if allies:
            nearest_ally = min(allies, key=lambda ally: math.hypot(ally.x - enemy.x, ally.y - enemy.y))
            nearest_ally_dx = (nearest_ally.x - enemy.x) / world.width
            nearest_ally_dy = (nearest_ally.y - enemy.y) / world.height
        collection_flag, escape_flag = self._mode_flags(world)
        player_nav_dx, player_nav_dy = self._navigation_direction(world, enemy.position, player.position)
        guard_target = world.escape_center() if world.can_player_escape() else world.current_player_goal_position()
        guard_nav_dx, guard_nav_dy = self._navigation_direction(world, enemy.position, guard_target)

        features = [
            enemy.x / world.width,
            enemy.y / world.height,
            enemy.facing_x,
            enemy.facing_y,
            rel_x / world.width,
            rel_y / world.height,
            1.0 if primary_visible else 0.0,
            1.0 if team_visible else 0.0,
            float(metrics.get("visible_enemy_ratio", 0.0)),
            nearest_ally_dx,
            nearest_ally_dy,
            world.remaining_count("note") / max(1, int(world.env_config["collectibles"]["notes"])),
            world.remaining_count("exam") / max(1, int(world.env_config["collectibles"]["exams"])),
            float(metrics.get("score_progress", 0.0)),
            float(metrics.get("objective_progress", 0.0)),
            world.time_remaining / world.max_episode_seconds,
            1.0 if world.can_player_escape() else 0.0,
            float(metrics.get("player_in_escape_zone", 0.0)),
            self._normalized_distance(world, float(metrics.get("distance_agents", 0.0))),
            self._normalized_distance(world, float(metrics.get("distance_player_to_target", 0.0))),
            self._normalized_distance(world, float(metrics.get("distance_player_to_escape", 0.0))),
            self._normalized_distance(world, float(metrics.get("distance_enemy_to_escape", 0.0))),
            float(metrics.get("exit_lead", 0.0)),
            float(metrics.get("collection_progress", 0.0)),
            float(metrics.get("team_detection_cooldown", 0.0)),
            world.enemy_pause_fraction(enemy) if hasattr(world, "enemy_pause_fraction") else 0.0,
            float(metrics.get("primary_threat_margin", 0.0)),
            collection_flag,
            escape_flag,
            player_nav_dx,
            player_nav_dy,
            guard_nav_dx,
            guard_nav_dy,
        ]
        features.extend(self._wall_rays(world, enemy.position, self.wall_rays))
        return np.asarray(features, dtype=np.float32)

    def build_player_obs(self, world) -> np.ndarray:
        player = world.player
        metrics = world.transition_metrics()
        nearest_enemy = world.nearest_enemy() if hasattr(world, "nearest_enemy") else world.enemy
        primary_enemy = world.enemy
        support_enemies = list(getattr(world, "support_enemies", []))
        nearest_support = (
            min(support_enemies, key=lambda enemy: math.hypot(enemy.x - player.x, enemy.y - player.y))
            if support_enemies
            else None
        )
        enemy_visible = self._enemy_visible_to_player(world)
        # Escape mode can disable partial observability so the player can plan
        # around enemy positions; collection mode keeps stealth masking.
        rel_x = nearest_enemy.x - player.x if (enemy_visible or not self.partial_observability) else 0.0
        rel_y = nearest_enemy.y - player.y if (enemy_visible or not self.partial_observability) else 0.0
        primary_rel_x = primary_enemy.x - player.x if (enemy_visible or not self.partial_observability) else 0.0
        primary_rel_y = primary_enemy.y - player.y if (enemy_visible or not self.partial_observability) else 0.0
        support_rel_x = 0.0
        support_rel_y = 0.0
        if nearest_support is not None and (enemy_visible or not self.partial_observability):
            support_rel_x = nearest_support.x - player.x
            support_rel_y = nearest_support.y - player.y
        escape_dx, escape_dy = self._relative_escape_offset(world, player.position)
        goal_nav_dx, goal_nav_dy = self._navigation_direction(world, player.position, world.current_player_goal_position())
        note_dx, note_dy = self._relative_collectible_offset(world, player.position, ("note",))
        exam_dx, exam_dy = self._relative_collectible_offset(world, player.position, ("exam",))
        power_dx, power_dy = self._relative_collectible_offset(world, player.position, ("coffee", "freeze"))
        collection_flag, escape_flag = self._mode_flags(world)

        features = [
            player.x / world.width,
            player.y / world.height,
            player.facing_x,
            player.facing_y,
            rel_x / world.width,
            rel_y / world.height,
            1.0 if enemy_visible else 0.0,
            float(metrics.get("player_visible_primary", 0.0)),
            float(metrics.get("visible_enemy_ratio", 0.0)),
            self._normalized_distance(world, float(metrics.get("distance_agents", 0.0))),
            note_dx,
            note_dy,
            exam_dx,
            exam_dy,
            power_dx,
            power_dy,
            float(metrics.get("objective_progress", 0.0)),
            float(metrics.get("score_progress", 0.0)),
            world.time_remaining / world.max_episode_seconds,
            1.0 if world.can_player_escape() else 0.0,
            float(metrics.get("player_in_escape_zone", 0.0)),
            float(metrics.get("collection_progress", 0.0)),
            self._normalized_distance(world, float(metrics.get("distance_player_to_target", 0.0))),
            self._normalized_distance(world, float(metrics.get("distance_player_to_escape", 0.0))),
            float(metrics.get("exit_lead", 0.0)),
            player.coffee_timer / max(1.0, float(world.env_config["world"]["coffee_duration_seconds"])),
            world.max_enemy_freeze_timer() / max(1.0, float(world.env_config["world"]["freeze_duration_seconds"])),
            goal_nav_dx,
            goal_nav_dy,
            collection_flag,
            escape_flag,
            primary_rel_x / world.width,
            primary_rel_y / world.height,
            support_rel_x / world.width,
            support_rel_y / world.height,
            escape_dx,
            escape_dy,
        ]
        features.extend(self._collectible_slots(world, player.position, "note", self.player_note_slots))
        features.extend(self._collectible_slots(world, player.position, "exam", self.player_exam_slots))
        features.extend(self._collectible_slots(world, player.position, "coffee", self.player_coffee_slots))
        features.extend(self._collectible_slots(world, player.position, "freeze", self.player_freeze_slots))
        features.extend(self._fan_rays(world, player.position, (player.facing_x, player.facing_y), self.player_enemy_rays))
        features.extend(self._wall_rays(world, player.position, self.wall_rays))
        return np.asarray(features, dtype=np.float32)

    def enemy_obs_dim(self) -> int:
        return 33 + self.wall_rays

    def player_obs_dim(self) -> int:
        return 37 + self._player_collectible_slot_dim() + self.player_enemy_rays + self.wall_rays

    def _player_collectible_slot_dim(self) -> int:
        slot_count = self.player_note_slots + self.player_exam_slots + self.player_coffee_slots + self.player_freeze_slots
        return slot_count * 3

    def player_collectible_slot_start(self) -> int:
        return 37

    def _collectible_slots(
        self,
        world,
        origin: tuple[float, float],
        kind: str,
        slot_count: int,
    ) -> list[float]:
        values: list[float] = []
        items = [item for item in world.collectibles if item.kind == kind]
        for item in items[:slot_count]:
            values.extend(
                [
                    (item.x - origin[0]) / world.width,
                    (item.y - origin[1]) / world.height,
                    1.0 if item.active else 0.0,
                ]
            )
        missing_slots = max(0, slot_count - len(items))
        values.extend([0.0, 0.0, 0.0] * missing_slots)
        return values

    def _relative_collectible_offset(self, world, origin: tuple[float, float], kinds: tuple[str, ...]) -> tuple[float, float]:
        target = world.nearest_collectible(origin, kinds=kinds)
        if target is None:
            return 0.0, 0.0
        return (target.x - origin[0]) / world.width, (target.y - origin[1]) / world.height

    def _relative_escape_offset(self, world, origin: tuple[float, float]) -> tuple[float, float]:
        escape_center = world.escape_center() if hasattr(world, "escape_center") else (
            world.escape_zone["x"] + (world.escape_zone["w"] / 2.0),
            world.escape_zone["y"] + (world.escape_zone["h"] / 2.0),
        )
        return (escape_center[0] - origin[0]) / world.width, (escape_center[1] - origin[1]) / world.height

    def _relative_player_goal_offset(self, world, origin: tuple[float, float]) -> tuple[float, float]:
        if not hasattr(world, "current_player_goal_position"):
            return 0.0, 0.0
        goal = world.current_player_goal_position()
        if goal is None:
            return 0.0, 0.0
        return (goal[0] - origin[0]) / world.width, (goal[1] - origin[1]) / world.height

    def _normalized_distance(self, world, distance: float) -> float:
        return float(distance / max(1.0, world.max_map_distance()))

    def _navigation_direction(
        self,
        world,
        origin: tuple[float, float],
        target: tuple[float, float] | None,
    ) -> tuple[float, float]:
        if target is None:
            return 0.0, 0.0
        if hasattr(world, "steer_towards_position"):
            return world.steer_towards_position(origin, target)
        return self._vector_between_positions(origin, target)

    def _mode_flags(self, world) -> tuple[float, float]:
        return (
            1.0 if getattr(world, "is_collection_mode", lambda: False)() else 0.0,
            1.0 if getattr(world, "is_escape_mode", lambda: False)() else 0.0,
        )

    def _enemy_visible_to_player(self, world) -> bool:
        player = world.player
        enemies = list(getattr(world, "all_enemies", lambda: [world.enemy])())
        for enemy in sorted(enemies, key=lambda item: math.hypot(player.x - item.x, player.y - item.y)):
            distance = math.hypot(player.x - enemy.x, player.y - enemy.y)
            if distance > self.max_ray_distance:
                continue
            if has_line_of_sight(player.position, enemy.position, world.obstacles):
                return True
        return False

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

    def _vector_between_positions(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
    ) -> tuple[float, float]:
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        length = math.hypot(dx, dy)
        if length <= 1e-8:
            return 0.0, 0.0
        return dx / length, dy / length
