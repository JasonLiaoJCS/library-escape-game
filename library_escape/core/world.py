from __future__ import annotations

import math
from typing import Any

import numpy as np

from library_escape.core.actions import action_to_direction
from library_escape.core.entities import (
    ActorState,
    NoteState,
    ObstacleState,
    Rectangle,
    StepMetrics,
    WorldSnapshot,
    WorldStatus,
    vec2,
)
from library_escape.core.navigation import build_walkable_grid
from library_escape.core.physics import angle_to_vector, length, line_of_sight, move_circle, normalize


class LibraryWorld:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        world_cfg = config["world"]
        self.width = float(world_cfg["width"])
        self.height = float(world_cfg["height"])
        self.grid_size = float(world_cfg["grid_size"])
        self.max_time = float(world_cfg["max_time"])
        self.physics_dt = float(world_cfg.get("physics_dt", 1.0 / 120.0))
        self.action_scheme = config["actions"]["scheme"]
        self.player_view_range = float(config["observations"].get("player_view_range", 260.0))
        self.player_view_angle = math.radians(float(config["observations"].get("player_view_angle_deg", 180.0)))

        self.rng = np.random.default_rng()
        self.time_elapsed = 0.0
        self.total_notes = 0
        self.notes_collected = 0
        self.player: ActorState
        self.enemy: ActorState
        self.exit_zone: Rectangle
        self.obstacles: list[ObstacleState] = []
        self.notes: list[NoteState] = []
        self.status = WorldStatus()
        self.last_metrics = StepMetrics()
        self.walkable_grid: list[list[bool]] = []
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time_elapsed = 0.0
        self.notes_collected = 0
        self.status = WorldStatus()
        self.obstacles = self._build_obstacles()
        self.exit_zone = self._build_exit()
        self.player = self._build_actor("player")
        self.enemy = self._build_actor("enemy")
        self.notes = self._build_notes()
        self.total_notes = len(self.notes)
        self.last_metrics = StepMetrics()
        radius = max(self.player.radius, self.enemy.radius)
        self.walkable_grid = build_walkable_grid(
            self.width,
            self.height,
            self.grid_size,
            self.obstacles,
            radius,
        )

    def _build_actor(self, role: str) -> ActorState:
        cfg = self.config["world"][role]
        position = np.asarray(cfg["start"], dtype=np.float32)
        velocity = vec2()
        facing = math.radians(float(cfg.get("facing_deg", 0.0)))
        return ActorState(
            name=cfg["name"],
            role=role,  # type: ignore[arg-type]
            position=position,
            velocity=velocity,
            facing=facing,
            radius=float(cfg["radius"]),
            max_speed=float(cfg["speed"]),
            view_range=float(cfg["vision_range"]),
            view_angle=math.radians(float(cfg["vision_angle_deg"])),
            catch_radius=float(cfg.get("catch_radius", 0.0)),
        )

    def _build_exit(self) -> Rectangle:
        cfg = self.config["world"]["exit_zone"]
        return Rectangle(float(cfg["x"]), float(cfg["y"]), float(cfg["w"]), float(cfg["h"]))

    def _build_notes(self) -> list[NoteState]:
        radius = float(self.config["world"]["note_radius"])
        notes = []
        for index, note_cfg in enumerate(self.config["world"]["notes"], start=1):
            notes.append(
                NoteState(
                    note_id=f"note_{index}",
                    position=np.asarray(note_cfg, dtype=np.float32),
                    radius=radius,
                )
            )
        return notes

    def _build_obstacles(self) -> list[ObstacleState]:
        obstacles = []
        for obstacle_cfg in self.config["world"]["obstacles"]:
            rect = Rectangle(
                float(obstacle_cfg["x"]),
                float(obstacle_cfg["y"]),
                float(obstacle_cfg["w"]),
                float(obstacle_cfg["h"]),
            )
            obstacles.append(
                ObstacleState(
                    kind=str(obstacle_cfg["type"]),
                    rect=rect,
                    blocks_movement=bool(obstacle_cfg.get("blocks_movement", True)),
                    blocks_sight=bool(obstacle_cfg.get("blocks_sight", True)),
                )
            )
        return obstacles

    @property
    def remaining_time(self) -> float:
        return max(0.0, self.max_time - self.time_elapsed)

    @property
    def notes_remaining(self) -> int:
        return sum(1 for note in self.notes if not note.collected)

    def snapshot(self) -> WorldSnapshot:
        enemy_sees_player = self.enemy_can_see_player()
        player_can_see_enemy = self.player_can_see_enemy()
        return WorldSnapshot(
            player_position=self.player.position.copy(),
            enemy_position=self.enemy.position.copy(),
            player_velocity=self.player.velocity.copy(),
            enemy_velocity=self.enemy.velocity.copy(),
            player_facing=self.player.facing,
            enemy_facing=self.enemy.facing,
            remaining_time=self.remaining_time,
            notes_remaining=self.notes_remaining,
            total_notes=self.total_notes,
            enemy_sees_player=enemy_sees_player,
            player_can_see_enemy=player_can_see_enemy,
            distance=self.player_enemy_distance(),
        )

    def all_notes_collected(self) -> bool:
        return self.notes_remaining == 0

    def player_enemy_distance(self) -> float:
        return length(self.player.position - self.enemy.position)

    def enemy_can_see_player(self) -> bool:
        vector = self.player.position - self.enemy.position
        distance = length(vector)
        if distance > self.enemy.view_range or distance <= 1e-8:
            return False
        facing_vec = angle_to_vector(self.enemy.facing)
        direction = normalize(vector)
        cos_limit = math.cos(self.enemy.view_angle / 2.0)
        return float(np.dot(direction, facing_vec)) >= cos_limit and line_of_sight(
            self.enemy.position,
            self.player.position,
            self.obstacles,
        )

    def player_can_see_enemy(self) -> bool:
        vector = self.enemy.position - self.player.position
        distance = length(vector)
        if distance > self.player_view_range or distance <= 1e-8:
            return False
        facing_vec = angle_to_vector(self.player.facing)
        direction = normalize(vector)
        cos_limit = math.cos(self.player_view_angle / 2.0)
        return float(np.dot(direction, facing_vec)) >= cos_limit and line_of_sight(
            self.player.position,
            self.enemy.position,
            self.obstacles,
        )

    def tick(
        self,
        player_action: Any,
        enemy_action: Any,
        dt: float | None = None,
    ) -> StepMetrics:
        if self.status.terminated or self.status.truncated:
            return StepMetrics()

        dt = float(dt or self.physics_dt)
        metrics = StepMetrics()
        previous_distance = self.player_enemy_distance()

        player_direction = action_to_direction(player_action, self.action_scheme)
        enemy_direction = action_to_direction(enemy_action, self.action_scheme)

        self.player.velocity = player_direction * self.player.max_speed
        self.enemy.velocity = enemy_direction * self.enemy.max_speed

        if length(self.player.velocity) > 1e-6:
            self.player.facing = math.atan2(float(self.player.velocity[1]), float(self.player.velocity[0]))
        if length(self.enemy.velocity) > 1e-6:
            self.enemy.facing = math.atan2(float(self.enemy.velocity[1]), float(self.enemy.velocity[0]))

        self.player.position, player_hits = move_circle(
            self.player.position,
            self.player.velocity,
            dt,
            self.player.radius,
            self.obstacles,
            self.width,
            self.height,
        )
        self.enemy.position, enemy_hits = move_circle(
            self.enemy.position,
            self.enemy.velocity,
            dt,
            self.enemy.radius,
            self.obstacles,
            self.width,
            self.height,
        )
        self.player.wall_hits += player_hits
        self.enemy.wall_hits += enemy_hits
        metrics.player_wall_hits += player_hits
        metrics.enemy_wall_hits += enemy_hits

        for note in self.notes:
            if note.collected:
                continue
            if length(note.position - self.player.position) <= note.radius + self.player.radius:
                note.collected = True
                self.notes_collected += 1
                metrics.note_collected += 1

        if self.enemy_can_see_player():
            metrics.player_visible_steps += 1
            self.enemy.last_seen_player = self.player.position.copy()
            self.enemy.last_seen_timer = 0.0
        else:
            self.enemy.last_seen_timer += dt

        new_distance = self.player_enemy_distance()
        metrics.distance_delta = previous_distance - new_distance

        catch_distance = self.player.radius + self.enemy.radius + self.enemy.catch_radius
        if new_distance <= catch_distance:
            metrics.caught = True
            self.status.terminated = True
            self.status.outcome = "caught"

        if self.all_notes_collected() and self.exit_zone.contains_circle(self.player.position, self.player.radius):
            metrics.escaped = True
            self.status.terminated = True
            self.status.outcome = "escaped"

        self.time_elapsed += dt
        if self.time_elapsed >= self.max_time and not self.status.terminated:
            metrics.timeout = True
            metrics.stalemate = self.notes_collected == 0
            self.status.truncated = True
            self.status.outcome = "timeout"

        self.last_metrics = metrics
        return metrics
