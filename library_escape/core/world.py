"""Core world simulation that is independent of pygame and RL wrappers."""

from __future__ import annotations

import math
import random
from collections import Counter

from .collectible import Collectible
from .enemy import EnemyState, VisionCone
from .events import StepEvents
from .obstacle import Obstacle
from .physics import move_circle, raycast_distance
from .character import PlayerState
from ..config import load_env_config, load_map_config


COLLECTIBLE_TEXTURES = {
    "note": "note",
    "exam": "exam",
    "coffee": "coffee",
    "freeze": "freeze",
}


class World:
    """Continuous-time world used by both the game client and RL environments."""

    def __init__(self, env_config: dict | None = None, seed: int | None = None) -> None:
        self.env_config = env_config or load_env_config()
        self.map_config = load_map_config(env_config=self.env_config)
        self.width = float(self.map_config["grid_width"])
        self.height = float(self.map_config["grid_height"])
        self.cell_size = int(self.map_config["cell_size"])
        self.physics_hz = int(self.env_config["timing"]["physics_hz"])
        self.physics_dt = 1.0 / self.physics_hz
        self.render_fps = int(self.env_config["timing"]["render_fps"])
        self.rl_frame_skip = int(self.env_config["timing"]["rl_frame_skip"])
        self.max_episode_seconds = float(self.env_config["timing"]["max_episode_seconds"])
        self.stalemate_seconds = float(self.env_config["timing"]["stalemate_seconds"])
        self.interaction_radius = float(self.env_config["world"]["interaction_radius"])
        self.capture_radius = float(self.env_config["world"]["capture_radius"])
        self.enemy_chase_speed_multiplier = float(self.env_config["enemy"]["chase_speed_multiplier"])
        self._rng = random.Random(seed)
        self._base_seed = seed
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)
            self._base_seed = seed

        self.obstacles = [Obstacle.from_dict(payload) for payload in self.map_config["obstacles"]]
        self.enemy_waypoints = [tuple(point) for point in self.map_config["enemy_waypoints"]]
        self.escape_zone = self.map_config["escape_zone"]
        self.collectible_spawn_points = [tuple(point) for point in self.map_config["collectible_spawn_points"]]

        player_spawn = tuple(self.map_config["player_spawn"])
        enemy_spawn = tuple(self.map_config["enemy_spawn"])
        world_cfg = self.env_config["world"]
        enemy_cfg = self.env_config["enemy"]
        randomization_cfg = self.env_config.get("randomization", {})

        player_speed = float(world_cfg["player_speed"])
        enemy_speed = float(world_cfg["enemy_speed"])
        vision_range = float(enemy_cfg["vision_range"])
        vision_angle = float(enemy_cfg["vision_angle_deg"])

        if randomization_cfg.get("enabled", False):
            player_spawn, enemy_spawn = self._sample_agent_spawns(
                min_distance=float(randomization_cfg["min_agent_spawn_distance"])
            )
            player_speed *= self._sample_uniform(randomization_cfg["player_speed_scale_range"])
            enemy_speed *= self._sample_uniform(randomization_cfg["enemy_speed_scale_range"])
            vision_range *= self._sample_uniform(randomization_cfg["vision_range_scale_range"])
            vision_angle += self._rng.uniform(
                -float(randomization_cfg["vision_angle_jitter_deg"]),
                float(randomization_cfg["vision_angle_jitter_deg"]),
            )

        self.player = PlayerState(
            name="player_0",
            x=float(player_spawn[0]) + 0.5,
            y=float(player_spawn[1]) + 0.5,
            radius=float(world_cfg["player_radius"]),
            base_speed=player_speed,
        )
        self.enemy = EnemyState(
            name="enemy_0",
            x=float(enemy_spawn[0]) + 0.5,
            y=float(enemy_spawn[1]) + 0.5,
            radius=float(world_cfg["enemy_radius"]),
            base_speed=enemy_speed,
            vision=VisionCone(
                range_cells=vision_range,
                angle_deg=vision_angle,
            ),
        )

        self.collectibles = self._spawn_collectibles()
        self.events = StepEvents()
        self.time_remaining = self.max_episode_seconds
        self.elapsed_time = 0.0
        self.stalemate_timer = 0.0
        self.terminated = False
        self.truncated = False
        self.outcome: str | None = None
        self.score = Counter()
        self.player_seen = False
        self.last_enemy_action = (0.0, 0.0)
        self.last_player_action = (0.0, 0.0)
        self.last_metrics = self.transition_metrics()

    def _spawn_collectibles(self) -> list[Collectible]:
        counts = self.env_config["collectibles"]
        points = list(self.collectible_spawn_points)
        self._rng.shuffle(points)
        cursor = 0
        spawned: list[Collectible] = []

        for kind in ("note", "exam", "coffee", "freeze"):
            count = int(counts.get(kind, 0))
            for _ in range(count):
                grid_x, grid_y = points[cursor]
                cursor += 1
                spawned.append(
                    Collectible(
                        kind=kind,
                        x=float(grid_x) + 0.5,
                        y=float(grid_y) + 0.5,
                        texture_key=COLLECTIBLE_TEXTURES[kind],
                    )
                )
        return spawned

    def clone_score(self) -> dict[str, int]:
        return dict(self.score)

    def active_collectibles(self, kind: str | None = None) -> list[Collectible]:
        items = [item for item in self.collectibles if item.active]
        if kind is None:
            return items
        return [item for item in items if item.kind == kind]

    def remaining_count(self, kind: str) -> int:
        return sum(1 for item in self.collectibles if item.active and item.kind == kind)

    def nearest_collectible(self, origin: tuple[float, float], kinds: tuple[str, ...] | None = None) -> Collectible | None:
        candidates = self.active_collectibles()
        if kinds is not None:
            candidates = [item for item in candidates if item.kind in kinds]
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.distance_to(origin))

    def can_player_escape(self) -> bool:
        world_cfg = self.env_config["world"]
        notes_ready = (not world_cfg["require_all_notes_to_escape"]) or self.remaining_count("note") == 0
        exams_ready = (not world_cfg["require_all_exams_to_escape"]) or self.remaining_count("exam") == 0
        return bool(notes_ready and exams_ready)

    def required_kinds(self) -> tuple[str, ...]:
        required = ["note"]
        if self.env_config["world"]["require_all_exams_to_escape"]:
            required.append("exam")
        return tuple(required)

    def required_total(self) -> int:
        counts = self.env_config["collectibles"]
        total = 0
        for kind in self.required_kinds():
            total += int(counts["exams"] if kind == "exam" else counts["notes"])
        return total

    def required_remaining(self) -> int:
        return sum(self.remaining_count(kind) for kind in self.required_kinds())

    def objective_progress_fraction(self) -> float:
        total = max(1, self.required_total())
        return 1.0 - (self.required_remaining() / total)

    def player_inside_escape_zone(self) -> bool:
        zone = self.escape_zone
        px, py = self.player.position
        return (
            zone["x"] <= px <= zone["x"] + zone["w"]
            and zone["y"] <= py <= zone["y"] + zone["h"]
        )

    def distance_player_to_escape(self) -> float:
        zone = self.escape_zone
        px, py = self.player.position
        closest_x = min(max(px, zone["x"]), zone["x"] + zone["w"])
        closest_y = min(max(py, zone["y"]), zone["y"] + zone["h"])
        return math.hypot(px - closest_x, py - closest_y)

    def nearest_required_collectible(self) -> Collectible | None:
        return self.nearest_collectible(self.player.position, kinds=self.required_kinds())

    def distance_player_to_target(self) -> float:
        target = self.nearest_required_collectible()
        if target is None:
            return 0.0
        return target.distance_to(self.player.position)

    def player_visible_to_enemy(self) -> bool:
        return self.enemy.vision.sees(
            owner_position=self.enemy.position,
            facing=(self.enemy.facing_x, self.enemy.facing_y),
            target_position=self.player.position,
            obstacles=self.obstacles,
        )

    def distance_between_agents(self) -> float:
        px, py = self.player.position
        ex, ey = self.enemy.position
        return math.hypot(px - ex, py - ey)

    def max_map_distance(self) -> float:
        return math.hypot(self.width, self.height)

    def alert_level(self) -> float:
        visibility = 1.0 if self.player_visible_to_enemy() else 0.0
        distance_pressure = 1.0 - min(1.0, self.distance_between_agents() / max(1.0, self.max_map_distance() * 0.45))
        return min(1.0, max(visibility * 0.7, distance_pressure))

    def enemy_mode(self) -> str:
        if self.enemy.freeze_timer > 0.0:
            return "frozen"
        if self.player_visible_to_enemy():
            return "chase"
        if self.enemy.last_seen_player is not None:
            return "search"
        return "patrol"

    def transition_metrics(self) -> dict[str, float]:
        return {
            "distance_agents": self.distance_between_agents(),
            "distance_player_to_target": self.distance_player_to_target(),
            "distance_player_to_escape": self.distance_player_to_escape(),
            "objective_progress": self.objective_progress_fraction(),
            "player_visible": 1.0 if self.player_visible_to_enemy() else 0.0,
            "can_escape": 1.0 if self.can_player_escape() else 0.0,
            "player_in_escape_zone": 1.0 if self.player_inside_escape_zone() else 0.0,
            "time_remaining": self.time_remaining,
        }

    def raycast_from(self, origin: tuple[float, float], angle_radians: float, max_distance: float) -> float:
        return raycast_distance(
            origin=origin,
            angle_radians=angle_radians,
            max_distance=max_distance,
            obstacles=self.obstacles,
            width=self.width,
            height=self.height,
        )

    def step(
        self,
        player_action: tuple[float, float],
        enemy_action: tuple[float, float],
        frame_skip: int = 1,
    ) -> StepEvents:
        aggregated = StepEvents()
        self.last_metrics = self.transition_metrics()
        for _ in range(frame_skip):
            step_events = self._substep(player_action, enemy_action, dt=self.physics_dt)
            aggregated.merge(step_events)
            if self.terminated or self.truncated:
                break
        self.events = aggregated
        return aggregated

    def _substep(
        self,
        player_action: tuple[float, float],
        enemy_action: tuple[float, float],
        dt: float,
    ) -> StepEvents:
        events = StepEvents()
        if self.terminated or self.truncated:
            return events

        self.last_player_action = player_action
        self.last_enemy_action = enemy_action

        previous_distance = self.distance_between_agents()

        self.player.coffee_timer = max(0.0, self.player.coffee_timer - dt)
        self.enemy.freeze_timer = max(0.0, self.enemy.freeze_timer - dt)

        player_speed_scale = float(self.env_config["world"]["coffee_speed_multiplier"]) if self.player.coffee_timer > 0.0 else 1.0
        enemy_speed_scale = self.enemy_chase_speed_multiplier if self.player_visible_to_enemy() else 1.0

        self.player.apply_action(player_action, speed_scale=player_speed_scale)
        if self.enemy.freeze_timer > 0.0:
            self.enemy.apply_action((0.0, 0.0))
        else:
            self.enemy.apply_action(enemy_action, speed_scale=enemy_speed_scale)

        self.player.x, self.player.y, player_collided = move_circle(
            x=self.player.x,
            y=self.player.y,
            radius=self.player.radius,
            velocity=(self.player.velocity_x, self.player.velocity_y),
            dt=dt,
            width=self.width,
            height=self.height,
            obstacles=self.obstacles,
        )
        self.enemy.x, self.enemy.y, enemy_collided = move_circle(
            x=self.enemy.x,
            y=self.enemy.y,
            radius=self.enemy.radius,
            velocity=(self.enemy.velocity_x, self.enemy.velocity_y),
            dt=dt,
            width=self.width,
            height=self.height,
            obstacles=self.obstacles,
        )

        if player_collided:
            events.player_wall_hits += 1
        if enemy_collided:
            events.enemy_wall_hits += 1

        if self.player_visible_to_enemy():
            self.player_seen = True
            self.enemy.last_seen_player = self.player.position
            events.visible_steps += 1
            events.progress_made = True
            self.stalemate_timer = 0.0

        self._collect_nearby_items(events)

        current_distance = self.distance_between_agents()
        events.distance_delta += previous_distance - current_distance
        if events.distance_delta > 1e-3:
            events.progress_made = True
        if abs(self.player.velocity_x) > 1e-6 or abs(self.player.velocity_y) > 1e-6:
            events.progress_made = True
        if abs(self.enemy.velocity_x) > 1e-6 or abs(self.enemy.velocity_y) > 1e-6:
            events.progress_made = True

        if current_distance <= self.capture_radius:
            self.terminated = True
            self.outcome = "caught"
            events.player_caught = True
            events.progress_made = True

        if (not self.terminated) and self.can_player_escape() and self.player_inside_escape_zone():
            self.terminated = True
            self.outcome = "escaped"
            events.player_escaped = True
            events.progress_made = True

        self.time_remaining = max(0.0, self.time_remaining - dt)
        self.elapsed_time += dt
        if self.time_remaining <= 0.0 and not self.terminated:
            self.truncated = True
            self.outcome = "time_expired"
            events.time_expired = True

        if events.progress_made:
            self.stalemate_timer = 0.0
        else:
            self.stalemate_timer += dt

        if self.stalemate_timer >= self.stalemate_seconds and not self.terminated and not self.truncated:
            self.truncated = True
            self.outcome = "stalemate"
            events.stalemate = True

        return events

    def _collect_nearby_items(self, events: StepEvents) -> None:
        for collectible in self.collectibles:
            if not collectible.active:
                continue
            if collectible.distance_to(self.player.position) > self.interaction_radius:
                continue
            collectible.active = False
            self.score[collectible.kind] += 1
            events.collected[collectible.kind] += 1
            events.progress_made = True
            self.stalemate_timer = 0.0

            if collectible.kind == "coffee":
                self.player.coffee_timer = float(self.env_config["world"]["coffee_duration_seconds"])
            elif collectible.kind == "freeze":
                self.enemy.freeze_timer = float(self.env_config["world"]["freeze_duration_seconds"])

    def info(self) -> dict[str, object]:
        return {
            "time_remaining": self.time_remaining,
            "elapsed_time": self.elapsed_time,
            "outcome": self.outcome,
            "score": self.clone_score(),
            "notes_remaining": self.remaining_count("note"),
            "exams_remaining": self.remaining_count("exam"),
            "player_seen": self.player_seen,
            "can_escape": self.can_player_escape(),
            "enemy_mode": self.enemy_mode(),
            "alert_level": self.alert_level(),
            "distance_to_enemy": self.distance_between_agents(),
            "distance_to_exit": self.distance_player_to_escape(),
            "objective_progress": self.objective_progress_fraction(),
        }

    def _sample_uniform(self, bounds: list[float] | tuple[float, float]) -> float:
        lower, upper = bounds
        return self._rng.uniform(float(lower), float(upper))

    def _free_spawn_cells(self) -> list[tuple[float, float]]:
        candidates = [tuple(self.map_config["player_spawn"]), tuple(self.map_config["enemy_spawn"])]
        candidates.extend(self.collectible_spawn_points)
        free_cells: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for cell in candidates:
            if cell in seen:
                continue
            seen.add(cell)
            cx = float(cell[0]) + 0.5
            cy = float(cell[1]) + 0.5
            if any(obstacle.contains_point(cx, cy) for obstacle in self.obstacles):
                continue
            free_cells.append((float(cell[0]), float(cell[1])))
        return free_cells

    def _sample_agent_spawns(self, min_distance: float) -> tuple[tuple[float, float], tuple[float, float]]:
        candidates = self._free_spawn_cells()
        base_player = tuple(self.map_config["player_spawn"])
        base_enemy = tuple(self.map_config["enemy_spawn"])
        if len(candidates) < 2:
            return base_player, base_enemy

        player_spawn = self._rng.choice(candidates)
        far_enough = [
            cell for cell in candidates
            if math.hypot(cell[0] - player_spawn[0], cell[1] - player_spawn[1]) >= min_distance
        ]
        if not far_enough:
            return base_player, base_enemy
        enemy_spawn = self._rng.choice(far_enough)
        return player_spawn, enemy_spawn
