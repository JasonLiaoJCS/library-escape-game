"""Core world simulation that is independent of pygame and RL wrappers."""

from __future__ import annotations

import math
import random
import heapq
from collections import Counter

from .collectible import Collectible
from .enemy import EnemyState, VisionCone
from .events import StepEvents
from .obstacle import Obstacle
from .physics import move_circle, raycast_distance
from .character import PlayerState
from ..config import load_env_config, load_map_config
from ..game_modes import normalize_game_mode


COLLECTIBLE_TEXTURES = {
    "note": "note",
    "exam": "exam",
    "coffee": "coffee",
    "freeze": "freeze",
}

COLLECTIBLE_COUNT_KEYS = {
    "note": "notes",
    "exam": "exams",
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
        self._refresh_runtime_config()
        self._rng = random.Random(seed)
        self._base_seed = seed
        self.reset(seed=seed)

    def _refresh_runtime_config(self) -> None:
        world_cfg = self.env_config["world"]
        enemy_cfg = self.env_config["enemy"]
        enemy_team_cfg = self.env_config.get("enemy_team", {})
        configured_mode = world_cfg.get("game_mode", world_cfg.get("ruleset", "escape"))
        self.game_mode = normalize_game_mode(str(configured_mode))
        self.ruleset = self.game_mode
        world_cfg["game_mode"] = self.game_mode
        world_cfg["ruleset"] = self.game_mode
        self.manual_collect_required = bool(world_cfg.get("manual_collect_required", False))
        self.collection_hold_seconds = float(world_cfg.get("collection_hold_seconds", 0.0))
        self.startup_grace_seconds = float(world_cfg.get("startup_grace_seconds", 0.0))
        self.coffee_effect = str(world_cfg.get("coffee_effect", "movement")).lower()
        self.coffee_collection_multiplier = float(
            world_cfg.get("coffee_collection_multiplier", world_cfg.get("coffee_speed_multiplier", 1.0))
        )
        self.enemy_detect_penalty_seconds = float(enemy_cfg.get("detect_penalty_seconds", 0.0))
        self.enemy_detect_pause_seconds = float(enemy_cfg.get("detect_pause_seconds", 0.0))
        self.enemy_chase_when_visible = bool(enemy_cfg.get("chase_when_visible", True))
        self.enemy_max_turn_rate_deg = float(enemy_cfg.get("max_turn_rate_deg_per_sec", 0.0))
        self.support_enemy_count = int(enemy_team_cfg.get("support_count", 0))
        self.support_speed_scale = float(enemy_team_cfg.get("support_speed_scale", 1.0))
        self.support_vision_range_scale = float(enemy_team_cfg.get("support_vision_range_scale", 1.0))
        self.support_vision_angle_scale = float(enemy_team_cfg.get("support_vision_angle_scale", 1.0))
        self.shared_last_seen = bool(enemy_team_cfg.get("shared_last_seen", True))
        self.team_detection_cooldown_seconds = float(enemy_team_cfg.get("shared_detection_cooldown_seconds", 1.0))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)
            self._base_seed = seed
        self._refresh_runtime_config()

        self.obstacles = [Obstacle.from_dict(payload) for payload in self.map_config["obstacles"]]
        self.enemy_waypoints = [tuple(point) for point in self.map_config["enemy_waypoints"]]
        self.support_enemy_spawns = [tuple(point) for point in self.map_config.get("support_enemy_spawns", [])]
        self.support_enemy_routes = [
            tuple(tuple(point) for point in route)
            for route in self.map_config.get("support_enemy_routes", [])
        ]
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
            if "support_count_range" in randomization_cfg:
                low, high = randomization_cfg["support_count_range"]
                self.support_enemy_count = self._rng.randint(int(low), int(high))

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
            patrol_route=tuple((float(x) + 0.5, float(y) + 0.5) for x, y in self.enemy_waypoints),
            is_primary=True,
        )
        self.support_enemies = self._spawn_support_enemies(
            base_speed=enemy_speed,
            base_range=vision_range,
            base_angle=vision_angle,
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
        self.team_detection_cooldown_timer = 0.0
        self.startup_grace_timer = self.startup_grace_seconds
        self.collection_target_index: int | None = None
        self.collection_progress = 0.0
        self.classic_detection_active = False
        self.last_metrics = self.transition_metrics()

    def _spawn_collectibles(self) -> list[Collectible]:
        counts = self.env_config["collectibles"]
        points = list(self.collectible_spawn_points)
        self._rng.shuffle(points)
        cursor = 0
        spawned: list[Collectible] = []

        for kind in ("note", "exam", "coffee", "freeze"):
            count = int(counts.get(COLLECTIBLE_COUNT_KEYS[kind], 0))
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

    def _spawn_support_enemies(self, base_speed: float, base_range: float, base_angle: float) -> list[EnemyState]:
        support_enemies: list[EnemyState] = []
        limit = min(self.support_enemy_count, len(self.support_enemy_spawns))
        for index in range(limit):
            spawn = self.support_enemy_spawns[index]
            route = self.support_enemy_routes[index] if index < len(self.support_enemy_routes) else tuple(self.enemy_waypoints)
            support_enemies.append(
                EnemyState(
                    name=f"enemy_{index + 1}",
                    x=float(spawn[0]) + 0.5,
                    y=float(spawn[1]) + 0.5,
                    radius=float(self.env_config["world"]["enemy_radius"]),
                    base_speed=base_speed * self.support_speed_scale,
                    vision=VisionCone(
                        range_cells=base_range * self.support_vision_range_scale,
                        angle_deg=base_angle * self.support_vision_angle_scale,
                    ),
                    patrol_route=tuple((float(x) + 0.5, float(y) + 0.5) for x, y in route),
                    is_primary=False,
                )
            )
        return support_enemies

    def all_enemies(self) -> list[EnemyState]:
        return [self.enemy, *getattr(self, "support_enemies", [])]

    def nearest_enemy(self) -> EnemyState:
        return min(self.all_enemies(), key=lambda enemy: math.hypot(self.player.x - enemy.x, self.player.y - enemy.y))

    def primary_enemy_sees_player(self) -> bool:
        return self._enemy_sees_player(self.enemy)

    def visible_enemies(self) -> list[EnemyState]:
        return [enemy for enemy in self.all_enemies() if self._enemy_sees_player(enemy)]

    def visible_enemy_count(self) -> int:
        return len(self.visible_enemies())

    def support_enemy_ratio(self) -> float:
        total_enemies = max(1, len(self.all_enemies()))
        return len(getattr(self, "support_enemies", [])) / total_enemies

    def nearest_enemy_distance_to_escape(self) -> float:
        return min(self._distance_enemy_to_escape(enemy) for enemy in self.all_enemies())

    def primary_enemy_distance(self) -> float:
        px, py = self.player.position
        return math.hypot(px - self.enemy.x, py - self.enemy.y)

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
        if self.is_collection_mode():
            return False
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

    def max_score_value(self) -> int:
        counts = self.env_config["collectibles"]
        return int(counts["notes"] * 8 + counts["exams"] * 12)

    def score_progress_fraction(self) -> float:
        return float(self.score_value() / max(1, self.max_score_value()))

    def team_detection_cooldown_fraction(self) -> float:
        return min(1.0, self.team_detection_cooldown_timer / max(1e-6, self.team_detection_cooldown_seconds))

    def enemy_pause_fraction(self, enemy: EnemyState) -> float:
        return min(1.0, enemy.detection_pause_timer / max(1e-6, self.enemy_detect_pause_seconds))

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

    def escape_center(self) -> tuple[float, float]:
        zone = self.escape_zone
        return zone["x"] + (zone["w"] / 2.0), zone["y"] + (zone["h"] / 2.0)

    def nearest_required_collectible(self) -> Collectible | None:
        return self.nearest_collectible(self.player.position, kinds=self.required_kinds())

    def current_collection_goal_collectible(self) -> Collectible | None:
        if not self.is_collection_mode():
            return None
        if self.collection_target_index is not None:
            if 0 <= self.collection_target_index < len(self.collectibles):
                collectible = self.collectibles[self.collection_target_index]
                if collectible.active:
                    return collectible

        candidates = self.active_collectibles()
        if not candidates:
            return None

        base_values = {
            "note": 1.00,
            "exam": 1.55,
            "coffee": 0.42,
            "freeze": 0.50,
        }
        visible_pressure = 1.0 if self.player_visible_to_enemy() else 0.0
        coffee_bias = 0.14 if self.player.coffee_timer <= 1e-6 else -0.10
        freeze_bias = 0.16 if visible_pressure > 0.0 else 0.02

        def score(item: Collectible) -> float:
            distance = max(0.35, item.distance_to(self.player.position))
            value = float(base_values.get(item.kind, 0.25))
            if item.kind == "coffee":
                value += coffee_bias
            elif item.kind == "freeze":
                value += freeze_bias
            return value / distance

        return max(candidates, key=score)

    def current_player_goal_position(self) -> tuple[float, float] | None:
        if self.can_player_escape():
            return self.escape_center()
        target = self.current_collection_goal_collectible() if self.is_collection_mode() else self.nearest_required_collectible()
        if target is None:
            return None
        return target.x, target.y

    def distance_player_to_target(self) -> float:
        goal_position = self.current_player_goal_position()
        if goal_position is None:
            return 0.0
        return self._distance_between_positions(self.player.position, goal_position)

    def distance_player_to_collectible_kind(self, kinds: tuple[str, ...]) -> float:
        target = self.nearest_collectible(self.player.position, kinds=kinds)
        if target is None:
            return 0.0
        return target.distance_to(self.player.position)

    def player_visible_to_enemy(self) -> bool:
        return any(self._enemy_sees_player(enemy) for enemy in self.all_enemies())

    def distance_between_agents(self) -> float:
        px, py = self.player.position
        return min(math.hypot(px - enemy.x, py - enemy.y) for enemy in self.all_enemies())

    def max_map_distance(self) -> float:
        return math.hypot(self.width, self.height)

    def is_classic_ruleset(self) -> bool:
        return self.is_collection_mode()

    def is_collection_mode(self) -> bool:
        return self.game_mode == "collection"

    def is_escape_mode(self) -> bool:
        return self.game_mode == "escape"

    def should_render_escape_zone(self) -> bool:
        return self.is_escape_mode()

    def score_value(self) -> int:
        return int(self.score["note"] * 8 + self.score["exam"] * 12)

    def collection_speed_multiplier(self) -> float:
        if self.player.coffee_timer <= 0.0:
            return 1.0
        if self.coffee_effect == "collection_speed":
            return self.coffee_collection_multiplier
        return 1.0

    def enemy_should_chase(self) -> bool:
        return self.enemy_chase_when_visible and self.is_escape_mode()

    def max_enemy_freeze_timer(self) -> float:
        return max(enemy.freeze_timer for enemy in self.all_enemies())

    def alert_level(self) -> float:
        if self.player_visible_to_enemy() or any(enemy.detection_pause_timer > 0.0 for enemy in self.all_enemies()):
            return 1.0
        return 0.0

    def enemy_mode(self) -> str:
        if all(enemy.freeze_timer > 0.0 for enemy in self.all_enemies()):
            return "frozen"
        if any(enemy.detection_pause_timer > 0.0 for enemy in self.all_enemies()):
            return "alerted"
        if self.player_visible_to_enemy() and self.enemy_should_chase():
            return "chase"
        if self.player_visible_to_enemy():
            return "spotted"
        if any(enemy.last_seen_player is not None for enemy in self.all_enemies()):
            return "search"
        return "patrol"

    def transition_metrics(self) -> dict[str, float]:
        visible_count = self.visible_enemy_count()
        total_enemies = max(1, len(self.all_enemies()))
        max_distance = max(1.0, self.max_map_distance())
        distance_agents = self.distance_between_agents()
        distance_primary = self.primary_enemy_distance()
        distance_player_to_escape = self.distance_player_to_escape()
        distance_enemy_to_escape = self.nearest_enemy_distance_to_escape()
        player_goal_position = self.current_player_goal_position()
        distance_primary_to_player_goal = (
            self._distance_between_positions(self.enemy.position, player_goal_position)
            if player_goal_position is not None
            else 0.0
        )
        exit_lead = (distance_enemy_to_escape - distance_player_to_escape) / max_distance
        exit_lead = max(-1.0, min(1.0, exit_lead))
        return {
            "distance_agents": distance_agents,
            "distance_primary_enemy": distance_primary,
            "distance_primary_to_player_goal": distance_primary_to_player_goal,
            "distance_player_to_target": self.distance_player_to_target(),
            "distance_player_to_note": self.distance_player_to_collectible_kind(("note",)),
            "distance_player_to_exam": self.distance_player_to_collectible_kind(("exam",)),
            "distance_player_to_powerup": self.distance_player_to_collectible_kind(("coffee", "freeze")),
            "distance_player_to_escape": distance_player_to_escape,
            "distance_enemy_to_escape": distance_enemy_to_escape,
            "objective_progress": self.objective_progress_fraction(),
            "score_progress": self.score_progress_fraction(),
            "collection_progress": self.collection_progress,
            "player_visible": 1.0 if self.player_visible_to_enemy() else 0.0,
            "player_visible_primary": 1.0 if self.primary_enemy_sees_player() else 0.0,
            "visible_enemy_count": float(visible_count),
            "visible_enemy_ratio": float(visible_count / total_enemies),
            "enemy_count": float(total_enemies),
            "enemy_count_ratio": float(total_enemies / max(1, 1 + len(self.support_enemy_spawns))),
            "support_enemy_count": float(len(getattr(self, "support_enemies", []))),
            "support_enemy_ratio": float(len(getattr(self, "support_enemies", [])) / total_enemies),
            "threat_margin": min(1.0, distance_agents / max_distance),
            "primary_threat_margin": min(1.0, distance_primary / max_distance),
            "exit_lead": exit_lead,
            "can_escape": 1.0 if self.can_player_escape() else 0.0,
            "player_in_escape_zone": 1.0 if self.player_inside_escape_zone() else 0.0,
            "notes_remaining_ratio": self.remaining_count("note") / max(1, int(self.env_config["collectibles"]["notes"])),
            "exams_remaining_ratio": self.remaining_count("exam") / max(1, int(self.env_config["collectibles"]["exams"])),
            "team_detection_cooldown": self.team_detection_cooldown_fraction(),
            "time_remaining": self.time_remaining,
        }

    def _enemy_sees_player(self, enemy: EnemyState) -> bool:
        if self.startup_grace_timer > 1e-6:
            return False
        return enemy.vision.sees(
            owner_position=enemy.position,
            facing=(enemy.facing_x, enemy.facing_y),
            target_position=self.player.position,
            obstacles=self.obstacles,
        )

    def _distance_enemy_to_escape(self, enemy: EnemyState) -> float:
        zone = self.escape_zone
        ex, ey = enemy.position
        closest_x = min(max(ex, zone["x"]), zone["x"] + zone["w"])
        closest_y = min(max(ey, zone["y"]), zone["y"] + zone["h"])
        return math.hypot(ex - closest_x, ey - closest_y)

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
        player_collect: bool | None = None,
    ) -> StepEvents:
        aggregated = StepEvents()
        self.last_metrics = self.transition_metrics()
        collect_pressed = (not self.manual_collect_required) if player_collect is None else bool(player_collect)
        player_start = self.player.position
        primary_enemy_start = self.enemy.position
        for _ in range(frame_skip):
            step_events = self._substep(player_action, enemy_action, dt=self.physics_dt, player_collect=collect_pressed)
            aggregated.merge(step_events)
            if self.terminated or self.truncated:
                break
        aggregated.player_net_displacement = self._distance_between_positions(player_start, self.player.position)
        aggregated.primary_enemy_net_displacement = self._distance_between_positions(primary_enemy_start, self.enemy.position)
        self.events = aggregated
        return aggregated

    def _substep(
        self,
        player_action: tuple[float, float],
        enemy_action: tuple[float, float],
        dt: float,
        player_collect: bool,
    ) -> StepEvents:
        events = StepEvents()
        if self.terminated or self.truncated:
            return events

        self.last_player_action = player_action
        self.last_enemy_action = enemy_action

        previous_distance = self.distance_between_agents()
        previous_primary_distance = self.primary_enemy_distance()
        player_start = self.player.position
        primary_enemy_start = self.enemy.position
        enemy_starts = {enemy.name: enemy.position for enemy in self.all_enemies()}

        self.player.coffee_timer = max(0.0, self.player.coffee_timer - dt)
        self.team_detection_cooldown_timer = max(0.0, self.team_detection_cooldown_timer - dt)
        self.startup_grace_timer = max(0.0, self.startup_grace_timer - dt)
        for enemy in self.all_enemies():
            enemy.freeze_timer = max(0.0, enemy.freeze_timer - dt)
            enemy.detection_pause_timer = max(0.0, enemy.detection_pause_timer - dt)
        classic_detector_before_move = self._classic_detection_enemy() if self.is_classic_ruleset() else None
        team_visible_before_move = self.player_visible_to_enemy()

        if self.player.coffee_timer > 0.0 and self.coffee_effect == "movement":
            player_speed_scale = float(self.env_config["world"]["coffee_speed_multiplier"])
        else:
            player_speed_scale = 1.0

        self.player.apply_action(player_action, speed_scale=player_speed_scale)
        enemy_turn_rate = self.enemy_max_turn_rate_deg if self.enemy_max_turn_rate_deg > 0.0 else None
        if self._should_pause_enemy(self.enemy, pause_for_detection=(classic_detector_before_move is self.enemy)):
            self.enemy.apply_action((0.0, 0.0))
        else:
            self.enemy.apply_action(
                enemy_action,
                speed_scale=self._enemy_speed_scale(self.enemy, team_visible_before_move),
                dt=dt,
                max_turn_rate_deg=enemy_turn_rate,
            )
        for support_enemy in self.support_enemies:
            visible_before = self._enemy_sees_player(support_enemy)
            if self._should_pause_enemy(support_enemy, pause_for_detection=(classic_detector_before_move is support_enemy)):
                support_enemy.apply_action((0.0, 0.0))
            else:
                support_action = self._support_enemy_action(support_enemy)
                support_enemy.apply_action(
                    support_action,
                    speed_scale=self._enemy_speed_scale(support_enemy, visible_before or support_enemy.last_seen_player is not None),
                    dt=dt,
                    max_turn_rate_deg=enemy_turn_rate,
                )

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
        enemy_collided = False
        for enemy in self.all_enemies():
            enemy.x, enemy.y, collided = move_circle(
                x=enemy.x,
                y=enemy.y,
                radius=enemy.radius,
                velocity=(enemy.velocity_x, enemy.velocity_y),
                dt=dt,
                width=self.width,
                height=self.height,
                obstacles=self.obstacles,
            )
            enemy_collided = enemy_collided or collided

        if player_collided:
            events.player_wall_hits += 1
        if enemy_collided:
            events.enemy_wall_hits += 1

        player_step_displacement = self._distance_between_positions(player_start, self.player.position)
        primary_enemy_step_displacement = self._distance_between_positions(primary_enemy_start, self.enemy.position)
        enemy_step_displacements = {
            enemy.name: self._distance_between_positions(enemy_starts[enemy.name], enemy.position)
            for enemy in self.all_enemies()
        }
        movement_progress_threshold = max(0.01, min(self.player.base_speed, self.enemy.base_speed) * dt * 0.30)
        events.player_path_length += player_step_displacement
        events.primary_enemy_path_length += primary_enemy_step_displacement
        events.player_net_displacement = player_step_displacement
        events.primary_enemy_net_displacement = primary_enemy_step_displacement

        visible_enemies = self.visible_enemies()
        if visible_enemies:
            self.player_seen = True
            events.visible_steps += 1
            events.visible_enemy_count += len(visible_enemies)
            if any(enemy.is_primary for enemy in visible_enemies):
                events.primary_visible_steps += 1
            events.support_visible_steps += sum(1 for enemy in visible_enemies if not enemy.is_primary)
            events.progress_made = True
            self.stalemate_timer = 0.0
            if self.is_classic_ruleset():
                self._trigger_classic_detection(
                    classic_detector_before_move or self._select_classic_detector(visible_enemies),
                    events,
                )
            else:
                if self.shared_last_seen:
                    for enemy in self.all_enemies():
                        enemy.last_seen_player = self.player.position
                else:
                    for enemy in visible_enemies:
                        enemy.last_seen_player = self.player.position
        elif self.is_classic_ruleset():
            self.classic_detection_active = False

        self._collect_nearby_items(events, dt=dt, player_collect=player_collect)

        current_distance = self.distance_between_agents()
        current_primary_distance = self.primary_enemy_distance()
        events.distance_delta += previous_distance - current_distance
        events.primary_distance_delta += previous_primary_distance - current_primary_distance
        if events.distance_delta > 1e-3:
            events.progress_made = True
        if events.primary_distance_delta > 1e-3:
            events.progress_made = True
        if player_step_displacement > movement_progress_threshold:
            events.progress_made = True
        if any(displacement > movement_progress_threshold for displacement in enemy_step_displacements.values()):
            events.progress_made = True

        if not self.is_classic_ruleset():
            if any(self._distance_player_to_enemy(enemy) <= self.capture_radius for enemy in self.all_enemies()):
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

    def _trigger_classic_detection(self, detector: EnemyState | None, events: StepEvents) -> None:
        if detector is None:
            self.classic_detection_active = False
            return
        if self.classic_detection_active or self.team_detection_cooldown_timer > 1e-6:
            return
        detector.last_seen_player = None
        detector.detection_pause_timer = max(detector.detection_pause_timer, self.enemy_detect_pause_seconds)
        if self.enemy_detect_penalty_seconds > 0.0:
            self.time_remaining = max(0.0, self.time_remaining - self.enemy_detect_penalty_seconds)
        self.team_detection_cooldown_timer = self.team_detection_cooldown_seconds
        self.classic_detection_active = True
        events.detection_events += 1
        events.progress_made = True

    def _enemy_speed_scale(self, enemy: EnemyState, chasing: bool) -> float:
        if enemy.freeze_timer > 0.0:
            return 0.0
        if self.enemy_should_chase() and chasing:
            return self.enemy_chase_speed_multiplier if enemy.is_primary else max(1.0, self.enemy_chase_speed_multiplier * 0.96)
        return 1.0

    def _should_pause_enemy(self, enemy: EnemyState, pause_for_detection: bool = False) -> bool:
        return enemy.freeze_timer > 0.0 or enemy.detection_pause_timer > 0.0 or pause_for_detection

    def _select_classic_detector(self, visible_enemies: list[EnemyState]) -> EnemyState | None:
        if not visible_enemies:
            return None
        primary_candidates = [enemy for enemy in visible_enemies if enemy.is_primary]
        candidates = primary_candidates or visible_enemies
        return min(candidates, key=lambda enemy: (self._distance_player_to_enemy(enemy), enemy.name))

    def _classic_detection_enemy(self) -> EnemyState | None:
        return self._select_classic_detector(self.visible_enemies())

    def _support_enemy_action(self, enemy: EnemyState) -> tuple[float, float]:
        if self.enemy_should_chase():
            intercept_target = self._support_enemy_intercept_target(enemy)
            if self._enemy_sees_player(enemy):
                enemy.last_seen_player = self.player.position
                return self.steer_towards_position(enemy.position, intercept_target)
            if enemy.last_seen_player is not None:
                if self._distance_between_positions(enemy.position, enemy.last_seen_player) <= 0.30:
                    enemy.last_seen_player = None
                else:
                    search_target = self._support_enemy_search_target(enemy, enemy.last_seen_player)
                    return self.steer_towards_position(enemy.position, search_target)

        route = enemy.patrol_route or tuple((float(x) + 0.5, float(y) + 0.5) for x, y in self.enemy_waypoints)
        if not route:
            return 0.0, 0.0
        waypoint = route[enemy.patrol_index % len(route)]
        patrol_target = self._support_enemy_patrol_target(enemy, waypoint)
        if self.navigation_goal_reached(enemy.position, patrol_target, threshold=0.35):
            enemy.patrol_index = (enemy.patrol_index + 1) % len(route)
            waypoint = route[enemy.patrol_index]
            patrol_target = self._support_enemy_patrol_target(enemy, waypoint)
        return self.steer_towards_position(enemy.position, patrol_target)

    def _support_enemy_intercept_target(self, enemy: EnemyState) -> tuple[float, float]:
        slot_index = max(0, [ally.name for ally in self.support_enemies].index(enemy.name) if enemy in self.support_enemies else 0)
        if self.can_player_escape():
            if slot_index == 0:
                return self._exit_guard_position()
            if slot_index == 1:
                return self._midpoint_target(self.player.position, self._exit_guard_position(), ratio=0.55)

        px, py = self.player.position
        base_angle = math.atan2(py - enemy.y, px - enemy.x)
        slot_offsets = (math.pi / 3.0, -math.pi / 3.0, math.pi * 0.85, -math.pi * 0.85)
        offset = slot_offsets[slot_index % len(slot_offsets)]
        radius = 2.2 if self.is_escape_mode() else 1.7
        target = (
            px + math.cos(base_angle + offset) * radius,
            py + math.sin(base_angle + offset) * radius,
        )
        return self._clamp_free_target(target)

    def _support_enemy_search_target(self, enemy: EnemyState, last_seen: tuple[float, float]) -> tuple[float, float]:
        slot_index = max(0, [ally.name for ally in self.support_enemies].index(enemy.name) if enemy in self.support_enemies else 0)
        spread_offsets = ((1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5))
        dx, dy = spread_offsets[slot_index % len(spread_offsets)]
        target = (last_seen[0] + dx, last_seen[1] + dy)
        if self.can_player_escape() and slot_index == 0:
            return self._exit_guard_position()
        return self._clamp_free_target(target)

    def _support_enemy_patrol_target(self, enemy: EnemyState, waypoint: tuple[float, float]) -> tuple[float, float]:
        if not (self.is_escape_mode() or self.is_collection_mode()):
            return waypoint

        goal_position = self.current_player_goal_position()
        if goal_position is None:
            return waypoint

        if self.is_collection_mode():
            score_progress = self.score_progress_fraction()
            objective_bias = 0.34 + (0.12 * score_progress)
            patrol_target = self._midpoint_target(waypoint, goal_position, ratio=objective_bias)
            if self.player_visible_to_enemy():
                patrol_target = self._midpoint_target(patrol_target, self.player.position, ratio=0.18)
            return patrol_target

        # Escape-mode support enemies should not be trapped in tiny local patrol loops.
        # We keep the original waypoint skeleton, but bias each waypoint toward the
        # player's live objective so support pressure sweeps toward notes early and
        # toward the exit lane later.
        if self.can_player_escape():
            return self._midpoint_target(waypoint, self._exit_guard_position(), ratio=0.78)

        progress = self.objective_progress_fraction()
        objective_bias = 0.48 + (0.18 * progress)
        patrol_target = self._midpoint_target(waypoint, goal_position, ratio=objective_bias)
        if progress >= 0.45:
            exit_bias = min(0.32, 0.10 + (0.22 * progress))
            patrol_target = self._midpoint_target(patrol_target, self._exit_guard_position(), ratio=exit_bias)
        return patrol_target

    def _exit_guard_position(self) -> tuple[float, float]:
        return self._clamp_free_target(self.escape_center())

    def _midpoint_target(self, a: tuple[float, float], b: tuple[float, float], ratio: float = 0.5) -> tuple[float, float]:
        return self._clamp_free_target((a[0] * (1.0 - ratio) + b[0] * ratio, a[1] * (1.0 - ratio) + b[1] * ratio))

    def _clamp_free_target(self, target: tuple[float, float]) -> tuple[float, float]:
        gx = min(max(int(target[0]), 0), int(self.width) - 1)
        gy = min(max(int(target[1]), 0), int(self.height) - 1)
        candidates = self._navigable_goal_cells((gx, gy))
        if candidates:
            candidate = min(candidates, key=lambda cell: self._grid_manhattan((gx, gy), cell))
            return candidate[0] + 0.5, candidate[1] + 0.5
        return min(max(target[0], 0.5), self.width - 0.5), min(max(target[1], 0.5), self.height - 0.5)

    def _vector_to(self, enemy: EnemyState, target: tuple[float, float]) -> tuple[float, float]:
        dx = target[0] - enemy.x
        dy = target[1] - enemy.y
        length = math.hypot(dx, dy)
        if length <= 1e-8:
            return 0.0, 0.0
        return dx / length, dy / length

    def _distance_between_positions(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _distance_player_to_enemy(self, enemy: EnemyState) -> float:
        return math.hypot(self.player.x - enemy.x, self.player.y - enemy.y)

    def steer_towards_position(self, origin: tuple[float, float], target: tuple[float, float]) -> tuple[float, float]:
        start_cell = (int(origin[0]), int(origin[1]))
        goal_cell = (int(target[0]), int(target[1]))
        goal_candidates = self._navigable_goal_cells(goal_cell)
        best_path: list[tuple[int, int]] | None = None
        best_candidate: tuple[int, int] | None = None
        best_cost: tuple[int, int] | None = None
        for candidate in goal_candidates:
            path = self._a_star_cells(start_cell, candidate)
            if candidate != start_cell and len(path) <= 1:
                continue
            cost = (len(path), self._grid_manhattan(goal_cell, candidate))
            if best_cost is not None and cost >= best_cost:
                continue
            best_path = path
            best_candidate = candidate
            best_cost = cost

        if best_path is not None and len(best_path) >= 2:
            next_cell = best_path[1]
            return self._vector_between_positions(origin, (next_cell[0] + 0.5, next_cell[1] + 0.5))
        if best_candidate is not None:
            return self._vector_between_positions(origin, (best_candidate[0] + 0.5, best_candidate[1] + 0.5))
        return self._vector_between_positions(origin, target)

    def _a_star_cells(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        frontier: list[tuple[int, tuple[int, int]]] = []
        heapq.heappush(frontier, (0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], int] = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for neighbor in self._grid_neighbors(current):
                new_cost = cost_so_far[current] + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._grid_manhattan(goal, neighbor)
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

    def _grid_neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        candidates = [
            (cell[0] + 1, cell[1]),
            (cell[0] - 1, cell[1]),
            (cell[0], cell[1] + 1),
            (cell[0], cell[1] - 1),
        ]
        return [candidate for candidate in candidates if self._is_free_cell(candidate)]

    def _grid_manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def navigation_goal_reached(
        self,
        origin: tuple[float, float],
        target: tuple[float, float],
        threshold: float = 0.30,
    ) -> bool:
        goal_cell = (int(target[0]), int(target[1]))
        for candidate in self._navigable_goal_cells(goal_cell):
            center = (candidate[0] + 0.5, candidate[1] + 0.5)
            if self._distance_between_positions(origin, center) <= threshold:
                return True
        return self._distance_between_positions(origin, target) <= threshold

    def _navigable_goal_cells(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        if self._is_free_cell(cell):
            return [cell]

        obstacle = self._obstacle_covering_cell(cell)
        if obstacle is None:
            candidates = [candidate for candidate in self._adjacent_cells(cell) if self._is_free_cell(candidate)]
            return candidates or [cell]

        perimeter: set[tuple[int, int]] = set()
        x0 = int(obstacle.x)
        y0 = int(obstacle.y)
        x1 = int(obstacle.x + obstacle.w) - 1
        y1 = int(obstacle.y + obstacle.h) - 1

        for gx in range(x0, x1 + 1):
            perimeter.add((gx, y0 - 1))
            perimeter.add((gx, y1 + 1))
        for gy in range(y0, y1 + 1):
            perimeter.add((x0 - 1, gy))
            perimeter.add((x1 + 1, gy))

        candidates = [candidate for candidate in perimeter if self._is_free_cell(candidate)]
        candidates.sort(key=lambda candidate: (self._grid_manhattan(cell, candidate), candidate[1], candidate[0]))
        return candidates or [cell]

    def _vector_between_positions(self, origin: tuple[float, float], target: tuple[float, float]) -> tuple[float, float]:
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        length = math.hypot(dx, dy)
        if length <= 1e-8:
            return 0.0, 0.0
        return dx / length, dy / length

    def _collect_nearby_items(self, events: StepEvents, dt: float, player_collect: bool) -> None:
        if self.manual_collect_required:
            self._update_manual_collection(events, dt=dt, player_collect=player_collect)
            return

        target_index = self._nearest_collectible_in_range()
        if target_index is None:
            self.collection_target_index = None
            self.collection_progress = 0.0
            return
        if self.collection_target_index != target_index:
            self.collection_target_index = target_index
            self.collection_progress = 0.0

        hold_seconds = max(1e-6, self.collection_hold_seconds)
        self.collection_progress = min(
            1.0,
            self.collection_progress + (dt * self.collection_speed_multiplier() / hold_seconds),
        )
        events.progress_made = True

        collectible = self.collectibles[target_index]
        if self.collection_progress >= 1.0 and collectible.active:
            self._collect_item(collectible, events)
            self.collection_target_index = None
            self.collection_progress = 0.0

    def _update_manual_collection(self, events: StepEvents, dt: float, player_collect: bool) -> None:
        if not player_collect:
            self.collection_target_index = None
            self.collection_progress = 0.0
            return

        target_index = self._nearest_collectible_in_range()
        if target_index is None:
            self.collection_target_index = None
            self.collection_progress = 0.0
            return

        if self.collection_target_index != target_index:
            self.collection_target_index = target_index
            self.collection_progress = 0.0

        hold_seconds = max(1e-6, self.collection_hold_seconds)
        self.collection_progress = min(
            1.0,
            self.collection_progress + (dt * self.collection_speed_multiplier() / hold_seconds),
        )
        events.progress_made = True

        collectible = self.collectibles[target_index]
        if self.collection_progress >= 1.0 and collectible.active:
            self._collect_item(collectible, events)
            self.collection_target_index = None
            self.collection_progress = 0.0

    def _nearest_collectible_in_range(self) -> int | None:
        best_index: int | None = None
        best_key: tuple[float, float, int] | None = None
        player_cell = self.player_grid_cell()
        player_position = self.player.position
        for index, collectible in enumerate(self.collectibles):
            if not collectible.active:
                continue
            if self.is_classic_ruleset():
                interaction_cells = self.interaction_cells_for_collectible(collectible)
                if player_cell not in interaction_cells:
                    continue
                item_distance = collectible.distance_to(player_position)
                edge_distance = min(self._cell_center_distance(player_cell, cell) for cell in interaction_cells)
                sort_key = (item_distance, edge_distance, index)
            else:
                item_distance = collectible.distance_to(player_position)
                if item_distance > self.interaction_radius:
                    continue
                sort_key = (item_distance, 0.0, index)
            if best_key is not None and sort_key >= best_key:
                continue
            best_index = index
            best_key = sort_key
        return best_index

    def player_grid_cell(self) -> tuple[int, int]:
        return int(self.player.x), int(self.player.y)

    def collectible_grid_cell(self, collectible: Collectible) -> tuple[int, int]:
        return int(collectible.x), int(collectible.y)

    def nearest_interactable_collectible(self) -> Collectible | None:
        index = self._nearest_collectible_in_range()
        if index is None:
            return None
        return self.collectibles[index]

    def interaction_cells_for_collectible(self, collectible: Collectible) -> list[tuple[int, int]]:
        collectible_cell = self.collectible_grid_cell(collectible)
        obstacle = self._obstacle_covering_cell(collectible_cell)
        if obstacle is None:
            return [cell for cell in self._adjacent_cells(collectible_cell) if self._is_free_cell(cell)]

        perimeter: set[tuple[int, int]] = set()
        x0 = int(obstacle.x)
        y0 = int(obstacle.y)
        x1 = int(obstacle.x + obstacle.w) - 1
        y1 = int(obstacle.y + obstacle.h) - 1

        for gx in range(x0, x1 + 1):
            perimeter.add((gx, y0 - 1))
            perimeter.add((gx, y1 + 1))
        for gy in range(y0, y1 + 1):
            perimeter.add((x0 - 1, gy))
            perimeter.add((x1 + 1, gy))

        return sorted(cell for cell in perimeter if self._is_free_cell(cell))

    def _is_adjacent_for_collection(self, player_cell: tuple[int, int], collectible_cell: tuple[int, int]) -> bool:
        dx = abs(player_cell[0] - collectible_cell[0])
        dy = abs(player_cell[1] - collectible_cell[1])
        return (dx == 1 and dy == 0) or (dx == 0 and dy == 1)

    def _cell_center_distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        ax, ay = a
        bx, by = b
        return math.hypot((ax + 0.5) - (bx + 0.5), (ay + 0.5) - (by + 0.5))

    def _obstacle_covering_cell(self, cell: tuple[int, int]) -> Obstacle | None:
        cx = cell[0] + 0.5
        cy = cell[1] + 0.5
        for obstacle in self.obstacles:
            if obstacle.contains_point(cx, cy):
                return obstacle
        return None

    def _adjacent_cells(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        return [
            (cell[0] + 1, cell[1]),
            (cell[0] - 1, cell[1]),
            (cell[0], cell[1] + 1),
            (cell[0], cell[1] - 1),
        ]

    def _is_free_cell(self, cell: tuple[int, int]) -> bool:
        gx, gy = cell
        if gx < 0 or gy < 0 or gx >= int(self.width) or gy >= int(self.height):
            return False
        cx = gx + 0.5
        cy = gy + 0.5
        return not any(obstacle.contains_point(cx, cy) for obstacle in self.obstacles)

    def _collect_item(self, collectible: Collectible, events: StepEvents) -> None:
        required_before = self.required_remaining()
        collectible.active = False
        self.score[collectible.kind] += 1
        events.collected[collectible.kind] += 1
        events.progress_made = True
        self.stalemate_timer = 0.0

        if collectible.kind == "coffee":
            self.player.coffee_timer = float(self.env_config["world"]["coffee_duration_seconds"])
        elif collectible.kind == "freeze":
            freeze_seconds = float(self.env_config["world"]["freeze_duration_seconds"])
            for enemy in self.all_enemies():
                enemy.freeze_timer = freeze_seconds

        if required_before > 0 and self.required_remaining() == 0:
            events.objective_completed = True

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
            "distance_to_primary_enemy": self.primary_enemy_distance(),
            "distance_to_exit": self.distance_player_to_escape(),
            "objective_progress": self.objective_progress_fraction(),
            "score_value": self.score_value(),
            "enemy_count": len(self.all_enemies()),
            "visible_enemy_count": self.visible_enemy_count(),
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
