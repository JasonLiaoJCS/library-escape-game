"""Replay serialization utilities."""

from __future__ import annotations

import gzip
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..config import resolve_repo_path
from ..core.collectible import Collectible
from ..core.enemy import EnemyState, VisionCone
from ..core.obstacle import Obstacle
from ..core.character import PlayerState

REPLAY_FORMAT = "library_escape_replay_v1"


def _serialize_obstacle(obstacle: Obstacle) -> dict:
    return {
        "kind": obstacle.kind,
        "texture_key": obstacle.texture_key,
        "x": obstacle.x,
        "y": obstacle.y,
        "w": obstacle.w,
        "h": obstacle.h,
    }


def _serialize_collectible(item: Collectible) -> dict:
    return {
        "kind": item.kind,
        "x": item.x,
        "y": item.y,
        "texture_key": item.texture_key,
        "active": item.active,
    }


def _serialize_world(world) -> dict:
    return {
        "time_remaining": world.time_remaining,
        "elapsed_time": world.elapsed_time,
        "terminated": world.terminated,
        "truncated": world.truncated,
        "outcome": world.outcome,
        "score": dict(world.score),
        "player": {
            "x": world.player.x,
            "y": world.player.y,
            "radius": world.player.radius,
            "base_speed": world.player.base_speed,
            "facing_x": world.player.facing_x,
            "facing_y": world.player.facing_y,
            "coffee_timer": world.player.coffee_timer,
        },
        "enemy": {
            "x": world.enemy.x,
            "y": world.enemy.y,
            "radius": world.enemy.radius,
            "base_speed": world.enemy.base_speed,
            "facing_x": world.enemy.facing_x,
            "facing_y": world.enemy.facing_y,
            "freeze_timer": world.enemy.freeze_timer,
            "last_seen_player": list(world.enemy.last_seen_player) if world.enemy.last_seen_player is not None else None,
            "vision_range": world.enemy.vision.range_cells,
            "vision_angle_deg": world.enemy.vision.angle_deg,
        },
        "collectibles": [_serialize_collectible(item) for item in world.collectibles],
        "metrics": {
            "can_escape": world.can_player_escape(),
            "player_visible": world.player_visible_to_enemy(),
            "alert_level": world.alert_level(),
            "enemy_mode": world.enemy_mode(),
            "distance_agents": world.distance_between_agents(),
            "distance_player_to_escape": world.distance_player_to_escape(),
            "objective_progress": world.objective_progress_fraction(),
        },
    }


class ReplayRecorder:
    def __init__(self, output_path: str | Path, world, metadata: dict | None = None) -> None:
        self.output_path = resolve_repo_path(output_path)
        self.env_config = world.env_config
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("recorded_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.metadata.setdefault("render_fps", world.render_fps)
        self.payload = {
            "format": REPLAY_FORMAT,
            "metadata": self.metadata,
            "env_config": self.env_config,
            "map": {
                "width": world.width,
                "height": world.height,
                "cell_size": world.cell_size,
                "escape_zone": world.escape_zone,
                "enemy_waypoints": [list(point) for point in world.enemy_waypoints],
                "obstacles": [_serialize_obstacle(obstacle) for obstacle in world.obstacles],
            },
            "frames": [],
        }

    def capture(self, world) -> None:
        self.payload["frames"].append(_serialize_world(world))

    def save(self) -> Path:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.output_path, "wt", encoding="utf-8") as handle:
            json.dump(self.payload, handle, ensure_ascii=True)
        return self.output_path


def load_replay(path: str | Path) -> dict:
    replay_path = resolve_repo_path(path)
    with gzip.open(replay_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != REPLAY_FORMAT:
        raise ValueError(f"Unsupported replay format in {replay_path}")
    return payload


@dataclass
class ReplayWorld:
    payload: dict

    def __post_init__(self) -> None:
        replay_map = self.payload["map"]
        self.env_config = self.payload["env_config"]
        self.width = float(replay_map["width"])
        self.height = float(replay_map["height"])
        self.cell_size = int(replay_map["cell_size"])
        self.escape_zone = replay_map["escape_zone"]
        self.enemy_waypoints = [tuple(point) for point in replay_map.get("enemy_waypoints", [])]
        self.obstacles = [Obstacle.from_dict(item) for item in replay_map["obstacles"]]
        self.render_fps = int(self.payload.get("metadata", {}).get("render_fps", self.env_config["timing"]["render_fps"]))
        self.frames = self.payload["frames"]
        self.frame_index = 0
        self.score = Counter()
        self.collectibles: list[Collectible] = []
        self.player = PlayerState(name="player_0", x=0.5, y=0.5, radius=0.3, base_speed=1.0)
        self.enemy = EnemyState(
            name="enemy_0",
            x=0.5,
            y=0.5,
            radius=0.3,
            base_speed=1.0,
            vision=VisionCone(range_cells=6.0, angle_deg=70.0),
        )
        self.time_remaining = 0.0
        self.elapsed_time = 0.0
        self.terminated = False
        self.truncated = False
        self.outcome = None
        self._metrics: dict[str, float | str | bool] = {}
        if self.frames:
            self.reset_to_frame(0)

    def reset_to_frame(self, frame_index: int) -> None:
        frame = self.frames[max(0, min(frame_index, len(self.frames) - 1))]
        self.frame_index = frame_index
        player = frame["player"]
        enemy = frame["enemy"]
        self.player = PlayerState(
            name="player_0",
            x=float(player["x"]),
            y=float(player["y"]),
            radius=float(player["radius"]),
            base_speed=float(player["base_speed"]),
            facing_x=float(player["facing_x"]),
            facing_y=float(player["facing_y"]),
            coffee_timer=float(player["coffee_timer"]),
        )
        self.enemy = EnemyState(
            name="enemy_0",
            x=float(enemy["x"]),
            y=float(enemy["y"]),
            radius=float(enemy["radius"]),
            base_speed=float(enemy["base_speed"]),
            facing_x=float(enemy["facing_x"]),
            facing_y=float(enemy["facing_y"]),
            freeze_timer=float(enemy["freeze_timer"]),
            last_seen_player=tuple(enemy["last_seen_player"]) if enemy["last_seen_player"] is not None else None,
            vision=VisionCone(
                range_cells=float(enemy["vision_range"]),
                angle_deg=float(enemy["vision_angle_deg"]),
            ),
        )
        self.collectibles = [
            Collectible(
                kind=str(item["kind"]),
                x=float(item["x"]),
                y=float(item["y"]),
                texture_key=str(item["texture_key"]),
                active=bool(item["active"]),
            )
            for item in frame["collectibles"]
        ]
        self.score = Counter({str(key): int(value) for key, value in frame["score"].items()})
        self.time_remaining = float(frame["time_remaining"])
        self.elapsed_time = float(frame["elapsed_time"])
        self.terminated = bool(frame["terminated"])
        self.truncated = bool(frame["truncated"])
        self.outcome = frame["outcome"]
        self._metrics = dict(frame["metrics"])

    def can_player_escape(self) -> bool:
        return bool(self._metrics.get("can_escape", False))

    def player_visible_to_enemy(self) -> bool:
        return bool(self._metrics.get("player_visible", False))

    def alert_level(self) -> float:
        return float(self._metrics.get("alert_level", 0.0))

    def enemy_mode(self) -> str:
        return str(self._metrics.get("enemy_mode", "patrol"))

    def distance_between_agents(self) -> float:
        return float(self._metrics.get("distance_agents", 0.0))

    def distance_player_to_escape(self) -> float:
        return float(self._metrics.get("distance_player_to_escape", 0.0))

    def objective_progress_fraction(self) -> float:
        return float(self._metrics.get("objective_progress", 0.0))
