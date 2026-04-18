"""World step event summaries used by reward functions and UI."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass(slots=True)
class StepEvents:
    visible_steps: int = 0
    primary_visible_steps: int = 0
    support_visible_steps: int = 0
    visible_enemy_count: int = 0
    detection_events: int = 0
    player_wall_hits: int = 0
    enemy_wall_hits: int = 0
    distance_delta: float = 0.0
    primary_distance_delta: float = 0.0
    collected: Counter[str] = field(default_factory=Counter)
    objective_completed: bool = False
    player_caught: bool = False
    player_escaped: bool = False
    time_expired: bool = False
    stalemate: bool = False
    progress_made: bool = False

    def merge(self, other: "StepEvents") -> None:
        self.visible_steps += other.visible_steps
        self.primary_visible_steps += other.primary_visible_steps
        self.support_visible_steps += other.support_visible_steps
        self.visible_enemy_count += other.visible_enemy_count
        self.detection_events += other.detection_events
        self.player_wall_hits += other.player_wall_hits
        self.enemy_wall_hits += other.enemy_wall_hits
        self.distance_delta += other.distance_delta
        self.primary_distance_delta += other.primary_distance_delta
        self.collected.update(other.collected)
        self.objective_completed = self.objective_completed or other.objective_completed
        self.player_caught = self.player_caught or other.player_caught
        self.player_escaped = self.player_escaped or other.player_escaped
        self.time_expired = self.time_expired or other.time_expired
        self.stalemate = self.stalemate or other.stalemate
        self.progress_made = self.progress_made or other.progress_made

    def count(self, item_kind: str) -> int:
        return int(self.collected[item_kind])
