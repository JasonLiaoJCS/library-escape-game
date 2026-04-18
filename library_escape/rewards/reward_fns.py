from __future__ import annotations

from dataclasses import asdict
from typing import Any

from library_escape.core.entities import StepMetrics


class RewardEngine:
    def __init__(self, reward_config: dict[str, Any]) -> None:
        self.reward_config = reward_config

    @staticmethod
    def _combine(base: StepMetrics, update: StepMetrics) -> StepMetrics:
        return StepMetrics(
            note_collected=base.note_collected + update.note_collected,
            player_visible_steps=base.player_visible_steps + update.player_visible_steps,
            player_wall_hits=base.player_wall_hits + update.player_wall_hits,
            enemy_wall_hits=base.enemy_wall_hits + update.enemy_wall_hits,
            distance_delta=base.distance_delta + update.distance_delta,
            caught=base.caught or update.caught,
            escaped=base.escaped or update.escaped,
            timeout=base.timeout or update.timeout,
            stalemate=base.stalemate or update.stalemate,
        )

    def combine(self, metrics_list: list[StepMetrics]) -> StepMetrics:
        merged = StepMetrics()
        for metrics in metrics_list:
            merged = self._combine(merged, metrics)
        return merged

    def single_agent_reward(
        self,
        role: str,
        metrics: StepMetrics,
        steps_taken: int,
    ) -> tuple[float, dict[str, float]]:
        player_reward, enemy_reward = self.multi_agent_rewards(metrics, steps_taken)
        if role == "player":
            return player_reward, player_reward_breakdown(self.reward_config["player"], metrics, steps_taken)
        return enemy_reward, enemy_reward_breakdown(self.reward_config["enemy"], metrics, steps_taken)

    def multi_agent_rewards(
        self,
        metrics: StepMetrics,
        steps_taken: int,
    ) -> tuple[float, float]:
        player_breakdown = player_reward_breakdown(self.reward_config["player"], metrics, steps_taken)
        enemy_breakdown = enemy_reward_breakdown(self.reward_config["enemy"], metrics, steps_taken)
        return sum(player_breakdown.values()), sum(enemy_breakdown.values())

    def multi_agent_breakdown(
        self,
        metrics: StepMetrics,
        steps_taken: int,
    ) -> dict[str, dict[str, float]]:
        return {
            "player_0": player_reward_breakdown(self.reward_config["player"], metrics, steps_taken),
            "enemy_0": enemy_reward_breakdown(self.reward_config["enemy"], metrics, steps_taken),
        }


def enemy_reward_breakdown(
    reward_cfg: dict[str, float],
    metrics: StepMetrics,
    steps_taken: int,
) -> dict[str, float]:
    return {
        "catch_player": reward_cfg["catch_player"] if metrics.caught else 0.0,
        "player_in_cone": reward_cfg["player_in_cone_per_step"] * metrics.player_visible_steps,
        "distance_shaping": reward_cfg["distance_shaping"] * metrics.distance_delta,
        "time_penalty": reward_cfg["time_penalty"] * steps_taken,
        "wall_penalty": reward_cfg["wall_penalty"] * metrics.enemy_wall_hits,
        "stalemate": reward_cfg["stalemate"] if metrics.stalemate else 0.0,
    }


def player_reward_breakdown(
    reward_cfg: dict[str, float],
    metrics: StepMetrics,
    steps_taken: int,
) -> dict[str, float]:
    return {
        "collect_note": reward_cfg["collect_note"] * metrics.note_collected,
        "escape": reward_cfg["escape"] if metrics.escaped else 0.0,
        "seen_penalty": reward_cfg["seen_per_step"] * metrics.player_visible_steps,
        "caught": reward_cfg["caught"] if metrics.caught else 0.0,
        "time_bonus": reward_cfg["time_bonus"] * steps_taken,
        "wall_penalty": reward_cfg["wall_penalty"] * metrics.player_wall_hits,
    }
