"""Reward shaping driven by YAML config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import load_rewards_config


@dataclass(slots=True)
class RewardEngine:
    reward_config: dict

    @classmethod
    def from_env_config(cls, env_config: dict) -> "RewardEngine":
        return cls(load_rewards_config(env_config["reward_path"]))

    def compute(
        self,
        world,
        events,
        prev_metrics: dict[str, Any] | None = None,
        next_metrics: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        reward_cfg = self.reward_config
        global_cfg = reward_cfg.get("global", {})
        enemy_cfg = reward_cfg["enemy"]
        player_cfg = reward_cfg["player"]
        anti_exploit_cfg = reward_cfg.get("anti_exploit", {})

        visibility_scale = events.visible_steps / max(1, world.rl_frame_skip)
        prev_metrics = prev_metrics or world.last_metrics
        next_metrics = next_metrics or world.transition_metrics()

        enemy_reward = 0.0
        enemy_reward += visibility_scale * float(enemy_cfg["player_in_cone_per_step"])
        enemy_reward += events.enemy_wall_hits * float(enemy_cfg["wall_penalty"])
        enemy_reward += float(enemy_cfg["time_penalty"])
        if abs(world.enemy.velocity_x) <= 1e-6 and abs(world.enemy.velocity_y) <= 1e-6:
            enemy_reward += float(enemy_cfg.get("idle_penalty", 0.0))
        if events.player_caught:
            enemy_reward += float(enemy_cfg["catch_player"])
        if events.time_expired or events.stalemate:
            enemy_reward += float(enemy_cfg.get("deny_escape_bonus", 0.0))
        if events.stalemate:
            enemy_reward += float(enemy_cfg["stalemate"])
        enemy_reward += self._potential_delta(
            gamma=float(global_cfg.get("gamma", 0.99)),
            enabled=bool(enemy_cfg.get("potential", {}).get("enabled", False)),
            prev_value=self._enemy_potential(world, prev_metrics),
            next_value=self._enemy_potential(world, next_metrics),
        )

        player_reward = 0.0
        player_reward += events.count("note") * float(player_cfg["collect_note"])
        player_reward += events.count("exam") * float(player_cfg.get("collect_exam", player_cfg["collect_note"]))
        player_reward += events.count("coffee") * float(player_cfg.get("collect_coffee", 0.0))
        player_reward += events.count("freeze") * float(player_cfg.get("collect_freeze", 0.0))
        player_reward += visibility_scale * float(player_cfg["seen_per_step"])
        player_reward += events.player_wall_hits * float(player_cfg["wall_penalty"])
        player_reward += float(player_cfg["time_bonus"])
        if abs(world.player.velocity_x) <= 1e-6 and abs(world.player.velocity_y) <= 1e-6:
            player_reward += float(player_cfg.get("idle_penalty", 0.0))
        if events.player_escaped:
            player_reward += float(player_cfg["escape"])
        if events.player_caught:
            player_reward += float(player_cfg["caught"])
        if events.stalemate:
            player_reward += float(player_cfg["stalemate"])
        player_reward += self._potential_delta(
            gamma=float(global_cfg.get("gamma", 0.99)),
            enabled=bool(player_cfg.get("potential", {}).get("enabled", False)),
            prev_value=self._player_potential(world, prev_metrics),
            next_value=self._player_potential(world, next_metrics),
        )
        if not events.progress_made:
            penalty = float(anti_exploit_cfg.get("no_progress_penalty", 0.0))
            player_reward += penalty
            enemy_reward += penalty

        zero_sum_mix = float(global_cfg.get("zero_sum_mix", 0.0))
        if zero_sum_mix > 0.0:
            player_adv = player_reward - enemy_reward
            enemy_adv = enemy_reward - player_reward
            player_reward = (1.0 - zero_sum_mix) * player_reward + zero_sum_mix * player_adv
            enemy_reward = (1.0 - zero_sum_mix) * enemy_reward + zero_sum_mix * enemy_adv

        clip_range = float(global_cfg.get("clip_range", 0.0))
        if clip_range > 0.0:
            player_reward = max(-clip_range, min(clip_range, player_reward))
            enemy_reward = max(-clip_range, min(clip_range, enemy_reward))

        return {"player_0": float(player_reward), "enemy_0": float(enemy_reward)}

    def _potential_delta(self, gamma: float, enabled: bool, prev_value: float, next_value: float) -> float:
        if not enabled:
            return 0.0
        return gamma * next_value - prev_value

    def _enemy_potential(self, world, metrics: dict[str, Any]) -> float:
        config = self.reward_config["enemy"].get("potential", {})
        max_distance = max(1.0, world.max_map_distance())
        capture_progress = 1.0 - min(1.0, float(metrics["distance_agents"]) / max_distance)
        visibility_lock = float(metrics["player_visible"])
        escape_pressure = 0.0
        if float(metrics["can_escape"]) > 0.5:
            escape_pressure = 1.0 - min(1.0, float(metrics["distance_player_to_escape"]) / max_distance)
        return (
            float(config.get("capture_progress", 0.0)) * capture_progress
            + float(config.get("visibility_lock", 0.0)) * visibility_lock
            + float(config.get("escape_pressure", 0.0)) * escape_pressure
        )

    def _player_potential(self, world, metrics: dict[str, Any]) -> float:
        config = self.reward_config["player"].get("potential", {})
        max_distance = max(1.0, world.max_map_distance())
        objective_progress = float(metrics["objective_progress"])
        target_navigation = 0.0
        if float(metrics["can_escape"]) > 0.5:
            target_navigation = 1.0 - min(1.0, float(metrics["distance_player_to_escape"]) / max_distance)
            target_weight = float(config.get("escape_navigation", 0.0))
        else:
            target_navigation = 1.0 - min(1.0, float(metrics["distance_player_to_target"]) / max_distance)
            target_weight = float(config.get("target_navigation", 0.0))
        stealth_margin = min(1.0, float(metrics["distance_agents"]) / max_distance)
        stealth_margin *= 0.35 if float(metrics["player_visible"]) > 0.5 else 1.0
        return (
            float(config.get("objective_progress", 0.0)) * objective_progress
            + target_weight * target_navigation
            + float(config.get("stealth_margin", 0.0)) * stealth_margin
        )


def compute_rewards(
    world,
    events,
    env_config: dict | None = None,
    prev_metrics: dict[str, Any] | None = None,
    next_metrics: dict[str, Any] | None = None,
) -> dict[str, float]:
    engine = RewardEngine.from_env_config(env_config or world.env_config)
    return engine.compute(world, events, prev_metrics=prev_metrics, next_metrics=next_metrics)
