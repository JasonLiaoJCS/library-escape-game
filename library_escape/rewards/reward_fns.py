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

        prev_metrics = prev_metrics or world.last_metrics
        next_metrics = next_metrics or world.transition_metrics()

        frame_scale = 1.0 / max(1, int(world.rl_frame_skip))
        enemy_count = max(1.0, float(next_metrics.get("enemy_count", 1.0)))
        primary_visibility = float(events.primary_visible_steps) * frame_scale
        support_visibility = float(events.support_visible_steps) * frame_scale
        visible_enemy_ratio = float(events.visible_enemy_count) / max(1.0, float(world.rl_frame_skip) * enemy_count)
        score_progress = float(next_metrics.get("score_progress", next_metrics.get("objective_progress", 0.0)))
        score_denial = 1.0 - score_progress

        enemy_reward = 0.0
        enemy_reward += primary_visibility * float(enemy_cfg.get("primary_visible_per_step", 0.0))
        enemy_reward += support_visibility * float(enemy_cfg.get("support_visible_per_step", 0.0))
        enemy_reward += visible_enemy_ratio * float(enemy_cfg.get("team_visible_ratio_per_step", 0.0))
        enemy_reward += float(events.detection_events) * float(enemy_cfg.get("detection_event_bonus", 0.0))
        enemy_reward += events.count("note") * float(enemy_cfg.get("player_collect_note_penalty", 0.0))
        enemy_reward += events.count("exam") * float(enemy_cfg.get("player_collect_exam_penalty", 0.0))
        enemy_reward += (events.count("coffee") + events.count("freeze")) * float(enemy_cfg.get("player_collect_powerup_penalty", 0.0))
        enemy_reward += events.enemy_wall_hits * float(enemy_cfg.get("wall_penalty", 0.0))
        enemy_reward += float(enemy_cfg.get("time_penalty", 0.0))
        if abs(world.enemy.velocity_x) <= 1e-6 and abs(world.enemy.velocity_y) <= 1e-6:
            enemy_reward += float(enemy_cfg.get("idle_penalty", 0.0))
        if events.player_caught:
            enemy_reward += float(enemy_cfg.get("catch_player", 0.0))
        if events.player_escaped:
            enemy_reward += float(enemy_cfg.get("lose_on_escape", 0.0))
        if events.time_expired:
            enemy_reward += float(enemy_cfg.get("timeout_win", 0.0))
            enemy_reward += score_denial * float(enemy_cfg.get("timeout_score_denial_bonus", 0.0))
        if events.stalemate:
            enemy_reward += float(enemy_cfg.get("stalemate", 0.0))
            enemy_reward += score_denial * float(enemy_cfg.get("stalemate_score_denial_bonus", 0.0))
        if events.objective_completed:
            enemy_reward += float(enemy_cfg.get("player_objective_complete_penalty", 0.0))
        enemy_reward += self._potential_delta(
            gamma=float(global_cfg.get("gamma", 0.99)),
            enabled=bool(enemy_cfg.get("potential", {}).get("enabled", False)),
            prev_value=self._enemy_potential(world, prev_metrics),
            next_value=self._enemy_potential(world, next_metrics),
        )

        player_reward = 0.0
        player_reward += events.count("note") * float(player_cfg.get("collect_note", 0.0))
        player_reward += events.count("exam") * float(player_cfg.get("collect_exam", 0.0))
        player_reward += events.count("coffee") * float(player_cfg.get("collect_coffee", 0.0))
        player_reward += events.count("freeze") * float(player_cfg.get("collect_freeze", 0.0))
        player_reward += float(events.detection_events) * float(player_cfg.get("detection_event_penalty", 0.0))
        player_reward += primary_visibility * float(player_cfg.get("primary_seen_per_step", 0.0))
        player_reward += support_visibility * float(player_cfg.get("support_seen_per_step", 0.0))
        player_reward += visible_enemy_ratio * float(player_cfg.get("multi_seen_penalty_per_step", 0.0))
        player_reward += events.player_wall_hits * float(player_cfg.get("wall_penalty", 0.0))
        player_reward += float(player_cfg.get("time_penalty", 0.0))
        if abs(world.player.velocity_x) <= 1e-6 and abs(world.player.velocity_y) <= 1e-6:
            player_reward += float(player_cfg.get("idle_penalty", 0.0))
        if events.player_escaped:
            player_reward += float(player_cfg.get("escape", 0.0))
        if events.player_caught:
            player_reward += float(player_cfg.get("caught", 0.0))
        if events.time_expired:
            player_reward += float(player_cfg.get("timeout_loss", 0.0))
            player_reward += score_progress * float(player_cfg.get("timeout_score_progress_bonus", 0.0))
        if events.stalemate:
            player_reward += float(player_cfg.get("stalemate", 0.0))
            player_reward += score_progress * float(player_cfg.get("stalemate_score_progress_bonus", 0.0))
        if events.objective_completed:
            player_reward += float(player_cfg.get("objective_complete_bonus", 0.0))
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
        capture_pressure = 1.0 - min(1.0, float(metrics.get("distance_agents", 0.0)) / max_distance)
        team_visibility = (
            0.65 * float(metrics.get("player_visible_primary", 0.0))
            + 0.35 * float(metrics.get("visible_enemy_ratio", 0.0))
        )
        objective_denial = 1.0 - self._mode_aware_progress(world, metrics)
        exit_guard = 0.0
        if float(metrics.get("can_escape", 0.0)) > 0.5 or float(metrics.get("objective_progress", 0.0)) >= 0.75:
            exit_guard = 1.0 - min(1.0, float(metrics.get("distance_enemy_to_escape", 0.0)) / max_distance)
        encirclement = capture_pressure * float(metrics.get("visible_enemy_ratio", 0.0))
        return (
            float(config.get("capture_pressure", 0.0)) * capture_pressure
            + float(config.get("team_visibility", 0.0)) * team_visibility
            + float(config.get("objective_denial", 0.0)) * objective_denial
            + float(config.get("exit_guard", 0.0)) * exit_guard
            + float(config.get("encirclement", 0.0)) * encirclement
        )

    def _player_potential(self, world, metrics: dict[str, Any]) -> float:
        config = self.reward_config["player"].get("potential", {})
        max_distance = max(1.0, world.max_map_distance())
        objective_progress = self._mode_aware_progress(world, metrics)
        if float(metrics.get("can_escape", 0.0)) > 0.5:
            target_navigation = 1.0 - min(1.0, float(metrics.get("distance_player_to_escape", 0.0)) / max_distance)
            target_weight = float(config.get("escape_navigation", 0.0))
        else:
            target_navigation = 1.0 - min(1.0, float(metrics.get("distance_player_to_target", 0.0)) / max_distance)
            target_weight = float(config.get("target_navigation", 0.0))

        threat_margin = min(1.0, float(metrics.get("distance_agents", 0.0)) / max_distance)
        threat_margin *= 1.0 - (0.65 * float(metrics.get("player_visible_primary", 0.0)))
        threat_margin *= 1.0 - (0.35 * float(metrics.get("visible_enemy_ratio", 0.0)))
        threat_margin = max(0.0, threat_margin)

        exit_window = 0.0
        if float(metrics.get("can_escape", 0.0)) > 0.5:
            exit_window = float(metrics.get("exit_lead", 0.0))

        return (
            float(config.get("objective_progress", 0.0)) * objective_progress
            + target_weight * target_navigation
            + float(config.get("threat_margin", 0.0)) * threat_margin
            + float(config.get("exit_window", 0.0)) * exit_window
        )

    def _mode_aware_progress(self, world, metrics: dict[str, Any]) -> float:
        if getattr(world, "is_collection_mode", lambda: False)():
            return float(metrics.get("score_progress", metrics.get("objective_progress", 0.0)))
        return float(metrics.get("objective_progress", 0.0))


def compute_rewards(
    world,
    events,
    env_config: dict | None = None,
    prev_metrics: dict[str, Any] | None = None,
    next_metrics: dict[str, Any] | None = None,
) -> dict[str, float]:
    engine = RewardEngine.from_env_config(env_config or world.env_config)
    return engine.compute(world, events, prev_metrics=prev_metrics, next_metrics=next_metrics)
