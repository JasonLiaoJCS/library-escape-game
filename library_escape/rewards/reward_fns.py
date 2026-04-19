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
        stationary_threshold = max(0.0, float(anti_exploit_cfg.get("stationary_threshold", 0.0)))
        idle_threshold = stationary_threshold * 0.75 if stationary_threshold > 0.0 else 0.0

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
        # Dense chase signal: reward primary enemy for physically closing the gap to the player.
        # Uses primary_distance_delta (positive when enemy closed the distance this macro-step).
        enemy_reward += float(events.primary_distance_delta) * float(enemy_cfg.get("chase_progress_per_unit", 0.0))
        # Search-mode shaping: when the player is not visible but the enemy is not idle,
        # small positive-for-moving bonus to break "spin in place" local optima.
        if float(events.primary_visible_steps) <= 0.0 and float(events.primary_enemy_net_displacement) > 0.0:
            enemy_reward += (
                float(events.primary_enemy_net_displacement)
                * float(enemy_cfg.get("search_move_per_unit", 0.0))
            )
        if (
            idle_threshold > 0.0
            and float(events.primary_enemy_net_displacement) < idle_threshold
            and world.enemy_pause_fraction(world.enemy) <= 1e-6
            and not self._enemy_guarding_exit(world, next_metrics)
        ):
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
        # Dense goal-progress signal: reward the player for shrinking the distance to the
        # currently-relevant goal (nearest required collectible, or escape zone when escape is unlocked).
        goal_progress_delta = self._goal_progress_delta(world, prev_metrics, next_metrics)
        player_reward += goal_progress_delta * float(player_cfg.get("goal_progress_per_unit", 0.0))
        # Evade signal: reward the player for increasing distance when currently visible to primary.
        if float(events.primary_visible_steps) > 0.0:
            player_reward += (
                -float(events.primary_distance_delta)
                * float(player_cfg.get("evade_progress_per_unit", 0.0))
            )
        if idle_threshold > 0.0 and float(events.player_net_displacement) < idle_threshold and float(next_metrics.get("collection_progress", 0.0)) <= 1e-6:
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

        player_reward += self._stationary_penalty(
            role="player",
            world=world,
            events=events,
            next_metrics=next_metrics,
            anti_exploit_cfg=anti_exploit_cfg,
        )
        enemy_reward += self._stationary_penalty(
            role="enemy",
            world=world,
            events=events,
            next_metrics=next_metrics,
            anti_exploit_cfg=anti_exploit_cfg,
        )
        player_reward += self._oscillation_penalty(
            role="player",
            world=world,
            events=events,
            next_metrics=next_metrics,
            anti_exploit_cfg=anti_exploit_cfg,
        )
        enemy_reward += self._oscillation_penalty(
            role="enemy",
            world=world,
            events=events,
            next_metrics=next_metrics,
            anti_exploit_cfg=anti_exploit_cfg,
        )

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
        # Use the primary enemy's own distance so the RL policy cannot freeload on
        # rule-based support enemies being close to the player.
        distance_source_key = (
            "distance_primary_enemy"
            if bool(config.get("use_primary_distance", True))
            else "distance_agents"
        )
        capture_pressure = 1.0 - min(1.0, float(metrics.get(distance_source_key, 0.0)) / max_distance)
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

    def _goal_progress_delta(
        self,
        world,
        prev_metrics: dict[str, Any],
        next_metrics: dict[str, Any],
    ) -> float:
        """Positive when the player shrinks the distance to the currently-relevant goal.

        In escape mode, the goal flips from "nearest required collectible" to
        "escape zone" the moment all prerequisites are collected. To avoid a huge
        discontinuity at that flip, we select the same key on both prev and next
        based on the *next* metrics' can_escape flag.
        """
        if float(next_metrics.get("can_escape", 0.0)) > 0.5 and getattr(world, "is_escape_mode", lambda: False)():
            key = "distance_player_to_escape"
        else:
            key = "distance_player_to_target"
        prev_distance = float(prev_metrics.get(key, 0.0))
        next_distance = float(next_metrics.get(key, 0.0))
        return prev_distance - next_distance

    def _stationary_penalty(
        self,
        *,
        role: str,
        world,
        events,
        next_metrics: dict[str, Any],
        anti_exploit_cfg: dict[str, Any],
    ) -> float:
        threshold = float(anti_exploit_cfg.get("stationary_threshold", 0.0))
        if threshold <= 0.0 or not self._quiet_step(role=role, world=world, events=events, next_metrics=next_metrics):
            return 0.0
        displacement = (
            float(events.player_net_displacement)
            if role == "player"
            else float(events.primary_enemy_net_displacement)
        )
        if displacement >= threshold:
            return 0.0
        if role == "enemy" and self._enemy_guarding_exit(world, next_metrics):
            return 0.0
        return float(anti_exploit_cfg.get(f"{role}_stationary_penalty", 0.0))

    def _oscillation_penalty(
        self,
        *,
        role: str,
        world,
        events,
        next_metrics: dict[str, Any],
        anti_exploit_cfg: dict[str, Any],
    ) -> float:
        path_threshold = float(anti_exploit_cfg.get("oscillation_path_threshold", 0.0))
        net_threshold = float(anti_exploit_cfg.get("oscillation_net_threshold", 0.0))
        if (
            path_threshold <= 0.0
            or net_threshold <= 0.0
            or not self._quiet_step(role=role, world=world, events=events, next_metrics=next_metrics)
        ):
            return 0.0
        path_length = float(events.player_path_length) if role == "player" else float(events.primary_enemy_path_length)
        net_displacement = (
            float(events.player_net_displacement)
            if role == "player"
            else float(events.primary_enemy_net_displacement)
        )
        if role == "enemy" and self._enemy_guarding_exit(world, next_metrics):
            return 0.0
        if path_length < path_threshold or net_displacement >= net_threshold:
            return 0.0
        return float(anti_exploit_cfg.get(f"{role}_oscillation_penalty", 0.0))

    def _quiet_step(self, *, role: str, world, events, next_metrics: dict[str, Any]) -> bool:
        # Note: we deliberately do NOT exempt visible_steps anymore. The worst pathology
        # observed was player/enemy oscillating while the player is in line of sight —
        # exempting visible_steps makes this invisible to the anti-exploit filter.
        if events.player_caught or events.player_escaped or events.objective_completed:
            return False
        if sum(events.collected.values()) > 0:
            return False
        if float(next_metrics.get("collection_progress", 0.0)) > 1e-6:
            return False
        if role == "enemy" and world.enemy_pause_fraction(world.enemy) > 1e-6:
            return False
        # Active detection trigger (classic/collection ruleset) is still a legitimate event.
        if events.detection_events > 0:
            return False
        return True

    def _enemy_guarding_exit(self, world, next_metrics: dict[str, Any]) -> bool:
        if float(next_metrics.get("can_escape", 0.0)) <= 0.5:
            return False
        return float(next_metrics.get("distance_enemy_to_escape", 999.0)) <= 1.25


def compute_rewards(
    world,
    events,
    env_config: dict | None = None,
    prev_metrics: dict[str, Any] | None = None,
    next_metrics: dict[str, Any] | None = None,
) -> dict[str, float]:
    engine = RewardEngine.from_env_config(env_config or world.env_config)
    return engine.compute(world, events, prev_metrics=prev_metrics, next_metrics=next_metrics)
