from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym

from library_escape.agents.rule_based_enemy import RuleBasedEnemyController, RuleBasedPlayerController
from library_escape.config import load_env_config, load_reward_config
from library_escape.core.entities import StepMetrics
from library_escape.core.world import LibraryWorld
from library_escape.env.obs_builder import ObservationBuilder
from library_escape.render import NullView, PygameView
from library_escape.rewards import RewardEngine


class LibraryEscapeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        role: str = "enemy",
        env_config: dict[str, Any] | None = None,
        reward_config: dict[str, Any] | None = None,
        render_mode: str | None = None,
        frame_skip: int = 4,
        opponent_controller: Any | None = None,
    ) -> None:
        super().__init__()
        self.role = role
        self.render_mode = render_mode
        self.env_config = deepcopy(env_config or load_env_config())
        self.reward_config = deepcopy(reward_config or load_reward_config())
        self.frame_skip = int(frame_skip)
        self.world = LibraryWorld(self.env_config)
        self.obs_builder = ObservationBuilder(self.env_config, self.world)
        self.reward_engine = RewardEngine(self.reward_config)
        self.opponent_controller = opponent_controller or self._default_opponent()
        self.view = self._build_view(render_mode)

        self.action_space = self.obs_builder.action_space(role)
        self.observation_space = self.obs_builder.observation_space(role)

    def _build_view(self, render_mode: str | None):
        if render_mode == "human" or render_mode == "rgb_array":
            return PygameView(self.env_config, render_mode=render_mode)
        render_cfg = self.env_config["render"]
        return NullView(int(render_cfg["window_width"]), int(render_cfg["window_height"]))

    def _default_opponent(self):
        if self.role == "enemy":
            return RuleBasedPlayerController(self.env_config)
        return RuleBasedEnemyController(self.env_config)

    def set_opponent_controller(self, opponent_controller: Any) -> None:
        self.opponent_controller = opponent_controller

    def _collect_info(self, metrics: StepMetrics, steps_taken: int, reward_breakdown: dict[str, float]) -> dict[str, Any]:
        snapshot = self.world.snapshot()
        return {
            "reward_breakdown": reward_breakdown,
            "steps_taken": steps_taken,
            "outcome": self.world.status.outcome,
            "notes_remaining": snapshot.notes_remaining,
            "remaining_time": snapshot.remaining_time,
            "enemy_sees_player": snapshot.enemy_sees_player,
            "player_can_see_enemy": snapshot.player_can_see_enemy,
            "distance": snapshot.distance,
            "metrics": {
                "note_collected": metrics.note_collected,
                "player_visible_steps": metrics.player_visible_steps,
                "player_wall_hits": metrics.player_wall_hits,
                "enemy_wall_hits": metrics.enemy_wall_hits,
                "distance_delta": metrics.distance_delta,
                "caught": metrics.caught,
                "escaped": metrics.escaped,
                "timeout": metrics.timeout,
                "stalemate": metrics.stalemate,
            },
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.world.reset(seed=seed)
        if hasattr(self.opponent_controller, "reset"):
            self.opponent_controller.reset()
        observation = self.obs_builder.build(self.role)
        info = self._collect_info(StepMetrics(), 0, {})
        return observation, info

    def step(self, action: Any):
        metrics_history: list[StepMetrics] = []
        steps_taken = 0
        for _ in range(self.frame_skip):
            if self.role == "enemy":
                player_action = self.opponent_controller.act(self.world)
                enemy_action = action
            else:
                player_action = action
                enemy_action = self.opponent_controller.act(self.world)

            metrics = self.world.tick(player_action, enemy_action)
            metrics_history.append(metrics)
            steps_taken += 1
            if self.world.status.terminated or self.world.status.truncated:
                break

        merged_metrics = self.reward_engine.combine(metrics_history)
        reward, reward_breakdown = self.reward_engine.single_agent_reward(self.role, merged_metrics, steps_taken)
        observation = self.obs_builder.build(self.role)
        info = self._collect_info(merged_metrics, steps_taken, reward_breakdown)
        terminated = self.world.status.terminated
        truncated = self.world.status.truncated
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        title = f"Single Agent: {self.role}"
        self.view.draw(self.world, title=title)
        if self.render_mode == "rgb_array":
            return self.view.rgb_array()
        return None

    def close(self) -> None:
        self.view.close()
