from __future__ import annotations

from copy import deepcopy
from typing import Any

from pettingzoo.utils import ParallelEnv

from library_escape.config import load_env_config, load_reward_config
from library_escape.core.constants import ENEMY_AGENT, PLAYER_AGENT
from library_escape.core.entities import StepMetrics
from library_escape.core.world import LibraryWorld
from library_escape.env.obs_builder import ObservationBuilder
from library_escape.render import NullView, PygameView
from library_escape.rewards import RewardEngine


class LibraryEscapeMAEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "library_escape_v0"}

    def __init__(
        self,
        env_config: dict[str, Any] | None = None,
        reward_config: dict[str, Any] | None = None,
        render_mode: str | None = None,
        frame_skip: int = 4,
    ) -> None:
        self.render_mode = render_mode
        self.env_config = deepcopy(env_config or load_env_config())
        self.reward_config = deepcopy(reward_config or load_reward_config())
        self.frame_skip = int(frame_skip)
        self.world = LibraryWorld(self.env_config)
        self.obs_builder = ObservationBuilder(self.env_config, self.world)
        self.reward_engine = RewardEngine(self.reward_config)
        self.possible_agents = [PLAYER_AGENT, ENEMY_AGENT]
        self.agents = self.possible_agents[:]
        self.observation_spaces = {
            PLAYER_AGENT: self.obs_builder.observation_space("player"),
            ENEMY_AGENT: self.obs_builder.observation_space("enemy"),
        }
        self.action_spaces = {
            PLAYER_AGENT: self.obs_builder.action_space("player"),
            ENEMY_AGENT: self.obs_builder.action_space("enemy"),
        }
        self.view = self._build_view(render_mode)

    def _build_view(self, render_mode: str | None):
        if render_mode == "human" or render_mode == "rgb_array":
            return PygameView(self.env_config, render_mode=render_mode)
        render_cfg = self.env_config["render"]
        return NullView(int(render_cfg["window_width"]), int(render_cfg["window_height"]))

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.world.reset(seed=seed)
        self.agents = self.possible_agents[:]
        observations = {
            PLAYER_AGENT: self.obs_builder.build("player"),
            ENEMY_AGENT: self.obs_builder.build("enemy"),
        }
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        metrics_history: list[StepMetrics] = []
        steps_taken = 0
        player_action = actions.get(PLAYER_AGENT, 0)
        enemy_action = actions.get(ENEMY_AGENT, 0)
        for _ in range(self.frame_skip):
            metrics = self.world.tick(player_action, enemy_action)
            metrics_history.append(metrics)
            steps_taken += 1
            if self.world.status.terminated or self.world.status.truncated:
                break

        merged_metrics = self.reward_engine.combine(metrics_history)
        reward_breakdowns = self.reward_engine.multi_agent_breakdown(merged_metrics, steps_taken)
        player_reward, enemy_reward = self.reward_engine.multi_agent_rewards(merged_metrics, steps_taken)
        rewards = {PLAYER_AGENT: player_reward, ENEMY_AGENT: enemy_reward}
        terminations = {agent: self.world.status.terminated for agent in self.possible_agents}
        truncations = {agent: self.world.status.truncated for agent in self.possible_agents}
        infos = {
            PLAYER_AGENT: {"reward_breakdown": reward_breakdowns[PLAYER_AGENT], "outcome": self.world.status.outcome},
            ENEMY_AGENT: {"reward_breakdown": reward_breakdowns[ENEMY_AGENT], "outcome": self.world.status.outcome},
        }
        observations = {
            PLAYER_AGENT: self.obs_builder.build("player"),
            ENEMY_AGENT: self.obs_builder.build("enemy"),
        }
        if self.world.status.terminated or self.world.status.truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos

    def render(self):
        self.view.draw(self.world, title="Multi-Agent")
        if self.render_mode == "rgb_array":
            return self.view.rgb_array()
        return None

    def close(self) -> None:
        self.view.close()
