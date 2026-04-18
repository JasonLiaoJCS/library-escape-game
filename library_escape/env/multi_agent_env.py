"""PettingZoo parallel environment for self-play experiments."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import load_env_config
from ..core.actions import action_to_vector
from ..core.world import World
from ..rewards.reward_fns import RewardEngine
from .action_masking import action_mask_for_world
from .obs_builder import ObsBuilder

try:  # pragma: no cover - import guard
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "pettingzoo and gymnasium are required for LibraryEscapeMAEnv. Install with `pip install -e .[rl]`."
    ) from exc


class LibraryEscapeMAEnv(ParallelEnv):
    metadata = {"name": "library_escape_v0"}

    def __init__(self, env_config: dict | None = None, seed: int | None = None) -> None:
        self.env_config = env_config or load_env_config()
        self.world = World(env_config=self.env_config, seed=seed)
        self.reward_engine = RewardEngine.from_env_config(self.env_config)
        self.obs_builder = ObsBuilder(self.env_config)
        self.possible_agents = ["player_0", "enemy_0"]
        self.agents = list(self.possible_agents)

    @lru_cache(maxsize=2)
    def observation_space(self, agent):
        obs_dim = self.obs_builder.player_obs_dim() if agent == "player_0" else self.obs_builder.enemy_obs_dim()
        return spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    @lru_cache(maxsize=2)
    def action_space(self, agent):
        action_cfg = self.env_config["action"]
        if action_cfg["type"] == "discrete":
            return spaces.Discrete(9)
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, seed: int | None = None, options=None):
        self.world.reset(seed=seed)
        self.agents = list(self.possible_agents)
        observations = {
            "player_0": self.obs_builder.build_player_obs(self.world),
            "enemy_0": self.obs_builder.build_enemy_obs(self.world),
        }
        infos = {
            "player_0": {**self.world.info(), "action_mask": action_mask_for_world(self.world, "player").tolist()},
            "enemy_0": {**self.world.info(), "action_mask": action_mask_for_world(self.world, "enemy").tolist()},
        }
        return observations, infos

    def step(self, actions):
        player_action = action_to_vector(actions.get("player_0", 0), self.env_config["action"])
        enemy_action = action_to_vector(actions.get("enemy_0", 0), self.env_config["action"])
        prev_metrics = self.world.transition_metrics()

        events = self.world.step(player_action=player_action, enemy_action=enemy_action, frame_skip=self.world.rl_frame_skip)
        next_metrics = self.world.transition_metrics()
        rewards = self.reward_engine.compute(self.world, events, prev_metrics=prev_metrics, next_metrics=next_metrics)

        observations = {
            "player_0": self.obs_builder.build_player_obs(self.world),
            "enemy_0": self.obs_builder.build_enemy_obs(self.world),
        }
        terminations = {agent: self.world.terminated for agent in self.possible_agents}
        truncations = {agent: self.world.truncated for agent in self.possible_agents}
        infos = {
            "player_0": {**self.world.info(), "action_mask": action_mask_for_world(self.world, "player").tolist()},
            "enemy_0": {**self.world.info(), "action_mask": action_mask_for_world(self.world, "enemy").tolist()},
        }

        if self.world.terminated or self.world.truncated:
            self.agents = []
        else:
            self.agents = list(self.possible_agents)

        return observations, rewards, terminations, truncations, infos
