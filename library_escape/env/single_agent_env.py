"""Gymnasium environment for training a single role against a scripted opponent."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import load_env_config
from ..core.actions import action_to_vector
from ..core.world import World
from ..rewards.reward_fns import RewardEngine
from .action_masking import action_mask_for_world
from .obs_builder import ObsBuilder

try:  # pragma: no cover - import guard
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "gymnasium is required for LibraryEscapeEnv. Install with `pip install -e .[rl]`."
    ) from exc


class LibraryEscapeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        controlled_agent: str = "enemy",
        opponent_controller=None,
        env_config: dict[str, Any] | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if controlled_agent not in {"enemy", "player"}:
            raise ValueError("controlled_agent must be 'enemy' or 'player'.")

        self.controlled_agent = controlled_agent
        self.opponent_controller = opponent_controller
        self.env_config = env_config or load_env_config()
        self.world = World(env_config=self.env_config, seed=seed)
        self.obs_builder = ObsBuilder(self.env_config)
        self.reward_engine = RewardEngine.from_env_config(self.env_config)
        self.render_mode = render_mode
        self._renderer = None

        self.action_space = self._build_action_space()
        obs_dim = self.obs_builder.player_obs_dim() if controlled_agent == "player" else self.obs_builder.enemy_obs_dim()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

    def _build_action_space(self):
        action_cfg = self.env_config["action"]
        if action_cfg["type"] == "discrete":
            return spaces.Discrete(9)
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.world.reset(seed=seed)
        if hasattr(self.opponent_controller, "reset"):
            self.opponent_controller.reset()
        observation = self._get_obs()
        info = self.world.info()
        info["action_mask"] = self.valid_action_mask().tolist()
        return observation, info

    def step(self, action):
        controlled_vector = action_to_vector(action, self.env_config["action"])
        opponent_vector = (0.0, 0.0)
        if self.opponent_controller is not None:
            opponent_vector = self.opponent_controller.act(self.world)
        prev_metrics = self.world.transition_metrics()

        if self.controlled_agent == "enemy":
            player_action = opponent_vector
            enemy_action = controlled_vector
        else:
            player_action = controlled_vector
            enemy_action = opponent_vector

        events = self.world.step(player_action=player_action, enemy_action=enemy_action, frame_skip=self.world.rl_frame_skip)
        next_metrics = self.world.transition_metrics()
        rewards = self.reward_engine.compute(self.world, events, prev_metrics=prev_metrics, next_metrics=next_metrics)
        reward = rewards["enemy_0"] if self.controlled_agent == "enemy" else rewards["player_0"]

        observation = self._get_obs()
        info = self.world.info()
        info["events"] = {
            "visible_steps": events.visible_steps,
            "player_wall_hits": events.player_wall_hits,
            "enemy_wall_hits": events.enemy_wall_hits,
            "collected": dict(events.collected),
            "distance_delta": events.distance_delta,
            "reward_player": rewards["player_0"],
            "reward_enemy": rewards["enemy_0"],
        }
        info["action_mask"] = self.valid_action_mask().tolist()
        return observation, reward, self.world.terminated, self.world.truncated, info

    def valid_action_mask(self) -> np.ndarray:
        return action_mask_for_world(self.world, self.controlled_agent)

    def action_masks(self) -> np.ndarray:
        return self.valid_action_mask()

    def _get_obs(self) -> np.ndarray:
        return self.obs_builder.build(self.world, self.controlled_agent)

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from ..render.pygame_view import PygameView

            self._renderer = PygameView(self.world)
        return self._renderer.render_frame(rgb_array=self.render_mode == "rgb_array")

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
