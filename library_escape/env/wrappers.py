"""Optional wrappers used by training scripts."""

from __future__ import annotations

try:  # pragma: no cover - import guard
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "gymnasium is required for wrappers. Install with `pip install -e .[rl]`."
    ) from exc


class FrameSkipOverrideWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip: int):
        super().__init__(env)
        self.frame_skip = frame_skip

    def step(self, action):
        original_skip = self.env.world.rl_frame_skip
        self.env.world.rl_frame_skip = self.frame_skip
        try:
            return self.env.step(action)
        finally:
            self.env.world.rl_frame_skip = original_skip
