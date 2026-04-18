from __future__ import annotations

import gymnasium as gym


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, frame_skip: int) -> None:
        super().__init__(env)
        self.frame_skip = int(frame_skip)
        if hasattr(env, "frame_skip"):
            env.frame_skip = self.frame_skip

    def step(self, action):
        if hasattr(self.env, "frame_skip"):
            self.env.frame_skip = self.frame_skip
        return self.env.step(action)
