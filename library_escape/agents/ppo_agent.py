from __future__ import annotations

from pathlib import Path
from typing import Any

from library_escape.env.obs_builder import ObservationBuilder
from library_escape.core.world import LibraryWorld

try:
    from stable_baselines3 import PPO
except ImportError:  # pragma: no cover - handled at runtime on missing dependency
    PPO = None  # type: ignore[assignment]


def _require_ppo() -> Any:
    if PPO is None:
        raise RuntimeError(
            "stable-baselines3 is not installed. Install project dependencies first: pip install -e ."
        )
    return PPO


def build_ppo_model(env: Any, training_cfg: dict[str, Any], tensorboard_log: str | None = None) -> Any:
    ppo_cls = _require_ppo()
    policy_kwargs = {"net_arch": training_cfg.get("net_arch", [128, 128])}
    return ppo_cls(
        "MlpPolicy",
        env,
        learning_rate=training_cfg.get("learning_rate", 3e-4),
        gamma=training_cfg.get("gamma", 0.99),
        gae_lambda=training_cfg.get("gae_lambda", 0.95),
        n_steps=training_cfg.get("n_steps", 512),
        batch_size=training_cfg.get("batch_size", 256),
        ent_coef=training_cfg.get("ent_coef", 0.01),
        clip_range=training_cfg.get("clip_range", 0.2),
        vf_coef=training_cfg.get("vf_coef", 0.5),
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device=training_cfg.get("device", "auto"),
    )


def load_ppo_model(path: str | Path) -> Any:
    ppo_cls = _require_ppo()
    return ppo_cls.load(str(path))


class SB3PolicyController:
    def __init__(
        self,
        model_path: str | Path,
        role: str,
        env_config: dict[str, Any],
        world: LibraryWorld,
        deterministic: bool = True,
    ) -> None:
        self.model = load_ppo_model(model_path)
        self.role = role
        self.obs_builder = ObservationBuilder(env_config, world)
        self.deterministic = deterministic

    def reset(self) -> None:
        return None

    def act(self, world: LibraryWorld) -> Any:
        self.obs_builder.world = world
        observation = self.obs_builder.build(self.role)
        action, _ = self.model.predict(observation, deterministic=self.deterministic)
        return action
