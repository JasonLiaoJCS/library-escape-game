"""Stable-Baselines3 policy wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import load_env_config, resolve_repo_path
from ..core.actions import action_to_vector
from ..env.action_masking import action_mask_for_world
from ..env.obs_builder import ObsBuilder


class SB3PolicyController:
    def __init__(
        self,
        role: str,
        model_path: str | Path,
        env_config: dict | None = None,
        deterministic: bool = True,
        obsnorm_path: str | Path | None = None,
    ) -> None:
        self.role = role
        self.env_config = env_config or load_env_config()
        self.obs_builder = ObsBuilder(self.env_config)
        self.deterministic = deterministic
        self.model_path = resolve_repo_path(model_path)
        self.model, self.uses_action_masks, self.algorithm_name = self._load_model(self.model_path)
        self.obs_normalizer = self._load_obs_normalizer(obsnorm_path)

    def reset(self) -> None:
        return None

    def act(self, world) -> tuple[float, float]:
        obs = self.obs_builder.build(world, self.role)
        if self.obs_normalizer is not None:
            obs = self._normalize_obs(obs, self.obs_normalizer)
        if self.uses_action_masks and str(self.env_config["action"]["type"]).lower() == "discrete":
            mask = action_mask_for_world(world, self.role)
            action, _ = self.model.predict(obs, deterministic=self.deterministic, action_masks=mask)
        else:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action_to_vector(action, self.env_config["action"])

    def _model_metadata_hints(self, path: Path) -> list[str]:
        candidates = [
            path.with_suffix(".meta.json"),
            path.parent / "model.meta.json",
        ]
        hints: list[str] = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            algorithm = payload.get("algorithm")
            if algorithm:
                hints.append(str(algorithm).lower())
        return hints

    def _load_model(self, path: Path):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "stable-baselines3 is required to load PPO policies. Install with `pip install -e .[rl]`."
            ) from exc

        loaders: list[tuple[str, object, bool]] = []
        hints = self._model_metadata_hints(path)
        maskable_first = any("maskable" in hint for hint in hints)
        if maskable_first:
            try:
                from sb3_contrib import MaskablePPO

                loaders.append(("maskable_ppo", MaskablePPO, True))
            except Exception:
                pass
        loaders.append(("ppo", PPO, False))
        if not maskable_first:
            try:
                from sb3_contrib import MaskablePPO

                loaders.append(("maskable_ppo", MaskablePPO, True))
            except Exception:
                pass

        last_error = None
        for name, cls, uses_action_masks in loaders:
            try:
                return cls.load(path), uses_action_masks, name
            except Exception as exc:  # pragma: no cover - fallback chain
                last_error = exc
        raise RuntimeError(f"Unable to load checkpoint at {path}: {last_error}") from last_error

    def _load_obs_normalizer(self, explicit_path: str | Path | None):
        candidate_paths = []
        if explicit_path is not None:
            candidate_paths.append(resolve_repo_path(explicit_path))
        candidate_paths.extend(
            [
                self.model_path.with_suffix(".obsnorm.npz"),
                self.model_path.parent / "obsnorm_latest.npz",
                self.model_path.parent.parent / "obsnorm_latest.npz",
            ]
        )
        for candidate in candidate_paths:
            if candidate.exists():
                payload = np.load(candidate)
                return {
                    "mean": payload["mean"].astype(np.float32),
                    "var": payload["var"].astype(np.float32),
                    "epsilon": float(payload["epsilon"][0]),
                    "clip_obs": float(payload["clip_obs"][0]),
                }
        return None

    def _normalize_obs(self, obs: np.ndarray, payload: dict[str, object]) -> np.ndarray:
        mean = payload["mean"]
        var = payload["var"]
        epsilon = float(payload["epsilon"])
        clip_obs = float(payload["clip_obs"])
        normalized = (obs - mean) / np.sqrt(var + epsilon)
        return np.clip(normalized, -clip_obs, clip_obs).astype(np.float32)
