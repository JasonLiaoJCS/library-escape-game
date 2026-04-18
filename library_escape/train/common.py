"""Shared helpers for training scripts."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..agents.opponent_pool import OpponentPool, WeightedControllerChooser
from ..agents.ppo_agent import SB3PolicyController
from ..agents.rule_based_enemy import RandomEnemyController, RuleBasedEnemyController
from ..agents.rule_based_player import HeuristicPlayerController, MixedPlayerController, RandomPlayerController
from ..config import load_env_config, load_training_config, resolve_repo_path
from ..eval.registry import discover_saved_models
from ..env.obs_builder import ObsBuilder
from ..env.single_agent_env import LibraryEscapeEnv
from ..game_modes import build_game_mode_env_config, normalize_game_mode


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def timestamped_run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@dataclass(frozen=True, slots=True)
class AlgorithmSpec:
    key: str
    summary_name: str
    model_cls: type | None
    eval_callback_cls: type | None
    uses_action_masks: bool = False


def apply_preset(
    mode_key: str,
    preset_name: str | None,
    env_config: dict[str, Any],
    train_config: dict[str, Any],
    root_training_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root_training_config = root_training_config or load_training_config()
    if not preset_name:
        return env_config, train_config
    preset = root_training_config.get("presets", {}).get(preset_name)
    if not preset:
        return env_config, train_config
    updated_train = deep_update(train_config, preset.get(mode_key, {}))
    updated_env = deep_update(env_config, preset.get("env_overrides", {}))
    game_mode = normalize_game_mode(updated_env.get("world", {}).get("game_mode", "escape"))
    updated_env = deep_update(updated_env, preset.get("env_overrides_by_game_mode", {}).get(game_mode, {}))
    updated_train = deep_update(updated_train, preset.get("train_overrides_by_game_mode", {}).get(game_mode, {}))
    return updated_env, updated_train


def apply_runtime_overrides(
    env_config: dict[str, Any],
    train_config: dict[str, Any],
    raw_overrides: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not raw_overrides:
        return env_config, train_config
    payload = json.loads(raw_overrides)
    updated_env = deep_update(env_config, payload.get("env", {}))
    updated_train = deep_update(train_config, payload.get("train", {}))
    return updated_env, updated_train


def activation_fn_from_name(name: str):
    import torch.nn as nn

    lookup = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    try:
        return lookup[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation function: {name}") from exc


def build_policy_kwargs(train_cfg: dict[str, Any]) -> dict[str, Any]:
    policy_cfg = train_cfg.get("policy", {})
    hidden_sizes = policy_cfg.get("hidden_sizes", [256, 256])
    activation_name = policy_cfg.get("activation_fn", "tanh")
    return {
        "net_arch": list(hidden_sizes),
        "activation_fn": activation_fn_from_name(str(activation_name)),
    }


def resolve_algorithm_spec(algorithm_key: str) -> AlgorithmSpec:
    key = str(algorithm_key).lower()
    if key in {"ppo", "league_ppo"}:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback

        return AlgorithmSpec(
            key=key,
            summary_name=key,
            model_cls=PPO,
            eval_callback_cls=EvalCallback,
            uses_action_masks=False,
        )
    if key in {"maskable_ppo", "league_maskable_ppo"}:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

        return AlgorithmSpec(
            key=key,
            summary_name=key,
            model_cls=MaskablePPO,
            eval_callback_cls=MaskableEvalCallback,
            uses_action_masks=True,
        )
    if key == "mappo_recipe":
        return AlgorithmSpec(
            key=key,
            summary_name=key,
            model_cls=None,
            eval_callback_cls=None,
            uses_action_masks=False,
        )
    raise ValueError(f"Unsupported training algorithm: {algorithm_key}")


def _cached_factory(builder: Callable[[], object]) -> Callable[[], object]:
    controller = None

    def _factory():
        nonlocal controller
        if controller is None:
            controller = builder()
        return controller

    return _factory


def discover_role_model_paths(role: str, game_mode: str, max_items: int = 8) -> list[Path]:
    normalized_mode = normalize_game_mode(game_mode)
    candidates = [item for item in discover_saved_models() if item.role == role and normalize_game_mode(item.game_mode) == normalized_mode]
    paths: list[Path] = []
    for candidate in candidates:
        if candidate.model_path.exists():
            paths.append(candidate.model_path)
        if len(paths) >= max_items:
            break
    return paths


def _expected_obs_dim(role: str, env_config: dict[str, Any]) -> int:
    builder = ObsBuilder(env_config)
    return builder.player_obs_dim() if role == "player" else builder.enemy_obs_dim()


def _obs_norm_dim_hint(model_path: Path) -> int | None:
    candidates = [
        model_path.with_suffix(".obsnorm.npz"),
        model_path.parent / "obsnorm_latest.npz",
        model_path.parent.parent / "obsnorm_latest.npz",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = np.load(candidate)
            mean = np.asarray(payload["mean"])
        except Exception:
            continue
        return int(mean.shape[0]) if mean.ndim >= 1 else None
    return None


def _build_history_pool(role: str, env_config: dict[str, Any], max_items: int, seed: int) -> OpponentPool | None:
    game_mode = str(env_config.get("world", {}).get("game_mode", "escape"))
    model_paths = discover_role_model_paths(role, game_mode=game_mode, max_items=max_items)
    expected_obs_dim = _expected_obs_dim(role, env_config)
    if not model_paths:
        return None
    pool = OpponentPool(max_size=max_items, seed=seed)
    for model_path in model_paths:
        observed_dim = _obs_norm_dim_hint(model_path)
        if observed_dim is not None and observed_dim != expected_obs_dim:
            continue
        pool.add(
            _cached_factory(
                lambda path=model_path, model_role=role, env_cfg=deepcopy(env_config): SB3PolicyController(
                    model_role,
                    path,
                    env_config=env_cfg,
                    deterministic=False,
                )
            )
        )
    return pool if len(pool) > 0 else None


def build_single_agent_opponent(
    controlled_agent: str,
    seed: int,
    env_config: dict[str, Any] | None = None,
    train_cfg: dict[str, Any] | None = None,
):
    env_config = env_config or load_env_config()
    if train_cfg is None:
        if controlled_agent == "enemy":
            return MixedPlayerController(
                controllers=[
                    HeuristicPlayerController(),
                    RandomPlayerController(seed=seed),
                ],
                seed=seed,
            )
        return RuleBasedEnemyController()

    curriculum_cfg = train_cfg.get("opponent_curriculum", {})
    history_limit = int(curriculum_cfg.get("max_history_pool", 8))
    random_seed_rng = random.Random(seed)

    if controlled_agent == "enemy":
        history_pool = _build_history_pool("player", env_config, max_items=history_limit, seed=seed)
        weighted_factories: list[tuple[Callable[[], object], float]] = [
            (_cached_factory(lambda: HeuristicPlayerController()), float(curriculum_cfg.get("heuristic_weight", 0.6))),
            (
                _cached_factory(lambda: RandomPlayerController(seed=random_seed_rng.randrange(1_000_000_000))),
                float(curriculum_cfg.get("random_weight", 0.2)),
            ),
        ]
        if history_pool is not None and curriculum_cfg.get("use_history_pool", True):
            weighted_factories.append((lambda pool=history_pool: pool.sample(), float(curriculum_cfg.get("history_weight", 0.2))))
        return WeightedControllerChooser(weighted_factories, seed=seed)

    history_pool = _build_history_pool("enemy", env_config, max_items=history_limit, seed=seed)
    weighted_factories = [
        (_cached_factory(lambda: RuleBasedEnemyController()), float(curriculum_cfg.get("heuristic_weight", 0.7))),
        (
            _cached_factory(lambda: RandomEnemyController(seed=random_seed_rng.randrange(1_000_000_000))),
            float(curriculum_cfg.get("random_weight", 0.1)),
        ),
    ]
    if history_pool is not None and curriculum_cfg.get("use_history_pool", True):
        weighted_factories.append((lambda pool=history_pool: pool.sample(), float(curriculum_cfg.get("history_weight", 0.2))))
    return WeightedControllerChooser(weighted_factories, seed=seed)


def make_single_agent_env_factory(
    controlled_agent: str,
    env_config: dict[str, Any],
    seed: int,
    rank: int,
    monitor_path: Path | None = None,
    opponent_factory: Callable[[], object] | None = None,
    use_action_masking: bool = False,
):
    def _init():
        from stable_baselines3.common.monitor import Monitor

        controller = opponent_factory() if opponent_factory is not None else build_single_agent_opponent(controlled_agent, seed + rank, env_config=env_config)
        env = LibraryEscapeEnv(
            controlled_agent=controlled_agent,
            opponent_controller=controller,
            env_config=deepcopy(env_config),
            seed=seed + rank,
        )
        if use_action_masking:
            from sb3_contrib.common.wrappers import ActionMasker

            env = ActionMasker(env, lambda wrapped_env: wrapped_env.unwrapped.action_masks())
        return Monitor(env, filename=str(monitor_path) if monitor_path else None)

    return _init


def build_vec_env(env_fns: list[Callable[[], object]]):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    return DummyVecEnv(env_fns) if len(env_fns) == 1 else SubprocVecEnv(env_fns, start_method="spawn")


def find_resume_vecnormalize_path(model_path: Path) -> Path | None:
    candidates = [
        model_path.parent / "vecnormalize.pkl",
        model_path.parent.parent / "vecnormalize.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def wrap_with_vec_normalize(vec_env, vec_cfg: dict[str, Any], training: bool, stats_path: Path | None = None):
    from stable_baselines3.common.vec_env import VecNormalize

    if not vec_cfg.get("enabled", False):
        return vec_env

    if stats_path and stats_path.exists() and stats_path.suffix == ".pkl":
        wrapped = VecNormalize.load(str(stats_path), vec_env)
        wrapped.training = training
        wrapped.norm_reward = bool(vec_cfg.get("norm_reward", True) and training)
        return wrapped

    return VecNormalize(
        vec_env,
        training=training,
        norm_obs=bool(vec_cfg.get("norm_obs", True)),
        norm_reward=bool(vec_cfg.get("norm_reward", True) and training),
        clip_obs=float(vec_cfg.get("clip_obs", 10.0)),
        clip_reward=float(vec_cfg.get("clip_reward", 10.0)),
        gamma=float(vec_cfg.get("gamma", 0.99)),
    )


def save_vecnormalize_artifacts(vec_env, model_output_path: Path) -> None:
    from stable_baselines3.common.vec_env import VecNormalize

    if not isinstance(vec_env, VecNormalize):
        return

    vec_path = model_output_path.parent / "vecnormalize.pkl"
    vec_env.save(str(vec_path))

    stats_path = model_output_path.with_suffix(".obsnorm.npz")
    np.savez(
        stats_path,
        mean=np.asarray(vec_env.obs_rms.mean, dtype=np.float64),
        var=np.asarray(vec_env.obs_rms.var, dtype=np.float64),
        epsilon=np.asarray([vec_env.epsilon], dtype=np.float64),
        clip_obs=np.asarray([vec_env.clip_obs], dtype=np.float64),
    )
    latest_stats_path = model_output_path.parent / "obsnorm_latest.npz"
    np.savez(
        latest_stats_path,
        mean=np.asarray(vec_env.obs_rms.mean, dtype=np.float64),
        var=np.asarray(vec_env.obs_rms.var, dtype=np.float64),
        epsilon=np.asarray([vec_env.epsilon], dtype=np.float64),
        clip_obs=np.asarray([vec_env.clip_obs], dtype=np.float64),
    )


def default_device(train_cfg: dict[str, Any]) -> str:
    return str(train_cfg.get("device", "auto"))


def interpolate_across_rounds(
    schedule_cfg: Any,
    round_idx: int,
    total_rounds: int,
    default: float,
) -> float:
    """Linear interpolation of a hyperparameter across self-play rounds.

    ``round_idx`` is 1-indexed. If ``schedule_cfg`` is a scalar or unrecognised
    shape, the scalar (or default) is returned unchanged.
    """
    if isinstance(schedule_cfg, (int, float)):
        return float(schedule_cfg)
    if not isinstance(schedule_cfg, dict):
        return float(default)
    schedule_type = str(schedule_cfg.get("type", "linear")).lower()
    start = float(schedule_cfg.get("start", default))
    end = float(schedule_cfg.get("end", start))
    if schedule_type == "constant" or total_rounds <= 1:
        return start
    progress = (round_idx - 1) / max(1, (total_rounds - 1))
    return start + (end - start) * progress


def apply_round_hyperparameter_schedules(
    base_train_cfg: dict[str, Any],
    round_idx: int,
    total_rounds: int,
) -> dict[str, Any]:
    """Return a deepcopy of ``base_train_cfg`` with per-round hparams filled in.

    Reads ``learning_rate_schedule``, ``clip_range_schedule`` and
    ``ent_coef_schedule`` entries and overwrites the corresponding scalar keys.
    Keys without a schedule are left untouched.
    """
    round_cfg = deepcopy(base_train_cfg)
    schedule_map = {
        "learning_rate": "learning_rate_schedule",
        "clip_range": "clip_range_schedule",
        "ent_coef": "ent_coef_schedule",
    }
    for scalar_key, schedule_key in schedule_map.items():
        schedule_cfg = base_train_cfg.get(schedule_key)
        if schedule_cfg is None:
            continue
        default_scalar = float(base_train_cfg.get(scalar_key, 0.0))
        round_cfg[scalar_key] = interpolate_across_rounds(
            schedule_cfg, round_idx, total_rounds, default_scalar
        )
    return round_cfg


def override_model_hyperparameters(
    model,
    *,
    learning_rate: float,
    clip_range: float,
    ent_coef: float,
) -> None:
    """Force a loaded SB3 PPO model to use the round's hyperparameters.

    ``PPO.load`` restores the schedules that were saved with the checkpoint,
    so a naive resume would ignore new learning-rate / clip-range / entropy
    settings. This helper replaces them with constants for the next phase.
    """
    from stable_baselines3.common.utils import get_schedule_fn

    model.learning_rate = float(learning_rate)
    model.lr_schedule = get_schedule_fn(float(learning_rate))
    model.clip_range = get_schedule_fn(float(clip_range))
    if hasattr(model, "clip_range_vf") and model.clip_range_vf is not None:
        model.clip_range_vf = get_schedule_fn(float(clip_range))
    model.ent_coef = float(ent_coef)


def load_role_configs(role: str, preset: str | None, game_mode: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    env_config = build_game_mode_env_config(game_mode=game_mode, manual_collect_required=False, interactive=False)
    root_training = load_training_config()
    train_config = deepcopy(root_training["single_agent"])
    env_config, train_config = apply_preset("single_agent", preset, env_config, train_config, root_training)
    if role not in {"enemy", "player"}:
        raise ValueError(f"Unsupported role: {role}")
    return env_config, train_config, root_training


def write_training_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(json.dumps(payload, default=str))
    summary_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=True), encoding="utf-8")


def write_model_metadata(model_path: Path, payload: dict[str, Any]) -> None:
    metadata_path = model_path.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(json.loads(json.dumps(payload, default=str)), indent=2, ensure_ascii=True), encoding="utf-8")


def resolve_single_agent_run_dir(role: str, train_cfg: dict[str, Any], run_name: str | None, game_mode: str) -> Path:
    checkpoint_key = "enemy_checkpoint_dir" if role == "enemy" else "player_checkpoint_dir"
    base_dir = resolve_repo_path(train_cfg[checkpoint_key])
    base_dir = base_dir / normalize_game_mode(game_mode)
    name = run_name or timestamped_run_name(role)
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_selfplay_run_dir(train_cfg: dict[str, Any], run_name: str | None, game_mode: str) -> Path:
    base_dir = resolve_repo_path(train_cfg["checkpoint_root_dir"])
    base_dir = base_dir / normalize_game_mode(game_mode)
    name = run_name or timestamped_run_name("selfplay")
    run_dir = base_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
