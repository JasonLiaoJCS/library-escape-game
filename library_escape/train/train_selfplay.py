"""Alternating self-play training for player and enemy policies."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from ..agents.opponent_pool import OpponentPool, PooledController, WeightedControllerChooser
from ..agents.ppo_agent import SB3PolicyController
from ..agents.rule_based_enemy import RandomEnemyController, RuleBasedEnemyController
from ..agents.rule_based_player import HeuristicPlayerController, RandomPlayerController
from ..config import REPO_ROOT, load_training_config, read_json, resolve_repo_path
from ..game_modes import build_game_mode_env_config, game_mode_label, normalize_game_mode
from .callbacks import EvalHistoryCallback, TrainingStatusCallback
from .common import (
    AlgorithmSpec,
    apply_preset,
    apply_round_hyperparameter_schedules,
    apply_runtime_overrides,
    build_policy_kwargs,
    build_vec_env,
    default_device,
    find_resume_vecnormalize_path,
    format_device_runtime,
    make_single_agent_env_factory,
    override_model_hyperparameters,
    resolve_algorithm_spec,
    resolve_device_runtime,
    resolve_selfplay_run_dir,
    save_vecnormalize_artifacts,
    wrap_with_vec_normalize,
    write_model_metadata,
    write_training_summary,
)
from .io_utils import atomic_write_json


@dataclass(frozen=True, slots=True)
class PhaseTrainingResult:
    model_path: Path
    training_mean_reward: float | None
    last_mean_reward: float | None
    best_mean_reward: float | None

    @property
    def balancing_reward(self) -> float | None:
        return self.training_mean_reward if self.training_mean_reward is not None else self.last_mean_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run alternating PPO self-play for Library Escape.")
    parser.add_argument("--preset", type=str, default="balanced")
    parser.add_argument("--game-mode", type=str, choices=("collection", "escape"), default="escape")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--timesteps-per-round", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-run", type=str, default=None, help="Resume self-play from an earlier run directory.")
    parser.add_argument("--resume-enemy", type=str, default=None, help="Resume enemy policy from a checkpoint path.")
    parser.add_argument("--resume-player", type=str, default=None, help="Resume player policy from a checkpoint path.")
    parser.add_argument("--overrides-json", type=str, default=None, help="Deep-merge overrides for env/train config.")
    return parser.parse_args()


def _cached_factory(builder):
    controller = None

    def _factory():
        nonlocal controller
        if controller is None:
            controller = builder()
        return controller

    return _factory


def _bootstrap_player_factory(seed: int, curriculum_cfg: dict) -> WeightedControllerChooser:
    return WeightedControllerChooser(
        [
            (_cached_factory(lambda: HeuristicPlayerController()), float(curriculum_cfg.get("bootstrap_heuristic_weight", 0.8))),
            (
                _cached_factory(lambda: RandomPlayerController(seed=seed)),
                float(curriculum_cfg.get("bootstrap_random_weight", 0.2)),
            ),
        ],
        seed=seed,
    )


def _bootstrap_enemy_factory(seed: int, curriculum_cfg: dict) -> WeightedControllerChooser:
    return WeightedControllerChooser(
        [
            (_cached_factory(lambda: RuleBasedEnemyController()), float(curriculum_cfg.get("bootstrap_heuristic_weight", 0.8))),
            (
                _cached_factory(lambda: RandomEnemyController(seed=seed)),
                float(curriculum_cfg.get("bootstrap_random_weight", 0.2)),
            ),
        ],
        seed=seed,
    )


def _make_selfplay_opponent_factory(pool: OpponentPool, fallback_factory, curriculum_cfg: dict):
    def _factory():
        controller = PooledController(
            pool=pool,
            fallback_factory=fallback_factory,
            latest_weight=float(curriculum_cfg.get("latest_weight", 0.5)),
            historical_weight=float(curriculum_cfg.get("historical_weight", 0.5)),
        )
        controller.reset()
        return controller

    return _factory


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _base_role_timestep_multipliers(train_cfg: dict) -> tuple[float, float]:
    role_cfg = train_cfg.get("role_timestep_multipliers", {})
    enemy_multiplier = float(role_cfg.get("enemy", 1.0))
    player_multiplier = float(role_cfg.get("player", 1.0))
    return max(0.01, enemy_multiplier), max(0.01, player_multiplier)


def _scaled_timesteps(base_timesteps: int, multiplier: float) -> int:
    return max(1, int(math.floor((base_timesteps * multiplier) + 0.5)))


def _round_timestep_plan(
    train_cfg: dict,
    *,
    round_idx: int,
    last_enemy_reward: float | None,
    last_player_reward: float | None,
) -> dict[str, object]:
    """Choose enemy/player phase lengths for the next self-play round.

    Enemy policies currently improve faster than player policies. The static
    multipliers give the player a larger default budget; adaptive adjustment
    then shifts additional time toward the weaker side using the previous
    round's training reward, falling back to eval reward if no training mean
    is available.
    """

    base_timesteps = int(train_cfg["timesteps_per_round"])
    enemy_multiplier, player_multiplier = _base_role_timestep_multipliers(train_cfg)
    adaptive_cfg = train_cfg.get("adaptive_timesteps", {})
    adaptive_enabled = bool(adaptive_cfg.get("enabled", False))
    reward_gap = None
    adjustment = 0.0

    if adaptive_enabled and last_enemy_reward is not None and last_player_reward is not None:
        warmup_rounds = int(adaptive_cfg.get("warmup_rounds", 1))
        if round_idx > warmup_rounds:
            gap_scale = max(1e-6, float(adaptive_cfg.get("reward_gap_scale", 300.0)))
            reward_gap = float(last_enemy_reward) - float(last_player_reward)
            adjustment = _clamp(reward_gap / gap_scale, -1.0, 1.0)
            enemy_multiplier *= 1.0 - adjustment * float(adaptive_cfg.get("max_enemy_adjustment", 0.25))
            player_multiplier *= 1.0 + adjustment * float(adaptive_cfg.get("max_player_adjustment", 0.35))

    min_multiplier = float(adaptive_cfg.get("min_multiplier", 0.01))
    max_multiplier = float(adaptive_cfg.get("max_multiplier", 10.0))
    enemy_multiplier = _clamp(enemy_multiplier, min_multiplier, max_multiplier)
    player_multiplier = _clamp(player_multiplier, min_multiplier, max_multiplier)

    return {
        "round": round_idx,
        "base_timesteps_per_round": base_timesteps,
        "enemy_multiplier": enemy_multiplier,
        "player_multiplier": player_multiplier,
        "enemy_timesteps": _scaled_timesteps(base_timesteps, enemy_multiplier),
        "player_timesteps": _scaled_timesteps(base_timesteps, player_multiplier),
        "adaptive_enabled": adaptive_enabled,
        "last_enemy_reward": last_enemy_reward,
        "last_player_reward": last_player_reward,
        "reward_gap": reward_gap,
        "adaptive_adjustment": adjustment,
    }


def _base_selfplay_round_timesteps(train_cfg: dict) -> int:
    base_timesteps = int(train_cfg["timesteps_per_round"])
    enemy_multiplier, player_multiplier = _base_role_timestep_multipliers(train_cfg)
    return _scaled_timesteps(base_timesteps, enemy_multiplier) + _scaled_timesteps(base_timesteps, player_multiplier)


def _latest_round_dir_with_models(run_dir: Path, role: str) -> Path | None:
    role_root = run_dir / role
    if not role_root.exists() or not role_root.is_dir():
        return None

    candidates: list[tuple[int, float, Path]] = []
    for child in role_root.iterdir():
        if not child.is_dir() or not child.name.startswith("round_"):
            continue
        suffix = child.name.split("round_", 1)[-1]
        try:
            round_idx = int(suffix)
        except ValueError:
            continue
        models_dir = child / "models"
        if not models_dir.exists() or not models_dir.is_dir():
            continue
        has_zip = any(p.suffix == ".zip" for p in models_dir.iterdir() if p.is_file())
        if not has_zip:
            continue
        candidates.append((round_idx, child.stat().st_mtime, child))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
    return candidates[0][2]


def _resolve_model_from_round_dir(round_dir: Path, role: str) -> Path | None:
    models_dir = round_dir / "models"
    if not models_dir.exists() or not models_dir.is_dir():
        return None

    preferred_prefix = f"{role}_round_"
    preferred_latest = sorted(
        [
            p
            for p in models_dir.iterdir()
            if p.is_file() and p.suffix == ".zip" and p.name.startswith(preferred_prefix) and p.name.endswith("_latest.zip")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if preferred_latest:
        return preferred_latest[0].resolve()

    best_model = models_dir / "best_model.zip"
    if best_model.exists():
        return best_model.resolve()

    all_zips = sorted(
        [p for p in models_dir.iterdir() if p.is_file() and p.suffix == ".zip"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if all_zips:
        return all_zips[0].resolve()

    return None


def _resolve_models_from_run_artifacts(run_dir: Path) -> tuple[Path | None, Path | None]:
    enemy_round_dir = _latest_round_dir_with_models(run_dir, "enemy")
    player_round_dir = _latest_round_dir_with_models(run_dir, "player")
    enemy_model = _resolve_model_from_round_dir(enemy_round_dir, "enemy") if enemy_round_dir else None
    player_model = _resolve_model_from_round_dir(player_round_dir, "player") if player_round_dir else None
    return enemy_model, player_model


def _resolve_resume_selfplay_models(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    resume_run_dir = resolve_repo_path(args.resume_run) if args.resume_run else None
    resume_enemy_model = resolve_repo_path(args.resume_enemy) if args.resume_enemy else None
    resume_player_model = resolve_repo_path(args.resume_player) if args.resume_player else None

    if resume_run_dir is not None:
        summary_path = resume_run_dir / "training_summary.json"
        if not summary_path.exists():
            # Allow passing a self-play game-mode root directory (for example,
            # checkpoints/selfplay/escape) by auto-selecting the latest run
            # that has a training summary.
            candidate_runs = []
            if resume_run_dir.exists() and resume_run_dir.is_dir():
                for child in resume_run_dir.iterdir():
                    child_summary = child / "training_summary.json"
                    if child.is_dir() and child_summary.exists():
                        candidate_runs.append((child_summary.stat().st_mtime, child))
            if candidate_runs:
                _, latest_run = max(candidate_runs, key=lambda item: (item[0], item[1].name))
                resume_run_dir = latest_run
                summary_path = resume_run_dir / "training_summary.json"
            else:
                recovered_enemy, recovered_player = _resolve_models_from_run_artifacts(resume_run_dir)
                if resume_enemy_model is None:
                    resume_enemy_model = recovered_enemy
                if resume_player_model is None:
                    resume_player_model = recovered_player
                if resume_enemy_model is None and resume_player_model is None:
                    raise FileNotFoundError(
                        "Resume run summary not found and no recoverable checkpoints found "
                        f"under run directory: {resume_run_dir}"
                    )
        if summary_path.exists():
            summary = read_json(summary_path)
            if resume_enemy_model is None and summary.get("final_enemy_model"):
                resume_enemy_model = resolve_repo_path(summary["final_enemy_model"])
            if resume_player_model is None and summary.get("final_player_model"):
                resume_player_model = resolve_repo_path(summary["final_player_model"])

        if resume_enemy_model is None or resume_player_model is None:
            recovered_enemy, recovered_player = _resolve_models_from_run_artifacts(resume_run_dir)
            if resume_enemy_model is None:
                resume_enemy_model = recovered_enemy
            if resume_player_model is None:
                resume_player_model = recovered_player

    for label, model_path in (("enemy", resume_enemy_model), ("player", resume_player_model)):
        if model_path is not None and not model_path.exists():
            raise FileNotFoundError(f"Resume {label} checkpoint not found: {model_path}")

    return resume_run_dir, resume_enemy_model, resume_player_model


def _train_phase(
    controlled_agent: str,
    env_config: dict,
    train_cfg: dict,
    algo_spec: AlgorithmSpec,
    phase_dir: Path,
    opponent_pool: OpponentPool,
    fallback_factory,
    seed: int,
    timesteps: int,
    phase_name: str,
    phase_index: int,
    phase_total: int,
    global_total_timesteps: int,
    global_step_offset: int,
    resume_model_path: Path | None = None,
    shared_progress_path: Path | None = None,
    shared_progress_history_path: Path | None = None,
    shared_eval_history_path: Path | None = None,
) -> PhaseTrainingResult:
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    if algo_spec.model_cls is None or algo_spec.eval_callback_cls is None:
        raise RuntimeError(f"Algorithm {algo_spec.key} does not support built-in self-play training.")

    models_dir = phase_dir / "models"
    monitor_dir = phase_dir / "monitor"
    tensorboard_dir = phase_dir / "tb"
    models_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    n_envs = int(train_cfg["n_envs"])
    vecnorm_cfg = deepcopy(train_cfg.get("vec_normalize", {}))
    if "gamma" not in vecnorm_cfg:
        vecnorm_cfg["gamma"] = float(train_cfg["gamma"])

    curriculum_cfg = train_cfg.get("opponent_curriculum", {})
    opponent_factory = _make_selfplay_opponent_factory(opponent_pool, fallback_factory, curriculum_cfg)
    env_fns = [
        make_single_agent_env_factory(
            controlled_agent=controlled_agent,
            env_config=env_config,
            seed=seed,
            rank=rank,
            monitor_path=monitor_dir / f"train_env_{rank}.monitor.csv",
            opponent_factory=opponent_factory,
            use_action_masking=algo_spec.uses_action_masks,
        )
        for rank in range(n_envs)
    ]
    vec_env = build_vec_env(env_fns)

    resume_stats_path = find_resume_vecnormalize_path(resume_model_path) if resume_model_path else None
    vec_env = wrap_with_vec_normalize(vec_env, vecnorm_cfg, training=True, stats_path=resume_stats_path)

    eval_env_fns = [
        make_single_agent_env_factory(
            controlled_agent=controlled_agent,
            env_config=env_config,
            seed=seed + 50_000,
            rank=0,
            monitor_path=monitor_dir / "eval_env.monitor.csv",
            opponent_factory=fallback_factory,
            use_action_masking=algo_spec.uses_action_masks,
        )
    ]
    eval_env = build_vec_env(eval_env_fns)
    eval_env = wrap_with_vec_normalize(eval_env, vecnorm_cfg, training=False, stats_path=resume_stats_path)

    progress_path = shared_progress_path or (phase_dir / "progress.json")
    progress_history_path = shared_progress_history_path or (phase_dir / "progress_history.jsonl")
    eval_history_path = shared_eval_history_path or (phase_dir / "eval_history.jsonl")

    status_tracker = TrainingStatusCallback(
        progress_path=progress_path,
        history_path=progress_history_path,
        total_timesteps=timesteps,
        phase_name=phase_name,
        phase_index=phase_index,
        phase_total=phase_total,
        global_step_offset=global_step_offset,
        global_total_timesteps=global_total_timesteps,
        log_interval_seconds=float(train_cfg["log_interval_seconds"]),
    )
    status_callback = status_tracker.callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(train_cfg["checkpoint_freq"]) // n_envs, 1),
        save_path=str(models_dir),
        name_prefix=f"{controlled_agent}_checkpoint",
        save_vecnormalize=bool(vecnorm_cfg.get("enabled", False)),
    )
    eval_callback = algo_spec.eval_callback_cls(
        eval_env=eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(phase_dir),
        eval_freq=max(int(train_cfg["eval_freq"]) // n_envs, 1),
        n_eval_episodes=int(train_cfg.get("eval_episodes", 8)),
        deterministic=True,
        render=False,
        callback_after_eval=EvalHistoryCallback(eval_history_path, phase_name=phase_name).callback,
        warn=False,
    )
    callbacks = CallbackList([status_callback, checkpoint_callback, eval_callback])

    if resume_model_path is not None:
        model = algo_spec.model_cls.load(str(resume_model_path), env=vec_env, device=default_device(train_cfg))
        model.tensorboard_log = str(tensorboard_dir)
        override_model_hyperparameters(
            model,
            learning_rate=float(train_cfg["learning_rate"]),
            clip_range=float(train_cfg["clip_range"]),
            ent_coef=float(train_cfg["ent_coef"]),
        )
    else:
        model = algo_spec.model_cls(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=float(train_cfg["learning_rate"]),
            n_steps=int(train_cfg["n_steps"]),
            batch_size=int(train_cfg["batch_size"]),
            gamma=float(train_cfg["gamma"]),
            gae_lambda=float(train_cfg["gae_lambda"]),
            clip_range=float(train_cfg["clip_range"]),
            ent_coef=float(train_cfg["ent_coef"]),
            vf_coef=float(train_cfg["vf_coef"]),
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            device=default_device(train_cfg),
            seed=seed,
            policy_kwargs=build_policy_kwargs(train_cfg),
        )

    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)
    final_path = models_dir / f"{phase_name}_latest"
    model.save(str(final_path))
    save_vecnormalize_artifacts(vec_env, final_path.with_suffix(".zip"))
    best_model_path = models_dir / "best_model.zip"
    if best_model_path.exists():
        save_vecnormalize_artifacts(vec_env, best_model_path)
    training_mean_reward = _finite_float((status_tracker.last_payload or {}).get("mean_episode_reward"))

    write_training_summary(
        phase_dir / "training_summary.json",
        {
            "mode": "self_play_phase",
            "algorithm": algo_spec.summary_name,
            "phase_name": phase_name,
            "controlled_agent": controlled_agent,
            "run_dir": phase_dir,
            "final_model": final_path.with_suffix(".zip"),
            "best_model": best_model_path if best_model_path.exists() else None,
            "phase_total_timesteps": timesteps,
            "global_step_offset": global_step_offset,
            "global_total_timesteps": global_total_timesteps,
            "training_mean_reward": training_mean_reward,
            "last_mean_reward": _finite_float(getattr(eval_callback, "last_mean_reward", None)),
            "best_mean_reward": _finite_float(getattr(eval_callback, "best_mean_reward", None)),
            "env_config": env_config,
            "train_config": train_cfg,
            "seed": seed,
            "resume_model_path": resume_model_path,
            "game_mode": env_config.get("world", {}).get("game_mode", "escape"),
            "game_mode_label": game_mode_label(env_config.get("world", {}).get("game_mode", "escape")),
        },
    )
    write_model_metadata(
        final_path.with_suffix(".zip"),
        {
            "role": controlled_agent,
            "algorithm": algo_spec.summary_name,
            "run_dir": phase_dir,
            "game_mode": env_config.get("world", {}).get("game_mode", "escape"),
            "game_mode_label": game_mode_label(env_config.get("world", {}).get("game_mode", "escape")),
        },
    )
    if best_model_path.exists():
        write_model_metadata(
            best_model_path,
            {
                "role": controlled_agent,
                "algorithm": algo_spec.summary_name,
                "run_dir": phase_dir,
                "game_mode": env_config.get("world", {}).get("game_mode", "escape"),
                "game_mode_label": game_mode_label(env_config.get("world", {}).get("game_mode", "escape")),
            },
        )

    vec_env.close()
    eval_env.close()
    return PhaseTrainingResult(
        model_path=final_path.with_suffix(".zip"),
        training_mean_reward=training_mean_reward,
        last_mean_reward=_finite_float(getattr(eval_callback, "last_mean_reward", None)),
        best_mean_reward=_finite_float(getattr(eval_callback, "best_mean_reward", None)),
    )


def _run_external_mappo_recipe(run_dir: Path, env_config: dict, train_cfg: dict, args: argparse.Namespace) -> None:
    recipe_cfg = train_cfg.get("opponent_curriculum", {}).get("mappo_recipe", {})
    command_template = str(recipe_cfg.get("external_command", "")).strip()
    if not command_template:
        raise RuntimeError(
            "Self-play algorithm is set to mappo_recipe, but no external_command is configured in "
            "configs/training.yaml or GUI overrides."
        )

    request_path = run_dir / "mappo_recipe_request.json"
    atomic_write_json(
        request_path,
        {
            "seed": args.seed,
            "preset": args.preset,
            "run_dir": str(run_dir),
            "rounds": int(train_cfg["rounds"]),
            "timesteps_per_round": int(train_cfg["timesteps_per_round"]),
            "n_envs": int(train_cfg["n_envs"]),
            "env_config": env_config,
            "train_config": train_cfg,
        },
        indent=2,
    )

    command = command_template.format(
        repo_root=str(REPO_ROOT),
        run_dir=str(run_dir),
        request_json=str(request_path),
        seed=args.seed,
        rounds=int(train_cfg["rounds"]),
        timesteps_per_round=int(train_cfg["timesteps_per_round"]),
        n_envs=int(train_cfg["n_envs"]),
    )
    subprocess.run(command, cwd=str(REPO_ROOT), shell=True, check=True)

    write_training_summary(
        run_dir / "training_summary.json",
        {
            "mode": "self_play_mappo_recipe",
            "algorithm": "mappo_recipe",
            "run_dir": run_dir,
            "request_json": request_path,
            "external_command": command,
            "env_config": env_config,
            "train_config": train_cfg,
            "seed": args.seed,
            "preset": args.preset,
            "game_mode": env_config.get("world", {}).get("game_mode", "escape"),
            "game_mode_label": game_mode_label(env_config.get("world", {}).get("game_mode", "escape")),
            "notes": recipe_cfg.get("notes", ""),
        },
    )


def main() -> None:
    args = parse_args()
    game_mode = normalize_game_mode(args.game_mode)
    env_config = build_game_mode_env_config(game_mode=game_mode, manual_collect_required=False, interactive=False)
    root_train_cfg = load_training_config()
    train_cfg = deepcopy(root_train_cfg["self_play"])
    env_config, train_cfg = apply_preset("self_play", args.preset, env_config, train_cfg, root_train_cfg)

    if args.rounds is not None:
        train_cfg["rounds"] = int(args.rounds)
    if args.timesteps_per_round is not None:
        train_cfg["timesteps_per_round"] = int(args.timesteps_per_round)
    if args.n_envs is not None:
        train_cfg["n_envs"] = int(args.n_envs)
    if args.device is not None:
        train_cfg["device"] = args.device
    env_config, train_cfg = apply_runtime_overrides(env_config, train_cfg, args.overrides_json)
    device_info = resolve_device_runtime(train_cfg)
    print(format_device_runtime(device_info))

    run_dir = resolve_selfplay_run_dir(train_cfg, args.run_name, game_mode)
    algo_spec = resolve_algorithm_spec(train_cfg.get("algorithm", "league_ppo"))
    if algo_spec.uses_action_masks and str(env_config["action"]["type"]).lower() != "discrete":
        raise RuntimeError("Maskable PPO self-play requires `action.type: discrete` in configs/env.yaml or GUI overrides.")
    if algo_spec.key == "mappo_recipe":
        _run_external_mappo_recipe(run_dir, env_config, train_cfg, args)
        print(f"MAPPO recipe run directory: {run_dir}")
        return

    enemy_root = run_dir / "enemy"
    player_root = run_dir / "player"
    enemy_root.mkdir(parents=True, exist_ok=True)
    player_root.mkdir(parents=True, exist_ok=True)

    rounds = int(train_cfg["rounds"])
    total_phases = rounds * 2
    base_round_timesteps = _base_selfplay_round_timesteps(train_cfg)
    curriculum_cfg = train_cfg.get("opponent_curriculum", {})

    player_pool = OpponentPool(max_size=int(train_cfg["opponent_pool_size"]), seed=args.seed)
    enemy_pool = OpponentPool(max_size=int(train_cfg["opponent_pool_size"]), seed=args.seed + 1000)

    progress_path = run_dir / "progress.json"
    progress_history_path = run_dir / "progress_history.jsonl"
    eval_history_path = run_dir / "eval_history.jsonl"

    resume_run_dir, current_enemy_model, current_player_model = _resolve_resume_selfplay_models(args)

    if current_enemy_model is not None:
        enemy_pool.add(
            _cached_factory(
                lambda path=current_enemy_model, env_cfg=deepcopy(env_config): SB3PolicyController(
                    "enemy",
                    path,
                    env_config=env_cfg,
                    deterministic=False,
                )
            )
        )
    if current_player_model is not None:
        player_pool.add(
            _cached_factory(
                lambda path=current_player_model, env_cfg=deepcopy(env_config): SB3PolicyController(
                    "player",
                    path,
                    env_config=env_cfg,
                    deterministic=False,
                )
            )
        )

    phase_counter = 0
    completed_timesteps = 0
    last_enemy_reward: float | None = None
    last_player_reward: float | None = None
    timestep_schedule: list[dict[str, object]] = []
    for round_idx in range(1, rounds + 1):
        round_train_cfg = apply_round_hyperparameter_schedules(
            train_cfg, round_idx=round_idx, total_rounds=rounds
        )
        round_timestep_plan = _round_timestep_plan(
            round_train_cfg,
            round_idx=round_idx,
            last_enemy_reward=last_enemy_reward,
            last_player_reward=last_player_reward,
        )
        enemy_timesteps = int(round_timestep_plan["enemy_timesteps"])
        player_timesteps = int(round_timestep_plan["player_timesteps"])
        timestep_schedule.append(dict(round_timestep_plan))
        remaining_rounds_after_this = max(0, rounds - round_idx)
        print(
            f"[round {round_idx}] lr={round_train_cfg.get('learning_rate')}, "
            f"clip_range={round_train_cfg.get('clip_range')}, "
            f"ent_coef={round_train_cfg.get('ent_coef')}"
        )
        print(
            f"[round {round_idx}] timesteps enemy={enemy_timesteps} "
            f"(x{round_timestep_plan['enemy_multiplier']:.2f}), "
            f"player={player_timesteps} (x{round_timestep_plan['player_multiplier']:.2f}), "
            f"last_reward enemy={last_enemy_reward}, player={last_player_reward}"
        )
        phase_counter += 1
        enemy_phase_dir = enemy_root / f"round_{round_idx:02d}"
        enemy_global_total = completed_timesteps + enemy_timesteps + player_timesteps + (
            remaining_rounds_after_this * base_round_timesteps
        )
        enemy_result = _train_phase(
            controlled_agent="enemy",
            env_config=env_config,
            train_cfg=round_train_cfg,
            algo_spec=algo_spec,
            phase_dir=enemy_phase_dir,
            opponent_pool=player_pool,
            fallback_factory=lambda round_seed=args.seed + round_idx: _bootstrap_player_factory(round_seed, curriculum_cfg),
            seed=args.seed + round_idx,
            timesteps=enemy_timesteps,
            phase_name=f"enemy_round_{round_idx:02d}",
            phase_index=phase_counter,
            phase_total=total_phases,
            global_total_timesteps=enemy_global_total,
            global_step_offset=completed_timesteps,
            resume_model_path=current_enemy_model,
            shared_progress_path=progress_path,
            shared_progress_history_path=progress_history_path,
            shared_eval_history_path=eval_history_path,
        )
        completed_timesteps += enemy_timesteps
        current_enemy_model = enemy_result.model_path
        last_enemy_reward = enemy_result.balancing_reward
        timestep_schedule[-1]["actual_enemy_training_mean_reward"] = enemy_result.training_mean_reward
        timestep_schedule[-1]["actual_enemy_last_mean_reward"] = enemy_result.last_mean_reward
        timestep_schedule[-1]["actual_enemy_best_mean_reward"] = enemy_result.best_mean_reward
        enemy_pool.add(
            _cached_factory(
                lambda path=enemy_result.model_path, env_cfg=deepcopy(env_config): SB3PolicyController(
                    "enemy",
                    path,
                    env_config=env_cfg,
                    deterministic=False,
                )
            )
        )

        phase_counter += 1
        player_phase_dir = player_root / f"round_{round_idx:02d}"
        player_global_total = completed_timesteps + player_timesteps + (remaining_rounds_after_this * base_round_timesteps)
        player_result = _train_phase(
            controlled_agent="player",
            env_config=env_config,
            train_cfg=round_train_cfg,
            algo_spec=algo_spec,
            phase_dir=player_phase_dir,
            opponent_pool=enemy_pool,
            fallback_factory=lambda round_seed=args.seed + 5000 + round_idx: _bootstrap_enemy_factory(round_seed, curriculum_cfg),
            seed=args.seed + 5000 + round_idx,
            timesteps=player_timesteps,
            phase_name=f"player_round_{round_idx:02d}",
            phase_index=phase_counter,
            phase_total=total_phases,
            global_total_timesteps=player_global_total,
            global_step_offset=completed_timesteps,
            resume_model_path=current_player_model,
            shared_progress_path=progress_path,
            shared_progress_history_path=progress_history_path,
            shared_eval_history_path=eval_history_path,
        )
        completed_timesteps += player_timesteps
        current_player_model = player_result.model_path
        last_player_reward = player_result.balancing_reward
        timestep_schedule[-1]["actual_player_training_mean_reward"] = player_result.training_mean_reward
        timestep_schedule[-1]["actual_player_last_mean_reward"] = player_result.last_mean_reward
        timestep_schedule[-1]["actual_player_best_mean_reward"] = player_result.best_mean_reward
        player_pool.add(
            _cached_factory(
                lambda path=player_result.model_path, env_cfg=deepcopy(env_config): SB3PolicyController(
                    "player",
                    path,
                    env_config=env_cfg,
                    deterministic=False,
                )
            )
        )

        print(f"[round {round_idx}] enemy -> {enemy_result.model_path}")
        print(f"[round {round_idx}] player -> {player_result.model_path}")

    write_training_summary(
        run_dir / "training_summary.json",
        {
            "mode": "self_play",
            "algorithm": algo_spec.summary_name,
            "device_info": device_info.to_dict(),
            "run_dir": run_dir,
            "enemy_root": enemy_root,
            "player_root": player_root,
            "final_enemy_model": current_enemy_model,
            "final_player_model": current_player_model,
            "actual_total_timesteps": completed_timesteps,
            "selfplay_timestep_schedule": timestep_schedule,
            "progress_path": progress_path,
            "progress_history_path": progress_history_path,
            "eval_history_path": eval_history_path,
            "env_config": env_config,
            "train_config": train_cfg,
            "seed": args.seed,
            "preset": args.preset,
            "game_mode": game_mode,
            "game_mode_label": game_mode_label(game_mode),
            "resume_from_run": resume_run_dir,
            "resume_enemy_model": current_enemy_model,
            "resume_player_model": current_player_model,
        },
    )


if __name__ == "__main__":
    main()
