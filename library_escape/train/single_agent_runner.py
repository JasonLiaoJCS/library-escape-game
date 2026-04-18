"""Shared runner for single-agent training modes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from .callbacks import EvalHistoryCallback, TrainingStatusCallback
from .common import (
    apply_runtime_overrides,
    build_policy_kwargs,
    build_single_agent_opponent,
    build_vec_env,
    default_device,
    find_resume_vecnormalize_path,
    load_role_configs,
    make_single_agent_env_factory,
    resolve_algorithm_spec,
    resolve_single_agent_run_dir,
    save_vecnormalize_artifacts,
    wrap_with_vec_normalize,
    write_model_metadata,
    write_training_summary,
)
from ..config import resolve_repo_path
from ..game_modes import game_mode_label, normalize_game_mode


def build_parser(role: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train a {role} PPO policy for Library Escape.")
    parser.add_argument("--preset", type=str, default="balanced")
    parser.add_argument("--game-mode", type=str, choices=("collection", "escape"), default="escape")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total training timesteps.")
    parser.add_argument("--n-envs", type=int, default=None, help="Override vectorized environment count.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Resume from a saved PPO checkpoint.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional output run directory name.")
    parser.add_argument("--overrides-json", type=str, default=None, help="Deep-merge overrides for env/train config.")
    return parser


def run_single_agent(role: str) -> None:
    if role not in {"enemy", "player"}:
        raise ValueError(f"Unsupported single-agent role: {role}")

    try:
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("stable-baselines3 is required for training. Install with `pip install -e .[rl]`.") from exc

    args = build_parser(role).parse_args()
    game_mode = normalize_game_mode(args.game_mode)
    env_config, train_cfg, _ = load_role_configs(role, args.preset, game_mode)

    if args.timesteps is not None:
        train_cfg["total_timesteps"] = int(args.timesteps)
    if args.n_envs is not None:
        train_cfg["n_envs"] = int(args.n_envs)
    if args.device is not None:
        train_cfg["device"] = args.device
    if args.resume is not None:
        train_cfg["resume_path"] = args.resume
    env_config, train_cfg = apply_runtime_overrides(env_config, train_cfg, args.overrides_json)
    algo_spec = resolve_algorithm_spec(train_cfg.get("algorithm", "ppo"))
    if algo_spec.uses_action_masks and str(env_config["action"]["type"]).lower() != "discrete":
        raise RuntimeError("MaskablePPO requires `action.type: discrete` in configs/env.yaml or GUI overrides.")

    run_dir = resolve_single_agent_run_dir(role, train_cfg, args.run_name, game_mode)
    models_dir = run_dir / "models"
    monitor_dir = run_dir / "monitor"
    tensorboard_dir = run_dir / "tb"
    models_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    n_envs = int(train_cfg["n_envs"])
    total_timesteps = int(train_cfg["total_timesteps"])
    vecnorm_cfg = deepcopy(train_cfg.get("vec_normalize", {}))
    if "gamma" not in vecnorm_cfg:
        vecnorm_cfg["gamma"] = float(train_cfg["gamma"])

    opponent_factory = lambda seed_offset=0: build_single_agent_opponent(role, args.seed + seed_offset)
    env_fns = [
        make_single_agent_env_factory(
            controlled_agent=role,
            env_config=env_config,
            seed=args.seed,
            rank=rank,
            monitor_path=monitor_dir / f"train_env_{rank}.monitor.csv",
            opponent_factory=lambda rank=rank: build_single_agent_opponent(
                role,
                args.seed + rank,
                env_config=env_config,
                train_cfg=train_cfg,
            ),
            use_action_masking=algo_spec.uses_action_masks,
        )
        for rank in range(n_envs)
    ]
    vec_env = build_vec_env(env_fns)

    resume_path = resolve_repo_path(train_cfg["resume_path"]) if train_cfg.get("resume_path") else None
    resume_stats_path = find_resume_vecnormalize_path(resume_path) if resume_path else None
    vec_env = wrap_with_vec_normalize(vec_env, vecnorm_cfg, training=True, stats_path=resume_stats_path)

    eval_env_fns = [
        make_single_agent_env_factory(
            controlled_agent=role,
            env_config=env_config,
            seed=args.seed + 10_000,
            rank=0,
            monitor_path=monitor_dir / "eval_env.monitor.csv",
            opponent_factory=lambda: build_single_agent_opponent(
                role,
                args.seed + 10_000,
                env_config=env_config,
                train_cfg=train_cfg,
            ),
            use_action_masking=algo_spec.uses_action_masks,
        )
    ]
    eval_env = build_vec_env(eval_env_fns)
    eval_env = wrap_with_vec_normalize(eval_env, vecnorm_cfg, training=False, stats_path=resume_stats_path)

    progress_path = run_dir / "progress.json"
    progress_history_path = run_dir / "progress_history.jsonl"
    eval_history_path = run_dir / "eval_history.jsonl"
    status_callback = TrainingStatusCallback(
        progress_path=progress_path,
        history_path=progress_history_path,
        total_timesteps=total_timesteps,
        phase_name=f"train_{role}",
        phase_index=1,
        phase_total=1,
        global_step_offset=0,
        global_total_timesteps=total_timesteps,
        log_interval_seconds=float(train_cfg["log_interval_seconds"]),
    ).callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(train_cfg["checkpoint_freq"]) // n_envs, 1),
        save_path=str(models_dir),
        name_prefix=f"{role}_checkpoint",
        save_vecnormalize=bool(vecnorm_cfg.get("enabled", False)),
    )
    if algo_spec.eval_callback_cls is None:
        raise RuntimeError(f"Algorithm {algo_spec.key} does not support built-in single-agent training.")

    eval_callback = algo_spec.eval_callback_cls(
        eval_env=eval_env,
        best_model_save_path=str(models_dir),
        log_path=str(run_dir),
        eval_freq=max(int(train_cfg["eval_freq"]) // n_envs, 1),
        n_eval_episodes=int(train_cfg["eval_episodes"]),
        deterministic=True,
        render=False,
        callback_after_eval=EvalHistoryCallback(eval_history_path, phase_name=f"train_{role}").callback,
        warn=False,
    )
    callback = CallbackList([status_callback, checkpoint_callback, eval_callback])

    if resume_path is not None:
        assert algo_spec.model_cls is not None
        model = algo_spec.model_cls.load(str(resume_path), env=vec_env, device=default_device(train_cfg))
    else:
        assert algo_spec.model_cls is not None
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
            seed=args.seed,
            policy_kwargs=build_policy_kwargs(train_cfg),
        )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    final_path = models_dir / f"{role}_latest"
    model.save(str(final_path))
    save_vecnormalize_artifacts(vec_env, final_path.with_suffix(".zip"))
    best_model_path = models_dir / "best_model.zip"
    if best_model_path.exists():
        save_vecnormalize_artifacts(vec_env, best_model_path)

    write_training_summary(
        run_dir / "training_summary.json",
        {
            "mode": f"single_agent_{role}",
            "algorithm": algo_spec.summary_name,
            "run_dir": run_dir,
            "final_model": final_path.with_suffix(".zip"),
            "best_model": best_model_path if best_model_path.exists() else None,
            "progress_path": progress_path,
            "progress_history_path": progress_history_path,
            "eval_history_path": eval_history_path,
            "env_config": env_config,
            "train_config": train_cfg,
            "seed": args.seed,
            "preset": args.preset,
            "game_mode": game_mode,
            "game_mode_label": game_mode_label(game_mode),
            "resume_path": resume_path,
        },
    )
    write_model_metadata(
        final_path.with_suffix(".zip"),
        {
            "role": role,
            "algorithm": algo_spec.summary_name,
            "run_dir": run_dir,
            "game_mode": game_mode,
            "game_mode_label": game_mode_label(game_mode),
        },
    )
    if best_model_path.exists():
        write_model_metadata(
            best_model_path,
            {
                "role": role,
                "algorithm": algo_spec.summary_name,
                "run_dir": run_dir,
                "game_mode": game_mode,
                "game_mode_label": game_mode_label(game_mode),
            },
        )

    vec_env.close()
    eval_env.close()
    print(f"Run directory: {run_dir}")
    print(f"Saved final {role} checkpoint to {final_path.with_suffix('.zip')}")
