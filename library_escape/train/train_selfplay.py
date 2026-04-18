"""Alternating self-play training for player and enemy policies."""

from __future__ import annotations

import argparse
import json
import subprocess
from copy import deepcopy
from pathlib import Path

from ..agents.opponent_pool import OpponentPool, PooledController, WeightedControllerChooser
from ..agents.ppo_agent import SB3PolicyController
from ..agents.rule_based_enemy import RandomEnemyController, RuleBasedEnemyController
from ..agents.rule_based_player import HeuristicPlayerController, RandomPlayerController
from ..config import REPO_ROOT, load_training_config
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
    make_single_agent_env_factory,
    override_model_hyperparameters,
    resolve_algorithm_spec,
    resolve_selfplay_run_dir,
    save_vecnormalize_artifacts,
    wrap_with_vec_normalize,
    write_model_metadata,
    write_training_summary,
)


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
) -> Path:
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

    callbacks = CallbackList(
        [
            TrainingStatusCallback(
                progress_path=progress_path,
                history_path=progress_history_path,
                total_timesteps=timesteps,
                phase_name=phase_name,
                phase_index=phase_index,
                phase_total=phase_total,
                global_step_offset=global_step_offset,
                global_total_timesteps=global_total_timesteps,
                log_interval_seconds=float(train_cfg["log_interval_seconds"]),
            ).callback,
            CheckpointCallback(
                save_freq=max(int(train_cfg["checkpoint_freq"]) // n_envs, 1),
                save_path=str(models_dir),
                name_prefix=f"{controlled_agent}_checkpoint",
                save_vecnormalize=bool(vecnorm_cfg.get("enabled", False)),
            ),
            algo_spec.eval_callback_cls(
                eval_env=eval_env,
                best_model_save_path=str(models_dir),
                log_path=str(phase_dir),
                eval_freq=max(int(train_cfg["eval_freq"]) // n_envs, 1),
                n_eval_episodes=int(train_cfg.get("eval_episodes", 8)),
                deterministic=True,
                render=False,
                callback_after_eval=EvalHistoryCallback(eval_history_path, phase_name=phase_name).callback,
                warn=False,
            ),
        ]
    )

    if resume_model_path is not None:
        model = algo_spec.model_cls.load(str(resume_model_path), env=vec_env, device=default_device(train_cfg))
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
    return final_path.with_suffix(".zip")


def _run_external_mappo_recipe(run_dir: Path, env_config: dict, train_cfg: dict, args: argparse.Namespace) -> None:
    recipe_cfg = train_cfg.get("opponent_curriculum", {}).get("mappo_recipe", {})
    command_template = str(recipe_cfg.get("external_command", "")).strip()
    if not command_template:
        raise RuntimeError(
            "Self-play algorithm is set to mappo_recipe, but no external_command is configured in "
            "configs/training.yaml or GUI overrides."
        )

    request_path = run_dir / "mappo_recipe_request.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps(
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
            ensure_ascii=True,
        ),
        encoding="utf-8",
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
    timesteps_per_round = int(train_cfg["timesteps_per_round"])
    total_phases = rounds * 2
    global_total_timesteps = timesteps_per_round * total_phases
    curriculum_cfg = train_cfg.get("opponent_curriculum", {})

    player_pool = OpponentPool(max_size=int(train_cfg["opponent_pool_size"]), seed=args.seed)
    enemy_pool = OpponentPool(max_size=int(train_cfg["opponent_pool_size"]), seed=args.seed + 1000)

    progress_path = run_dir / "progress.json"
    progress_history_path = run_dir / "progress_history.jsonl"
    eval_history_path = run_dir / "eval_history.jsonl"

    current_enemy_model: Path | None = None
    current_player_model: Path | None = None

    phase_counter = 0
    for round_idx in range(1, rounds + 1):
        round_train_cfg = apply_round_hyperparameter_schedules(
            train_cfg, round_idx=round_idx, total_rounds=rounds
        )
        print(
            f"[round {round_idx}] lr={round_train_cfg.get('learning_rate')}, "
            f"clip_range={round_train_cfg.get('clip_range')}, "
            f"ent_coef={round_train_cfg.get('ent_coef')}"
        )
        phase_counter += 1
        enemy_phase_dir = enemy_root / f"round_{round_idx:02d}"
        enemy_path = _train_phase(
            controlled_agent="enemy",
            env_config=env_config,
            train_cfg=round_train_cfg,
            algo_spec=algo_spec,
            phase_dir=enemy_phase_dir,
            opponent_pool=player_pool,
            fallback_factory=lambda round_seed=args.seed + round_idx: _bootstrap_player_factory(round_seed, curriculum_cfg),
            seed=args.seed + round_idx,
            timesteps=timesteps_per_round,
            phase_name=f"enemy_round_{round_idx:02d}",
            phase_index=phase_counter,
            phase_total=total_phases,
            global_total_timesteps=global_total_timesteps,
            global_step_offset=(phase_counter - 1) * timesteps_per_round,
            resume_model_path=current_enemy_model,
            shared_progress_path=progress_path,
            shared_progress_history_path=progress_history_path,
            shared_eval_history_path=eval_history_path,
        )
        current_enemy_model = enemy_path
        enemy_pool.add(
            _cached_factory(
                lambda path=enemy_path, env_cfg=deepcopy(env_config): SB3PolicyController(
                    "enemy",
                    path,
                    env_config=env_cfg,
                    deterministic=False,
                )
            )
        )

        phase_counter += 1
        player_phase_dir = player_root / f"round_{round_idx:02d}"
        player_path = _train_phase(
            controlled_agent="player",
            env_config=env_config,
            train_cfg=round_train_cfg,
            algo_spec=algo_spec,
            phase_dir=player_phase_dir,
            opponent_pool=enemy_pool,
            fallback_factory=lambda round_seed=args.seed + 5000 + round_idx: _bootstrap_enemy_factory(round_seed, curriculum_cfg),
            seed=args.seed + 5000 + round_idx,
            timesteps=timesteps_per_round,
            phase_name=f"player_round_{round_idx:02d}",
            phase_index=phase_counter,
            phase_total=total_phases,
            global_total_timesteps=global_total_timesteps,
            global_step_offset=(phase_counter - 1) * timesteps_per_round,
            resume_model_path=current_player_model,
            shared_progress_path=progress_path,
            shared_progress_history_path=progress_history_path,
            shared_eval_history_path=eval_history_path,
        )
        current_player_model = player_path
        player_pool.add(
            _cached_factory(
                lambda path=player_path, env_cfg=deepcopy(env_config): SB3PolicyController(
                    "player",
                    path,
                    env_config=env_cfg,
                    deterministic=False,
                )
            )
        )

        print(f"[round {round_idx}] enemy -> {enemy_path}")
        print(f"[round {round_idx}] player -> {player_path}")

    write_training_summary(
        run_dir / "training_summary.json",
        {
            "mode": "self_play",
            "algorithm": algo_spec.summary_name,
            "run_dir": run_dir,
            "enemy_root": enemy_root,
            "player_root": player_root,
            "final_enemy_model": current_enemy_model,
            "final_player_model": current_player_model,
            "progress_path": progress_path,
            "progress_history_path": progress_history_path,
            "eval_history_path": eval_history_path,
            "env_config": env_config,
            "train_config": train_cfg,
            "seed": args.seed,
            "preset": args.preset,
            "game_mode": game_mode,
            "game_mode_label": game_mode_label(game_mode),
        },
    )


if __name__ == "__main__":
    main()
