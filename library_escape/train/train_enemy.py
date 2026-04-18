from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import time

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from library_escape.agents.ppo_agent import build_ppo_model
from library_escape.agents.rule_based_enemy import MixtureController, RandomController, RuleBasedPlayerController
from library_escape.config import load_env_config, load_reward_config, load_training_config
from library_escape.env.single_agent_env import LibraryEscapeEnv
from library_escape.paths import MODEL_DIR
from library_escape.train.progress import TrainingEtaCallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PPO enemy agent against a scripted player.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total training timesteps.")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the trained enemy model.")
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--render", action="store_true", help="Render training episodes. This is much slower.")
    return parser


def main() -> None:
    training_cfg = deepcopy(load_training_config())
    env_cfg = deepcopy(load_env_config())
    reward_cfg = deepcopy(load_reward_config())
    args = build_parser().parse_args()

    single_cfg = training_cfg["single_agent"]
    if args.timesteps is not None:
        single_cfg["total_timesteps"] = args.timesteps
    if args.frame_skip is not None:
        training_cfg["shared"]["frame_skip"] = args.frame_skip

    frame_skip = int(training_cfg["shared"]["frame_skip"])
    n_envs = int(single_cfg.get("n_envs", 1))
    vec_type = str(single_cfg.get("vector_env", "dummy")).lower()
    vec_cls = SubprocVecEnv if vec_type == "subproc" else DummyVecEnv
    render_mode = "human" if args.render else None
    if args.render:
        n_envs = 1
        vec_cls = DummyVecEnv

    opponent_mix = single_cfg.get("opponent_mix", {})
    weighted_controllers = []
    if float(opponent_mix.get("rule_based", 0.0)) > 0.0:
        weighted_controllers.append((float(opponent_mix["rule_based"]), RuleBasedPlayerController(env_cfg)))
    if float(opponent_mix.get("random", 0.0)) > 0.0:
        weighted_controllers.append((float(opponent_mix["random"]), RandomController(env_cfg)))
    if not weighted_controllers:
        weighted_controllers.append((1.0, RuleBasedPlayerController(env_cfg)))

    def make_env(rank: int):
        def _init():
            controllers = []
            if float(opponent_mix.get("rule_based", 0.0)) > 0.0:
                controllers.append((float(opponent_mix["rule_based"]), RuleBasedPlayerController(env_cfg)))
            if float(opponent_mix.get("random", 0.0)) > 0.0:
                controllers.append((float(opponent_mix["random"]), RandomController(env_cfg, seed=rank)))
            if not controllers:
                controllers.append((1.0, RuleBasedPlayerController(env_cfg)))
            opponent_controller = MixtureController(
                env_cfg,
                controllers,
                seed=int(training_cfg["shared"]["seed"]) + rank,
            )
            return LibraryEscapeEnv(
                role="enemy",
                env_config=env_cfg,
                reward_config=reward_cfg,
                frame_skip=frame_skip,
                opponent_controller=opponent_controller,
                render_mode=render_mode,
            )

        return _init

    env = vec_cls([make_env(rank) for rank in range(n_envs)])
    log_dir = Path(single_cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    effective_cfg = {**single_cfg, "device": training_cfg["shared"]["device"]}
    rollout_target = max(1, math.ceil(int(single_cfg["total_timesteps"]) / max(1, n_envs)))
    effective_cfg["n_steps"] = min(int(effective_cfg.get("n_steps", 512)), rollout_target)
    effective_cfg["batch_size"] = min(int(effective_cfg.get("batch_size", 256)), int(effective_cfg["n_steps"]))
    model = build_ppo_model(env, effective_cfg, tensorboard_log=str(log_dir))
    total_timesteps = int(single_cfg["total_timesteps"])
    progress_callback = TrainingEtaCallback(
        label="Enemy PPO",
        segment_total_timesteps=total_timesteps,
        global_total_timesteps=total_timesteps,
        global_completed_before=0,
        global_start_time=time.monotonic(),
    )
    model.learn(total_timesteps=total_timesteps, callback=progress_callback)

    output = args.output or MODEL_DIR / "enemy" / "enemy_ppo"
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    env.close()
    print(f"Saved enemy PPO model to {output}.zip")


if __name__ == "__main__":
    main()
