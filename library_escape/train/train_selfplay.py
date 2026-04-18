from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random
import time

from library_escape.agents.ppo_agent import SB3PolicyController, build_ppo_model
from library_escape.agents.opponent_pool import OpponentPool
from library_escape.agents.rule_based_enemy import RuleBasedEnemyController, RuleBasedPlayerController
from library_escape.config import load_env_config, load_reward_config, load_training_config
from library_escape.env.single_agent_env import LibraryEscapeEnv
from library_escape.train.progress import TrainingEtaCallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Alternate PPO training between player and enemy for self-play.")
    parser.add_argument("--rounds", type=int, default=None, help="Override number of self-play rounds.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override timesteps per round.")
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--render", action="store_true", help="Render training episodes. This is much slower.")
    return parser


def _effective_training_cfg(net_arch: list[int], device: str, total_timesteps: int) -> dict[str, object]:
    n_steps = min(512, total_timesteps)
    return {
        "net_arch": net_arch,
        "device": device,
        "n_steps": n_steps,
        "batch_size": min(256, n_steps),
    }


def _train_player(
    env_cfg,
    reward_cfg,
    frame_skip: int,
    total_timesteps: int,
    checkpoint_dir: Path,
    opponent_path: Path | None,
    net_arch: list[int],
    device: str,
    render: bool,
    progress_callback: TrainingEtaCallback,
):
    env = LibraryEscapeEnv(
        role="player",
        env_config=env_cfg,
        reward_config=reward_cfg,
        frame_skip=frame_skip,
        render_mode="human" if render else None,
    )
    if opponent_path is not None:
        env.set_opponent_controller(SB3PolicyController(opponent_path, "enemy", env_cfg, env.world))
    else:
        env.set_opponent_controller(RuleBasedEnemyController(env_cfg))
    model = build_ppo_model(env, _effective_training_cfg(net_arch, device, total_timesteps))
    model.learn(total_timesteps=total_timesteps, callback=progress_callback)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output = checkpoint_dir / f"player_round_{len(list(checkpoint_dir.glob('*.zip'))) + 1}"
    model.save(str(output))
    env.close()
    return Path(f"{output}.zip")


def _train_enemy(
    env_cfg,
    reward_cfg,
    frame_skip: int,
    total_timesteps: int,
    checkpoint_dir: Path,
    opponent_path: Path | None,
    net_arch: list[int],
    device: str,
    render: bool,
    progress_callback: TrainingEtaCallback,
):
    env = LibraryEscapeEnv(
        role="enemy",
        env_config=env_cfg,
        reward_config=reward_cfg,
        frame_skip=frame_skip,
        render_mode="human" if render else None,
    )
    if opponent_path is not None:
        env.set_opponent_controller(SB3PolicyController(opponent_path, "player", env_cfg, env.world))
    else:
        env.set_opponent_controller(RuleBasedPlayerController(env_cfg))
    model = build_ppo_model(env, _effective_training_cfg(net_arch, device, total_timesteps))
    model.learn(total_timesteps=total_timesteps, callback=progress_callback)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output = checkpoint_dir / f"enemy_round_{len(list(checkpoint_dir.glob('*.zip'))) + 1}"
    model.save(str(output))
    env.close()
    return Path(f"{output}.zip")


def main() -> None:
    training_cfg = deepcopy(load_training_config())
    env_cfg = deepcopy(load_env_config())
    reward_cfg = deepcopy(load_reward_config())
    args = build_parser().parse_args()

    ma_cfg = training_cfg["multi_agent"]
    rounds = args.rounds or int(ma_cfg["rounds"])
    timesteps = args.timesteps or int(ma_cfg["timesteps_per_round"])
    frame_skip = args.frame_skip or int(training_cfg["shared"]["frame_skip"])
    player_dir = Path(ma_cfg["player_checkpoint_dir"])
    enemy_dir = Path(ma_cfg["enemy_checkpoint_dir"])
    player_pool = OpponentPool(int(ma_cfg["pool_size"]))
    enemy_pool = OpponentPool(int(ma_cfg["pool_size"]))
    rng = random.Random(int(training_cfg["shared"]["seed"]))
    global_start_time = time.monotonic()
    global_total_timesteps = rounds * timesteps * 2

    for round_index in range(1, rounds + 1):
        sampled_enemy = rng.choice(enemy_pool.all()) if enemy_pool.all() else None
        player_global_before = ((round_index - 1) * 2) * timesteps
        print(
            f"[SELFPLAY] Round {round_index}/{rounds} | phase=player | "
            f"global {player_global_before}/{global_total_timesteps}",
            flush=True,
        )
        latest_player = _train_player(
            env_cfg,
            reward_cfg,
            frame_skip,
            timesteps,
            player_dir,
            sampled_enemy,
            list(ma_cfg.get("player_net_arch", [128, 128])),
            str(training_cfg["shared"].get("device", "auto")),
            args.render,
            TrainingEtaCallback(
                label=f"Round {round_index}/{rounds} player",
                segment_total_timesteps=timesteps,
                global_total_timesteps=global_total_timesteps,
                global_completed_before=player_global_before,
                global_start_time=global_start_time,
            ),
        )
        player_pool.add(latest_player)

        sampled_player = rng.choice(player_pool.all()) if player_pool.all() else None
        enemy_global_before = ((round_index - 1) * 2 + 1) * timesteps
        print(
            f"[SELFPLAY] Round {round_index}/{rounds} | phase=enemy | "
            f"global {enemy_global_before}/{global_total_timesteps}",
            flush=True,
        )
        latest_enemy = _train_enemy(
            env_cfg,
            reward_cfg,
            frame_skip,
            timesteps,
            enemy_dir,
            sampled_player,
            list(ma_cfg.get("enemy_net_arch", [128, 128])),
            str(training_cfg["shared"].get("device", "auto")),
            args.render,
            TrainingEtaCallback(
                label=f"Round {round_index}/{rounds} enemy",
                segment_total_timesteps=timesteps,
                global_total_timesteps=global_total_timesteps,
                global_completed_before=enemy_global_before,
                global_start_time=global_start_time,
            ),
        )
        enemy_pool.add(latest_enemy)
        print(f"Completed self-play round {round_index}: player={latest_player.name}, enemy={latest_enemy.name}")


if __name__ == "__main__":
    main()
