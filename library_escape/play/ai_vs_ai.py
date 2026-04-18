"""Run Library Escape with AI controlling both sides."""

from __future__ import annotations

import argparse

import pygame

from ..agents.ppo_agent import SB3PolicyController
from ..agents.rule_based_enemy import RuleBasedEnemyController
from ..agents.rule_based_player import HeuristicPlayerController
from ..config import load_env_config
from ..core.world import World
from ..replay.io import ReplayRecorder
from ..render.pygame_view import PygameView


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch AI vs AI in Library Escape.")
    parser.add_argument("--player-model", type=str, default=None, help="Optional PPO checkpoint for the player.")
    parser.add_argument("--enemy-model", type=str, default=None, help="Optional PPO checkpoint for the enemy.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hidden-window", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--record-replay", type=str, default=None, help="Optional replay output path (.ler.gz).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = load_env_config()
    world = World(env_config=env_config, seed=args.seed)
    renderer = PygameView(world, title="Library Escape - AI vs AI", hidden=args.hidden_window)
    player_controller = (
        SB3PolicyController("player", args.player_model, env_config=env_config)
        if args.player_model
        else HeuristicPlayerController()
    )
    enemy_controller = (
        SB3PolicyController("enemy", args.enemy_model, env_config=env_config)
        if args.enemy_model
        else RuleBasedEnemyController()
    )
    replay = None
    if args.record_replay:
        replay = ReplayRecorder(
            args.record_replay,
            world,
            metadata={
                "mode": "ai_vs_ai",
                "seed": args.seed,
                "player_model": args.player_model,
                "enemy_model": args.enemy_model,
            },
        )
        replay.capture(world)

    clock = pygame.time.Clock()
    accumulator = 0.0
    wall_elapsed = 0.0
    running = True

    while running:
        frame_dt = min(clock.tick(world.render_fps) / 1000.0, 0.25)
        wall_elapsed += frame_dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r and (world.terminated or world.truncated):
                world.reset(seed=args.seed)
                if hasattr(player_controller, "reset"):
                    player_controller.reset()
                if hasattr(enemy_controller, "reset"):
                    enemy_controller.reset()

        if args.max_seconds is not None and wall_elapsed >= args.max_seconds:
            running = False

        if not (world.terminated or world.truncated):
            accumulator += frame_dt
            while accumulator >= world.physics_dt:
                player_action = player_controller.act(world)
                enemy_action = enemy_controller.act(world)
                world.step(player_action=player_action, enemy_action=enemy_action, frame_skip=1)
                accumulator -= world.physics_dt
                if world.terminated or world.truncated:
                    break

        renderer.render_frame()
        if replay is not None:
            replay.capture(world)

    renderer.close()
    pygame.quit()
    if replay is not None:
        saved_path = replay.save()
        print(f"Saved replay to {saved_path}")


if __name__ == "__main__":
    main()
