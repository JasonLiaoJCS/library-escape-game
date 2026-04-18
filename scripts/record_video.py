"""Record a frame sequence from AI-vs-AI mode without extra dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from library_escape.agents.ppo_agent import SB3PolicyController
from library_escape.agents.rule_based_enemy import RuleBasedEnemyController
from library_escape.agents.rule_based_player import HeuristicPlayerController
from library_escape.config import load_env_config, resolve_repo_path
from library_escape.core.world import World
from library_escape.render.pygame_view import PygameView


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save a sequence of rendered frames to disk.")
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--output-dir", type=str, default="videos/frames")
    parser.add_argument("--player-model", type=str, default=None)
    parser.add_argument("--enemy-model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic-policy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = load_env_config()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    world = World(env_config=env_config, seed=args.seed)
    renderer = PygameView(world, title="Library Escape Recorder", hidden=True)
    player_controller = (
        SB3PolicyController(
            "player",
            args.player_model,
            env_config=env_config,
            deterministic=args.deterministic_policy,
            decision_repeat_steps=max(1, int(env_config["timing"]["rl_frame_skip"])),
        )
        if args.player_model
        else HeuristicPlayerController()
    )
    enemy_controller = (
        SB3PolicyController(
            "enemy",
            args.enemy_model,
            env_config=env_config,
            deterministic=args.deterministic_policy,
            decision_repeat_steps=max(1, int(env_config["timing"]["rl_frame_skip"])),
        )
        if args.enemy_model
        else RuleBasedEnemyController()
    )

    for frame_idx in range(args.frames):
        if world.terminated or world.truncated:
            break
        player_action = player_controller.act(world)
        enemy_action = enemy_controller.act(world)
        world.step(player_action=player_action, enemy_action=enemy_action, frame_skip=1)
        renderer.render_frame()
        pygame.image.save(renderer.canvas, str(output_dir / f"frame_{frame_idx:05d}.png"))

    renderer.close()
    pygame.quit()
    print(f"Saved frames to {output_dir}")


if __name__ == "__main__":
    main()
