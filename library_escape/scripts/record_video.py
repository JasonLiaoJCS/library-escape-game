from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from library_escape.agents.ppo_agent import SB3PolicyController
from library_escape.agents.rule_based_enemy import RuleBasedEnemyController, RuleBasedPlayerController
from library_escape.config import load_env_config
from library_escape.core.world import LibraryWorld
from library_escape.render import PygameView


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record an AI match as a PNG frame sequence.")
    parser.add_argument("--player-model", type=Path, default=None)
    parser.add_argument("--enemy-model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("recordings/frames"))
    parser.add_argument("--max-steps", type=int, default=600)
    return parser


def _build_controller(path: Path | None, role: str, env_config, world):
    if path is not None:
        return SB3PolicyController(path, role, env_config, world)
    if role == "enemy":
        return RuleBasedEnemyController(env_config)
    return RuleBasedPlayerController(env_config)


def main() -> None:
    args = build_parser().parse_args()
    env_config = deepcopy(load_env_config())
    world = LibraryWorld(env_config)
    player_controller = _build_controller(args.player_model, "player", env_config, world)
    enemy_controller = _build_controller(args.enemy_model, "enemy", env_config, world)
    view = PygameView(env_config, render_mode="rgb_array")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.max_steps):
        player_action = player_controller.act(world)
        enemy_action = enemy_controller.act(world)
        world.tick(player_action, enemy_action)
        view.draw(world, title="Recorded Match")
        frame_path = args.output_dir / f"frame_{step:05d}.png"
        import pygame

        pygame.image.save(view.surface, str(frame_path))
        if world.status.terminated or world.status.truncated:
            break

    view.close()
    print(f"Saved frames to {args.output_dir}")


if __name__ == "__main__":
    main()
