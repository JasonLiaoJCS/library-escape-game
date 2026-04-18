from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import pygame

from library_escape.agents.ppo_agent import SB3PolicyController
from library_escape.agents.rule_based_enemy import RuleBasedEnemyController, RuleBasedPlayerController
from library_escape.config import load_env_config
from library_escape.core.world import LibraryWorld
from library_escape.input import KeyboardActionAdapter
from library_escape.render import NullView, PygameView


def _choose_enemy_controller(args: argparse.Namespace, env_config: dict[str, Any], world: LibraryWorld):
    if args.enemy_model:
        return SB3PolicyController(args.enemy_model, "enemy", env_config, world)
    return RuleBasedEnemyController(env_config)


def run_game(args: argparse.Namespace) -> str | None:
    env_config = deepcopy(load_env_config())
    if args.action_scheme:
        env_config["actions"]["scheme"] = args.action_scheme
    world = LibraryWorld(env_config)
    enemy_controller = _choose_enemy_controller(args, env_config, world)
    keyboard = KeyboardActionAdapter(env_config["actions"]["scheme"])
    autoplay_controller = RuleBasedPlayerController(env_config)

    if args.headless:
        view = NullView(env_config["render"]["window_width"], env_config["render"]["window_height"])
        max_steps = args.max_steps or int(world.max_time / world.physics_dt)
        for _ in range(max_steps):
            player_action = autoplay_controller.act(world)
            enemy_action = enemy_controller.act(world)
            world.tick(player_action, enemy_action)
            if world.status.terminated or world.status.truncated:
                break
        view.close()
        return world.status.outcome

    view = PygameView(env_config, render_mode="human")
    running = True
    accumulator = 0.0
    previous_time = time.perf_counter()
    fixed_dt = world.physics_dt
    steps = 0
    max_steps = args.max_steps

    while running and not (world.status.terminated or world.status.truncated):
        current_time = time.perf_counter()
        accumulator += min(current_time - previous_time, 0.1)
        previous_time = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        pressed = pygame.key.get_pressed()
        player_action = keyboard.action_from_pressed(pressed)
        while accumulator >= fixed_dt:
            enemy_action = enemy_controller.act(world)
            world.tick(player_action, enemy_action)
            accumulator -= fixed_dt
            steps += 1
            if max_steps is not None and steps >= max_steps:
                running = False
                break
            if world.status.terminated or world.status.truncated:
                break
        view.draw(world, title="Human vs AI Enemy")

    outcome = world.status.outcome
    view.close()
    return outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Library Escape against an AI enemy.")
    parser.add_argument("--enemy-model", type=Path, default=None, help="Path to a trained enemy PPO checkpoint (.zip).")
    parser.add_argument("--action-scheme", choices=["discrete", "continuous"], default=None)
    parser.add_argument("--headless", action="store_true", help="Run without a window using rule-based autoplay.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional hard cap for smoke tests.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outcome = run_game(args)
    if outcome:
        print(f"Game finished with outcome: {outcome}")


if __name__ == "__main__":
    main()
