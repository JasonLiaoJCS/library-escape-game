from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from library_escape.agents.ppo_agent import SB3PolicyController
from library_escape.agents.rule_based_enemy import RuleBasedEnemyController, RuleBasedPlayerController
from library_escape.config import load_env_config
from library_escape.core.world import LibraryWorld
from library_escape.render import NullView, PygameView


def _build_controller(path: Path | None, role: str, env_config: dict[str, Any], world: LibraryWorld):
    if path:
        return SB3PolicyController(path, role, env_config, world)
    if role == "enemy":
        return RuleBasedEnemyController(env_config)
    return RuleBasedPlayerController(env_config)


def run_episode(
    player_model: Path | None = None,
    enemy_model: Path | None = None,
    render: bool = False,
    max_steps: int | None = None,
) -> str | None:
    env_config = deepcopy(load_env_config())
    world = LibraryWorld(env_config)
    player_controller = _build_controller(player_model, "player", env_config, world)
    enemy_controller = _build_controller(enemy_model, "enemy", env_config, world)
    view = PygameView(env_config, render_mode="human") if render else NullView(
        env_config["render"]["window_width"],
        env_config["render"]["window_height"],
    )

    cap = max_steps or int(world.max_time / world.physics_dt)
    for _ in range(cap):
        player_action = player_controller.act(world)
        enemy_action = enemy_controller.act(world)
        world.tick(player_action, enemy_action)
        if render:
            view.draw(world, title="AI vs AI")
        if world.status.terminated or world.status.truncated:
            break

    outcome = world.status.outcome
    view.close()
    return outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AI vs AI matches in Library Escape.")
    parser.add_argument("--player-model", type=Path, default=None)
    parser.add_argument("--enemy-model", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outcome = run_episode(
        player_model=args.player_model,
        enemy_model=args.enemy_model,
        render=args.render,
        max_steps=args.max_steps,
    )
    print(f"AI vs AI outcome: {outcome}")


if __name__ == "__main__":
    main()
