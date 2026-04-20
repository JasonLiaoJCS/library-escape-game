"""Action masking helpers for discrete-control RL agents."""

from __future__ import annotations

import numpy as np

from ..core.actions import DISCRETE_ACTIONS, discrete_to_vector
from ..core.physics import move_circle


def _speed_scale_for_role(world, role: str) -> float:
    if role == "player":
        if world.player.coffee_timer > 0.0:
            return float(world.env_config["world"]["coffee_speed_multiplier"])
        return 1.0
    if world.enemy.freeze_timer > 0.0:
        return 0.0
    if world.player_visible_to_enemy():
        return float(world.enemy_chase_speed_multiplier)
    return 1.0


def action_mask_for_world(world, role: str) -> np.ndarray:
    """Return a boolean-ish mask for the current world state.

    The mask is designed for the discrete 9-action movement space. It
    invalidates movement directions that would immediately result in no motion
    because of walls or boundaries. No-op is only left valid when the actor is
    forced to idle or the player is actively collecting.
    """

    action_cfg = world.env_config["action"]
    if str(action_cfg["type"]).lower() != "discrete":
        return np.ones(9, dtype=np.int8)

    if role not in {"player", "enemy"}:
        raise ValueError(f"Unsupported role for action masking: {role}")

    actor = world.player if role == "player" else world.enemy
    if role == "enemy" and (world.enemy.freeze_timer > 0.0 or getattr(world.enemy, "detection_pause_timer", 0.0) > 0.0):
        mask = np.zeros(len(DISCRETE_ACTIONS), dtype=np.int8)
        mask[0] = 1
        return mask

    speed_scale = _speed_scale_for_role(world, role)
    mask = np.zeros(len(DISCRETE_ACTIONS), dtype=np.int8)
    allow_noop = bool(action_cfg.get("allow_noop", False))
    if role == "player":
        if getattr(world, "collection_progress", 0.0) > 1e-6:
            allow_noop = True
        elif hasattr(world, "nearest_interactable_collectible") and world.nearest_interactable_collectible() is not None:
            allow_noop = True
    if allow_noop:
        mask[0] = 1

    for action_id in sorted(DISCRETE_ACTIONS):
        if action_id == 0:
            continue
        direction = discrete_to_vector(action_id)
        velocity = (
            direction[0] * actor.base_speed * speed_scale,
            direction[1] * actor.base_speed * speed_scale,
        )
        next_x, next_y, _ = move_circle(
            x=actor.x,
            y=actor.y,
            radius=actor.radius,
            velocity=velocity,
            dt=world.physics_dt,
            width=world.width,
            height=world.height,
            obstacles=world.obstacles,
        )
        moved = abs(next_x - actor.x) > 1e-4 or abs(next_y - actor.y) > 1e-4
        if moved:
            mask[action_id] = 1

    if not np.any(mask):
        mask[0] = 1
    return mask
