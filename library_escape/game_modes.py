"""Helpers for user-facing game modes and their config presets."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import load_env_config, resolve_repo_path


GAME_MODE_ALIASES = {
    "collection": "collection",
    "classic": "collection",
    "score": "collection",
    "escape": "escape",
    "rl": "escape",
    "escape_rl": "escape",
}


def normalize_game_mode(raw_mode: str | None, default: str = "escape") -> str:
    if raw_mode is None:
        return GAME_MODE_ALIASES.get(default.lower(), "escape")
    return GAME_MODE_ALIASES.get(str(raw_mode).strip().lower(), GAME_MODE_ALIASES.get(default.lower(), "escape"))


def game_mode_label(game_mode: str) -> str:
    normalized = normalize_game_mode(game_mode)
    return "Collection" if normalized == "collection" else "Escape"


def reward_path_for_game_mode(game_mode: str) -> str:
    normalized = normalize_game_mode(game_mode)
    return f"configs/rewards_{normalized}.yaml"


def apply_game_mode_overrides(
    env_config: dict[str, Any],
    *,
    game_mode: str,
    manual_collect_required: bool | None = None,
    interactive: bool = False,
) -> dict[str, Any]:
    normalized = normalize_game_mode(game_mode)
    world_cfg = env_config.setdefault("world", {})
    enemy_cfg = env_config.setdefault("enemy", {})
    enemy_team_cfg = env_config.setdefault("enemy_team", {})
    observation_cfg = env_config.setdefault("observation", {})
    ui_cfg = env_config.setdefault("ui", {})

    world_cfg["game_mode"] = normalized
    world_cfg["ruleset"] = normalized
    env_config["reward_path"] = reward_path_for_game_mode(normalized)
    world_cfg["collection_hold_seconds"] = 0.85

    if manual_collect_required is None:
        manual_collect_required = interactive and normalized == "collection"
    world_cfg["manual_collect_required"] = bool(manual_collect_required)

    if normalized == "collection":
        world_cfg["player_speed"] = 4.00
        world_cfg["enemy_speed"] = 3.65
        world_cfg["startup_grace_seconds"] = 1.00
        world_cfg["require_all_notes_to_escape"] = True
        world_cfg["require_all_exams_to_escape"] = True
        world_cfg["coffee_effect"] = "collection_speed"
        world_cfg["coffee_collection_multiplier"] = 1.50
        enemy_cfg["vision_range"] = 5.5
        enemy_cfg["vision_angle_deg"] = 52.0
        enemy_cfg["detect_penalty_seconds"] = 8.0
        enemy_cfg["detect_pause_seconds"] = 1.20
        enemy_cfg["chase_when_visible"] = False
        enemy_cfg["max_turn_rate_deg_per_sec"] = 0.0
        enemy_team_cfg["support_count"] = 4
        enemy_team_cfg["support_speed_scale"] = 1.04
        enemy_team_cfg["support_vision_range_scale"] = 1.00
        enemy_team_cfg["support_vision_angle_scale"] = 0.96
        observation_cfg["partial_observability"] = False
        if interactive:
            ui_cfg["show_detection_meter"] = False
            ui_cfg["show_last_seen_marker"] = False
            ui_cfg["show_enemy_mode"] = False
            ui_cfg["show_tactical_panel"] = False
            ui_cfg["show_exit_label"] = False
    else:
        world_cfg["player_speed"] = 4.55
        world_cfg["enemy_speed"] = 3.55
        world_cfg["startup_grace_seconds"] = 0.75
        world_cfg["require_all_notes_to_escape"] = True
        world_cfg["require_all_exams_to_escape"] = False
        world_cfg["coffee_effect"] = "movement"
        world_cfg["coffee_collection_multiplier"] = 1.0
        enemy_cfg["vision_range"] = 5.3
        enemy_cfg["vision_angle_deg"] = 54.0
        enemy_cfg["detect_penalty_seconds"] = 0.0
        enemy_cfg["detect_pause_seconds"] = 0.0
        enemy_cfg["chase_when_visible"] = True
        enemy_cfg["chase_speed_multiplier"] = 1.08
        enemy_cfg["max_turn_rate_deg_per_sec"] = 0.0
        enemy_team_cfg["support_count"] = 1
        enemy_team_cfg["support_speed_scale"] = 0.82
        enemy_team_cfg["support_vision_range_scale"] = 0.80
        enemy_team_cfg["support_vision_angle_scale"] = 0.80
        observation_cfg["partial_observability"] = False
        if interactive:
            ui_cfg.setdefault("show_exit_label", True)

    return env_config


def build_game_mode_env_config(
    *,
    game_mode: str,
    manual_collect_required: bool | None = None,
    interactive: bool = False,
) -> dict[str, Any]:
    env_config = copy.deepcopy(load_env_config())
    return apply_game_mode_overrides(
        env_config,
        game_mode=game_mode,
        manual_collect_required=manual_collect_required,
        interactive=interactive,
    )


def read_model_game_mode(model_path: str | Path) -> str | None:
    model = resolve_repo_path(model_path)
    candidate_paths = [
        model.with_suffix(".meta.json"),
        model.parent / "model.meta.json",
    ]
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        game_mode = payload.get("game_mode")
        if game_mode:
            return normalize_game_mode(str(game_mode))
    return None


def infer_game_mode_from_models(*model_paths: str | Path | None, default: str = "escape") -> str:
    inferred = {
        normalize_game_mode(mode)
        for mode in (read_model_game_mode(path) for path in model_paths if path)
        if mode
    }
    if len(inferred) == 1:
        return inferred.pop()
    return normalize_game_mode(default)
