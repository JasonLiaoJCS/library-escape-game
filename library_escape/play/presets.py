"""Gameplay-oriented config presets for interactive play modes."""

from __future__ import annotations

from ..game_modes import build_game_mode_env_config, normalize_game_mode


def build_play_env_config(
    *,
    game_mode: str | None = None,
    ruleset: str | None = None,
    manual_collect_required: bool | None = None,
) -> dict:
    selected_mode = normalize_game_mode(game_mode or ruleset or "collection")
    return build_game_mode_env_config(
        game_mode=selected_mode,
        manual_collect_required=manual_collect_required,
        interactive=True,
    )
