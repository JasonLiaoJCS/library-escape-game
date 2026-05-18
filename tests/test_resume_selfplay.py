from __future__ import annotations

import argparse
import json
from pathlib import Path

from library_escape.train.train_selfplay import (
    _base_selfplay_round_timesteps,
    _resolve_resume_selfplay_models,
    _round_timestep_plan,
)


def test_resolve_resume_selfplay_models_from_run_summary(tmp_path: Path):
    enemy_model = tmp_path / "enemy_latest.zip"
    player_model = tmp_path / "player_latest.zip"
    enemy_model.write_bytes(b"dummy")
    player_model.write_bytes(b"dummy")

    run_dir = tmp_path / "selfplay_run"
    run_dir.mkdir()
    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "final_enemy_model": str(enemy_model),
                "final_player_model": str(player_model),
            }
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        resume_run=str(run_dir),
        resume_enemy=None,
        resume_player=None,
    )

    resolved_run, resolved_enemy, resolved_player = _resolve_resume_selfplay_models(args)

    assert resolved_run == run_dir.resolve()
    assert resolved_enemy == enemy_model.resolve()
    assert resolved_player == player_model.resolve()


def test_resolve_resume_selfplay_models_from_game_mode_root(tmp_path: Path):
    run_root = tmp_path / "escape"
    run_root.mkdir()

    old_enemy_model = tmp_path / "enemy_old.zip"
    old_player_model = tmp_path / "player_old.zip"
    old_enemy_model.write_bytes(b"old")
    old_player_model.write_bytes(b"old")
    old_run_dir = run_root / "selfplay_20260424_175841"
    old_run_dir.mkdir()
    (old_run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "final_enemy_model": str(old_enemy_model),
                "final_player_model": str(old_player_model),
            }
        ),
        encoding="utf-8",
    )

    new_enemy_model = tmp_path / "enemy_new.zip"
    new_player_model = tmp_path / "player_new.zip"
    new_enemy_model.write_bytes(b"new")
    new_player_model.write_bytes(b"new")
    new_run_dir = run_root / "selfplay_20260425_215517"
    new_run_dir.mkdir()
    (new_run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "final_enemy_model": str(new_enemy_model),
                "final_player_model": str(new_player_model),
            }
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        resume_run=str(run_root),
        resume_enemy=None,
        resume_player=None,
    )

    resolved_run, resolved_enemy, resolved_player = _resolve_resume_selfplay_models(args)

    assert resolved_run == new_run_dir.resolve()
    assert resolved_enemy == new_enemy_model.resolve()
    assert resolved_player == new_player_model.resolve()


def test_resolve_resume_selfplay_models_without_top_level_summary(tmp_path: Path):
    run_dir = tmp_path / "selfplay_run"
    enemy_models = run_dir / "enemy" / "round_08" / "models"
    player_models = run_dir / "player" / "round_07" / "models"
    enemy_models.mkdir(parents=True)
    player_models.mkdir(parents=True)

    enemy_latest = enemy_models / "enemy_round_08_latest.zip"
    player_latest = player_models / "player_round_07_latest.zip"
    enemy_latest.write_bytes(b"enemy")
    player_latest.write_bytes(b"player")

    args = argparse.Namespace(
        resume_run=str(run_dir),
        resume_enemy=None,
        resume_player=None,
    )

    resolved_run, resolved_enemy, resolved_player = _resolve_resume_selfplay_models(args)

    assert resolved_run == run_dir.resolve()
    assert resolved_enemy == enemy_latest.resolve()
    assert resolved_player == player_latest.resolve()


def test_selfplay_timestep_plan_defaults_to_more_player_training():
    train_cfg = {
        "timesteps_per_round": 1000,
        "role_timestep_multipliers": {"enemy": 0.75, "player": 1.45},
        "adaptive_timesteps": {"enabled": True, "warmup_rounds": 1},
    }

    plan = _round_timestep_plan(
        train_cfg,
        round_idx=1,
        last_enemy_reward=None,
        last_player_reward=None,
    )

    assert plan["enemy_timesteps"] == 750
    assert plan["player_timesteps"] == 1450
    assert _base_selfplay_round_timesteps(train_cfg) == 2200


def test_selfplay_timestep_plan_boosts_player_when_enemy_is_ahead():
    train_cfg = {
        "timesteps_per_round": 1000,
        "role_timestep_multipliers": {"enemy": 0.75, "player": 1.45},
        "adaptive_timesteps": {
            "enabled": True,
            "warmup_rounds": 1,
            "reward_gap_scale": 300.0,
            "max_enemy_adjustment": 0.25,
            "max_player_adjustment": 0.35,
            "min_multiplier": 0.50,
            "max_multiplier": 2.25,
        },
    }

    plan = _round_timestep_plan(
        train_cfg,
        round_idx=2,
        last_enemy_reward=100.0,
        last_player_reward=-200.0,
    )

    assert plan["reward_gap"] == 300.0
    assert plan["adaptive_adjustment"] == 1.0
    assert plan["enemy_timesteps"] == 563
    assert plan["player_timesteps"] == 1958
