from __future__ import annotations

import argparse
import json
from pathlib import Path

from library_escape.train.train_selfplay import _resolve_resume_selfplay_models


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
