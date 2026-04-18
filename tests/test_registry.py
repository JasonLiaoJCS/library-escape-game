import json

from library_escape.eval.registry import discover_saved_models


def test_discover_saved_models_finds_single_and_selfplay_outputs(tmp_path):
    enemy_run = tmp_path / "enemy_run"
    enemy_run.mkdir()
    enemy_model = enemy_run / "enemy_latest.zip"
    enemy_model.write_bytes(b"zip")
    (enemy_run / "training_summary.json").write_text(
        json.dumps(
            {
                "mode": "single_agent_enemy",
                "algorithm": "maskable_ppo",
                "final_model": str(enemy_model),
            }
        ),
        encoding="utf-8",
    )

    selfplay_run = tmp_path / "selfplay_run"
    selfplay_run.mkdir()
    player_model = selfplay_run / "player_latest.zip"
    enemy_sp_model = selfplay_run / "enemy_latest.zip"
    player_model.write_bytes(b"zip")
    enemy_sp_model.write_bytes(b"zip")
    (selfplay_run / "training_summary.json").write_text(
        json.dumps(
            {
                "mode": "self_play",
                "algorithm": "league_maskable_ppo",
                "final_player_model": str(player_model),
                "final_enemy_model": str(enemy_sp_model),
            }
        ),
        encoding="utf-8",
    )

    discovered = discover_saved_models(tmp_path)
    roles = sorted(candidate.role for candidate in discovered)
    assert roles.count("enemy") == 2
    assert roles.count("player") == 1
