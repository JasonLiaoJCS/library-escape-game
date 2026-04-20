import os
import time
from pathlib import Path

from library_escape.gui import app as gui_app
from library_escape.gui.app import compact_training_run, read_json, write_interrupted_training_summary


def _write_file(path: Path, contents: str = "x", *, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_write_interrupted_training_summary_for_selfplay_prefers_latest_written_models(tmp_path):
    run_dir = tmp_path / "selfplay_escape_demo"
    now = time.time()

    enemy_old = _write_file(run_dir / "enemy" / "round_01" / "models" / "enemy_round_01_latest.zip", mtime=now - 20)
    player_old = _write_file(run_dir / "player" / "round_01" / "models" / "player_round_01_latest.zip", mtime=now - 20)
    enemy_new = _write_file(run_dir / "enemy" / "round_02" / "models" / "enemy_checkpoint_2048_steps.zip", mtime=now - 5)
    player_new = _write_file(run_dir / "player" / "round_02" / "models" / "player_checkpoint_1024_steps.zip", mtime=now - 2)
    _write_file(run_dir / "enemy" / "round_02" / "models" / "obsnorm_latest.npz", mtime=now - 2)
    _write_file(run_dir / "player" / "round_02" / "models" / "obsnorm_latest.npz", mtime=now - 2)
    _write_file(
        run_dir / "progress.json",
        '{"phase_name":"enemy_round_02","global_progress":0.42,"phase_progress":0.84,"elapsed_seconds":123.0}',
    )

    payload = write_interrupted_training_summary(
        run_dir,
        {
            "mode": "selfplay",
            "game_mode": "escape",
            "algorithm": "league_maskable_ppo",
            "preset": "overnight",
            "seed": "7",
            "resume_path": None,
        },
    )

    saved = read_json(run_dir / "training_summary.json")
    assert payload["mode"] == "self_play"
    assert saved["status"] == "stopped_early"
    assert Path(saved["final_enemy_model"]) == enemy_new
    assert Path(saved["final_player_model"]) == player_new
    assert enemy_old.exists()
    assert player_old.exists()
    assert enemy_new.with_suffix(".meta.json").exists()
    assert player_new.with_suffix(".meta.json").exists()


def test_compact_training_run_keeps_latest_playable_artifacts_and_summary(tmp_path):
    run_dir = tmp_path / "enemy_escape_demo"
    final_model = _write_file(run_dir / "models" / "enemy_latest.zip")
    _write_file(run_dir / "models" / "enemy_latest.meta.json")
    _write_file(run_dir / "models" / "enemy_latest.obsnorm.npz")
    _write_file(run_dir / "models" / "obsnorm_latest.npz")
    _write_file(run_dir / "models" / "vecnormalize.pkl")
    extra_checkpoint = _write_file(run_dir / "models" / "enemy_checkpoint_4096_steps.zip")
    _write_file(run_dir / "models" / "enemy_checkpoint_vecnormalize_4096_steps.pkl")
    tb_file = _write_file(run_dir / "tb" / "run_1" / "events.out.tfevents.fake")
    monitor_file = _write_file(run_dir / "monitor" / "train_env_0.monitor.csv")
    _write_file(run_dir / "progress.json", '{"global_progress":1.0}')
    _write_file(run_dir / "progress_history.jsonl", '{"global_progress":1.0}\n')
    _write_file(run_dir / "eval_history.jsonl", '{"last_mean_reward":1.0}\n')
    _write_file(
        run_dir / "training_summary.json",
        (
            "{"
            f'"mode":"single_agent_enemy","final_model":"{final_model.as_posix()}",'
            f'"progress_path":"{(run_dir / "progress.json").as_posix()}",'
            f'"progress_history_path":"{(run_dir / "progress_history.jsonl").as_posix()}",'
            f'"eval_history_path":"{(run_dir / "eval_history.jsonl").as_posix()}",'
            '"game_mode":"escape","algorithm":"maskable_ppo"'
            "}"
        ),
    )

    result = compact_training_run(run_dir)

    saved = read_json(run_dir / "training_summary.json")
    assert result["deleted_files"] >= 4
    assert final_model.exists()
    assert (run_dir / "models" / "enemy_latest.meta.json").exists()
    assert (run_dir / "models" / "enemy_latest.obsnorm.npz").exists()
    assert (run_dir / "models" / "obsnorm_latest.npz").exists()
    assert (run_dir / "models" / "vecnormalize.pkl").exists()
    assert not extra_checkpoint.exists()
    assert not tb_file.exists()
    assert not monitor_file.exists()
    assert not (run_dir / "progress_history.jsonl").exists()
    assert not (run_dir / "eval_history.jsonl").exists()
    assert saved["artifacts_compacted"] is True
    assert saved["progress_history_path"] is None
    assert saved["eval_history_path"] is None


def test_tensorboard_wait_hint_reports_first_rollout_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(gui_app, "inspect_scalar_data", lambda run_dir: (1, False))

    hint = gui_app.tensorboard_wait_hint(
        tmp_path,
        progress_payload={"phase_timesteps": 31760},
        context={
            "algorithm": "league_maskable_ppo",
            "train_config": {"n_envs": 16, "n_steps": 2048},
        },
    )

    assert hint is not None
    assert "32,768" in hint
    assert "31,760" in hint
    assert "1,008" in hint


def test_resolve_effective_train_config_for_gui_applies_preset_and_overrides():
    train_cfg = gui_app.resolve_effective_train_config_for_gui(
        "selfplay",
        "overnight_4090",
        "escape",
        {"train": {"algorithm": "league_maskable_ppo"}},
        n_envs_override="16",
    )

    assert train_cfg["n_steps"] == 2048
    assert train_cfg["n_envs"] == 16
    assert train_cfg["algorithm"] == "league_maskable_ppo"
