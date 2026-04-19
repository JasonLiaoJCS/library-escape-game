import json
import time

import library_escape.train.callbacks as callbacks
import library_escape.train.io_utils as io_utils


class _DummyModel:
    ep_info_buffer = []


class _DummyCallback:
    def __init__(self, num_timesteps: int = 128) -> None:
        self.num_timesteps = num_timesteps
        self.model = _DummyModel()


def test_write_json_atomic_retries_when_replace_is_temporarily_denied(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress.json"
    original_replace = io_utils.os.replace
    attempts = {"count": 0}

    def flaky_replace(src, dst):
        if attempts["count"] < 2:
            attempts["count"] += 1
            raise PermissionError("file temporarily locked")
        return original_replace(src, dst)

    monkeypatch.setattr(io_utils, "_retry_sleep", lambda attempt: None)
    monkeypatch.setattr(io_utils.os, "replace", flaky_replace)

    assert io_utils.atomic_write_json(progress_path, {"global_progress": 0.5}) is True
    assert json.loads(progress_path.read_text(encoding="utf-8"))["global_progress"] == 0.5
    assert attempts["count"] == 2


def test_write_json_atomic_falls_back_to_in_place_write_when_replace_keeps_failing(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress.json"

    def always_fail_replace(src, dst):
        raise PermissionError("rename blocked")

    monkeypatch.setattr(io_utils, "_retry_sleep", lambda attempt: None)
    monkeypatch.setattr(io_utils.os, "replace", always_fail_replace)

    assert io_utils.atomic_write_json(progress_path, {"phase_name": "enemy_round_01"}) is True
    assert json.loads(progress_path.read_text(encoding="utf-8"))["phase_name"] == "enemy_round_01"
    assert not progress_path.with_name(f"{progress_path.name}.{io_utils.os.getpid()}.tmp").exists()


def test_training_status_callback_keeps_running_when_progress_writes_fail(tmp_path, monkeypatch):
    status = callbacks.TrainingStatusCallback(
        progress_path=tmp_path / "progress.json",
        history_path=tmp_path / "progress_history.jsonl",
        total_timesteps=1024,
        phase_name="enemy_round_01",
        log_interval_seconds=0.0,
    )
    dummy = _DummyCallback(num_timesteps=256)

    monkeypatch.setattr(callbacks, "_write_json_atomic", lambda path, payload: False)
    monkeypatch.setattr(callbacks, "_append_jsonl", lambda path, payload: False)
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    status._start_time = time.perf_counter() - 1.0
    status._last_flush = 0.0

    assert status._on_step(dummy) is True
