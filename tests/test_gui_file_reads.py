from pathlib import Path

from library_escape.gui.app import read_json


def test_read_json_retries_transient_permission_error(tmp_path, monkeypatch):
    progress_path = tmp_path / "progress.json"
    progress_path.write_text('{"global_progress": 0.5, "phase_name": "train_enemy"}', encoding="utf-8")

    original_read_text = Path.read_text
    calls = {"count": 0}

    def flaky_read_text(self, *args, **kwargs):
        if self == progress_path and calls["count"] == 0:
            calls["count"] += 1
            raise PermissionError("progress file is temporarily locked")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    payload = read_json(progress_path)

    assert payload["global_progress"] == 0.5
    assert payload["phase_name"] == "train_enemy"
    assert calls["count"] == 1
