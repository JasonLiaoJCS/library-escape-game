"""Training callbacks for progress, ETA, and evaluation logging."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def format_seconds(seconds: float) -> str:
    if seconds is None:  # type: ignore[comparison-overlap]
        return "estimating..."
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class TrainingStatusCallback:
    """SB3 callback that writes progress/ETA to JSON and JSONL files."""

    def __init__(
        self,
        progress_path: Path,
        history_path: Path,
        total_timesteps: int,
        phase_name: str,
        phase_index: int = 1,
        phase_total: int = 1,
        global_step_offset: int = 0,
        global_total_timesteps: int | None = None,
        log_interval_seconds: float = 1.0,
        verbose: int = 0,
    ) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _InnerCallback(BaseCallback):
            def __init__(self, outer) -> None:
                super().__init__(verbose=verbose)
                self.outer = outer

            def _on_training_start(self) -> None:
                self.outer._on_training_start(self)

            def _on_step(self) -> bool:
                return self.outer._on_step(self)

            def _on_training_end(self) -> None:
                self.outer._on_training_end(self)

        self.callback = _InnerCallback(self)
        self.progress_path = progress_path
        self.history_path = history_path
        self.total_timesteps = total_timesteps
        self.phase_name = phase_name
        self.phase_index = phase_index
        self.phase_total = phase_total
        self.global_step_offset = global_step_offset
        self.global_total_timesteps = global_total_timesteps or total_timesteps
        self.log_interval_seconds = log_interval_seconds
        self._start_time = 0.0
        self._last_flush = 0.0
        self._last_terminal_print = 0.0

    def _build_payload(self, callback, done: bool = False) -> dict:
        elapsed = max(time.perf_counter() - self._start_time, 1e-6)
        current_steps = int(callback.num_timesteps)
        global_steps = min(self.global_total_timesteps, self.global_step_offset + current_steps)
        fps = current_steps / elapsed if elapsed > 0 else 0.0
        remaining_steps = max(0, self.global_total_timesteps - global_steps)
        eta_seconds = None if current_steps <= 0 or fps <= 1e-6 else (remaining_steps / fps)
        progress = global_steps / max(1, self.global_total_timesteps)

        mean_episode_reward = None
        mean_episode_length = None
        ep_info_buffer = getattr(callback.model, "ep_info_buffer", None)
        if ep_info_buffer:
            rewards = [float(item["r"]) for item in ep_info_buffer if "r" in item]
            lengths = [float(item["l"]) for item in ep_info_buffer if "l" in item]
            if rewards:
                mean_episode_reward = sum(rewards) / len(rewards)
            if lengths:
                mean_episode_length = sum(lengths) / len(lengths)

        payload = {
            "timestamp": time.time(),
            "phase_name": self.phase_name,
            "phase_index": self.phase_index,
            "phase_total": self.phase_total,
            "phase_progress": min(1.0, current_steps / max(1, self.total_timesteps)),
            "global_progress": min(1.0, progress),
            "elapsed_seconds": elapsed,
            "eta_seconds": eta_seconds,
            "eta_hms": format_seconds(eta_seconds),
            "fps": fps,
            "phase_timesteps": current_steps,
            "phase_total_timesteps": self.total_timesteps,
            "global_timesteps": global_steps,
            "global_total_timesteps": self.global_total_timesteps,
            "mean_episode_reward": mean_episode_reward,
            "mean_episode_length": mean_episode_length,
            "done": done,
        }
        return payload

    def _flush(self, callback, done: bool = False) -> None:
        payload = self._build_payload(callback, done=done)
        _write_json_atomic(self.progress_path, payload)
        _append_jsonl(self.history_path, payload)
        now = time.perf_counter()
        if now - self._last_terminal_print >= 5.0 or done:
            self._last_terminal_print = now
            print(
                f"[{self.phase_name}] {payload['global_progress'] * 100:5.1f}% | "
                f"step {payload['global_timesteps']}/{payload['global_total_timesteps']} | "
                f"fps {payload['fps']:.0f} | ETA {payload['eta_hms']}"
            )

    def _on_training_start(self, callback) -> None:
        self._start_time = time.perf_counter()
        self._last_flush = 0.0
        self._last_terminal_print = 0.0
        self._flush(callback, done=False)

    def _on_step(self, callback) -> bool:
        now = time.perf_counter()
        if now - self._last_flush >= self.log_interval_seconds:
            self._last_flush = now
            self._flush(callback, done=False)
        return True

    def _on_training_end(self, callback) -> None:
        self._flush(callback, done=True)


class EvalHistoryCallback:
    """Child callback for EvalCallback that writes eval metrics to JSONL."""

    def __init__(self, output_path: Path, phase_name: str) -> None:
        from stable_baselines3.common.callbacks import BaseCallback

        class _InnerCallback(BaseCallback):
            def __init__(self, outer) -> None:
                super().__init__(verbose=0)
                self.outer = outer

            def _on_step(self) -> bool:
                return self.outer._on_step(self)

        self.callback = _InnerCallback(self)
        self.output_path = output_path
        self.phase_name = phase_name

    def _on_step(self, callback) -> bool:
        parent = callback.parent
        payload = {
            "timestamp": time.time(),
            "phase_name": self.phase_name,
            "num_timesteps": int(parent.num_timesteps),
            "last_mean_reward": float(parent.last_mean_reward),
            "best_mean_reward": float(parent.best_mean_reward),
            "n_eval_episodes": int(parent.n_eval_episodes),
        }
        _append_jsonl(self.output_path, payload)
        return True
