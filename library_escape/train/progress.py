from __future__ import annotations

import time

from stable_baselines3.common.callbacks import BaseCallback


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class TrainingEtaCallback(BaseCallback):
    def __init__(
        self,
        label: str,
        segment_total_timesteps: int,
        global_total_timesteps: int | None = None,
        global_completed_before: int = 0,
        global_start_time: float | None = None,
        print_every_seconds: float = 5.0,
    ) -> None:
        super().__init__(verbose=0)
        self.label = label
        self.segment_total_timesteps = max(1, int(segment_total_timesteps))
        self.global_total_timesteps = (
            max(1, int(global_total_timesteps)) if global_total_timesteps is not None else None
        )
        self.global_completed_before = max(0, int(global_completed_before))
        self.global_start_time = global_start_time or time.monotonic()
        self.print_every_seconds = float(print_every_seconds)
        self.segment_start_time = self.global_start_time
        self.last_print_time = self.global_start_time

    def _on_training_start(self) -> None:
        self.segment_start_time = time.monotonic()
        self.last_print_time = self.segment_start_time - self.print_every_seconds
        self._print_status(force=True)

    def _on_step(self) -> bool:
        self._print_status(force=False)
        return True

    def _on_training_end(self) -> None:
        self._print_status(force=True, finished=True)

    def _print_status(self, force: bool, finished: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_print_time < self.print_every_seconds:
            return

        segment_completed = min(self.num_timesteps, self.segment_total_timesteps)
        segment_remaining = max(0, self.segment_total_timesteps - segment_completed)
        segment_elapsed = max(1e-6, now - self.segment_start_time)
        segment_fps = segment_completed / segment_elapsed
        segment_pct = (segment_completed / self.segment_total_timesteps) * 100.0
        if finished:
            segment_eta_text = format_seconds(0.0)
        elif segment_completed == 0:
            segment_eta_text = "calculating"
        else:
            segment_eta_text = format_seconds(segment_remaining / max(segment_fps, 1e-6))

        message = (
            f"[ETA] {self.label} | phase {segment_completed}/{self.segment_total_timesteps} "
            f"({segment_pct:5.1f}%) | phase_eta {segment_eta_text}"
        )

        if self.global_total_timesteps is not None:
            global_completed = min(
                self.global_completed_before + segment_completed,
                self.global_total_timesteps,
            )
            global_remaining = max(0, self.global_total_timesteps - global_completed)
            global_elapsed = max(1e-6, now - self.global_start_time)
            global_fps = global_completed / global_elapsed if global_completed > 0 else segment_fps
            global_pct = (global_completed / self.global_total_timesteps) * 100.0
            if finished and global_remaining == 0:
                global_eta_text = format_seconds(0.0)
            elif global_completed == 0:
                global_eta_text = "calculating"
            else:
                global_eta_text = format_seconds(global_remaining / max(global_fps, 1e-6))
            message += (
                f" | overall {global_completed}/{self.global_total_timesteps} "
                f"({global_pct:5.1f}%) | overall_eta {global_eta_text}"
            )

        if finished:
            message += " | finished"

        print(message, flush=True)
        self.last_print_time = now
