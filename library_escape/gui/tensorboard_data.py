"""Helpers for reading TensorBoard scalar data inside the GUI."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


def load_scalar_series(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        return {}

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for event_file in run_dir.rglob("events.out.tfevents.*"):
        try:
            accumulator = event_accumulator.EventAccumulator(
                str(event_file),
                size_guidance={"scalars": 50_000},
            )
            accumulator.Reload()
        except Exception:
            continue
        for tag in accumulator.Tags().get("scalars", []):
            try:
                events = accumulator.Scalars(tag)
            except Exception:
                continue
            for event in events:
                series[tag].append((int(event.step), float(event.value)))

    for tag, points in series.items():
        points.sort(key=lambda item: item[0])
        deduped: list[tuple[int, float]] = []
        last_point: tuple[int, float] | None = None
        for point in points:
            if last_point == point:
                continue
            deduped.append(point)
            last_point = point
        series[tag] = deduped
    return dict(series)
