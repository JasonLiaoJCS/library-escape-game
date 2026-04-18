"""Collectibles and scoring pickups."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Collectible:
    kind: str
    x: float
    y: float
    texture_key: str
    active: bool = True

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def distance_to(self, point: tuple[float, float]) -> float:
        px, py = point
        dx = self.x - px
        dy = self.y - py
        return float((dx * dx + dy * dy) ** 0.5)
