"""Static world geometry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Obstacle:
    kind: str
    texture_key: str
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    @classmethod
    def from_dict(cls, payload: dict[str, float | str]) -> "Obstacle":
        return cls(
            kind=str(payload["kind"]),
            texture_key=str(payload["texture_key"]),
            x=float(payload["x"]),
            y=float(payload["y"]),
            w=float(payload["w"]),
            h=float(payload["h"]),
        )
