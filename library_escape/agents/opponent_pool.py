from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable


class OpponentPool:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.paths: deque[Path] = deque(maxlen=max_size)

    def add(self, path: str | Path) -> None:
        self.paths.append(Path(path))

    def extend(self, values: Iterable[str | Path]) -> None:
        for value in values:
            self.add(value)

    def latest(self) -> Path | None:
        return self.paths[-1] if self.paths else None

    def all(self) -> list[Path]:
        return list(self.paths)
