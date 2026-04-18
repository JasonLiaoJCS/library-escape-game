from __future__ import annotations

import numpy as np

from library_escape.core.world import LibraryWorld


class NullView:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def draw(self, world: LibraryWorld, title: str | None = None) -> None:
        return None

    def rgb_array(self) -> np.ndarray:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def close(self) -> None:
        return None
