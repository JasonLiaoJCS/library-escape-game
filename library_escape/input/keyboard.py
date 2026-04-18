from __future__ import annotations

from typing import Any

import numpy as np
import pygame

from library_escape.core.actions import normalize_vector


class KeyboardActionAdapter:
    def __init__(self, action_scheme: str) -> None:
        self.action_scheme = action_scheme

    def action_from_pressed(self, pressed: Any) -> Any:
        dx = float(pressed[pygame.K_d] or pressed[pygame.K_RIGHT]) - float(
            pressed[pygame.K_a] or pressed[pygame.K_LEFT]
        )
        dy = float(pressed[pygame.K_s] or pressed[pygame.K_DOWN]) - float(
            pressed[pygame.K_w] or pressed[pygame.K_UP]
        )
        direction = normalize_vector(np.array([dx, dy], dtype=np.float32))
        if self.action_scheme == "continuous":
            return direction

        mapping = {
            (0, 0): 0,
            (0, -1): 1,
            (1, -1): 2,
            (1, 0): 3,
            (1, 1): 4,
            (0, 1): 5,
            (-1, 1): 6,
            (-1, 0): 7,
            (-1, -1): 8,
        }
        key = (int(np.sign(direction[0])), int(np.sign(direction[1])))
        return mapping[key]
