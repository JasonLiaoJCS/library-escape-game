"""Keyboard to movement-vector adapter."""

from __future__ import annotations

import pygame

from ..core.actions import normalize_vector


class KeyboardController:
    def current_action(self) -> tuple[float, float]:
        pressed = pygame.key.get_pressed()
        x = float(pressed[pygame.K_d] or pressed[pygame.K_RIGHT]) - float(pressed[pygame.K_a] or pressed[pygame.K_LEFT])
        y = float(pressed[pygame.K_s] or pressed[pygame.K_DOWN]) - float(pressed[pygame.K_w] or pressed[pygame.K_UP])
        return normalize_vector((x, y))

    def collect_pressed(self) -> bool:
        pressed = pygame.key.get_pressed()
        return bool(pressed[pygame.K_e])
