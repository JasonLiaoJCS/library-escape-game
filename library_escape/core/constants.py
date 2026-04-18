from __future__ import annotations

import math


DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 600
DEFAULT_RENDER_FPS = 60
DEFAULT_PHYSICS_DT = 1.0 / 120.0

PLAYER_AGENT = "player_0"
ENEMY_AGENT = "enemy_0"

BACKGROUND_COLOR = (228, 213, 178)
GRID_COLOR = (214, 199, 163)
PLAYER_COLOR = (88, 160, 255)
ENEMY_COLOR = (219, 88, 88)
NOTE_COLOR = (255, 235, 120)
EXIT_COLOR = (122, 220, 175)
VISION_COLOR = (255, 130, 90, 55)
VISION_OUTLINE = (255, 130, 90)
HUD_COLOR = (29, 23, 20)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FULL_CIRCLE = math.tau

# Discrete actions: stop, N, NE, E, SE, S, SW, W, NW.
DISCRETE_ACTION_VECTORS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, -1.0),
    (1.0, -1.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 0.0),
    (-1.0, -1.0),
)

RAY_ANGLES: tuple[float, ...] = (
    0.0,
    math.pi / 4.0,
    math.pi / 2.0,
    3.0 * math.pi / 4.0,
    math.pi,
    5.0 * math.pi / 4.0,
    3.0 * math.pi / 2.0,
    7.0 * math.pi / 4.0,
)
