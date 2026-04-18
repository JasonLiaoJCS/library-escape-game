from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pygame

from library_escape.core.constants import (
    BACKGROUND_COLOR,
    BLACK,
    ENEMY_COLOR,
    EXIT_COLOR,
    GRID_COLOR,
    HUD_COLOR,
    NOTE_COLOR,
    PLAYER_COLOR,
    VISION_COLOR,
    VISION_OUTLINE,
    WHITE,
)
from library_escape.core.physics import angle_to_vector
from library_escape.core.world import LibraryWorld
from library_escape.paths import FONT_DIR, IMG_DIR


class PygameView:
    def __init__(self, config: dict[str, Any], render_mode: str = "human") -> None:
        pygame.init()
        pygame.font.init()
        self.render_mode = render_mode
        render_cfg = config["render"]
        self.width = int(render_cfg["window_width"])
        self.height = int(render_cfg["window_height"])
        self.fps = int(render_cfg["fps"])
        self.show_grid = bool(render_cfg.get("show_grid", False))
        self.show_debug_text = bool(render_cfg.get("show_debug_text", False))
        self.clock = pygame.time.Clock()

        flags = 0
        if render_mode == "human":
            self.surface = pygame.display.set_mode((self.width, self.height), flags)
            pygame.display.set_caption("Library Escape")
        else:
            self.surface = pygame.Surface((self.width, self.height))

        self.background = self._load_scaled_image(IMG_DIR / "playground_background.jpg", (self.width, self.height))
        self.table_texture = self._load_scaled_image(IMG_DIR / "table1x2.png", (120, 60))
        self.bookshelf_texture = self._load_scaled_image(IMG_DIR / "bookshelf_image.jpg", (80, 160))
        self.font = self._load_font("Action_Man_Bold.ttf", 28)
        self.small_font = self._load_font("arial.ttf", 18)

    def _load_font(self, filename: str, size: int) -> pygame.font.Font:
        path = FONT_DIR / filename
        if path.exists():
            return pygame.font.Font(str(path), size)
        return pygame.font.SysFont("arial", size)

    def _load_scaled_image(self, path: Path, size: tuple[int, int]) -> pygame.Surface | None:
        if not path.exists():
            return None
        try:
            image = pygame.image.load(str(path))
            if pygame.display.get_surface() is not None:
                image = image.convert_alpha()
            return pygame.transform.smoothscale(image, size)
        except pygame.error:
            return None

    def _draw_vision_cone(self, actor_pos: np.ndarray, facing: float, view_range: float, view_angle: float) -> None:
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        direction_1 = angle_to_vector(facing - view_angle / 2.0) * view_range
        direction_2 = angle_to_vector(facing + view_angle / 2.0) * view_range
        points = [
            (float(actor_pos[0]), float(actor_pos[1])),
            (float(actor_pos[0] + direction_1[0]), float(actor_pos[1] + direction_1[1])),
            (float(actor_pos[0] + direction_2[0]), float(actor_pos[1] + direction_2[1])),
        ]
        pygame.draw.polygon(overlay, VISION_COLOR, points)
        pygame.draw.polygon(overlay, VISION_OUTLINE, points, width=2)
        self.surface.blit(overlay, (0, 0))

    def _draw_actor(self, position: np.ndarray, radius: float, color: tuple[int, int, int], facing: float, name: str) -> None:
        center = (int(position[0]), int(position[1]))
        pygame.draw.circle(self.surface, color, center, int(radius))
        heading = angle_to_vector(facing) * (radius + 14.0)
        tip = (int(position[0] + heading[0]), int(position[1] + heading[1]))
        pygame.draw.line(self.surface, WHITE, center, tip, width=3)
        label = self.small_font.render(name, True, WHITE)
        rect = label.get_rect(midbottom=(center[0], center[1] - int(radius) - 6))
        self.surface.blit(label, rect)

    def _draw_obstacle(self, obstacle: Any) -> None:
        rect = pygame.Rect(
            int(obstacle.rect.x),
            int(obstacle.rect.y),
            int(obstacle.rect.w),
            int(obstacle.rect.h),
        )
        texture = None
        if obstacle.kind == "table":
            texture = self.table_texture
        elif obstacle.kind == "bookshelf":
            texture = self.bookshelf_texture

        if texture is not None:
            scaled = pygame.transform.smoothscale(texture, rect.size)
            self.surface.blit(scaled, rect.topleft)
        else:
            pygame.draw.rect(self.surface, (109, 78, 43), rect, border_radius=6)
        pygame.draw.rect(self.surface, BLACK, rect, width=2, border_radius=6)

    def _draw_grid(self, world: LibraryWorld) -> None:
        if not self.show_grid:
            return
        grid_size = int(world.grid_size)
        for x in range(0, self.width, grid_size):
            pygame.draw.line(self.surface, GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, grid_size):
            pygame.draw.line(self.surface, GRID_COLOR, (0, y), (self.width, y))

    def draw(self, world: LibraryWorld, title: str | None = None) -> None:
        if self.background is not None:
            self.surface.blit(self.background, (0, 0))
        else:
            self.surface.fill(BACKGROUND_COLOR)

        self._draw_grid(world)

        exit_rect = pygame.Rect(
            int(world.exit_zone.x),
            int(world.exit_zone.y),
            int(world.exit_zone.w),
            int(world.exit_zone.h),
        )
        pygame.draw.rect(self.surface, EXIT_COLOR, exit_rect, border_radius=10)
        pygame.draw.rect(self.surface, BLACK, exit_rect, width=2, border_radius=10)

        for obstacle in world.obstacles:
            self._draw_obstacle(obstacle)

        for note in world.notes:
            if note.collected:
                continue
            center = (int(note.position[0]), int(note.position[1]))
            pygame.draw.circle(self.surface, NOTE_COLOR, center, int(note.radius) + 4)
            pygame.draw.circle(self.surface, WHITE, center, int(note.radius))
            pygame.draw.circle(self.surface, BLACK, center, int(note.radius), width=2)

        self._draw_vision_cone(
            world.enemy.position,
            world.enemy.facing,
            world.enemy.view_range,
            world.enemy.view_angle,
        )
        self._draw_actor(world.player.position, world.player.radius, PLAYER_COLOR, world.player.facing, world.player.name)
        self._draw_actor(world.enemy.position, world.enemy.radius, ENEMY_COLOR, world.enemy.facing, world.enemy.name)

        title_text = title or "Library Escape"
        hud = [
            title_text,
            f"Time: {world.remaining_time:05.1f}",
            f"Notes: {world.notes_collected}/{world.total_notes}",
        ]
        if world.status.outcome:
            hud.append(f"Outcome: {world.status.outcome}")
        for index, line in enumerate(hud):
            text = self.font.render(line, True, HUD_COLOR)
            self.surface.blit(text, (18, 12 + index * 28))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)

    def rgb_array(self) -> np.ndarray:
        frame = pygame.surfarray.array3d(self.surface)
        return np.transpose(frame, (1, 0, 2))

    def close(self) -> None:
        pygame.quit()
