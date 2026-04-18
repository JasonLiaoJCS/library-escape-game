"""Pygame renderer for the continuous-time Python port."""

from __future__ import annotations

import math

import numpy as np
import pygame

from ..assets import FONT_PATH, asset_path


class PygameView:
    def __init__(self, world, title: str = "Library Escape", hidden: bool = False) -> None:
        self.world = world
        self.width_px = int(world.width * world.cell_size)
        self.height_px = int(world.height * world.cell_size)
        self.hidden = hidden
        self._ensure_pygame(title)
        self.font = self._load_font(28)
        self.big_font = self._load_font(44)
        self.background = self._load_scaled(asset_path("background"), (self.width_px, self.height_px))
        self.object_cache: dict[str, pygame.Surface] = {}
        self.sprite_cache: dict[str, pygame.Surface] = {}

    def _ensure_pygame(self, title: str) -> None:
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        display_surface = pygame.display.get_surface()
        if display_surface is None:
            flags = pygame.HIDDEN if self.hidden else 0
            self.screen = pygame.display.set_mode((self.width_px, self.height_px), flags=flags)
            pygame.display.set_caption(title)
        else:
            self.screen = display_surface
        self.canvas = pygame.Surface((self.width_px, self.height_px), pygame.SRCALPHA)

    def _load_font(self, size: int) -> pygame.font.Font:
        if FONT_PATH.exists():
            return pygame.font.Font(str(FONT_PATH), size)
        return pygame.font.SysFont("arial", size)

    def _load_scaled(self, path, size: tuple[int, int]) -> pygame.Surface:
        try:
            image = pygame.image.load(str(path))
            if pygame.display.get_surface() is not None:
                image = image.convert_alpha()
        except Exception:
            image = pygame.Surface(size, pygame.SRCALPHA)
            image.fill((60, 60, 60, 255))
            return image
        return pygame.transform.smoothscale(image, size)

    def render_frame(self, rgb_array: bool = False):
        self.canvas.blit(self.background, (0, 0))
        self._draw_escape_zone()
        self._draw_obstacles()
        self._draw_collectibles()
        self._draw_vision_cone()
        self._draw_last_seen_marker()
        self._draw_actor("player", self.world.player)
        self._draw_actor("enemy", self.world.enemy)
        self._draw_alert_overlay()
        self._draw_hud()
        self._draw_outcome_overlay()

        if rgb_array:
            return np.transpose(pygame.surfarray.array3d(self.canvas), (1, 0, 2))

        self.screen.blit(self.canvas, (0, 0))
        pygame.display.flip()
        return None

    def close(self) -> None:
        if pygame.display.get_surface() is not None:
            pygame.display.quit()

    def _draw_escape_zone(self) -> None:
        zone = self.world.escape_zone
        rect = pygame.Rect(
            int(zone["x"] * self.world.cell_size),
            int(zone["y"] * self.world.cell_size),
            int(zone["w"] * self.world.cell_size),
            int(zone["h"] * self.world.cell_size),
        )
        color = (80, 180, 80, 90) if self.world.can_player_escape() else (110, 110, 110, 70)
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill(color)
        self.canvas.blit(overlay, rect.topleft)
        pygame.draw.rect(self.canvas, (220, 255, 220), rect, width=2)
        label = "EXIT OPEN" if self.world.can_player_escape() else "EXIT LOCKED"
        text = self.font.render(label, True, (240, 255, 240))
        self.canvas.blit(text, (rect.x + 8, max(0, rect.y - 28)))

    def _draw_obstacles(self) -> None:
        for obstacle in self.world.obstacles:
            rect = pygame.Rect(
                int(obstacle.x * self.world.cell_size),
                int(obstacle.y * self.world.cell_size),
                int(obstacle.w * self.world.cell_size),
                int(obstacle.h * self.world.cell_size),
            )
            surface = self.object_cache.get(obstacle.texture_key)
            if surface is None:
                surface = self._load_scaled(asset_path(obstacle.texture_key), rect.size)
                self.object_cache[obstacle.texture_key] = surface
            self.canvas.blit(surface, rect.topleft)

    def _draw_collectibles(self) -> None:
        for collectible in self.world.collectibles:
            if not collectible.active:
                continue
            size = int(self.world.cell_size * 0.65)
            rect = pygame.Rect(0, 0, size, size)
            rect.center = (
                int(collectible.x * self.world.cell_size),
                int(collectible.y * self.world.cell_size),
            )
            glow = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 255, 255, 50), (size, size), size // 2 + 8)
            self.canvas.blit(glow, (rect.centerx - size, rect.centery - size))
            surface = self.object_cache.get(collectible.texture_key)
            if surface is None:
                surface = self._load_scaled(asset_path(collectible.texture_key), rect.size)
                self.object_cache[collectible.texture_key] = surface
            self.canvas.blit(surface, rect.topleft)

    def _draw_vision_cone(self) -> None:
        enemy = self.world.enemy
        center = (
            int(enemy.x * self.world.cell_size),
            int(enemy.y * self.world.cell_size),
        )
        facing_angle = math.atan2(enemy.facing_y, enemy.facing_x)
        half_angle = math.radians(enemy.vision.angle_deg / 2.0)
        max_radius = int(enemy.vision.range_cells * self.world.cell_size)

        overlay = pygame.Surface((self.width_px, self.height_px), pygame.SRCALPHA)
        points = [center]
        segments = 20
        for index in range(segments + 1):
            angle = facing_angle - half_angle + (index / segments) * (half_angle * 2.0)
            points.append(
                (
                    int(center[0] + math.cos(angle) * max_radius),
                    int(center[1] + math.sin(angle) * max_radius),
                )
            )
        pygame.draw.polygon(overlay, (220, 60, 60, 55), points)
        pygame.draw.lines(overlay, (255, 120, 120, 110), False, points[1:], width=2)
        self.canvas.blit(overlay, (0, 0))

    def _draw_last_seen_marker(self) -> None:
        if not self.world.env_config.get("ui", {}).get("show_last_seen_marker", True):
            return
        if self.world.enemy.last_seen_player is None or self.world.player_visible_to_enemy():
            return
        marker_pos = (
            int(self.world.enemy.last_seen_player[0] * self.world.cell_size),
            int(self.world.enemy.last_seen_player[1] * self.world.cell_size),
        )
        pulse = 10 + int(5 * (1.0 + math.sin(pygame.time.get_ticks() * 0.006)))
        pygame.draw.circle(self.canvas, (255, 100, 80), marker_pos, pulse, width=3)
        label = self.font.render("LAST SEEN", True, (255, 230, 220))
        self.canvas.blit(label, (marker_pos[0] + 14, marker_pos[1] - 12))

    def _draw_actor(self, actor_type: str, actor) -> None:
        sprite_key = self._sprite_key(actor_type, actor.facing_x, actor.facing_y)
        width = int(self.world.cell_size * 0.9)
        height = int(self.world.cell_size * 1.45)
        rect = pygame.Rect(0, 0, width, height)
        rect.midbottom = (
            int(actor.x * self.world.cell_size),
            int((actor.y + actor.radius) * self.world.cell_size),
        )
        surface = self.sprite_cache.get(sprite_key)
        if surface is None:
            surface = self._load_scaled(asset_path(sprite_key), rect.size)
            self.sprite_cache[sprite_key] = surface
        if actor_type == "player":
            glow = pygame.Surface((width * 2, height), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (255, 240, 120, 55), glow.get_rect())
            self.canvas.blit(glow, (rect.centerx - width, rect.bottom - height // 2))
        self.canvas.blit(surface, rect.topleft)

    def _draw_hud(self) -> None:
        lines = [
            f"Time: {self.world.time_remaining:05.1f}s",
            f"Notes: {self.world.score['note']} / {self.world.env_config['collectibles']['notes']}",
            f"Exams: {self.world.score['exam']} / {self.world.env_config['collectibles']['exams']}",
            f"Coffee: {self.world.player.coffee_timer:04.1f}s",
            f"Freeze: {self.world.enemy.freeze_timer:04.1f}s",
        ]
        for index, line in enumerate(lines):
            text = self.font.render(line, True, (255, 255, 255))
            background = pygame.Surface((text.get_width() + 16, text.get_height() + 8), pygame.SRCALPHA)
            background.fill((0, 0, 0, 140))
            self.canvas.blit(background, (16, 16 + index * 34))
            self.canvas.blit(text, (24, 20 + index * 34))

        tip = "Collect notes, unlock exit, then escape."
        tip_text = self.font.render(tip, True, (255, 255, 255))
        tip_bg = pygame.Surface((tip_text.get_width() + 18, tip_text.get_height() + 8), pygame.SRCALPHA)
        tip_bg.fill((0, 0, 0, 140))
        self.canvas.blit(tip_bg, (16, self.height_px - 52))
        self.canvas.blit(tip_text, (24, self.height_px - 48))

        if self.world.env_config.get("ui", {}).get("show_detection_meter", True):
            self._draw_detection_meter()
        if self.world.env_config.get("ui", {}).get("show_tactical_panel", True):
            self._draw_tactical_panel()

    def _draw_outcome_overlay(self) -> None:
        if not (self.world.terminated or self.world.truncated):
            return
        overlay = pygame.Surface((self.width_px, self.height_px), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.canvas.blit(overlay, (0, 0))
        title = self.world.outcome.replace("_", " ").upper() if self.world.outcome else "EPISODE END"
        title_text = self.big_font.render(title, True, (255, 255, 255))
        body = self.font.render("Press R to restart or ESC to quit.", True, (240, 240, 240))
        self.canvas.blit(title_text, (self.width_px // 2 - title_text.get_width() // 2, self.height_px // 2 - 70))
        self.canvas.blit(body, (self.width_px // 2 - body.get_width() // 2, self.height_px // 2))

    def _sprite_key(self, actor_type: str, facing_x: float, facing_y: float) -> str:
        if abs(facing_x) > abs(facing_y):
            return f"{actor_type}_{'right' if facing_x >= 0 else 'left'}"
        return f"{actor_type}_{'down' if facing_y >= 0 else 'up'}"

    def _draw_detection_meter(self) -> None:
        alert = self.world.alert_level()
        meter_rect = pygame.Rect(self.width_px - 280, 20, 240, 24)
        pygame.draw.rect(self.canvas, (0, 0, 0, 150), meter_rect, border_radius=8)
        fill_rect = meter_rect.copy()
        fill_rect.width = max(0, int((meter_rect.width - 4) * alert))
        fill_rect.inflate_ip(-4, -4)
        color = (90, 205, 110) if alert < 0.35 else (255, 182, 66) if alert < 0.7 else (255, 90, 90)
        pygame.draw.rect(self.canvas, color, fill_rect, border_radius=6)
        pygame.draw.rect(self.canvas, (255, 255, 255), meter_rect, width=2, border_radius=8)
        label = self.font.render("DETECTION", True, (245, 248, 255))
        self.canvas.blit(label, (meter_rect.x, meter_rect.y - 30))

    def _draw_tactical_panel(self) -> None:
        panel_rect = pygame.Rect(self.width_px - 320, 70, 280, 150)
        panel = pygame.Surface(panel_rect.size, pygame.SRCALPHA)
        panel.fill((6, 10, 14, 180))
        self.canvas.blit(panel, panel_rect.topleft)
        pygame.draw.rect(self.canvas, (90, 140, 190), panel_rect, width=2, border_radius=10)

        lines = [
            f"Enemy mode: {self.world.enemy_mode().upper()}",
            f"Alert: {self.world.alert_level():.2f}",
            f"Dist to enemy: {self.world.distance_between_agents():.2f}",
            f"Dist to exit: {self.world.distance_player_to_escape():.2f}",
            f"Objective progress: {self.world.objective_progress_fraction() * 100:4.1f}%",
        ]
        for index, line in enumerate(lines):
            text = self.font.render(line, True, (234, 241, 249))
            self.canvas.blit(text, (panel_rect.x + 16, panel_rect.y + 14 + index * 26))

    def _draw_alert_overlay(self) -> None:
        alert = self.world.alert_level()
        if alert <= 0.35:
            return
        alpha = int(95 * alert)
        overlay = pygame.Surface((self.width_px, self.height_px), pygame.SRCALPHA)
        overlay.fill((120, 0, 0, alpha))
        self.canvas.blit(overlay, (0, 0))
