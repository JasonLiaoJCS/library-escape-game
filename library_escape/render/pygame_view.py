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

        hud_size = max(18, min(28, int(world.cell_size * 0.72)))
        self.small_font = self._load_font(max(14, int(hud_size * 0.75)))
        self.font = self._load_font(hud_size)
        self.big_font = self._load_font(max(34, int(world.cell_size * 1.18)))

        self.background = self._load_background()
        self.object_cache: dict[str, pygame.Surface] = {}
        self.sprite_cache: dict[str, pygame.Surface] = {}
        self.detect_penalty_label = self._build_detect_penalty_label()

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

    def _load_scaled(self, path, size: tuple[int, int], smooth: bool = False) -> pygame.Surface:
        try:
            image = pygame.image.load(str(path))
            if pygame.display.get_surface() is not None:
                image = image.convert_alpha()
        except Exception:
            image = pygame.Surface(size, pygame.SRCALPHA)
            image.fill((60, 60, 60, 255))
            return image
        if smooth:
            return pygame.transform.smoothscale(image, size)
        return pygame.transform.scale(image, size)

    def _load_background(self) -> pygame.Surface:
        try:
            path = asset_path("background")
            image = pygame.image.load(str(path))
            if pygame.display.get_surface() is not None:
                image = image.convert()
        except Exception:
            return self._load_scaled(asset_path("background_alt"), (self.width_px, self.height_px), smooth=True)

        return pygame.transform.smoothscale(image, (self.width_px, self.height_px))

    def _build_detect_penalty_label(self) -> str:
        penalty = float(getattr(self.world, "enemy_detect_penalty_seconds", 0.0))
        if penalty <= 0.0:
            return "SPOTTED!"
        whole = int(round(penalty))
        if abs(penalty - whole) <= 1e-6:
            return f"SPOTTED!  -{whole}s"
        return f"SPOTTED!  -{penalty:.1f}s"

    def render_frame(self, rgb_array: bool = False):
        self.canvas.blit(self.background, (0, 0))
        self._draw_escape_zone()
        self._draw_obstacles()
        self._draw_collectibles()
        self._draw_collection_progress()
        self._draw_vision_cones()
        self._draw_last_seen_marker()
        self._draw_actor("player", self.world.player)
        for support_enemy in getattr(self.world, "support_enemies", []):
            self._draw_actor("enemy", support_enemy, primary=False)
        self._draw_actor("enemy", self.world.enemy, primary=True)
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
        if not self._should_render_escape_zone():
            return

        zone = self.world.escape_zone
        rect = pygame.Rect(
            int(zone["x"] * self.world.cell_size),
            int(zone["y"] * self.world.cell_size),
            int(zone["w"] * self.world.cell_size),
            int(zone["h"] * self.world.cell_size),
        )
        color = (88, 176, 86, 92) if self.world.can_player_escape() else (110, 110, 110, 70)
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill(color)
        self.canvas.blit(overlay, rect.topleft)
        pygame.draw.rect(self.canvas, (232, 246, 226), rect, width=2)

        if not self._ui_flag("show_exit_label", True):
            return

        label = "EXIT OPEN" if self.world.can_player_escape() else "EXIT LOCKED"
        text = self.small_font.render(label, True, (244, 250, 240))
        label_x = min(max(8, rect.x + 6), self.width_px - text.get_width() - 8)
        label_y = max(0, rect.y - text.get_height() - 6)
        self._blit_panel((label_x - 8, label_y - 2), (text.get_width() + 16, text.get_height() + 6), (0, 0, 0, 135))
        self.canvas.blit(text, (label_x, label_y))

    def _draw_obstacles(self) -> None:
        for obstacle in self.world.obstacles:
            rect = pygame.Rect(
                int(obstacle.x * self.world.cell_size),
                int(obstacle.y * self.world.cell_size),
                int(obstacle.w * self.world.cell_size),
                int(obstacle.h * self.world.cell_size),
            )
            surface = self.object_cache.get(obstacle.texture_key)
            if surface is None or surface.get_size() != rect.size:
                surface = self._load_scaled(asset_path(obstacle.texture_key), rect.size)
                self.object_cache[obstacle.texture_key] = surface
            self.canvas.blit(surface, rect.topleft)

    def _draw_collectibles(self) -> None:
        for collectible in self.world.collectibles:
            if not collectible.active:
                continue
            rect = self._collectible_rect(collectible)
            glow = pygame.Surface((rect.width + 18, rect.height + 18), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (255, 255, 255, 42), glow.get_rect())
            self.canvas.blit(glow, (rect.x - 9, rect.y - 9))
            shadow = pygame.Surface((rect.width, rect.height // 2), pygame.SRCALPHA)
            pygame.draw.ellipse(shadow, (0, 0, 0, 36), shadow.get_rect())
            self.canvas.blit(shadow, (rect.x, rect.bottom - shadow.get_height() // 2))
            cache_key = f"pickup:{collectible.texture_key}:{rect.width}x{rect.height}"
            surface = self.object_cache.get(cache_key)
            if surface is None or surface.get_size() != rect.size:
                surface = self._load_scaled(asset_path(collectible.texture_key), rect.size)
                self.object_cache[cache_key] = surface
            self.canvas.blit(surface, rect.topleft)

    def _draw_collection_progress(self) -> None:
        target_index = getattr(self.world, "collection_target_index", None)
        progress = float(getattr(self.world, "collection_progress", 0.0))
        if target_index is None or progress <= 0.0 or target_index >= len(self.world.collectibles):
            return

        collectible = self.world.collectibles[target_index]
        if not collectible.active:
            return

        rect = self._collectible_rect(collectible)
        center_x = rect.centerx
        center_y = rect.centery
        radius = int(self.world.cell_size * 0.58)
        arc_rect = pygame.Rect(center_x - radius, center_y - radius, radius * 2, radius * 2)
        pygame.draw.arc(
            self.canvas,
            (255, 255, 255),
            arc_rect,
            -math.pi / 2,
            -math.pi / 2 + (math.tau * min(1.0, progress)),
            width=4,
        )

    def _draw_vision_cones(self) -> None:
        overlay = pygame.Surface((self.width_px, self.height_px), pygame.SRCALPHA)
        for enemy in getattr(self.world, "support_enemies", []):
            self._draw_single_vision_cone(overlay, enemy, fill_alpha=24, outline_alpha=48)
        self._draw_single_vision_cone(overlay, self.world.enemy, fill_alpha=46, outline_alpha=86)
        self.canvas.blit(overlay, (0, 0))

    def _draw_single_vision_cone(self, overlay: pygame.Surface, enemy, fill_alpha: int, outline_alpha: int) -> None:
        center = self._vision_origin(enemy)
        facing_angle = math.atan2(enemy.facing_y, enemy.facing_x)
        half_angle = math.radians(enemy.vision.angle_deg / 2.0)
        max_radius = int(enemy.vision.range_cells * self.world.cell_size)
        points = [center]
        segments = 28
        for index in range(segments + 1):
            angle = facing_angle - half_angle + (index / segments) * (half_angle * 2.0)
            points.append(
                (
                    int(center[0] + math.cos(angle) * max_radius),
                    int(center[1] + math.sin(angle) * max_radius),
                )
            )
        pygame.draw.polygon(overlay, (210, 64, 64, fill_alpha), points)
        pygame.draw.lines(overlay, (255, 118, 118, outline_alpha), False, points[1:], width=2)

    def _draw_last_seen_marker(self) -> None:
        if not self._ui_flag("show_last_seen_marker", True):
            return
        if self.world.enemy.last_seen_player is None or self.world.player_visible_to_enemy():
            return
        marker_pos = (
            int(self.world.enemy.last_seen_player[0] * self.world.cell_size),
            int(self.world.enemy.last_seen_player[1] * self.world.cell_size),
        )
        pulse = 8 + int(4 * (1.0 + math.sin(pygame.time.get_ticks() * 0.006)))
        pygame.draw.circle(self.canvas, (255, 110, 84), marker_pos, pulse, width=3)
        label = self.small_font.render("LAST SEEN", True, (255, 236, 226))
        self.canvas.blit(label, (marker_pos[0] + 12, marker_pos[1] - label.get_height() // 2))

    def _draw_actor(self, actor_type: str, actor, primary: bool = True) -> None:
        sprite_key = self._sprite_key(actor_type, actor.facing_x, actor.facing_y)
        width = max(28, int(self.world.cell_size * 0.80))
        height = max(56, int(self.world.cell_size * 1.60))
        rect = pygame.Rect(0, 0, width, height)
        rect.midbottom = (
            int(actor.x * self.world.cell_size),
            int((actor.y + actor.radius) * self.world.cell_size),
        )
        surface = self.sprite_cache.get(sprite_key)
        if surface is None or surface.get_size() != rect.size:
            surface = self._load_scaled(asset_path(sprite_key), rect.size)
            self.sprite_cache[sprite_key] = surface
        if actor_type == "player":
            glow = pygame.Surface((width * 2, height), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (255, 239, 120, 34), glow.get_rect())
            self.canvas.blit(glow, (rect.centerx - width, rect.bottom - height // 2))
        elif primary:
            glow = pygame.Surface((width + 14, height // 2), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (255, 96, 96, 28), glow.get_rect())
            self.canvas.blit(glow, (rect.x - 7, rect.bottom - glow.get_height() // 2))
        self.canvas.blit(surface, rect.topleft)

    def _draw_hud(self) -> None:
        if self._is_classic_ruleset():
            self._draw_classic_hud()
        else:
            self._draw_modern_hud()

    def _draw_classic_hud(self) -> None:
        timer_text = self.big_font.render(f"TIME: {self._format_clock(self.world.time_remaining)}", True, (255, 255, 255))
        self._blit_panel((16, 12), (timer_text.get_width() + 18, timer_text.get_height() + 10), (0, 0, 0, 138))
        self.canvas.blit(timer_text, (24, 16))

        counters = [
            f"NOTES: {self.world.score['note']} / {self.world.env_config['collectibles']['notes']}",
            f"EXAMS: {self.world.score['exam']} / {self.world.env_config['collectibles']['exams']}",
        ]
        for index, line in enumerate(counters):
            text = self.font.render(line, True, (255, 255, 255))
            top = 22 + timer_text.get_height() + index * (text.get_height() + 10)
            self._blit_panel((16, top), (text.get_width() + 18, text.get_height() + 8), (0, 0, 0, 128))
            self.canvas.blit(text, (24, top + 2))

        self._draw_score_box()
        self._draw_status_icons()

        if self.world.alert_level() > 0.0:
            alert_text = self.font.render(self.detect_penalty_label, True, (255, 248, 240))
            box_size = (alert_text.get_width() + 24, alert_text.get_height() + 12)
            box_x = (self.width_px - box_size[0]) // 2
            self._blit_panel((box_x, 16), box_size, (120, 16, 16, 180))
            self.canvas.blit(alert_text, (box_x + 12, 22))

        tip = "WASD move   E collect   avoid the vision cone"
        tip_text = self.small_font.render(tip.upper(), True, (255, 255, 255))
        tip_x = 16
        tip_y = self.height_px - tip_text.get_height() - 18
        self._blit_panel((tip_x, tip_y - 4), (tip_text.get_width() + 16, tip_text.get_height() + 8), (0, 0, 0, 138))
        self.canvas.blit(tip_text, (tip_x + 8, tip_y))

        prompt_target = None
        if self._is_classic_ruleset() and hasattr(self.world, "nearest_interactable_collectible"):
            prompt_target = self.world.nearest_interactable_collectible()
        if prompt_target is not None:
            prompt = self.small_font.render("HOLD E", True, (255, 248, 232))
            prompt_rect = self._collectible_rect(prompt_target)
            prompt_x = max(8, min(self.width_px - prompt.get_width() - 8, prompt_rect.x - 2))
            prompt_y = max(50, prompt_rect.y - prompt.get_height() - 8)
            self._blit_panel((prompt_x - 6, prompt_y - 2), (prompt.get_width() + 12, prompt.get_height() + 6), (0, 0, 0, 144))
            self.canvas.blit(prompt, (prompt_x, prompt_y))

    def _draw_modern_hud(self) -> None:
        lines = [
            f"Time: {self.world.time_remaining:05.1f}s",
            f"Notes: {self.world.score['note']} / {self.world.env_config['collectibles']['notes']}",
            f"Exams: {self.world.score['exam']} / {self.world.env_config['collectibles']['exams']}",
            f"Coffee: {self.world.player.coffee_timer:04.1f}s",
            f"Freeze: {self.world.max_enemy_freeze_timer():04.1f}s",
        ]
        top = 16
        for line in lines:
            text = self.font.render(line, True, (255, 255, 255))
            self._blit_panel((16, top), (text.get_width() + 16, text.get_height() + 8), (0, 0, 0, 132))
            self.canvas.blit(text, (24, top + 2))
            top += text.get_height() + 10

        tip = "Collect notes, unlock exit, then escape."
        tip_text = self.small_font.render(tip, True, (255, 255, 255))
        tip_w = min(self.width_px - 24, tip_text.get_width() + 18)
        tip_y = self.height_px - tip_text.get_height() - 18
        self._blit_panel((16, tip_y - 4), (tip_w, tip_text.get_height() + 8), (0, 0, 0, 132))
        self.canvas.blit(tip_text, (24, tip_y))

        if self._ui_flag("show_detection_meter", True):
            self._draw_detection_meter()
        if self._ui_flag("show_tactical_panel", True):
            self._draw_tactical_panel()

    def _draw_score_box(self) -> None:
        score_text = self.font.render(f"SCORE: {self._score_value()}", True, (255, 215, 0))
        box_w = score_text.get_width() + 22
        box_h = score_text.get_height() + 10
        box_x = self.width_px - box_w - 16
        self._blit_panel((box_x, 16), (box_w, box_h), (0, 0, 0, 138))
        self.canvas.blit(score_text, (box_x + 11, 20))

    def _draw_status_icons(self) -> None:
        icon_specs = []
        if self.world.player.coffee_timer > 0.0:
            icon_specs.append(("coffee", self.world.player.coffee_timer))
        if self.world.max_enemy_freeze_timer() > 0.0:
            icon_specs.append(("freeze", self.world.max_enemy_freeze_timer()))

        icon_size = max(36, int(self.world.cell_size * 1.10))
        margin = 12
        top = 56 + self.font.get_height()
        for texture_key, seconds_left in icon_specs:
            x = self.width_px - icon_size - 20
            self._blit_panel((x - 64, top - 2), (icon_size + 66, icon_size + 4), (0, 0, 0, 128))
            timer_text = self.small_font.render(f"{seconds_left:04.1f}s", True, (255, 255, 255))
            self.canvas.blit(timer_text, (x - 56, top + (icon_size - timer_text.get_height()) // 2))
            icon = self.object_cache.get(f"icon:{texture_key}:{icon_size}")
            if icon is None:
                icon = self._load_scaled(asset_path(texture_key), (icon_size, icon_size))
                self.object_cache[f"icon:{texture_key}:{icon_size}"] = icon
            self.canvas.blit(icon, (x, top))
            top += icon_size + margin

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
        meter_width = min(220, self.width_px - 48)
        meter_rect = pygame.Rect(self.width_px - meter_width - 24, 20, meter_width, 22)
        pygame.draw.rect(self.canvas, (0, 0, 0, 152), meter_rect, border_radius=8)
        fill_width = max(0, int((meter_rect.width - 4) * alert))
        fill_rect = pygame.Rect(meter_rect.x + 2, meter_rect.y + 2, fill_width, meter_rect.height - 4)
        color = (90, 205, 110) if alert < 0.35 else (255, 182, 66) if alert < 0.7 else (255, 90, 90)
        pygame.draw.rect(self.canvas, color, fill_rect, border_radius=6)
        pygame.draw.rect(self.canvas, (255, 255, 255), meter_rect, width=2, border_radius=8)
        label = self.small_font.render("DETECTION", True, (245, 248, 255))
        self.canvas.blit(label, (meter_rect.x, meter_rect.y - label.get_height() - 4))

    def _draw_tactical_panel(self) -> None:
        lines = [
            f"Enemy mode: {self.world.enemy_mode().upper()}",
            f"Alert: {self.world.alert_level():.2f}",
            f"Dist to enemy: {self.world.distance_between_agents():.2f}",
            f"Enemies: {len(getattr(self.world, 'all_enemies', lambda: [self.world.enemy])())}",
            f"Visible enemies: {getattr(self.world, 'visible_enemy_count', lambda: int(self.world.player_visible_to_enemy()))()}",
            f"Dist to exit: {self.world.distance_player_to_escape():.2f}",
            f"Objective: {self.world.objective_progress_fraction() * 100:4.1f}%",
        ]
        text_surfaces = [self.small_font.render(line, True, (234, 241, 249)) for line in lines]
        panel_width = min(self.width_px - 40, max(surface.get_width() for surface in text_surfaces) + 24)
        panel_height = sum(surface.get_height() for surface in text_surfaces) + 24 + (len(text_surfaces) - 1) * 6
        panel_rect = pygame.Rect(self.width_px - panel_width - 20, 62, panel_width, panel_height)
        self._blit_panel(panel_rect.topleft, panel_rect.size, (6, 10, 14, 176))
        pygame.draw.rect(self.canvas, (90, 140, 190), panel_rect, width=2, border_radius=10)

        y = panel_rect.y + 12
        for surface in text_surfaces:
            self.canvas.blit(surface, (panel_rect.x + 12, y))
            y += surface.get_height() + 6

    def _draw_alert_overlay(self) -> None:
        alert = self.world.alert_level()
        if alert <= 0.0:
            return
        alpha = 72 if self._is_classic_ruleset() else int(95 * alert)
        overlay = pygame.Surface((self.width_px, self.height_px), pygame.SRCALPHA)
        overlay.fill((120, 0, 0, alpha))
        self.canvas.blit(overlay, (0, 0))

    def _vision_origin(self, enemy) -> tuple[int, int]:
        center_x = int(enemy.x * self.world.cell_size)
        center_y = int(enemy.y * self.world.cell_size)
        if abs(enemy.facing_x) >= abs(enemy.facing_y):
            center_y -= int(self.world.cell_size / 2.15)
        return center_x, center_y

    def _collectible_rect(self, collectible) -> pygame.Rect:
        cell_x = int(collectible.x - 0.5)
        cell_y = int(collectible.y - 0.5)
        bob = int(round(math.sin((pygame.time.get_ticks() * 0.005) + (cell_x * 0.7) + (cell_y * 0.3)) * 1.2))
        return pygame.Rect(
            cell_x * self.world.cell_size,
            cell_y * self.world.cell_size - bob,
            self.world.cell_size,
            self.world.cell_size,
        )

    def _blit_panel(self, position: tuple[int, int], size: tuple[int, int], color: tuple[int, int, int, int]) -> None:
        panel = pygame.Surface(size, pygame.SRCALPHA)
        panel.fill(color)
        self.canvas.blit(panel, position)

    def _format_clock(self, seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        minutes = total_seconds // 60
        remainder = total_seconds % 60
        return f"{minutes:02d}:{remainder:02d}"

    def _ui_flag(self, key: str, default: bool) -> bool:
        return bool(self.world.env_config.get("ui", {}).get(key, default))

    def _should_render_escape_zone(self) -> bool:
        if hasattr(self.world, "should_render_escape_zone"):
            return bool(self.world.should_render_escape_zone())
        ruleset = str(self.world.env_config.get("world", {}).get("game_mode", self.world.env_config.get("world", {}).get("ruleset", "escape"))).lower()
        return ruleset != "collection"

    def _is_classic_ruleset(self) -> bool:
        if hasattr(self.world, "is_classic_ruleset"):
            return bool(self.world.is_classic_ruleset())
        ruleset = str(self.world.env_config.get("world", {}).get("game_mode", self.world.env_config.get("world", {}).get("ruleset", "escape"))).lower()
        return ruleset == "collection"

    def _score_value(self) -> int:
        if hasattr(self.world, "score_value"):
            return int(self.world.score_value())
        return int(self.world.score.get("note", 0) * 8 + self.world.score.get("exam", 0) * 12)
