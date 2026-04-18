"""Runtime audio helpers for the Python port.

This restores the legacy C++ audio files that still exist in the repository:

- ``audio/game_bgm.wav``
- ``audio/detected_by_enemy.mp3``
- ``audio/score_*.mp3``

The original C++ code also referenced ``sounds/collect.wav`` and
``sounds/powerup.wav``, but those files are currently not present in the repo.
We keep explicit hooks for them so the game can use them automatically if the
files are added back later.
"""

from __future__ import annotations

from pathlib import Path

import pygame

from .config import REPO_ROOT


LEGACY_AUDIO_FILES = {
    "bgm": REPO_ROOT / "audio" / "game_bgm.wav",
    "detected": REPO_ROOT / "audio" / "detected_by_enemy.mp3",
    "score_aplus": REPO_ROOT / "audio" / "score_aplus.mp3",
    "score_a": REPO_ROOT / "audio" / "score_a.mp3",
    "score_aminus": REPO_ROOT / "audio" / "score_aminus.mp3",
    "score_bplus": REPO_ROOT / "audio" / "score_bplus.mp3",
    "score_b": REPO_ROOT / "audio" / "score_b.mp3",
    "score_bminus": REPO_ROOT / "audio" / "score_bminus.mp3",
    "score_cplus": REPO_ROOT / "audio" / "score_cplus.mp3",
    "score_c": REPO_ROOT / "audio" / "score_c.mp3",
    "score_cminus": REPO_ROOT / "audio" / "score_cminus.mp3",
    "score_f": REPO_ROOT / "audio" / "score_f.mp3",
}

LEGACY_MISSING_AUDIO_FILES = {
    "collect": REPO_ROOT / "sounds" / "collect.wav",
    "powerup": REPO_ROOT / "sounds" / "powerup.wav",
}


def available_audio_files() -> dict[str, Path]:
    return {key: path for key, path in LEGACY_AUDIO_FILES.items() if path.exists()}


def missing_legacy_audio_files() -> dict[str, Path]:
    return {key: path for key, path in LEGACY_MISSING_AUDIO_FILES.items() if not path.exists()}


def grade_for_score(score_value: int, max_score_value: int) -> str:
    percent = 0.0 if max_score_value <= 0 else (float(score_value) / float(max_score_value)) * 100.0
    if percent >= 100.0:
        return "A+"
    if percent >= 90.0:
        return "A"
    if percent >= 85.0:
        return "A-"
    if percent >= 80.0:
        return "B+"
    if percent >= 75.0:
        return "B"
    if percent >= 70.0:
        return "B-"
    if percent >= 65.0:
        return "C+"
    if percent >= 60.0:
        return "C"
    return "F"


def score_sound_key(grade: str) -> str:
    normalized = grade.strip().upper()
    lookup = {
        "A+": "score_aplus",
        "A": "score_a",
        "A-": "score_aminus",
        "B+": "score_bplus",
        "B": "score_b",
        "B-": "score_bminus",
        "C+": "score_cplus",
        "C": "score_c",
        "C-": "score_cminus",
        "F": "score_f",
    }
    return lookup.get(normalized, "score_f")


class GameAudioController:
    def __init__(self, world, *, music_volume: float = 0.35, sfx_volume: float = 0.75) -> None:
        self.music_volume = max(0.0, min(1.0, float(music_volume)))
        self.sfx_volume = max(0.0, min(1.0, float(sfx_volume)))
        self.available = False
        self._sound_cache: dict[str, pygame.mixer.Sound] = {}
        self._round_finished = False
        self._previous_visibility = False
        self._try_initialize()
        self.reset_round(world)
        self._play_music_loop()

    def _try_initialize(self) -> None:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self.available = True
        except Exception:
            self.available = False

    def _load_sound(self, key: str) -> pygame.mixer.Sound | None:
        if not self.available:
            return None
        if key in self._sound_cache:
            return self._sound_cache[key]
        path = LEGACY_AUDIO_FILES.get(key, LEGACY_MISSING_AUDIO_FILES.get(key))
        if path is None or not path.exists():
            return None
        try:
            sound = pygame.mixer.Sound(str(path))
        except Exception:
            return None
        sound.set_volume(self.sfx_volume)
        self._sound_cache[key] = sound
        return sound

    def _play_sound(self, key: str) -> None:
        sound = self._load_sound(key)
        if sound is None:
            return
        try:
            sound.play()
        except Exception:
            return

    def _play_music_loop(self) -> None:
        if not self.available:
            return
        bgm = LEGACY_AUDIO_FILES["bgm"]
        if not bgm.exists():
            return
        try:
            if pygame.mixer.music.get_busy():
                return
            pygame.mixer.music.load(str(bgm))
            pygame.mixer.music.set_volume(self.music_volume)
            pygame.mixer.music.play(-1)
        except Exception:
            return

    def reset_round(self, world) -> None:
        self._round_finished = False
        self._previous_visibility = bool(world.player_visible_to_enemy())

    def update(self, world, events) -> None:
        if not self.available:
            return

        current_visibility = bool(world.player_visible_to_enemy())
        if int(getattr(events, "detection_events", 0)) > 0 or (current_visibility and not self._previous_visibility):
            self._play_sound("detected")

        if getattr(events, "count", None) is not None:
            collected_notes = int(events.count("note")) + int(events.count("exam"))
            collected_powerups = int(events.count("coffee")) + int(events.count("freeze"))
            if collected_notes > 0:
                self._play_sound("collect")
            elif collected_powerups > 0:
                self._play_sound("powerup")

        if not self._round_finished and (world.terminated or world.truncated):
            self._round_finished = True
            grade = grade_for_score(int(world.score_value()), int(world.max_score_value()))
            self._play_sound(score_sound_key(grade))

        self._previous_visibility = current_visibility

    def close(self) -> None:
        if not self.available:
            return
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
