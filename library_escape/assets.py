"""Asset paths used by the Python renderer."""

from __future__ import annotations

from pathlib import Path

from .config import REPO_ROOT


ASSET_PATHS = {
    "background": REPO_ROOT / "imgs" / "playground_background.jpg",
    "background_alt": REPO_ROOT / "imgs" / "playground_background(1).jpg",
    "table_bag_1x1": REPO_ROOT / "imgs" / "table-bag-1x1.png",
    "table_bag_2x2": REPO_ROOT / "imgs" / "table-bag-2x2.png",
    "table_book_1x1": REPO_ROOT / "imgs" / "table-book-1x1.png",
    "table_book_2x2": REPO_ROOT / "imgs" / "table-book-2x2.png",
    "bookshelf_2x1": REPO_ROOT / "imgs" / "bookshelf-2x1.png",
    "bookshelf_3x1": REPO_ROOT / "imgs" / "bookshelf-3x1.png",
    "bookshelf_4x1": REPO_ROOT / "imgs" / "bookshelf-4x1.png",
    "bookshelf_4x2": REPO_ROOT / "imgs" / "bookshelf-4x2.png",
    "bookshelf_up": REPO_ROOT / "imgs" / "bookshelf-up-4x14.png",
    "bookshelf_left": REPO_ROOT / "imgs" / "bookshelf-left-1x14.png",
    "bookshelf_right": REPO_ROOT / "imgs" / "bookshelf-right-1x14.png",
    "note": REPO_ROOT / "imgs" / "note.png",
    "exam": REPO_ROOT / "imgs" / "pastexam.png",
    "coffee": REPO_ROOT / "imgs" / "coffee.png",
    "freeze": REPO_ROOT / "imgs" / "freeze.png",
    "player_up": REPO_ROOT / "imgs" / "character_up.png",
    "player_down": REPO_ROOT / "imgs" / "character_down.png",
    "player_left": REPO_ROOT / "imgs" / "character_left.png",
    "player_right": REPO_ROOT / "imgs" / "character_right.png",
    "enemy_up": REPO_ROOT / "imgs" / "character_up.png",
    "enemy_down": REPO_ROOT / "imgs" / "character_down.png",
    "enemy_left": REPO_ROOT / "imgs" / "character_left.png",
    "enemy_right": REPO_ROOT / "imgs" / "character_right.png",
}

FONT_PATH = REPO_ROOT / "fonts" / "Action_Man_Bold.ttf"


def asset_path(key: str) -> Path:
    return ASSET_PATHS[key]
