"""Headless renderer used by training or smoke tests."""

from __future__ import annotations


class NullView:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def render_frame(self, rgb_array: bool = False):
        return None

    def close(self) -> None:
        return None
