"""Replay viewer and frame exporter."""

from __future__ import annotations

import argparse
from pathlib import Path

import pygame

from ..config import resolve_repo_path
from ..render.pygame_view import PygameView
from .io import ReplayWorld, load_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View or export a Library Escape replay.")
    parser.add_argument("--replay", type=str, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--hidden-window", action="store_true")
    parser.add_argument("--export-frames", type=str, default=None)
    return parser.parse_args()


def export_frames(world: ReplayWorld, output_dir: Path, hidden_window: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    renderer = PygameView(world, title="Library Escape Replay Export", hidden=hidden_window)
    for frame_index in range(len(world.frames)):
        world.reset_to_frame(frame_index)
        renderer.render_frame()
        pygame.image.save(renderer.canvas, str(output_dir / f"frame_{frame_index:05d}.png"))
    renderer.close()
    pygame.quit()


def play_replay(world: ReplayWorld, speed: float, hidden_window: bool) -> None:
    renderer = PygameView(world, title="Library Escape Replay Viewer", hidden=hidden_window)
    clock = pygame.time.Clock()
    paused = False
    frame_index = 0
    accumulator = 0.0
    seconds_per_frame = 1.0 / max(1, int(world.render_fps))
    running = True

    while running:
        dt = min(clock.tick(max(1, int(world.render_fps * max(speed, 0.1)))) / 1000.0, 0.25)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = not paused
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                frame_index = min(len(world.frames) - 1, frame_index + 1)
                world.reset_to_frame(frame_index)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                frame_index = max(0, frame_index - 1)
                world.reset_to_frame(frame_index)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                frame_index = 0
                world.reset_to_frame(frame_index)

        if not paused and frame_index < len(world.frames) - 1:
            accumulator += dt * max(speed, 0.1)
            while accumulator >= seconds_per_frame and frame_index < len(world.frames) - 1:
                accumulator -= seconds_per_frame
                frame_index += 1
                world.reset_to_frame(frame_index)

        renderer.render_frame()

    renderer.close()
    pygame.quit()


def main() -> None:
    args = parse_args()
    payload = load_replay(args.replay)
    world = ReplayWorld(payload)
    if args.export_frames:
        export_frames(world, resolve_repo_path(args.export_frames), hidden_window=True)
        print(f"Exported {len(world.frames)} frames to {resolve_repo_path(args.export_frames)}")
        return
    play_replay(world, speed=float(args.speed), hidden_window=args.hidden_window)


if __name__ == "__main__":
    main()
