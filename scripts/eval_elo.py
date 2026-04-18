"""Evaluate checkpoints head-to-head and optionally build an Elo leaderboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from library_escape.eval.elo import build_leaderboard, play_match, save_leaderboard
from library_escape.eval.registry import ModelCandidate
from library_escape.game_modes import read_model_game_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Library Escape checkpoints and estimate Elo ratings.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--player-model", type=str, default=None)
    parser.add_argument("--enemy-model", type=str, default=None)
    parser.add_argument("--root", type=str, default=None, help="Checkpoint root to scan when building a leaderboard.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path for the leaderboard.")
    parser.add_argument("--max-players", type=int, default=6)
    parser.add_argument("--max-enemies", type=int, default=6)
    parser.add_argument("--k-factor", type=float, default=24.0)
    return parser.parse_args()


def _candidate(role: str, raw_path: str) -> ModelCandidate:
    path = Path(raw_path).resolve()
    return ModelCandidate(
        role=role,
        label=path.stem,
        model_path=path,
        run_dir=path.parent,
        summary_path=path.parent / "training_summary.json",
        mode="manual",
        algorithm="unknown",
        game_mode=read_model_game_mode(path) or "escape",
    )


def main() -> None:
    args = parse_args()
    if args.root:
        payload = build_leaderboard(
            root=Path(args.root).resolve(),
            episodes=int(args.episodes),
            max_players=int(args.max_players),
            max_enemies=int(args.max_enemies),
            k_factor=float(args.k_factor),
            progress=print,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.output:
            save_leaderboard(Path(args.output).resolve(), payload)
            print(f"saved={Path(args.output).resolve()}")
        return

    player_candidate = _candidate("player", args.player_model) if args.player_model else None
    enemy_candidate = _candidate("enemy", args.enemy_model) if args.enemy_model else None
    result = play_match(player_candidate, enemy_candidate, episodes=int(args.episodes))
    print(f"player_wins={result['player_wins']}")
    print(f"enemy_wins={result['enemy_wins']}")
    print(f"draws={result['draws']}")
    print(f"player_score={float(result['score']):.4f}")


if __name__ == "__main__":
    main()
