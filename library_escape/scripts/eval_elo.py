from __future__ import annotations

import argparse
from pathlib import Path

from library_escape.play.ai_vs_ai import run_episode


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def outcome_score(outcome: str | None) -> tuple[float, float]:
    if outcome == "escaped":
        return 1.0, 0.0
    if outcome == "caught":
        return 0.0, 1.0
    return 0.5, 0.5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate Elo by repeating AI vs AI matches.")
    parser.add_argument("--player-model", type=Path, default=None)
    parser.add_argument("--enemy-model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--k-factor", type=float, default=24.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    player_rating = 1000.0
    enemy_rating = 1000.0
    outcomes: dict[str, int] = {"escaped": 0, "caught": 0, "timeout": 0, "other": 0}

    for _ in range(args.episodes):
        outcome = run_episode(player_model=args.player_model, enemy_model=args.enemy_model, render=False)
        player_score, enemy_score = outcome_score(outcome)
        player_expected = expected_score(player_rating, enemy_rating)
        enemy_expected = expected_score(enemy_rating, player_rating)
        player_rating += args.k_factor * (player_score - player_expected)
        enemy_rating += args.k_factor * (enemy_score - enemy_expected)
        if outcome in outcomes:
            outcomes[outcome] += 1
        else:
            outcomes["other"] += 1

    print(f"Player Elo: {player_rating:.1f}")
    print(f"Enemy Elo: {enemy_rating:.1f}")
    print(f"Outcomes: {outcomes}")


if __name__ == "__main__":
    main()
