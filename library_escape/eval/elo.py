"""Elo leaderboard utilities for player and enemy checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..agents.ppo_agent import SB3PolicyController
from ..agents.rule_based_enemy import RuleBasedEnemyController
from ..agents.rule_based_player import HeuristicPlayerController
from ..config import load_env_config
from ..core.world import World
from .registry import ModelCandidate, discover_saved_models


@dataclass(slots=True)
class LeaderboardEntry:
    key: str
    role: str
    label: str
    rating: float = 1500.0
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    win_rate: float = 0.0
    source: str = ""
    model_path: str | None = None
    algorithm: str = "baseline"

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "role": self.role,
            "label": self.label,
            "rating": round(self.rating, 2),
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "source": self.source,
            "model_path": self.model_path,
            "algorithm": self.algorithm,
        }


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + (10.0 ** ((rating_b - rating_a) / 400.0)))


def _update_elo(player_entry: LeaderboardEntry, enemy_entry: LeaderboardEntry, score: float, k_factor: float) -> None:
    expected_player = _expected_score(player_entry.rating, enemy_entry.rating)
    expected_enemy = 1.0 - expected_player
    player_entry.rating += k_factor * (score - expected_player)
    enemy_entry.rating += k_factor * ((1.0 - score) - expected_enemy)


def _make_player_controller(candidate: ModelCandidate | None, env_config: dict):
    if candidate is None:
        return HeuristicPlayerController()
    return SB3PolicyController("player", candidate.model_path, env_config=env_config)


def _make_enemy_controller(candidate: ModelCandidate | None, env_config: dict):
    if candidate is None:
        return RuleBasedEnemyController()
    return SB3PolicyController("enemy", candidate.model_path, env_config=env_config)


def play_match(
    player_candidate: ModelCandidate | None,
    enemy_candidate: ModelCandidate | None,
    episodes: int,
    seeds: list[int] | None = None,
) -> dict[str, float | int]:
    env_config = load_env_config()
    player_controller = _make_player_controller(player_candidate, env_config)
    enemy_controller = _make_enemy_controller(enemy_candidate, env_config)
    seeds = seeds or list(range(episodes))

    player_wins = 0
    enemy_wins = 0
    draws = 0
    for seed in seeds[:episodes]:
        world = World(env_config=env_config, seed=seed)
        if hasattr(player_controller, "reset"):
            player_controller.reset()
        if hasattr(enemy_controller, "reset"):
            enemy_controller.reset()
        while not (world.terminated or world.truncated):
            player_action = player_controller.act(world)
            enemy_action = enemy_controller.act(world)
            world.step(player_action=player_action, enemy_action=enemy_action, frame_skip=world.rl_frame_skip)

        if world.outcome == "escaped":
            player_wins += 1
        elif world.outcome == "caught":
            enemy_wins += 1
        else:
            draws += 1

    score = (player_wins + 0.5 * draws) / max(1, episodes)
    return {
        "player_wins": player_wins,
        "enemy_wins": enemy_wins,
        "draws": draws,
        "score": score,
    }


def build_leaderboard(
    root: Path | None = None,
    episodes: int = 8,
    max_players: int = 6,
    max_enemies: int = 6,
    k_factor: float = 24.0,
    progress: Callable[[str], None] | None = None,
) -> dict:
    discovered = discover_saved_models(root)
    player_candidates = [item for item in discovered if item.role == "player"][:max_players]
    enemy_candidates = [item for item in discovered if item.role == "enemy"][:max_enemies]

    player_entries: dict[str, LeaderboardEntry] = {
        "baseline_player": LeaderboardEntry(
            key="baseline_player",
            role="player",
            label="heuristic_player",
            source="baseline",
        )
    }
    enemy_entries: dict[str, LeaderboardEntry] = {
        "baseline_enemy": LeaderboardEntry(
            key="baseline_enemy",
            role="enemy",
            label="rule_enemy",
            source="baseline",
        )
    }

    for index, candidate in enumerate(player_candidates, start=1):
        player_entries[f"player_{index}"] = LeaderboardEntry(
            key=f"player_{index}",
            role="player",
            label=candidate.label,
            source=candidate.mode,
            model_path=str(candidate.model_path),
            algorithm=candidate.algorithm,
        )
    for index, candidate in enumerate(enemy_candidates, start=1):
        enemy_entries[f"enemy_{index}"] = LeaderboardEntry(
            key=f"enemy_{index}",
            role="enemy",
            label=candidate.label,
            source=candidate.mode,
            model_path=str(candidate.model_path),
            algorithm=candidate.algorithm,
        )

    player_sources: list[tuple[LeaderboardEntry, ModelCandidate | None]] = [(player_entries["baseline_player"], None)]
    player_sources.extend((player_entries[f"player_{idx}"], candidate) for idx, candidate in enumerate(player_candidates, start=1))
    enemy_sources: list[tuple[LeaderboardEntry, ModelCandidate | None]] = [(enemy_entries["baseline_enemy"], None)]
    enemy_sources.extend((enemy_entries[f"enemy_{idx}"], candidate) for idx, candidate in enumerate(enemy_candidates, start=1))

    match_rows: list[dict] = []
    total_matches = len(player_sources) * len(enemy_sources)
    match_counter = 0

    for player_entry, player_candidate in player_sources:
        for enemy_entry, enemy_candidate in enemy_sources:
            match_counter += 1
            if progress is not None:
                progress(f"[{match_counter}/{total_matches}] {player_entry.label} vs {enemy_entry.label}")
            result = play_match(player_candidate, enemy_candidate, episodes=episodes)
            score = float(result["score"])
            _update_elo(player_entry, enemy_entry, score=score, k_factor=k_factor)
            player_entry.games += episodes
            enemy_entry.games += episodes
            player_entry.wins += int(result["player_wins"])
            player_entry.draws += int(result["draws"])
            player_entry.losses += int(result["enemy_wins"])
            enemy_entry.wins += int(result["enemy_wins"])
            enemy_entry.draws += int(result["draws"])
            enemy_entry.losses += int(result["player_wins"])
            match_rows.append(
                {
                    "player": player_entry.label,
                    "enemy": enemy_entry.label,
                    "player_score": round(score, 4),
                    "player_wins": int(result["player_wins"]),
                    "enemy_wins": int(result["enemy_wins"]),
                    "draws": int(result["draws"]),
                }
            )

    for entry in list(player_entries.values()) + list(enemy_entries.values()):
        entry.win_rate = (entry.wins + 0.5 * entry.draws) / max(1, entry.games)

    player_table = sorted((entry.as_dict() for entry in player_entries.values()), key=lambda item: item["rating"], reverse=True)
    enemy_table = sorted((entry.as_dict() for entry in enemy_entries.values()), key=lambda item: item["rating"], reverse=True)

    return {
        "episodes_per_match": episodes,
        "k_factor": k_factor,
        "max_players": max_players,
        "max_enemies": max_enemies,
        "players": player_table,
        "enemies": enemy_table,
        "matches": match_rows,
    }


def save_leaderboard(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
