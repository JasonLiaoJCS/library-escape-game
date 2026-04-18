"""Discover saved checkpoints and expose them as structured candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import REPO_ROOT


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    role: str
    label: str
    model_path: Path
    run_dir: Path
    summary_path: Path
    mode: str
    algorithm: str


def _resolve_model_path(raw_path: str | None, summary_dir: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve() if not (summary_dir / path).exists() else (summary_dir / path).resolve()


def discover_saved_models(root: Path | None = None) -> list[ModelCandidate]:
    root = (root or (REPO_ROOT / "checkpoints")).resolve()
    candidates: list[ModelCandidate] = []
    if not root.exists():
        return candidates

    for summary_path in sorted(root.rglob("training_summary.json")):
        summary = summary_path.read_text(encoding="utf-8")
        try:
            import json

            payload = json.loads(summary)
        except Exception:
            continue

        mode = str(payload.get("mode", ""))
        if mode == "self_play_phase":
            continue

        algorithm = str(payload.get("algorithm", payload.get("train_config", {}).get("algorithm", "ppo")))
        summary_dir = summary_path.parent
        run_label = summary_dir.name

        if mode == "single_agent_enemy":
            model_path = _resolve_model_path(payload.get("final_model"), summary_dir)
            if model_path and model_path.exists():
                candidates.append(
                    ModelCandidate(
                        role="enemy",
                        label=run_label,
                        model_path=model_path,
                        run_dir=summary_dir,
                        summary_path=summary_path,
                        mode=mode,
                        algorithm=algorithm,
                    )
                )
        elif mode == "single_agent_player":
            model_path = _resolve_model_path(payload.get("final_model"), summary_dir)
            if model_path and model_path.exists():
                candidates.append(
                    ModelCandidate(
                        role="player",
                        label=run_label,
                        model_path=model_path,
                        run_dir=summary_dir,
                        summary_path=summary_path,
                        mode=mode,
                        algorithm=algorithm,
                    )
                )
        elif mode in {"self_play", "self_play_mappo_recipe"}:
            player_model = _resolve_model_path(payload.get("final_player_model"), summary_dir)
            enemy_model = _resolve_model_path(payload.get("final_enemy_model"), summary_dir)
            if player_model and player_model.exists():
                candidates.append(
                    ModelCandidate(
                        role="player",
                        label=f"{run_label}:player",
                        model_path=player_model,
                        run_dir=summary_dir,
                        summary_path=summary_path,
                        mode=mode,
                        algorithm=algorithm,
                    )
                )
            if enemy_model and enemy_model.exists():
                candidates.append(
                    ModelCandidate(
                        role="enemy",
                        label=f"{run_label}:enemy",
                        model_path=enemy_model,
                        run_dir=summary_dir,
                        summary_path=summary_path,
                        mode=mode,
                        algorithm=algorithm,
                    )
                )

    candidates.sort(key=lambda item: item.model_path.stat().st_mtime if item.model_path.exists() else 0.0, reverse=True)
    return candidates
