"""Tkinter launcher for playing, training, replay, and run analysis."""

from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from copy import deepcopy
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..config import REPO_ROOT, load_training_config, resolve_repo_path
from ..eval.registry import discover_saved_models
from ..game_modes import game_mode_label, infer_game_mode_from_models, normalize_game_mode, read_model_game_mode
from ..gui.tensorboard_data import inspect_scalar_data, load_scalar_series
from ..replay.io import load_replay
from ..train.callbacks import format_seconds
from ..train.common import deep_update, timestamped_run_name, write_model_metadata, write_training_summary

try:  # pragma: no cover - optional GUI plotting dependency
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - fallback if matplotlib unavailable
    FigureCanvasTkAgg = None
    Figure = None


_JSON_READ_RETRIES = 3
_JSON_READ_RETRY_DELAY_SECONDS = 0.05
APP_PALETTE = {
    "bg": "#0b1220",
    "surface": "#111b2e",
    "surface_alt": "#16243b",
    "surface_soft": "#1b2d48",
    "border": "#2a3b57",
    "text": "#e8eef8",
    "muted": "#9fb0c8",
    "accent": "#4aa3ff",
    "accent_active": "#71b8ff",
    "warm": "#ff9a62",
    "warm_active": "#ffb184",
    "success": "#7fd18a",
}
PREFERRED_UI_FONTS = ("Segoe UI Variable Text", "Segoe UI", "Arial")
PREFERRED_MONO_FONTS = ("Cascadia Mono", "Consolas", "Courier New")
LOG_POLL_INTERVAL_MS = 100


def enable_windows_high_dpi() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext") and user32.SetProcessDpiAwarenessContext(-4):
            return
    except Exception:
        pass
    try:
        shcore = ctypes.windll.shcore
        if hasattr(shcore, "SetProcessDpiAwareness"):
            shcore.SetProcessDpiAwareness(2)
            return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _pick_font(preferred: tuple[str, ...], available: set[str]) -> str:
    for family in preferred:
        if family in available:
            return family
    return preferred[-1]


def drain_log_queue(log_queue: queue.Queue[str], widget: tk.Text) -> int:
    lines: list[str] = []
    while True:
        try:
            lines.append(log_queue.get_nowait())
        except queue.Empty:
            break
    if not lines:
        return 0
    widget.insert(tk.END, "".join(lines))
    widget.see(tk.END)
    return len(lines)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    for attempt in range(_JSON_READ_RETRIES):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (PermissionError, OSError, json.JSONDecodeError):
            if attempt == _JSON_READ_RETRIES - 1:
                return {}
            time.sleep(_JSON_READ_RETRY_DELAY_SECONDS)
    return {}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    for attempt in range(_JSON_READ_RETRIES):
        rows: list[dict] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            return rows
        except FileNotFoundError:
            return []
        except (PermissionError, OSError, json.JSONDecodeError):
            if attempt == _JSON_READ_RETRIES - 1:
                return []
            time.sleep(_JSON_READ_RETRY_DELAY_SECONDS)
    return []


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name.lower())
    return int(match.group(1)) if match else -1


def _path_from_summary(value: object) -> Path | None:
    if value in (None, "", "None"):
        return None
    try:
        candidate = resolve_repo_path(str(value))
    except Exception:
        candidate = Path(str(value))
    return candidate if candidate.exists() else None


def find_latest_playable_model(search_root: Path, role: str) -> Path | None:
    if not search_root.exists():
        return None
    role_token = role.lower()
    candidates: list[Path] = []
    for path in search_root.rglob("*.zip"):
        name = path.name.lower()
        if "latest.zip" in name and role_token in name:
            candidates.append(path)
            continue
        if "_checkpoint_" in name and role_token in name and name.endswith("_steps.zip"):
            candidates.append(path)
            continue
        if name == "best_model.zip":
            candidates.append(path)
    if not candidates:
        return None
    preferred_candidates = [path for path in candidates if path.name.lower() != "best_model.zip"]
    if preferred_candidates:
        candidates = preferred_candidates
    candidates.sort(key=lambda item: (item.stat().st_mtime, "latest.zip" in item.name.lower(), _checkpoint_step(item)), reverse=True)
    return candidates[0]


def playable_model_artifacts(model_path: Path | None) -> set[Path]:
    if model_path is None or not model_path.exists():
        return set()
    artifacts = {model_path.resolve()}
    for candidate in (
        model_path.with_suffix(".meta.json"),
        model_path.with_suffix(".obsnorm.npz"),
        model_path.parent / "obsnorm_latest.npz",
        model_path.parent / "vecnormalize.pkl",
        model_path.parent.parent / "obsnorm_latest.npz",
        model_path.parent.parent / "vecnormalize.pkl",
    ):
        if candidate.exists():
            artifacts.add(candidate.resolve())
    return artifacts


def ensure_model_metadata(model_path: Path | None, *, role: str, algorithm: str, run_dir: Path, game_mode: str) -> None:
    if model_path is None or not model_path.exists():
        return
    metadata_path = model_path.with_suffix(".meta.json")
    if metadata_path.exists():
        return
    write_model_metadata(
        model_path,
        {
            "role": role,
            "algorithm": algorithm,
            "run_dir": run_dir,
            "game_mode": game_mode,
            "game_mode_label": game_mode_label(game_mode),
        },
    )


def build_interrupted_training_summary(run_dir: Path, context: dict[str, object], progress_payload: dict | None = None) -> dict[str, object]:
    mode = str(context.get("mode", "enemy"))
    game_mode = normalize_game_mode(str(context.get("game_mode", "escape")))
    algorithm = str(context.get("algorithm", "ppo"))
    progress_payload = progress_payload or read_json(run_dir / "progress.json")
    payload: dict[str, object] = {
        "algorithm": algorithm,
        "run_dir": run_dir,
        "progress_path": run_dir / "progress.json",
        "progress_history_path": run_dir / "progress_history.jsonl",
        "eval_history_path": run_dir / "eval_history.jsonl",
        "seed": context.get("seed"),
        "preset": context.get("preset"),
        "game_mode": game_mode,
        "game_mode_label": game_mode_label(game_mode),
        "status": "stopped_early",
        "completed": False,
        "stopped_by_user": True,
        "phase_name": progress_payload.get("phase_name"),
        "phase_progress": progress_payload.get("phase_progress"),
        "global_progress": progress_payload.get("global_progress"),
        "elapsed_seconds": progress_payload.get("elapsed_seconds"),
    }
    if isinstance(context.get("train_config"), dict):
        payload["train_config"] = context["train_config"]
    if mode == "selfplay":
        enemy_model = find_latest_playable_model(run_dir / "enemy", "enemy")
        player_model = find_latest_playable_model(run_dir / "player", "player")
        ensure_model_metadata(enemy_model, role="enemy", algorithm=algorithm, run_dir=run_dir, game_mode=game_mode)
        ensure_model_metadata(player_model, role="player", algorithm=algorithm, run_dir=run_dir, game_mode=game_mode)
        payload.update(
            {
                "mode": "self_play",
                "enemy_root": run_dir / "enemy",
                "player_root": run_dir / "player",
                "final_enemy_model": enemy_model,
                "final_player_model": player_model,
                "resume_from_run": context.get("resume_path"),
                "resume_enemy_model": enemy_model,
                "resume_player_model": player_model,
            }
        )
        return payload

    final_model = find_latest_playable_model(run_dir / "models", mode)
    ensure_model_metadata(final_model, role=mode, algorithm=algorithm, run_dir=run_dir, game_mode=game_mode)
    payload.update(
        {
            "mode": f"single_agent_{mode}",
            "final_model": final_model,
            "best_model": None,
            "resume_path": context.get("resume_path"),
        }
    )
    return payload


def write_interrupted_training_summary(run_dir: Path, context: dict[str, object], progress_payload: dict | None = None) -> dict[str, object]:
    payload = build_interrupted_training_summary(run_dir, context, progress_payload)
    write_training_summary(run_dir / "training_summary.json", payload)
    return payload


def compact_training_run(run_dir: Path) -> dict[str, int]:
    run_dir = run_dir.resolve()
    summary_path = run_dir / "training_summary.json"
    summary = read_json(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary not found: {summary_path}")

    keep_paths: set[Path] = {summary_path.resolve()}
    progress_path = run_dir / "progress.json"
    if progress_path.exists():
        keep_paths.add(progress_path.resolve())

    mode = str(summary.get("mode", ""))
    retained_models: list[Path] = []
    if mode == "self_play":
        enemy_model = _path_from_summary(summary.get("final_enemy_model")) or find_latest_playable_model(run_dir / "enemy", "enemy")
        player_model = _path_from_summary(summary.get("final_player_model")) or find_latest_playable_model(run_dir / "player", "player")
        if enemy_model is not None:
            summary["final_enemy_model"] = str(enemy_model)
            retained_models.append(enemy_model)
            keep_paths.update(playable_model_artifacts(enemy_model))
        if player_model is not None:
            summary["final_player_model"] = str(player_model)
            retained_models.append(player_model)
            keep_paths.update(playable_model_artifacts(player_model))
        summary["progress_history_path"] = None
        summary["eval_history_path"] = None
    else:
        role = "enemy" if "enemy" in mode else "player"
        final_model = _path_from_summary(summary.get("final_model")) or find_latest_playable_model(run_dir / "models", role)
        if final_model is not None:
            summary["final_model"] = str(final_model)
            retained_models.append(final_model)
            keep_paths.update(playable_model_artifacts(final_model))
        summary["best_model"] = None
        summary["progress_history_path"] = None
        summary["eval_history_path"] = None

    summary["artifacts_compacted"] = True
    summary["artifacts_compacted_at"] = time.time()
    summary["retained_models"] = [str(path) for path in retained_models]
    write_training_summary(summary_path, summary)

    deleted_files = 0
    deleted_bytes = 0
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in keep_paths:
            continue
        deleted_bytes += path.stat().st_size
        path.unlink()
        deleted_files += 1

    for directory in sorted((path for path in run_dir.rglob("*") if path.is_dir()), key=lambda item: len(item.parts), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            continue

    return {"deleted_files": deleted_files, "deleted_bytes": deleted_bytes, "kept_files": len([path for path in keep_paths if path.exists()])}


def open_path(path: Path) -> None:
    resolved = path.resolve()
    if sys.platform.startswith("win"):
        os.startfile(str(resolved))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(resolved)])
        return
    subprocess.Popen(["xdg-open", str(resolved)])


def python_executable() -> Path:
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return venv_python
    return Path(sys.executable)


def checkpoint_root_for_mode(mode: str, game_mode: str) -> Path:
    training_cfg = load_training_config()
    if mode == "enemy":
        return resolve_repo_path(training_cfg["single_agent"]["enemy_checkpoint_dir"]) / normalize_game_mode(game_mode)
    if mode == "player":
        return resolve_repo_path(training_cfg["single_agent"]["player_checkpoint_dir"]) / normalize_game_mode(game_mode)
    return resolve_repo_path(training_cfg["self_play"]["checkpoint_root_dir"]) / normalize_game_mode(game_mode)


def resolve_effective_train_config_for_gui(
    mode: str,
    preset: str | None,
    game_mode: str,
    overrides: dict[str, object] | None = None,
    *,
    n_envs_override: str | None = None,
    timesteps_override: str | None = None,
    rounds_override: str | None = None,
) -> dict[str, object]:
    training_cfg = load_training_config()
    section_key = "self_play" if mode == "selfplay" else "single_agent"
    train_cfg = deepcopy(training_cfg.get(section_key, {}))
    preset_cfg = training_cfg.get("presets", {}).get(preset or "", {})
    train_cfg = deep_update(train_cfg, preset_cfg.get(section_key, {}))
    train_cfg = deep_update(
        train_cfg,
        preset_cfg.get("train_overrides_by_game_mode", {}).get(normalize_game_mode(game_mode), {}),
    )
    if overrides and isinstance(overrides.get("train"), dict):
        train_cfg = deep_update(train_cfg, overrides["train"])
    if n_envs_override:
        train_cfg["n_envs"] = int(n_envs_override)
    if timesteps_override:
        key = "timesteps_per_round" if mode == "selfplay" else "total_timesteps"
        train_cfg[key] = int(timesteps_override)
    if mode == "selfplay" and rounds_override:
        train_cfg["rounds"] = int(rounds_override)
    return train_cfg


def expected_tensorboard_first_dump_step(
    *,
    summary: dict | None = None,
    context: dict[str, object] | None = None,
) -> int | None:
    train_cfg = None
    if summary and isinstance(summary.get("train_config"), dict):
        train_cfg = summary["train_config"]
    elif context and isinstance(context.get("train_config"), dict):
        train_cfg = context["train_config"]
    if not isinstance(train_cfg, dict):
        return None

    algorithm = str(
        (summary or {}).get("algorithm")
        or (context or {}).get("algorithm")
        or train_cfg.get("algorithm", "")
    ).lower()
    if "ppo" not in algorithm:
        return None
    try:
        n_envs = int(train_cfg.get("n_envs", 0))
        n_steps = int(train_cfg.get("n_steps", 0))
    except (TypeError, ValueError):
        return None
    if n_envs <= 0 or n_steps <= 0:
        return None
    return n_envs * n_steps


def tensorboard_wait_hint(
    run_dir: Path,
    *,
    summary: dict | None = None,
    progress_payload: dict | None = None,
    context: dict[str, object] | None = None,
) -> str | None:
    event_file_count, has_scalar_data = inspect_scalar_data(run_dir)
    if event_file_count <= 0:
        return "TensorBoard started, but this run has not written any event files yet."
    if has_scalar_data:
        return None

    first_dump_step = expected_tensorboard_first_dump_step(summary=summary, context=context)
    current_steps = 0
    if progress_payload:
        raw_steps = progress_payload.get("phase_timesteps", progress_payload.get("global_timesteps", 0))
        try:
            current_steps = int(raw_steps)
        except (TypeError, ValueError):
            current_steps = 0
    if first_dump_step is not None and current_steps < first_dump_step:
        remaining = first_dump_step - current_steps
        return (
            "TensorBoard found event files, but no scalar data is available yet. "
            "For PPO and MaskablePPO this is normal before the first rollout/update finishes. "
            f"Expect the first dashboard around step {first_dump_step:,} "
            f"(current {current_steps:,}, about {remaining:,} more)."
        )
    return (
        "TensorBoard found event files, but they do not contain scalar summaries yet. "
        "If training is still running, wait for the next rollout/update to finish and refresh the page."
    )


def _is_archived_checkpoint_path(path: Path) -> bool:
    return "_archive" in path.parts


def discover_training_runs(root: Path) -> list[dict]:
    runs: list[dict] = []
    if not root.exists():
        return runs
    for summary_path in sorted(root.rglob("training_summary.json")):
        if _is_archived_checkpoint_path(summary_path):
            continue
        payload = read_json(summary_path)
        if payload.get("mode") == "self_play_phase":
            continue
        runs.append({"path": summary_path.parent, "summary": payload})
    runs.sort(key=lambda item: str(item["path"]), reverse=True)
    return runs


class LauncherApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Library Escape Control Center")
        self._configure_window()
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self._configure_fonts()
        self._configure_theme()

        self.notebook = ttk.Notebook(self, style="App.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=18, pady=18)

        self.play_frame = PlayFrame(self.notebook)
        self.train_frame = TrainFrame(self.notebook)
        self.results_frame = ResultsFrame(self.notebook)
        self.tensorboard_frame = TensorBoardFrame(self.notebook)
        self.leaderboard_frame = LeaderboardFrame(self.notebook)
        self.replay_frame = ReplayFrame(self.notebook)
        self.config_frame = ConfigFrame(self.notebook)

        self.notebook.add(self.play_frame, text="Play")
        self.notebook.add(self.train_frame, text="Train")
        self.notebook.add(self.results_frame, text="Results")
        self.notebook.add(self.tensorboard_frame, text="TensorBoard")
        self.notebook.add(self.leaderboard_frame, text="Leaderboard")
        self.notebook.add(self.replay_frame, text="Replay")
        self.notebook.add(self.config_frame, text="Config")

    def _configure_window(self) -> None:
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        target_width = min(max(int(screen_width * 0.84), 1180), 1840)
        target_height = min(max(int(screen_height * 0.86), 820), 1280)
        pos_x = max((screen_width - target_width) // 2, 16)
        pos_y = max((screen_height - target_height) // 2, 16)
        self.geometry(f"{target_width}x{target_height}+{pos_x}+{pos_y}")
        self.minsize(1080, 720)
        self.configure(bg=APP_PALETTE["bg"])

    def _configure_fonts(self) -> None:
        available_fonts = set(tkfont.families(self))
        ui_family = _pick_font(PREFERRED_UI_FONTS, available_fonts)
        mono_family = _pick_font(PREFERRED_MONO_FONTS, available_fonts)
        dpi = self.winfo_fpixels("1i")
        base_size = 10 if dpi < 120 else 11
        small_size = max(base_size - 1, 9)
        heading_size = base_size + 1
        hero_size = base_size + 5

        tkfont.nametofont("TkDefaultFont").configure(family=ui_family, size=base_size)
        tkfont.nametofont("TkTextFont").configure(family=ui_family, size=base_size)
        tkfont.nametofont("TkMenuFont").configure(family=ui_family, size=base_size)
        tkfont.nametofont("TkHeadingFont").configure(family=ui_family, size=heading_size, weight="bold")
        tkfont.nametofont("TkFixedFont").configure(family=mono_family, size=max(base_size, 10))

        tkfont.Font(name="AppCaptionFont", exists=False, family=ui_family, size=small_size)
        tkfont.Font(name="AppSectionFont", exists=False, family=ui_family, size=heading_size, weight="bold")
        tkfont.Font(name="AppHeroFont", exists=False, family=ui_family, size=hero_size, weight="bold")
        tkfont.Font(name="AppTabFont", exists=False, family=ui_family, size=base_size, weight="bold")

    def _configure_theme(self) -> None:
        palette = APP_PALETTE
        self.style.configure(".", background=palette["bg"], foreground=palette["text"], fieldbackground=palette["surface"])
        self.style.configure("TFrame", background=palette["bg"])
        self.style.configure("Surface.TFrame", background=palette["surface"])
        self.style.configure("Card.TFrame", background=palette["surface_alt"])
        self.style.configure("TLabel", background=palette["bg"], foreground=palette["text"])
        self.style.configure("Subdued.TLabel", background=palette["surface_alt"], foreground=palette["muted"])
        self.style.configure("MetricLabel.TLabel", background=palette["surface_alt"], foreground=palette["muted"], font="AppCaptionFont")
        self.style.configure("MetricValue.TLabel", background=palette["surface_alt"], foreground=palette["text"], font="AppSectionFont")
        self.style.configure("Hero.TLabel", background=palette["surface_alt"], foreground="#f8fbff", font="AppHeroFont")
        self.style.configure("TLabelframe", background=palette["bg"], foreground="#f8fbff", padding=14)
        self.style.configure("TLabelframe.Label", background=palette["bg"], foreground="#f8fbff", font="AppSectionFont")
        self.style.configure("TButton", background=palette["accent"], foreground="#08111d", padding=(14, 10), font="AppTabFont")
        self.style.map(
            "TButton",
            background=[("active", palette["accent_active"]), ("pressed", palette["accent_active"])],
            foreground=[("disabled", palette["muted"])],
        )
        self.style.configure("Accent.TButton", background=palette["warm"], foreground="#141922", padding=(14, 10), font="AppTabFont")
        self.style.map("Accent.TButton", background=[("active", palette["warm_active"]), ("pressed", palette["warm_active"])])
        self.style.configure("TCheckbutton", background=palette["bg"], foreground=palette["text"])
        self.style.configure("TEntry", fieldbackground=palette["surface"], foreground=palette["text"], padding=(10, 8))
        self.style.configure(
            "TCombobox",
            fieldbackground=palette["surface"],
            background=palette["surface"],
            foreground=palette["text"],
            arrowsize=14,
            padding=(10, 8),
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", palette["surface"])],
            background=[("readonly", palette["surface"])],
            foreground=[("readonly", palette["text"])],
        )
        self.style.configure(
            "Treeview",
            background=palette["surface"],
            foreground=palette["text"],
            fieldbackground=palette["surface"],
            rowheight=30,
        )
        self.style.configure("Treeview.Heading", background=palette["surface_alt"], foreground="#f8fbff", font="AppTabFont")
        self.style.configure("Horizontal.TProgressbar", troughcolor=palette["surface"], background=palette["accent"], thickness=16)
        self.style.configure("App.TNotebook", background=palette["bg"], borderwidth=0)
        self.style.configure(
            "App.TNotebook.Tab",
            background=palette["surface"],
            foreground=palette["muted"],
            padding=(18, 10),
            font="AppTabFont",
        )
        self.style.map(
            "App.TNotebook.Tab",
            background=[("selected", palette["surface_alt"]), ("active", palette["surface_soft"])],
            foreground=[("selected", "#f8fbff"), ("active", "#f8fbff")],
        )
        self.style.configure("Vertical.TScrollbar", background=palette["surface_alt"], troughcolor=palette["bg"])
        self.style.configure("Horizontal.TScrollbar", background=palette["surface_alt"], troughcolor=palette["bg"])

        self.option_add("*Font", "TkDefaultFont")
        self.option_add("*Listbox.background", palette["surface"])
        self.option_add("*Listbox.foreground", palette["text"])
        self.option_add("*Listbox.selectBackground", palette["accent"])
        self.option_add("*Listbox.selectForeground", "#f8fbff")
        self.option_add("*Listbox.highlightThickness", 1)
        self.option_add("*Listbox.highlightBackground", palette["border"])
        self.option_add("*Listbox.relief", "flat")
        self.option_add("*Listbox.borderWidth", 0)
        self.option_add("*TCombobox*Listbox.background", palette["surface"])
        self.option_add("*TCombobox*Listbox.foreground", palette["text"])
        self.option_add("*TCombobox*Listbox.selectBackground", palette["accent"])
        self.option_add("*TCombobox*Listbox.selectForeground", "#f8fbff")


class ScrollableFrame(ttk.Frame):
    def __init__(self, master, *, background: str | None = None) -> None:
        super().__init__(master, style="TFrame")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._background = background or APP_PALETTE["bg"]
        self.canvas = tk.Canvas(
            self,
            bg=self._background,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")

        self.content = ttk.Frame(self.canvas, style="TFrame")
        self.content.columnconfigure(0, weight=1)
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._sync_content_width)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)
        self.content.bind("<Enter>", self._bind_mousewheel)
        self.content.bind("<Leave>", self._unbind_mousewheel)

    def _sync_scroll_region(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_content_width(self, event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _bind_mousewheel(self, _event=None) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event) -> None:
        if hasattr(event, "delta") and event.delta:
            step = -1 * int(event.delta / 120) if event.delta else 0
        elif getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = 0
        if step:
            self.canvas.yview_scroll(step, "units")


class BasePanel(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master, padding=12)

    def section(
        self,
        row: int,
        column: int,
        title: str,
        columnspan: int = 1,
        rowspan: int = 1,
        *,
        parent=None,
        padx: int = 8,
        pady: int = 8,
    ) -> ttk.LabelFrame:
        host = parent or self
        frame = ttk.LabelFrame(host, text=title, padding=14)
        frame.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, sticky="nsew", padx=padx, pady=pady)
        return frame

    def text_box(self, master, height: int = 12, wrap: str = "word", monospace: bool = False) -> tk.Text:
        widget = tk.Text(
            master,
            bg=APP_PALETTE["surface"],
            fg=APP_PALETTE["text"],
            insertbackground="#ffffff",
            wrap=wrap,
            font="TkFixedFont" if monospace else "TkTextFont",
            height=height,
            relief="flat",
            bd=0,
            padx=14,
            pady=12,
            spacing1=1,
            spacing3=2,
            highlightthickness=1,
            highlightbackground=APP_PALETTE["border"],
            highlightcolor=APP_PALETTE["accent"],
            selectbackground=APP_PALETTE["accent"],
            selectforeground="#f8fbff",
        )
        return widget

    def scrollable_frame(self, master, *, background: str | None = None) -> ScrollableFrame:
        return ScrollableFrame(master, background=background)


class PlayFrame(BasePanel):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        left = self.section(0, 0, "Launch Mode")
        right = self.section(0, 1, "Selected Files")
        status = self.section(1, 0, "Status", columnspan=2)

        self.mode_var = tk.StringVar(value="Human vs Rule Enemy")
        self.game_mode_var = tk.StringVar(value="collection")
        self.seed_var = tk.StringVar(value="")
        self.deterministic_policy_var = tk.BooleanVar(value=True)
        self.enemy_model_var = tk.StringVar()
        self.player_model_var = tk.StringVar()
        self.record_replay_var = tk.BooleanVar(value=False)
        self.replay_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready.")
        self.game_mode_hint_var = tk.StringVar(value="")

        ttk.Label(left, text="Mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            left,
            textvariable=self.mode_var,
            values=["Human vs Rule Enemy", "Human vs Enemy Checkpoint", "AI vs AI"],
            state="readonly",
            width=30,
        ).grid(row=1, column=0, sticky="ew", pady=(2, 10))

        ttk.Label(left, text="Game mode").grid(row=2, column=0, sticky="w")
        ttk.Combobox(
            left,
            textvariable=self.game_mode_var,
            values=["collection", "escape"],
            state="readonly",
            width=30,
        ).grid(row=3, column=0, sticky="ew", pady=(2, 4))
        ttk.Label(left, textvariable=self.game_mode_hint_var, justify="left", wraplength=360).grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(left, text="Seed").grid(row=5, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.seed_var, width=12).grid(row=6, column=0, sticky="w", pady=(2, 10))
        ttk.Checkbutton(left, text="Deterministic checkpoint playback", variable=self.deterministic_policy_var).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        ttk.Checkbutton(left, text="Record replay", variable=self.record_replay_var).grid(row=8, column=0, sticky="w")
        ttk.Entry(left, textvariable=self.replay_path_var).grid(row=9, column=0, sticky="ew", pady=(6, 6))
        ttk.Button(left, text="Replay Path", command=self.browse_replay_path).grid(row=9, column=1, padx=(8, 0))

        button_row = ttk.Frame(left)
        button_row.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(button_row, text="Start Game", style="Accent.TButton", command=self.launch).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Open Replays", command=lambda: open_path(REPO_ROOT / "replays")).pack(side="left", padx=(0, 8))
        ttk.Button(button_row, text="Open Checkpoints", command=lambda: open_path(REPO_ROOT / "checkpoints")).pack(side="left")
        left.columnconfigure(0, weight=1)

        ttk.Label(right, text="Enemy checkpoint").grid(row=0, column=0, sticky="w")
        ttk.Entry(right, textvariable=self.enemy_model_var).grid(row=1, column=0, sticky="ew", pady=(2, 6))
        ttk.Button(right, text="Browse Enemy", command=lambda: self.browse_model(self.enemy_model_var)).grid(row=1, column=1, padx=(8, 0))

        ttk.Label(right, text="Player checkpoint").grid(row=2, column=0, sticky="w")
        ttk.Entry(right, textvariable=self.player_model_var).grid(row=3, column=0, sticky="ew", pady=(2, 6))
        ttk.Button(right, text="Browse Player", command=lambda: self.browse_model(self.player_model_var)).grid(row=3, column=1, padx=(8, 0))

        tips = (
            "Human vs Rule Enemy: no checkpoint required.\n"
            "Human vs Enemy Checkpoint: choose an enemy .zip model.\n"
            "AI vs AI: choose either or both checkpoints. Empty fields fall back to rule-based agents.\n"
            "Seed left blank = random reset each launch. Fill a seed only when you want reproducible playback.\n"
            "Deterministic playback is ON by default for stable argmax-style playback. Turn it off only when you want policy variety.\n"
            "Collection = score-focused stealth mode.\n"
            "Escape = collect required notes, then break out through the exit.\n"
            "Checkpoint playback should usually use the same Game mode the model was trained in.\n"
            "If replay recording is enabled, the session is saved as a compressed .ler.gz file."
        )
        ttk.Label(right, text=tips, justify="left").grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 8))
        right.columnconfigure(0, weight=1)

        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.game_mode_var.trace_add("write", lambda *_: self._sync_game_mode_hint())
        self._sync_game_mode_hint()

    def browse_model(self, variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(
            title="Select checkpoint",
            filetypes=[("ZIP checkpoint", "*.zip"), ("All files", "*.*")],
            initialdir=str(REPO_ROOT / "checkpoints"),
        )
        if selected:
            variable.set(selected)

    def browse_replay_path(self) -> None:
        default_dir = REPO_ROOT / "replays"
        default_dir.mkdir(parents=True, exist_ok=True)
        suggested = default_dir / f"{timestamped_run_name('session')}.ler.gz"
        selected = filedialog.asksaveasfilename(
            title="Replay output",
            defaultextension=".ler.gz",
            initialdir=str(default_dir),
            initialfile=suggested.name,
            filetypes=[("Library Escape Replay", "*.ler.gz"), ("All files", "*.*")],
        )
        if selected:
            self.replay_path_var.set(selected)

    def launch(self) -> None:
        cmd = [str(python_executable())]
        mode = self.mode_var.get()
        selected_mode = normalize_game_mode(self.game_mode_var.get())
        seed = self.seed_var.get().strip()
        if mode == "AI vs AI":
            cmd += ["-m", "library_escape.play.ai_vs_ai"]
            if self.player_model_var.get().strip():
                cmd += ["--player-model", self.player_model_var.get().strip()]
            if self.enemy_model_var.get().strip():
                cmd += ["--enemy-model", self.enemy_model_var.get().strip()]
        else:
            cmd += ["-m", "library_escape.play.human_vs_ai"]
            if mode == "Human vs Enemy Checkpoint":
                if not self.enemy_model_var.get().strip():
                    messagebox.showwarning("Checkpoint required", "Please choose an enemy checkpoint first.")
                    return
                cmd += ["--enemy-model", self.enemy_model_var.get().strip()]

        if seed:
            cmd += ["--seed", seed]
        if self.deterministic_policy_var.get():
            cmd += ["--deterministic-policy"]

        if not self._confirm_model_game_mode(mode, selected_mode):
            return
        cmd += ["--game-mode", selected_mode]

        if self.record_replay_var.get():
            replay_path = self.replay_path_var.get().strip()
            if not replay_path:
                replay_dir = REPO_ROOT / "replays"
                replay_dir.mkdir(parents=True, exist_ok=True)
                replay_path = str(replay_dir / f"{timestamped_run_name('session')}.ler.gz")
                self.replay_path_var.set(replay_path)
            cmd += ["--record-replay", replay_path]

        subprocess.Popen(cmd, cwd=str(REPO_ROOT))
        self.status_var.set(f"Launched {game_mode_label(selected_mode)} mode: " + " ".join(cmd[1:]))

    def _sync_game_mode_hint(self) -> None:
        mode = normalize_game_mode(self.game_mode_var.get())
        if mode == "collection":
            self.game_mode_hint_var.set(
                "Collection mode: score-focused stealth. The player tries to collect as many notes/exams as possible "
                "before time runs out, while enemies try to spot the player and drain time."
            )
            return
        self.game_mode_hint_var.set(
            "Escape mode: objective-focused pursuit. The player must collect the required notes first, then reach the "
            "exit. Trained checkpoints should usually be played back in the same Game mode they were trained for."
        )

    def _confirm_model_game_mode(self, launch_mode: str, selected_mode: str) -> bool:
        model_paths: list[str] = []
        if launch_mode in {"Human vs Enemy Checkpoint", "AI vs AI"} and self.enemy_model_var.get().strip():
            model_paths.append(self.enemy_model_var.get().strip())
        if launch_mode == "AI vs AI" and self.player_model_var.get().strip():
            model_paths.append(self.player_model_var.get().strip())
        mismatches: list[str] = []
        for model_path in model_paths:
            model_mode = read_model_game_mode(model_path)
            if model_mode and model_mode != selected_mode:
                mismatches.append(f"{Path(model_path).name}: trained for {game_mode_label(model_mode)}")
        if not mismatches:
            return True
        return bool(
            messagebox.askyesno(
                "Mode mismatch",
                "Selected checkpoints were trained for a different game mode:\n\n"
                + "\n".join(mismatches)
                + f"\n\nYou are trying to launch {game_mode_label(selected_mode)} mode. Continue anyway?",
            )
        )


class TrainFrame(BasePanel):
    SINGLE_ALGOS = ("ppo", "maskable_ppo")
    SELFPLAY_ALGOS = ("league_ppo", "league_maskable_ppo", "mappo_recipe")

    def __init__(self, master) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.main_split = ttk.Panedwindow(self, orient="horizontal")
        self.main_split.grid(row=0, column=0, sticky="nsew")

        sidebar = ttk.Frame(self.main_split, style="TFrame")
        sidebar.configure(width=430)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(0, weight=1)

        workspace = ttk.Frame(self.main_split, style="TFrame")
        workspace.columnconfigure(0, weight=1)
        workspace.rowconfigure(0, weight=1)

        self.main_split.add(sidebar, weight=3)
        self.main_split.add(workspace, weight=7)

        self.sidebar_tabs = ttk.Notebook(sidebar, style="App.TNotebook")
        self.sidebar_tabs.grid(row=0, column=0, sticky="nsew")

        self.setup_scroll = self.scrollable_frame(self.sidebar_tabs)
        self.curriculum_scroll = self.scrollable_frame(self.sidebar_tabs)
        self.sidebar_tabs.add(self.setup_scroll, text="Setup")
        self.sidebar_tabs.add(self.curriculum_scroll, text="Curriculum")

        controls = self.section(0, 0, "Training Setup", parent=self.setup_scroll.content, padx=0, pady=0)
        advanced = self.section(0, 0, "Algorithms And Curriculum", parent=self.curriculum_scroll.content, padx=0, pady=0)

        self.workspace_split = ttk.Panedwindow(workspace, orient="vertical")
        self.workspace_split.grid(row=0, column=0, sticky="nsew")

        monitor_host = ttk.Frame(self.workspace_split, style="TFrame")
        monitor_host.configure(height=280)
        monitor_host.columnconfigure(0, weight=1)
        monitor_host.rowconfigure(0, weight=1)

        logs_host = ttk.Frame(self.workspace_split, style="TFrame")
        logs_host.configure(height=420)
        logs_host.columnconfigure(0, weight=1)
        logs_host.rowconfigure(0, weight=1)

        self.workspace_split.add(monitor_host, weight=2)
        self.workspace_split.add(logs_host, weight=5)

        monitor = self.section(0, 0, "Training Monitor", parent=monitor_host, padx=0, pady=0)
        logs = self.section(0, 0, "Training Logs", parent=logs_host, padx=0, pady=0)
        logs.rowconfigure(0, weight=1)
        logs.columnconfigure(0, weight=1)

        training_cfg = load_training_config()
        single_agent_cfg = training_cfg.get("single_agent", {})
        self_play_cfg = training_cfg.get("self_play", {})
        single_curriculum_cfg = single_agent_cfg.get("opponent_curriculum", {})
        self_play_curriculum_cfg = self_play_cfg.get("opponent_curriculum", {})
        presets = sorted(training_cfg.get("presets", {}).keys())
        self.mode_var = tk.StringVar(value="enemy")
        self.game_mode_var = tk.StringVar(value="escape")
        self.preset_var = tk.StringVar(value=presets[1] if len(presets) > 1 else (presets[0] if presets else "balanced"))
        self.run_name_var = tk.StringVar()
        self.timesteps_var = tk.StringVar()
        self.rounds_var = tk.StringVar()
        self.n_envs_var = tk.StringVar()
        self.seed_var = tk.StringVar(value="7")
        self.device_var = tk.StringVar(value="cuda")
        self.resume_path_var = tk.StringVar()
        self.resume_label_var = tk.StringVar(value="Resume checkpoint (optional)")
        self.default_single_algorithm = str(single_agent_cfg.get("algorithm", "ppo"))
        self.default_selfplay_algorithm = str(self_play_cfg.get("algorithm", "league_ppo"))
        self.algorithm_var = tk.StringVar(value=self.default_single_algorithm)
        self.use_history_var = tk.BooleanVar(value=bool(single_curriculum_cfg.get("use_history_pool", True)))
        self.sa_random_weight_var = tk.StringVar(value=str(single_curriculum_cfg.get("random_weight", 0.20)))
        self.sa_heuristic_weight_var = tk.StringVar(value=str(single_curriculum_cfg.get("heuristic_weight", 0.60)))
        self.sa_history_weight_var = tk.StringVar(value=str(single_curriculum_cfg.get("history_weight", 0.20)))
        self.sa_history_max_var = tk.StringVar(value=str(single_curriculum_cfg.get("max_history_pool", 8)))
        self.sp_bootstrap_random_var = tk.StringVar(value=str(self_play_curriculum_cfg.get("bootstrap_random_weight", 0.20)))
        self.sp_bootstrap_heuristic_var = tk.StringVar(value=str(self_play_curriculum_cfg.get("bootstrap_heuristic_weight", 0.80)))
        self.sp_latest_weight_var = tk.StringVar(value=str(self_play_curriculum_cfg.get("latest_weight", 0.50)))
        self.sp_historical_weight_var = tk.StringVar(value=str(self_play_curriculum_cfg.get("historical_weight", 0.50)))
        self.sp_history_max_var = tk.StringVar(value=str(self_play_cfg.get("opponent_pool_size", 6)))
        self.mappo_command_var = tk.StringVar(value=str(self_play_curriculum_cfg.get("mappo_recipe", {}).get("external_command", "")))
        self.mappo_notes_var = tk.StringVar(value=str(self_play_curriculum_cfg.get("mappo_recipe", {}).get("notes", "")))

        self.run_dir_var = tk.StringVar(value="No active run")
        self.phase_var = tk.StringVar(value="Idle")
        self.progress_var = tk.StringVar(value="0%")
        self.elapsed_var = tk.StringVar(value="00:00")
        self.eta_var = tk.StringVar(value="--:--")
        self.fps_var = tk.StringVar(value="0")
        self.reward_var = tk.StringVar(value="n/a")
        self.game_mode_hint_var = tk.StringVar(value="")

        ttk.Label(controls, text="Mode").grid(row=0, column=0, sticky="w")
        self.mode_combo = ttk.Combobox(controls, textvariable=self.mode_var, values=["enemy", "player", "selfplay"], state="readonly", width=18)
        self.mode_combo.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(controls, text="Game mode").grid(row=2, column=0, sticky="w")
        ttk.Combobox(controls, textvariable=self.game_mode_var, values=["collection", "escape"], state="readonly", width=18).grid(row=3, column=0, sticky="ew", pady=(2, 4))
        ttk.Label(controls, textvariable=self.game_mode_hint_var, justify="left", wraplength=320).grid(row=4, column=0, sticky="w", pady=(0, 8))
        ttk.Label(controls, text="Preset").grid(row=5, column=0, sticky="w")
        ttk.Combobox(controls, textvariable=self.preset_var, values=presets, state="readonly", width=18).grid(row=6, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(controls, text="Run name").grid(row=7, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.run_name_var, width=22).grid(row=8, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(controls, text="Timesteps / round").grid(row=9, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.timesteps_var, width=22).grid(row=10, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(controls, text="Rounds (selfplay)").grid(row=11, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.rounds_var, width=22).grid(row=12, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(controls, text="Vector envs").grid(row=13, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.n_envs_var, width=22).grid(row=14, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(controls, text="Seed").grid(row=15, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.seed_var, width=22).grid(row=16, column=0, sticky="ew", pady=(2, 8))
        ttk.Label(controls, text="Device").grid(row=17, column=0, sticky="w")
        ttk.Combobox(controls, textvariable=self.device_var, values=["auto", "cpu", "cuda"], state="readonly", width=18).grid(row=18, column=0, sticky="ew", pady=(2, 10))
        ttk.Label(controls, textvariable=self.resume_label_var).grid(row=19, column=0, sticky="w")
        resume_row = ttk.Frame(controls)
        resume_row.grid(row=20, column=0, sticky="ew", pady=(2, 10))
        resume_row.columnconfigure(0, weight=1)
        ttk.Entry(resume_row, textvariable=self.resume_path_var, width=22).grid(row=0, column=0, sticky="ew")
        ttk.Button(resume_row, text="Browse", command=self.browse_resume_path).grid(row=0, column=1, padx=(8, 0))

        button_bar = ttk.Frame(controls)
        button_bar.grid(row=21, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(button_bar, text="Start Training", style="Accent.TButton", command=self.start_training).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Stop", command=self.stop_training).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Compact Run", command=self.compact_current_run).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Open Run Folder", command=self.open_run_folder).pack(side="left")
        ttk.Button(controls, text="Open TensorBoard Server", command=self.open_tensorboard).grid(row=22, column=0, sticky="ew", pady=(10, 0))
        controls.columnconfigure(0, weight=1)

        ttk.Label(advanced, text="Algorithm").grid(row=0, column=0, sticky="w")
        self.algorithm_combo = ttk.Combobox(advanced, textvariable=self.algorithm_var, values=list(self.SINGLE_ALGOS), state="readonly", width=22)
        self.algorithm_combo.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        ttk.Checkbutton(advanced, text="Use history pool (single-agent)", variable=self.use_history_var).grid(row=2, column=0, sticky="w", pady=(0, 6))
        ttk.Label(advanced, text="Single-agent random / heuristic / history").grid(row=3, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.sa_random_weight_var).grid(row=4, column=0, sticky="ew", pady=(2, 4))
        ttk.Entry(advanced, textvariable=self.sa_heuristic_weight_var).grid(row=5, column=0, sticky="ew", pady=(0, 4))
        ttk.Entry(advanced, textvariable=self.sa_history_weight_var).grid(row=6, column=0, sticky="ew", pady=(0, 4))
        ttk.Entry(advanced, textvariable=self.sa_history_max_var).grid(row=7, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(advanced, text="Self-play bootstrap random / heuristic").grid(row=8, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.sp_bootstrap_random_var).grid(row=9, column=0, sticky="ew", pady=(2, 4))
        ttk.Entry(advanced, textvariable=self.sp_bootstrap_heuristic_var).grid(row=10, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(advanced, text="Self-play latest / historical").grid(row=11, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.sp_latest_weight_var).grid(row=12, column=0, sticky="ew", pady=(2, 4))
        ttk.Entry(advanced, textvariable=self.sp_historical_weight_var).grid(row=13, column=0, sticky="ew", pady=(0, 4))
        ttk.Entry(advanced, textvariable=self.sp_history_max_var).grid(row=14, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(advanced, text="MAPPO recipe command").grid(row=15, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.mappo_command_var).grid(row=16, column=0, sticky="ew", pady=(2, 6))
        ttk.Label(advanced, text="MAPPO notes").grid(row=17, column=0, sticky="w")
        ttk.Entry(advanced, textvariable=self.mappo_notes_var).grid(row=18, column=0, sticky="ew", pady=(2, 0))
        advanced.columnconfigure(0, weight=1)

        monitor.columnconfigure(0, weight=1)
        hero = ttk.Frame(monitor, style="Card.TFrame", padding=18)
        hero.grid(row=0, column=0, sticky="ew")
        hero.columnconfigure(0, weight=1)

        ttk.Label(hero, text="Active phase", style="MetricLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(hero, textvariable=self.phase_var, style="Hero.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 4))
        ttk.Label(
            hero,
            textvariable=self.run_dir_var,
            style="Subdued.TLabel",
            justify="left",
            wraplength=960,
        ).grid(row=2, column=0, sticky="ew", pady=(0, 12))
        self.progress_bar = ttk.Progressbar(hero, mode="determinate", maximum=100)
        self.progress_bar.grid(row=3, column=0, sticky="ew")

        metrics = ttk.Frame(monitor, style="TFrame")
        metrics.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        for column in range(3):
            metrics.columnconfigure(column, weight=1)

        metric_pairs = [
            ("Progress", self.progress_var),
            ("Elapsed", self.elapsed_var),
            ("ETA", self.eta_var),
            ("FPS", self.fps_var),
            ("Mean reward", self.reward_var),
        ]
        for idx, (label, variable) in enumerate(metric_pairs):
            self._metric_card(metrics, idx // 3, idx % 3, label, variable)

        self.log_text = self.text_box(logs, height=20, wrap="none", monospace=True)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        v_scrollbar = ttk.Scrollbar(logs, orient="vertical", command=self.log_text.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar = ttk.Scrollbar(logs, orient="horizontal", command=self.log_text.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.log_text.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.current_run_dir: Path | None = None
        self.progress_path: Path | None = None
        self.summary_path: Path | None = None
        self.current_training_context: dict[str, object] = {}
        self.stop_requested = False

        self.mode_var.trace_add("write", lambda *_: self._sync_mode_fields())
        self.game_mode_var.trace_add("write", lambda *_: self._sync_game_mode_hint())
        self._sync_mode_fields()
        self._sync_game_mode_hint()
        self.after_idle(self._set_default_split_positions)

    def _metric_card(self, master, row: int, column: int, label: str, variable: tk.StringVar) -> None:
        card = ttk.Frame(master, style="Card.TFrame", padding=14)
        card.grid(row=row, column=column, sticky="nsew", padx=6, pady=6)
        ttk.Label(card, text=label, style="MetricLabel.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(card, textvariable=variable, style="MetricValue.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))

    def _set_default_split_positions(self) -> None:
        try:
            total_width = max(self.winfo_width(), self.winfo_reqwidth())
            sidebar_width = min(max(int(total_width * 0.34), 360), 520)
            self.main_split.sashpos(0, sidebar_width)
            total_height = max(self.winfo_height(), self.winfo_reqheight())
            monitor_height = min(max(int(total_height * 0.32), 220), 340)
            self.workspace_split.sashpos(0, monitor_height)
        except tk.TclError:
            return

    def _sync_mode_fields(self) -> None:
        mode = self.mode_var.get()
        if mode == "selfplay":
            values = list(self.SELFPLAY_ALGOS)
            if self.algorithm_var.get() not in values:
                self.algorithm_var.set(self.default_selfplay_algorithm if self.default_selfplay_algorithm in values else values[0])
            self.resume_label_var.set("Resume self-play run (optional)")
        else:
            values = list(self.SINGLE_ALGOS)
            if self.algorithm_var.get() not in values:
                self.algorithm_var.set(self.default_single_algorithm if self.default_single_algorithm in values else values[0])
            self.resume_label_var.set("Resume checkpoint (optional)")
        self.algorithm_combo.configure(values=values)

    def browse_resume_path(self) -> None:
        mode = self.mode_var.get()
        if mode == "selfplay":
            initial_dir = REPO_ROOT / "checkpoints" / "selfplay"
            selected = filedialog.askdirectory(title="Select self-play run folder", initialdir=str(initial_dir))
        else:
            initial_dir = REPO_ROOT / "checkpoints"
            selected = filedialog.askopenfilename(
                title="Select checkpoint",
                filetypes=[("ZIP checkpoint", "*.zip"), ("All files", "*.*")],
                initialdir=str(initial_dir),
            )
        if selected:
            self.resume_path_var.set(selected)

    def _build_overrides_payload(self) -> dict:
        mode = self.mode_var.get()
        if mode == "selfplay":
            return {
                "train": {
                    "algorithm": self.algorithm_var.get(),
                    "opponent_pool_size": int(self.sp_history_max_var.get().strip() or "6"),
                    "opponent_curriculum": {
                        "bootstrap_random_weight": float(self.sp_bootstrap_random_var.get().strip() or "0.2"),
                        "bootstrap_heuristic_weight": float(self.sp_bootstrap_heuristic_var.get().strip() or "0.8"),
                        "latest_weight": float(self.sp_latest_weight_var.get().strip() or "0.5"),
                        "historical_weight": float(self.sp_historical_weight_var.get().strip() or "0.5"),
                        "max_history_pool": int(self.sp_history_max_var.get().strip() or "6"),
                        "mappo_recipe": {
                            "enabled": self.algorithm_var.get() == "mappo_recipe",
                            "external_command": self.mappo_command_var.get().strip(),
                            "notes": self.mappo_notes_var.get().strip(),
                        },
                    },
                }
            }
        return {
            "train": {
                "algorithm": self.algorithm_var.get(),
                "opponent_curriculum": {
                    "use_history_pool": bool(self.use_history_var.get()),
                    "random_weight": float(self.sa_random_weight_var.get().strip() or "0.2"),
                    "heuristic_weight": float(self.sa_heuristic_weight_var.get().strip() or "0.6"),
                    "history_weight": float(self.sa_history_weight_var.get().strip() or "0.2"),
                    "max_history_pool": int(self.sa_history_max_var.get().strip() or "8"),
                },
            }
        }

    def _build_train_command(self) -> list[str]:
        mode = self.mode_var.get()
        game_mode = normalize_game_mode(self.game_mode_var.get())
        seed = self.seed_var.get().strip() or "7"
        run_name = self.run_name_var.get().strip() or timestamped_run_name(mode)
        self.run_name_var.set(run_name)

        if mode == "selfplay":
            module = "library_escape.train.train_selfplay"
        elif mode == "player":
            module = "library_escape.train.train_player"
        else:
            module = "library_escape.train.train_enemy"

        cmd = [
            str(python_executable()),
            "-u",
            "-m",
            module,
            "--preset",
            self.preset_var.get(),
            "--game-mode",
            game_mode,
            "--seed",
            seed,
            "--run-name",
            run_name,
        ]
        if self.timesteps_var.get().strip():
            flag = "--timesteps-per-round" if mode == "selfplay" else "--timesteps"
            cmd += [flag, self.timesteps_var.get().strip()]
        if self.rounds_var.get().strip() and mode == "selfplay":
            cmd += ["--rounds", self.rounds_var.get().strip()]
        if self.n_envs_var.get().strip():
            cmd += ["--n-envs", self.n_envs_var.get().strip()]
        if self.device_var.get().strip():
            cmd += ["--device", self.device_var.get().strip()]
        if self.resume_path_var.get().strip():
            if mode == "selfplay":
                cmd += ["--resume-run", self.resume_path_var.get().strip()]
            else:
                cmd += ["--resume", self.resume_path_var.get().strip()]

        overrides = self._build_overrides_payload()
        cmd += ["--overrides-json", json.dumps(overrides, separators=(",", ":"), ensure_ascii=True)]
        effective_train_cfg = resolve_effective_train_config_for_gui(
            mode,
            self.preset_var.get(),
            game_mode,
            overrides,
            n_envs_override=self.n_envs_var.get().strip() or None,
            timesteps_override=self.timesteps_var.get().strip() or None,
            rounds_override=self.rounds_var.get().strip() or None,
        )

        self.current_run_dir = checkpoint_root_for_mode(mode, game_mode) / run_name
        self.progress_path = self.current_run_dir / "progress.json"
        self.summary_path = self.current_run_dir / "training_summary.json"
        self.run_dir_var.set(str(self.current_run_dir))
        self.current_training_context = {
            "mode": mode,
            "game_mode": game_mode,
            "algorithm": self.algorithm_var.get(),
            "preset": self.preset_var.get(),
            "seed": seed,
            "resume_path": self.resume_path_var.get().strip() or None,
            "train_config": effective_train_cfg,
        }
        return cmd

    def _sync_game_mode_hint(self) -> None:
        mode = normalize_game_mode(self.game_mode_var.get())
        if mode == "collection":
            self.game_mode_hint_var.set(
                "Collection training: optimize score under time pressure. Use this when you want agents to learn "
                "stealthy collection, denial, pressure, and score suppression rather than literal escape."
            )
            return
        self.game_mode_hint_var.set(
            "Escape training: optimize win/lose objective completion. Use this when you want agents to learn "
            "route planning, pursuit, denial, guarding the exit, and coordinated catch-versus-escape behavior."
        )

    def start_training(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("Training active", "A training job is already running.")
            return

        cmd = self._build_train_command()
        self.stop_requested = False
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "Launching:\n" + " ".join(cmd) + "\n\n")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        threading.Thread(target=self._pump_logs, daemon=True).start()
        self.after(LOG_POLL_INTERVAL_MS, self.poll_training)

    def _pump_logs(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(line)

    def poll_training(self) -> None:
        drain_log_queue(self.log_queue, self.log_text)

        if self.progress_path and self.progress_path.exists():
            payload = read_json(self.progress_path)
            if payload:
                progress = float(payload.get("global_progress", 0.0))
                self.progress_bar["value"] = progress * 100.0
                self.phase_var.set(str(payload.get("phase_name", "Idle")))
                self.progress_var.set(f"{progress * 100.0:5.1f}%")
                self.elapsed_var.set(format_seconds(float(payload.get("elapsed_seconds", 0.0))))
                self.eta_var.set(str(payload.get("eta_hms", "--:--")))
                self.fps_var.set(f"{float(payload.get('fps', 0.0)):.0f}")
                mean_reward = payload.get("mean_episode_reward")
                self.reward_var.set("n/a" if mean_reward is None else f"{float(mean_reward):.2f}")

        if self.process is None:
            return
        if self.process.poll() is None:
            self.after(LOG_POLL_INTERVAL_MS, self.poll_training)
        else:
            self.after(LOG_POLL_INTERVAL_MS, self.poll_training)
            if self.stop_requested and self.current_run_dir is not None and self.summary_path is not None and not self.summary_path.exists():
                payload = write_interrupted_training_summary(
                    self.current_run_dir,
                    self.current_training_context,
                    read_json(self.progress_path) if self.progress_path is not None else None,
                )
                self.log_text.insert(
                    tk.END,
                    "\nWrote interrupted training summary for Results view: "
                    + str(self.current_run_dir / "training_summary.json")
                    + "\n",
                )
                if payload.get("final_model") is None and payload.get("final_enemy_model") is None:
                    self.log_text.insert(tk.END, "No saved checkpoint was found yet; the run will appear in Results without a playable model.\n")
            self.log_text.insert(tk.END, f"\nProcess exited with code {self.process.returncode}\n")
            self.log_text.see(tk.END)
            self.stop_requested = False
            self.process = None

    def stop_training(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.stop_requested = True
        self.process.terminate()
        self.log_text.insert(tk.END, "\nRequested training stop.\n")
        self.log_text.see(tk.END)

    def compact_current_run(self) -> None:
        if self.current_run_dir is None:
            messagebox.showinfo("No run", "Start, stop, or select a training run first.")
            return
        if not (self.current_run_dir / "training_summary.json").exists():
            messagebox.showwarning("Summary missing", "This run does not have a training_summary.json yet.")
            return
        if not messagebox.askyesno(
            "Compact run",
            "Keep only the latest playable checkpoint and minimal metadata for this run?\n\n"
            "This will delete extra checkpoints, TensorBoard logs, monitor CSVs, and other large artifacts.",
        ):
            return
        result = compact_training_run(self.current_run_dir)
        freed_mb = result["deleted_bytes"] / (1024 * 1024)
        self.log_text.insert(
            tk.END,
            f"\nCompacted run {self.current_run_dir.name}: deleted {result['deleted_files']} files, freed {freed_mb:.1f} MB.\n",
        )
        self.log_text.see(tk.END)

    def open_run_folder(self) -> None:
        if self.current_run_dir is None:
            messagebox.showinfo("No run", "No run directory is selected yet.")
            return
        open_path(self.current_run_dir)

    def open_tensorboard(self) -> None:
        if self.current_run_dir is None:
            messagebox.showinfo("No run", "Start or select a run first.")
            return
        cmd = [str(python_executable()), "-m", "tensorboard.main", "--logdir", str(self.current_run_dir)]
        subprocess.Popen(cmd, cwd=str(REPO_ROOT))
        self.log_text.insert(tk.END, "\nLaunched TensorBoard server for: " + str(self.current_run_dir) + "\n")
        summary = read_json(self.summary_path) if self.summary_path is not None else {}
        progress_payload = read_json(self.progress_path) if self.progress_path is not None else {}
        hint = tensorboard_wait_hint(
            self.current_run_dir,
            summary=summary,
            progress_payload=progress_payload,
            context=self.current_training_context,
        )
        if hint:
            self.log_text.insert(tk.END, hint + "\n")
        self.log_text.see(tk.END)


class ResultsFrame(BasePanel):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        browser = self.section(0, 0, "Runs")
        detail = self.section(0, 1, "Details")
        detail.rowconfigure(1, weight=1)
        detail.columnconfigure(0, weight=1)

        self.root_var = tk.StringVar(value=str(REPO_ROOT / "checkpoints"))
        self.summary_path_var = tk.StringVar(value="No run selected")
        self.deterministic_playback_var = tk.BooleanVar(value=True)

        ttk.Label(browser, text="Checkpoint root").grid(row=0, column=0, sticky="w")
        ttk.Entry(browser, textvariable=self.root_var).grid(row=1, column=0, sticky="ew", pady=(2, 6))
        ttk.Button(browser, text="Browse Root", command=self.browse_root).grid(row=1, column=1, padx=(8, 0))
        ttk.Button(browser, text="Refresh", command=self.refresh_runs).grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Checkbutton(
            browser,
            text="Deterministic playback (default)",
            variable=self.deterministic_playback_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 8))

        run_list_host = ttk.Frame(browser, style="TFrame")
        run_list_host.grid(row=4, column=0, columnspan=2, sticky="nsew")
        run_list_host.columnconfigure(0, weight=1)
        run_list_host.rowconfigure(0, weight=1)

        self.run_list = tk.Listbox(
            run_list_host,
            bg=APP_PALETTE["surface"],
            fg=APP_PALETTE["text"],
            selectbackground=APP_PALETTE["accent"],
            selectforeground="#f8fbff",
            height=24,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=APP_PALETTE["border"],
            activestyle="none",
            font="TkTextFont",
        )
        self.run_list.grid(row=0, column=0, sticky="nsew")
        run_list_scrollbar = ttk.Scrollbar(run_list_host, orient="vertical", command=self.run_list.yview)
        run_list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.run_list.configure(yscrollcommand=run_list_scrollbar.set)
        self.run_list.bind("<<ListboxSelect>>", lambda _event: self.load_selected_run())
        browser.rowconfigure(4, weight=1)
        browser.columnconfigure(0, weight=1)

        button_bar = ttk.Frame(browser)
        button_bar.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(button_bar, text="Open Folder", command=self.open_selected_folder).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Compact Run", command=self.compact_selected_run).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Evaluate", command=self.evaluate_selected).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Play AI vs AI", command=self.play_selected).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="TensorBoard Server", command=self.tensorboard_selected).pack(side="left")

        ttk.Label(detail, textvariable=self.summary_path_var).grid(row=0, column=0, sticky="w")

        upper = ttk.Panedwindow(detail, orient="vertical")
        upper.grid(row=1, column=0, sticky="nsew")

        text_frame = ttk.Frame(upper, style="TFrame")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        plot_frame = ttk.Frame(upper)
        upper.add(text_frame, weight=1)
        upper.add(plot_frame, weight=2)

        self.detail_text = self.text_box(text_frame, height=14)
        self.detail_text.grid(row=0, column=0, sticky="nsew")
        detail_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.detail_text.yview)
        detail_scrollbar.grid(row=0, column=1, sticky="ns")
        self.detail_text.configure(yscrollcommand=detail_scrollbar.set)

        self.plot_frame = plot_frame
        self.figure_canvas = None

        self.selected_run_dir: Path | None = None
        self.run_summaries: list[dict] = []
        self.refresh_runs()

    def browse_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.root_var.get() or str(REPO_ROOT / "checkpoints"))
        if selected:
            self.root_var.set(selected)
            self.refresh_runs()

    def refresh_runs(self) -> None:
        root = Path(self.root_var.get())
        self.run_summaries = discover_training_runs(root)
        self.run_list.delete(0, tk.END)
        for item in self.run_summaries:
            mode = item["summary"].get("mode", "run")
            algorithm = item["summary"].get("algorithm", "ppo")
            game_mode = normalize_game_mode(item["summary"].get("game_mode", item["summary"].get("env_config", {}).get("world", {}).get("game_mode", "escape")))
            self.run_list.insert(tk.END, f"{item['path'].name}  [{game_mode_label(game_mode)} | {mode} | {algorithm}]")

    def load_selected_run(self) -> None:
        selection = self.run_list.curselection()
        if not selection:
            return
        entry = self.run_summaries[selection[0]]
        self.selected_run_dir = entry["path"]
        summary = entry["summary"]
        self.summary_path_var.set(str(self.selected_run_dir))

        lines = [
            f"Mode: {summary.get('mode')}",
            f"Status: {summary.get('status', 'completed')}",
            f"Game mode: {game_mode_label(normalize_game_mode(summary.get('game_mode', summary.get('env_config', {}).get('world', {}).get('game_mode', 'escape'))))}",
            f"Algorithm: {summary.get('algorithm', 'n/a')}",
            f"Run dir: {self.selected_run_dir}",
            f"Final model: {summary.get('final_model') or summary.get('final_enemy_model')}",
            f"Preset: {summary.get('preset', 'n/a')}",
            f"Seed: {summary.get('seed', 'n/a')}",
            f"Compacted: {'yes' if summary.get('artifacts_compacted') else 'no'}",
            "",
            "Summary JSON:",
            json.dumps(summary, indent=2, ensure_ascii=False),
        ]
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, "\n".join(lines))
        self._render_plots()

    def _render_plots(self) -> None:
        if self.selected_run_dir is None or Figure is None or FigureCanvasTkAgg is None:
            return
        progress_rows = read_jsonl(self.selected_run_dir / "progress_history.jsonl")
        eval_rows = read_jsonl(self.selected_run_dir / "eval_history.jsonl")

        if self.figure_canvas is not None:
            self.figure_canvas.get_tk_widget().destroy()

        figure = Figure(figsize=(7.4, 5.2), dpi=120, facecolor=APP_PALETTE["bg"])
        ax1 = figure.add_subplot(211)
        ax2 = figure.add_subplot(212)
        for axis in (ax1, ax2):
            axis.set_facecolor(APP_PALETTE["surface"])
            axis.tick_params(colors=APP_PALETTE["text"])
            axis.title.set_color("#f8fbff")
            axis.xaxis.label.set_color(APP_PALETTE["text"])
            axis.yaxis.label.set_color(APP_PALETTE["text"])

        if progress_rows:
            xs = [row.get("global_timesteps", row.get("phase_timesteps", 0)) for row in progress_rows]
            ys = [row.get("mean_episode_reward") if row.get("mean_episode_reward") is not None else 0.0 for row in progress_rows]
            ax1.plot(xs, ys, color=APP_PALETTE["accent"], linewidth=2.0)
        ax1.set_title("Training Reward Trend")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Mean Episode Reward")

        if eval_rows:
            ex = [row.get("num_timesteps", 0) for row in eval_rows]
            ey = [row.get("last_mean_reward", 0.0) for row in eval_rows]
            by = [row.get("best_mean_reward", 0.0) for row in eval_rows]
            ax2.plot(ex, ey, color=APP_PALETTE["warm"], linewidth=2.0, label="Eval reward")
            ax2.plot(ex, by, color=APP_PALETTE["success"], linewidth=1.6, linestyle="--", label="Best reward")
            ax2.legend(facecolor=APP_PALETTE["bg"], edgecolor=APP_PALETTE["border"], labelcolor=APP_PALETTE["text"])
        ax2.set_title("Evaluation Reward")
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Mean Eval Reward")

        figure.tight_layout(pad=1.6)
        self.figure_canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    def open_selected_folder(self) -> None:
        if self.selected_run_dir is None:
            return
        open_path(self.selected_run_dir)

    def compact_selected_run(self) -> None:
        if self.selected_run_dir is None:
            return
        if not messagebox.askyesno(
            "Compact run",
            "Keep only the latest playable checkpoint and minimal metadata for this selected run?\n\n"
            "This cannot be undone from the GUI.",
        ):
            return
        result = compact_training_run(self.selected_run_dir)
        freed_mb = result["deleted_bytes"] / (1024 * 1024)
        messagebox.showinfo(
            "Run compacted",
            f"Deleted {result['deleted_files']} files and freed {freed_mb:.1f} MB.\n\n"
            f"Kept {result['kept_files']} essential files for playback and Results.",
        )
        self.refresh_runs()
        self.summary_path_var.set(str(self.selected_run_dir))
        self.load_selected_run()

    def tensorboard_selected(self) -> None:
        if self.selected_run_dir is None:
            return
        subprocess.Popen([str(python_executable()), "-m", "tensorboard.main", "--logdir", str(self.selected_run_dir)], cwd=str(REPO_ROOT))
        summary = read_json(self.selected_run_dir / "training_summary.json")
        progress_payload = read_json(self.selected_run_dir / "progress.json")
        hint = tensorboard_wait_hint(self.selected_run_dir, summary=summary, progress_payload=progress_payload)
        if hint:
            messagebox.showinfo("TensorBoard status", hint)

    def _selected_models(self) -> tuple[str | None, str | None]:
        if self.selected_run_dir is None:
            return None, None
        summary = read_json(self.selected_run_dir / "training_summary.json")
        mode = summary.get("mode")
        if mode == "single_agent_enemy":
            return None, str(summary.get("final_model"))
        if mode == "single_agent_player":
            return str(summary.get("final_model")), None
        if mode == "self_play":
            return str(summary.get("final_player_model")), str(summary.get("final_enemy_model"))
        return None, None

    def evaluate_selected(self) -> None:
        if self.selected_run_dir is None:
            return
        player_model, enemy_model = self._selected_models()
        cmd = [str(python_executable()), str(REPO_ROOT / "scripts" / "eval_elo.py"), "--episodes", "12"]
        if player_model:
            cmd += ["--player-model", player_model]
        if enemy_model:
            cmd += ["--enemy-model", enemy_model]
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        output = completed.stdout if completed.stdout else completed.stderr
        messagebox.showinfo("Evaluation", output or "No output.")

    def play_selected(self) -> None:
        if self.selected_run_dir is None:
            return
        player_model, enemy_model = self._selected_models()
        summary = read_json(self.selected_run_dir / "training_summary.json")
        game_mode = normalize_game_mode(summary.get("game_mode", summary.get("env_config", {}).get("world", {}).get("game_mode", "escape")))
        cmd = [str(python_executable()), "-m", "library_escape.play.ai_vs_ai"]
        if player_model:
            cmd += ["--player-model", player_model]
        if enemy_model:
            cmd += ["--enemy-model", enemy_model]
        if self.deterministic_playback_var.get():
            cmd += ["--deterministic-policy"]
        cmd += ["--game-mode", game_mode]
        subprocess.Popen(cmd, cwd=str(REPO_ROOT))


class TensorBoardFrame(BasePanel):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        browser = self.section(0, 0, "Run Browser")
        detail = self.section(0, 1, "Built-In TensorBoard Scalars")
        detail.rowconfigure(1, weight=1)
        detail.columnconfigure(0, weight=0)
        detail.columnconfigure(1, weight=1)

        self.root_var = tk.StringVar(value=str(REPO_ROOT / "checkpoints"))
        self.summary_var = tk.StringVar(value="Select a run to inspect TensorBoard scalars.")

        ttk.Label(browser, text="Checkpoint root").grid(row=0, column=0, sticky="w")
        ttk.Entry(browser, textvariable=self.root_var).grid(row=1, column=0, sticky="ew", pady=(2, 6))
        ttk.Button(browser, text="Browse Root", command=self.browse_root).grid(row=1, column=1, padx=(8, 0))
        ttk.Button(browser, text="Refresh Runs", command=self.refresh_runs).grid(row=2, column=0, sticky="ew", pady=(0, 8))

        run_list_host = ttk.Frame(browser, style="TFrame")
        run_list_host.grid(row=3, column=0, columnspan=2, sticky="nsew")
        run_list_host.columnconfigure(0, weight=1)
        run_list_host.rowconfigure(0, weight=1)

        self.run_list = tk.Listbox(
            run_list_host,
            bg=APP_PALETTE["surface"],
            fg=APP_PALETTE["text"],
            selectbackground=APP_PALETTE["accent"],
            selectforeground="#f8fbff",
            height=22,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=APP_PALETTE["border"],
            activestyle="none",
            font="TkTextFont",
        )
        self.run_list.grid(row=0, column=0, sticky="nsew")
        run_list_scrollbar = ttk.Scrollbar(run_list_host, orient="vertical", command=self.run_list.yview)
        run_list_scrollbar.grid(row=0, column=1, sticky="ns")
        self.run_list.configure(yscrollcommand=run_list_scrollbar.set)
        self.run_list.bind("<<ListboxSelect>>", lambda _event: self.load_selected_run())
        browser.rowconfigure(3, weight=1)
        browser.columnconfigure(0, weight=1)

        ttk.Label(detail, textvariable=self.summary_var).grid(row=0, column=0, columnspan=2, sticky="w")
        tag_list_host = ttk.Frame(detail, style="TFrame")
        tag_list_host.grid(row=1, column=0, sticky="nsew")
        tag_list_host.columnconfigure(0, weight=1)
        tag_list_host.rowconfigure(0, weight=1)
        self.tag_list = tk.Listbox(
            tag_list_host,
            bg=APP_PALETTE["surface"],
            fg=APP_PALETTE["text"],
            selectbackground=APP_PALETTE["accent"],
            selectforeground="#f8fbff",
            selectmode=tk.EXTENDED,
            width=32,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=APP_PALETTE["border"],
            activestyle="none",
            font="TkTextFont",
        )
        self.tag_list.grid(row=0, column=0, sticky="nsew")
        tag_scrollbar = ttk.Scrollbar(tag_list_host, orient="vertical", command=self.tag_list.yview)
        tag_scrollbar.grid(row=0, column=1, sticky="ns")
        self.tag_list.configure(yscrollcommand=tag_scrollbar.set)
        self.tag_list.bind("<<ListboxSelect>>", lambda _event: self.render_plot())

        right = ttk.Frame(detail)
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=0)

        self.plot_host = ttk.Frame(right)
        self.plot_host.grid(row=0, column=0, sticky="nsew")
        self.figure_canvas = None
        self.stats_text = self.text_box(right, height=8)
        self.stats_text.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        stats_scrollbar = ttk.Scrollbar(right, orient="vertical", command=self.stats_text.yview)
        stats_scrollbar.grid(row=1, column=1, sticky="ns", pady=(10, 0))
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)

        self.selected_run_dir: Path | None = None
        self.run_summaries: list[dict] = []
        self.scalar_series: dict[str, list[tuple[int, float]]] = {}
        self.refresh_runs()

    def browse_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.root_var.get())
        if selected:
            self.root_var.set(selected)
            self.refresh_runs()

    def refresh_runs(self) -> None:
        self.run_summaries = discover_training_runs(Path(self.root_var.get()))
        self.run_list.delete(0, tk.END)
        for item in self.run_summaries:
            mode = item["summary"].get("mode", "run")
            game_mode = normalize_game_mode(item["summary"].get("game_mode", item["summary"].get("env_config", {}).get("world", {}).get("game_mode", "escape")))
            self.run_list.insert(tk.END, f"{item['path'].name}  [{game_mode_label(game_mode)} | {mode}]")

    def load_selected_run(self) -> None:
        selection = self.run_list.curselection()
        if not selection:
            return
        entry = self.run_summaries[selection[0]]
        self.selected_run_dir = entry["path"]
        self.scalar_series = load_scalar_series(self.selected_run_dir)
        self.summary_var.set(f"{self.selected_run_dir} | {len(self.scalar_series)} scalar tags")
        self.tag_list.delete(0, tk.END)
        tags = sorted(self.scalar_series)
        for tag in tags:
            self.tag_list.insert(tk.END, tag)

        preferred_patterns = ["rollout/ep_rew_mean", "eval/mean_reward", "train/value_loss", "train/entropy_loss"]
        selected_indices: list[int] = []
        for pattern in preferred_patterns:
            for idx, tag in enumerate(tags):
                if pattern in tag and idx not in selected_indices:
                    selected_indices.append(idx)
                    break
        if not selected_indices and tags:
            selected_indices = list(range(min(3, len(tags))))
        for idx in selected_indices:
            self.tag_list.selection_set(idx)
        self.render_plot()

    def render_plot(self) -> None:
        if Figure is None or FigureCanvasTkAgg is None:
            return
        selected_indices = list(self.tag_list.curselection())
        if not selected_indices or not self.scalar_series:
            return
        tags = [self.tag_list.get(index) for index in selected_indices[:4]]

        if self.figure_canvas is not None:
            self.figure_canvas.get_tk_widget().destroy()

        figure = Figure(figsize=(8.2, 5.2), dpi=120, facecolor=APP_PALETTE["bg"])
        axes = [figure.add_subplot(len(tags), 1, idx + 1) for idx in range(len(tags))]
        if len(tags) == 1:
            axes = [axes[0]]
        stats_lines: list[str] = []

        for axis, tag in zip(axes, tags):
            axis.set_facecolor(APP_PALETTE["surface"])
            axis.tick_params(colors=APP_PALETTE["text"])
            axis.title.set_color("#f8fbff")
            axis.xaxis.label.set_color(APP_PALETTE["text"])
            axis.yaxis.label.set_color(APP_PALETTE["text"])

            points = self.scalar_series.get(tag, [])
            xs = [step for step, _ in points]
            ys = [value for _, value in points]
            axis.plot(xs, ys, color=APP_PALETTE["accent"], linewidth=1.8)
            axis.set_title(tag)
            axis.set_xlabel("Step")
            axis.set_ylabel("Value")
            if ys:
                stats_lines.append(f"{tag}\n  latest={ys[-1]:.5f}  min={min(ys):.5f}  max={max(ys):.5f}")

        figure.tight_layout(pad=1.5)
        self.figure_canvas = FigureCanvasTkAgg(figure, master=self.plot_host)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, "\n\n".join(stats_lines))


class LeaderboardFrame(BasePanel):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        controls = self.section(0, 0, "Leaderboard Builder")
        boards = self.section(0, 1, "Ratings")
        logs = self.section(1, 0, "Matches And Logs", columnspan=2)
        boards.columnconfigure(0, weight=1)
        boards.columnconfigure(1, weight=1)
        boards.rowconfigure(1, weight=1)
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(0, weight=1)

        self.root_var = tk.StringVar(value=str(REPO_ROOT / "checkpoints"))
        self.output_var = tk.StringVar(value=str(REPO_ROOT / "checkpoints" / "leaderboard.json"))
        self.episodes_var = tk.StringVar(value="8")
        self.max_players_var = tk.StringVar(value="6")
        self.max_enemies_var = tk.StringVar(value="6")
        self.k_factor_var = tk.StringVar(value="24")
        self.status_var = tk.StringVar(value="Ready.")

        ttk.Label(controls, text="Checkpoint root").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.root_var).grid(row=1, column=0, sticky="ew", pady=(2, 6))
        ttk.Button(controls, text="Browse Root", command=self.browse_root).grid(row=1, column=1, padx=(8, 0))
        ttk.Label(controls, text="Output JSON").grid(row=2, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.output_var).grid(row=3, column=0, sticky="ew", pady=(2, 6))
        ttk.Button(controls, text="Browse Output", command=self.browse_output).grid(row=3, column=1, padx=(8, 0))

        ttk.Label(controls, text="Episodes").grid(row=4, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.episodes_var).grid(row=5, column=0, sticky="ew", pady=(2, 6))
        ttk.Label(controls, text="Max players").grid(row=6, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.max_players_var).grid(row=7, column=0, sticky="ew", pady=(2, 6))
        ttk.Label(controls, text="Max enemies").grid(row=8, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.max_enemies_var).grid(row=9, column=0, sticky="ew", pady=(2, 6))
        ttk.Label(controls, text="K-factor").grid(row=10, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.k_factor_var).grid(row=11, column=0, sticky="ew", pady=(2, 10))

        button_bar = ttk.Frame(controls)
        button_bar.grid(row=12, column=0, columnspan=2, sticky="ew")
        ttk.Button(button_bar, text="Build Leaderboard", style="Accent.TButton", command=self.build_leaderboard).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Load Existing", command=self.load_existing).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Open Output", command=self.open_output).pack(side="left")
        ttk.Label(controls, textvariable=self.status_var).grid(row=13, column=0, columnspan=2, sticky="w", pady=(12, 0))
        controls.columnconfigure(0, weight=1)

        ttk.Label(boards, text="Players").grid(row=0, column=0, sticky="w")
        ttk.Label(boards, text="Enemies").grid(row=0, column=1, sticky="w")
        self.player_tree = self._tree(boards)
        self.enemy_tree = self._tree(boards)
        self.player_tree.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.enemy_tree.grid(row=1, column=1, sticky="nsew")

        self.log_text = self.text_box(logs, height=20)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(logs, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()

    def _tree(self, master) -> ttk.Treeview:
        tree = ttk.Treeview(master, columns=("name", "rating", "games", "wins", "source"), show="headings", height=14)
        tree.heading("name", text="Name")
        tree.heading("rating", text="Rating")
        tree.heading("games", text="Games")
        tree.heading("wins", text="WinRate")
        tree.heading("source", text="Source")
        tree.column("name", width=180, anchor="w")
        tree.column("rating", width=90, anchor="center")
        tree.column("games", width=80, anchor="center")
        tree.column("wins", width=80, anchor="center")
        tree.column("source", width=180, anchor="w")
        return tree

    def browse_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.root_var.get())
        if selected:
            self.root_var.set(selected)

    def browse_output(self) -> None:
        selected = filedialog.asksaveasfilename(
            title="Leaderboard output",
            defaultextension=".json",
            initialfile=Path(self.output_var.get()).name,
            initialdir=str(Path(self.output_var.get()).parent),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.output_var.set(selected)

    def build_leaderboard(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("Busy", "A leaderboard build is already running.")
            return
        cmd = [
            str(python_executable()),
            str(REPO_ROOT / "scripts" / "eval_elo.py"),
            "--root",
            self.root_var.get(),
            "--episodes",
            self.episodes_var.get().strip() or "8",
            "--max-players",
            self.max_players_var.get().strip() or "6",
            "--max-enemies",
            self.max_enemies_var.get().strip() or "6",
            "--k-factor",
            self.k_factor_var.get().strip() or "24",
            "--output",
            self.output_var.get(),
        ]
        self.log_text.delete("1.0", tk.END)
        self.process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self.status_var.set("Building leaderboard...")
        threading.Thread(target=self._pump_logs, daemon=True).start()
        self.after(LOG_POLL_INTERVAL_MS, self._poll_process)

    def _pump_logs(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(line)

    def _poll_process(self) -> None:
        drain_log_queue(self.log_queue, self.log_text)
        if self.process is None:
            return
        if self.process.poll() is None:
            self.after(LOG_POLL_INTERVAL_MS, self._poll_process)
            return
        self.log_text.insert(tk.END, f"\nProcess exited with code {self.process.returncode}\n")
        self.log_text.see(tk.END)
        self.status_var.set("Leaderboard updated." if self.process.returncode == 0 else "Leaderboard build failed.")
        self.process = None
        if Path(self.output_var.get()).exists():
            self.load_existing()

    def load_existing(self) -> None:
        output_path = Path(self.output_var.get())
        if not output_path.exists():
            messagebox.showinfo("Missing file", "Leaderboard JSON does not exist yet.")
            return
        payload = read_json(output_path)
        for tree in (self.player_tree, self.enemy_tree):
            for item in tree.get_children():
                tree.delete(item)
        for row in payload.get("players", []):
            self.player_tree.insert("", tk.END, values=(row["label"], row["rating"], row["games"], row["win_rate"], row["source"]), iid=row["key"])
        for row in payload.get("enemies", []):
            self.enemy_tree.insert("", tk.END, values=(row["label"], row["rating"], row["games"], row["win_rate"], row["source"]), iid=row["key"])

        match_lines = [
            f"Episodes per match: {payload.get('episodes_per_match', 'n/a')}",
            f"K-factor: {payload.get('k_factor', 'n/a')}",
            "",
            "Matches:",
        ]
        for row in payload.get("matches", [])[:120]:
            match_lines.append(
                f"{row['player']} vs {row['enemy']} | score={row['player_score']:.3f} | "
                f"PW={row['player_wins']} EW={row['enemy_wins']} D={row['draws']}"
            )
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "\n".join(match_lines))

    def open_output(self) -> None:
        open_path(Path(self.output_var.get()))


class ReplayFrame(BasePanel):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        browser = self.section(0, 0, "Replay Browser")
        detail = self.section(0, 1, "Replay Details")
        detail.columnconfigure(0, weight=1)
        detail.rowconfigure(1, weight=1)

        self.root_var = tk.StringVar(value=str(REPO_ROOT / "replays"))
        self.selected_path_var = tk.StringVar(value="No replay selected")
        self.export_dir_var = tk.StringVar(value=str(REPO_ROOT / "videos" / "frames"))

        ttk.Label(browser, text="Replay root").grid(row=0, column=0, sticky="w")
        ttk.Entry(browser, textvariable=self.root_var).grid(row=1, column=0, sticky="ew", pady=(2, 6))
        ttk.Button(browser, text="Browse Root", command=self.browse_root).grid(row=1, column=1, padx=(8, 0))
        ttk.Button(browser, text="Refresh", command=self.refresh_replays).grid(row=2, column=0, sticky="ew", pady=(0, 8))

        replay_list_host = ttk.Frame(browser, style="TFrame")
        replay_list_host.grid(row=3, column=0, columnspan=2, sticky="nsew")
        replay_list_host.columnconfigure(0, weight=1)
        replay_list_host.rowconfigure(0, weight=1)

        self.replay_list = tk.Listbox(
            replay_list_host,
            bg=APP_PALETTE["surface"],
            fg=APP_PALETTE["text"],
            selectbackground=APP_PALETTE["accent"],
            selectforeground="#f8fbff",
            height=24,
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=APP_PALETTE["border"],
            activestyle="none",
            font="TkTextFont",
        )
        self.replay_list.grid(row=0, column=0, sticky="nsew")
        replay_scrollbar = ttk.Scrollbar(replay_list_host, orient="vertical", command=self.replay_list.yview)
        replay_scrollbar.grid(row=0, column=1, sticky="ns")
        self.replay_list.configure(yscrollcommand=replay_scrollbar.set)
        self.replay_list.bind("<<ListboxSelect>>", lambda _event: self.load_selected_replay())
        browser.rowconfigure(3, weight=1)
        browser.columnconfigure(0, weight=1)

        button_bar = ttk.Frame(browser)
        button_bar.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(button_bar, text="Open Folder", command=self.open_root).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Play Replay", command=self.play_selected).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Export Frames", command=self.export_selected).pack(side="left")

        ttk.Label(detail, textvariable=self.selected_path_var).grid(row=0, column=0, sticky="w")
        detail.rowconfigure(1, weight=1)
        self.detail_text = self.text_box(detail, height=24)
        self.detail_text.grid(row=1, column=0, sticky="nsew")
        detail_scrollbar = ttk.Scrollbar(detail, orient="vertical", command=self.detail_text.yview)
        detail_scrollbar.grid(row=1, column=1, sticky="ns")
        self.detail_text.configure(yscrollcommand=detail_scrollbar.set)
        export_bar = ttk.Frame(detail)
        export_bar.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Entry(export_bar, textvariable=self.export_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(export_bar, text="Browse Export", command=self.browse_export).pack(side="left", padx=(8, 0))

        self.replays: list[Path] = []
        self.selected_replay: Path | None = None
        self.refresh_replays()

    def browse_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.root_var.get())
        if selected:
            self.root_var.set(selected)
            self.refresh_replays()

    def browse_export(self) -> None:
        selected = filedialog.askdirectory(initialdir=str(Path(self.export_dir_var.get()).parent))
        if selected:
            self.export_dir_var.set(selected)

    def refresh_replays(self) -> None:
        root = Path(self.root_var.get())
        self.replays = sorted(root.rglob("*.ler.gz"), reverse=True) if root.exists() else []
        self.replay_list.delete(0, tk.END)
        for replay_path in self.replays:
            self.replay_list.insert(tk.END, replay_path.name)

    def load_selected_replay(self) -> None:
        selection = self.replay_list.curselection()
        if not selection:
            return
        self.selected_replay = self.replays[selection[0]]
        payload = load_replay(self.selected_replay)
        self.selected_path_var.set(str(self.selected_replay))
        metadata = payload.get("metadata", {})
        lines = [
            f"Frames: {len(payload.get('frames', []))}",
            f"Mode: {metadata.get('mode', 'n/a')}",
            f"Seed: {metadata.get('seed', 'n/a')}",
            f"Recorded at: {metadata.get('recorded_at', 'n/a')}",
            f"Render FPS: {metadata.get('render_fps', 'n/a')}",
            "",
            "Metadata:",
            json.dumps(metadata, indent=2, ensure_ascii=False),
        ]
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, "\n".join(lines))

    def play_selected(self) -> None:
        if self.selected_replay is None:
            return
        cmd = [str(python_executable()), "-m", "library_escape.replay.viewer", "--replay", str(self.selected_replay)]
        subprocess.Popen(cmd, cwd=str(REPO_ROOT))

    def export_selected(self) -> None:
        if self.selected_replay is None:
            return
        export_dir = Path(self.export_dir_var.get())
        if export_dir.suffix:
            export_dir = export_dir.parent
        replay_stem = self.selected_replay.name.replace(".ler.gz", "")
        target_dir = export_dir / replay_stem
        cmd = [
            str(python_executable()),
            "-m",
            "library_escape.replay.viewer",
            "--replay",
            str(self.selected_replay),
            "--export-frames",
            str(target_dir),
        ]
        subprocess.Popen(cmd, cwd=str(REPO_ROOT))

    def open_root(self) -> None:
        open_path(Path(self.root_var.get()))


class ConfigFrame(BasePanel):
    def __init__(self, master) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        scroll_host = self.scrollable_frame(self)
        scroll_host.grid(row=0, column=0, sticky="nsew")
        docs = self.section(0, 0, "Quick Access", parent=scroll_host.content)
        tips = self.section(1, 0, "What To Edit", parent=scroll_host.content)

        buttons = [
            ("Quick Start Guide", REPO_ROOT / "docs" / "OPERATION_QUICKSTART.md"),
            ("Collection Rewards", REPO_ROOT / "configs" / "rewards_collection.yaml"),
            ("Escape Rewards", REPO_ROOT / "configs" / "rewards_escape.yaml"),
            ("Environment", REPO_ROOT / "configs" / "env.yaml"),
            ("Training", REPO_ROOT / "configs" / "training.yaml"),
            ("Map", REPO_ROOT / "configs" / "map.json"),
            ("Game Modes Helper", REPO_ROOT / "library_escape" / "game_modes.py"),
            ("Operation Guide", REPO_ROOT / "docs" / "OPERATION_GUIDE.md"),
            ("Project Root", REPO_ROOT),
        ]
        for idx, (label, path) in enumerate(buttons):
            ttk.Button(docs, text=label, command=lambda path=path: open_path(path)).grid(row=idx, column=0, sticky="ew", pady=4)
        docs.columnconfigure(0, weight=1)

        copy = (
            "Quick operation guide: docs/OPERATION_QUICKSTART.md\n"
            "Full technical guide: docs/OPERATION_GUIDE.md\n"
            "\n"
            "Collection-mode rewards: configs/rewards_collection.yaml\n"
            "Escape-mode rewards: configs/rewards_escape.yaml\n"
            "Observation / action / frame skip: configs/env.yaml\n"
            "Training presets / algorithms / curriculum / MAPPO recipe: configs/training.yaml\n"
            "Map layout and spawns: configs/map.json\n"
            "User-facing mode mapping: library_escape/game_modes.py\n"
            "\n"
            "In both Play and Train, choose a Game mode first:\n"
            "Collection = score / pressure / stealth collection.\n"
            "Escape = required notes first, then escape through the exit.\n"
            "\n"
            "Use Train for enemy / player / self-play with PPO, Maskable PPO, or external MAPPO recipe.\n"
            "Checkpoints should usually be played in the same Game mode they were trained in.\n"
            "Use TensorBoard for built-in scalar inspection.\n"
            "Use Leaderboard to build automatic player/enemy Elo tables.\n"
            "Use Replay to browse .ler.gz files and export frame sequences."
        )
        ttk.Label(tips, text=copy, justify="left", wraplength=980).grid(row=0, column=0, sticky="w")


def main() -> None:
    enable_windows_high_dpi()
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
