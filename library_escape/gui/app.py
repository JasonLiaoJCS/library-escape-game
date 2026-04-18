"""Tkinter launcher for playing, training, replay, and run analysis."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from ..config import REPO_ROOT, load_training_config, resolve_repo_path
from ..eval.registry import discover_saved_models
from ..game_modes import game_mode_label, infer_game_mode_from_models, normalize_game_mode, read_model_game_mode
from ..gui.tensorboard_data import load_scalar_series
from ..replay.io import load_replay
from ..train.callbacks import format_seconds
from ..train.common import timestamped_run_name

try:  # pragma: no cover - optional GUI plotting dependency
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover - fallback if matplotlib unavailable
    FigureCanvasTkAgg = None
    Figure = None


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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
        self.geometry("1520x960")
        self.minsize(1320, 860)
        self.configure(bg="#10151b")
        self.style = ttk.Style(self)
        self.style.theme_use("clam")
        self._configure_theme()

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=14, pady=14)

        self.play_frame = PlayFrame(notebook)
        self.train_frame = TrainFrame(notebook)
        self.results_frame = ResultsFrame(notebook)
        self.tensorboard_frame = TensorBoardFrame(notebook)
        self.leaderboard_frame = LeaderboardFrame(notebook)
        self.replay_frame = ReplayFrame(notebook)
        self.config_frame = ConfigFrame(notebook)

        notebook.add(self.play_frame, text="Play")
        notebook.add(self.train_frame, text="Train")
        notebook.add(self.results_frame, text="Results")
        notebook.add(self.tensorboard_frame, text="TensorBoard")
        notebook.add(self.leaderboard_frame, text="Leaderboard")
        notebook.add(self.replay_frame, text="Replay")
        notebook.add(self.config_frame, text="Config")

    def _configure_theme(self) -> None:
        self.style.configure(".", background="#10151b", foreground="#e9eef5", fieldbackground="#171d24")
        self.style.configure("TFrame", background="#10151b")
        self.style.configure("TLabelframe", background="#10151b", foreground="#f3f6fb")
        self.style.configure("TLabelframe.Label", background="#10151b", foreground="#f3f6fb")
        self.style.configure("TLabel", background="#10151b", foreground="#e9eef5")
        self.style.configure("TButton", background="#2a8cff", foreground="#f8fbff", padding=8)
        self.style.configure("Accent.TButton", background="#ff8d4d", foreground="#10151b", padding=8)
        self.style.configure("TEntry", fieldbackground="#171d24", foreground="#eef4ff")
        self.style.configure("TCombobox", fieldbackground="#171d24", foreground="#eef4ff")
        self.style.configure("Treeview", background="#0b1015", foreground="#d9e4f2", fieldbackground="#0b1015")
        self.style.configure("Treeview.Heading", background="#18222c", foreground="#f2f6fb")
        self.style.configure("Horizontal.TProgressbar", troughcolor="#171d24", background="#2a8cff")
        self.option_add("*Font", "{Segoe UI} 10")


class BasePanel(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master, padding=12)

    def section(self, row: int, column: int, title: str, columnspan: int = 1, rowspan: int = 1) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(self, text=title, padding=12)
        frame.grid(row=row, column=column, columnspan=columnspan, rowspan=rowspan, sticky="nsew", padx=8, pady=8)
        return frame

    def text_box(self, master, height: int = 12) -> tk.Text:
        widget = tk.Text(master, bg="#0b1015", fg="#d9e4f2", insertbackground="#ffffff", wrap="word", font=("Consolas", 10), height=height)
        return widget


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
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)

        controls = self.section(0, 0, "Training Setup")
        advanced = self.section(0, 1, "Algorithms And Curriculum")
        monitor = self.section(0, 2, "Progress")
        logs = self.section(1, 0, "Logs", columnspan=3)
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
        self.device_var = tk.StringVar(value="auto")
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

        button_bar = ttk.Frame(controls)
        button_bar.grid(row=19, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(button_bar, text="Start Training", style="Accent.TButton", command=self.start_training).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Stop", command=self.stop_training).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Open Run Folder", command=self.open_run_folder).pack(side="left")
        ttk.Button(controls, text="Open TensorBoard Server", command=self.open_tensorboard).grid(row=20, column=0, sticky="ew", pady=(10, 0))
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

        self.progress_bar = ttk.Progressbar(monitor, mode="determinate", maximum=100)
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(4, 10))
        monitor.columnconfigure(0, weight=1)
        monitor.columnconfigure(1, weight=1)

        metric_pairs = [
            ("Run dir", self.run_dir_var),
            ("Phase", self.phase_var),
            ("Progress", self.progress_var),
            ("Elapsed", self.elapsed_var),
            ("ETA", self.eta_var),
            ("FPS", self.fps_var),
            ("Mean reward", self.reward_var),
        ]
        for idx, (label, variable) in enumerate(metric_pairs, start=1):
            ttk.Label(monitor, text=label).grid(row=idx, column=0, sticky="w", pady=2)
            ttk.Label(monitor, textvariable=variable).grid(row=idx, column=1, sticky="w", pady=2)

        self.log_text = self.text_box(logs, height=24)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(logs, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.current_run_dir: Path | None = None
        self.progress_path: Path | None = None

        self.mode_var.trace_add("write", lambda *_: self._sync_mode_fields())
        self.game_mode_var.trace_add("write", lambda *_: self._sync_game_mode_hint())
        self._sync_mode_fields()
        self._sync_game_mode_hint()

    def _sync_mode_fields(self) -> None:
        mode = self.mode_var.get()
        if mode == "selfplay":
            values = list(self.SELFPLAY_ALGOS)
            if self.algorithm_var.get() not in values:
                self.algorithm_var.set(self.default_selfplay_algorithm if self.default_selfplay_algorithm in values else values[0])
        else:
            values = list(self.SINGLE_ALGOS)
            if self.algorithm_var.get() not in values:
                self.algorithm_var.set(self.default_single_algorithm if self.default_single_algorithm in values else values[0])
        self.algorithm_combo.configure(values=values)

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

        overrides = self._build_overrides_payload()
        cmd += ["--overrides-json", json.dumps(overrides, separators=(",", ":"), ensure_ascii=True)]

        self.current_run_dir = checkpoint_root_for_mode(mode, game_mode) / run_name
        self.progress_path = self.current_run_dir / "progress.json"
        self.run_dir_var.set(str(self.current_run_dir))
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
        self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, "Launching:\n" + " ".join(cmd) + "\n\n")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._pump_logs, daemon=True).start()
        self.after(250, self.poll_training)

    def _pump_logs(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(line)

    def poll_training(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)

        if self.progress_path and self.progress_path.exists():
            payload = read_json(self.progress_path)
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
            self.after(500, self.poll_training)
        else:
            self.after(200, self.poll_training)
            self.log_text.insert(tk.END, f"\nProcess exited with code {self.process.returncode}\n")
            self.log_text.see(tk.END)
            self.process = None

    def stop_training(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        self.log_text.insert(tk.END, "\nRequested training stop.\n")
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

        self.run_list = tk.Listbox(browser, bg="#0b1015", fg="#d9e4f2", selectbackground="#2a8cff", height=24)
        self.run_list.grid(row=4, column=0, columnspan=2, sticky="nsew")
        self.run_list.bind("<<ListboxSelect>>", lambda _event: self.load_selected_run())
        browser.rowconfigure(4, weight=1)
        browser.columnconfigure(0, weight=1)

        button_bar = ttk.Frame(browser)
        button_bar.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(button_bar, text="Open Folder", command=self.open_selected_folder).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Evaluate", command=self.evaluate_selected).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Play AI vs AI", command=self.play_selected).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="TensorBoard Server", command=self.tensorboard_selected).pack(side="left")

        ttk.Label(detail, textvariable=self.summary_path_var).grid(row=0, column=0, sticky="w")

        upper = ttk.Panedwindow(detail, orient="vertical")
        upper.grid(row=1, column=0, sticky="nsew")

        text_frame = ttk.Frame(upper)
        plot_frame = ttk.Frame(upper)
        upper.add(text_frame, weight=1)
        upper.add(plot_frame, weight=2)

        self.detail_text = self.text_box(text_frame, height=14)
        self.detail_text.pack(fill="both", expand=True)

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
            f"Game mode: {game_mode_label(normalize_game_mode(summary.get('game_mode', summary.get('env_config', {}).get('world', {}).get('game_mode', 'escape'))))}",
            f"Algorithm: {summary.get('algorithm', 'n/a')}",
            f"Run dir: {self.selected_run_dir}",
            f"Final model: {summary.get('final_model') or summary.get('final_enemy_model')}",
            f"Preset: {summary.get('preset', 'n/a')}",
            f"Seed: {summary.get('seed', 'n/a')}",
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

        figure = Figure(figsize=(7.4, 5.2), dpi=100, facecolor="#10151b")
        ax1 = figure.add_subplot(211)
        ax2 = figure.add_subplot(212)
        for axis in (ax1, ax2):
            axis.set_facecolor("#0b1015")
            axis.tick_params(colors="#d9e4f2")
            axis.title.set_color("#f4f8ff")
            axis.xaxis.label.set_color("#d9e4f2")
            axis.yaxis.label.set_color("#d9e4f2")

        if progress_rows:
            xs = [row.get("global_timesteps", row.get("phase_timesteps", 0)) for row in progress_rows]
            ys = [row.get("mean_episode_reward") if row.get("mean_episode_reward") is not None else 0.0 for row in progress_rows]
            ax1.plot(xs, ys, color="#2a8cff", linewidth=2.0)
        ax1.set_title("Training Reward Trend")
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Mean Episode Reward")

        if eval_rows:
            ex = [row.get("num_timesteps", 0) for row in eval_rows]
            ey = [row.get("last_mean_reward", 0.0) for row in eval_rows]
            by = [row.get("best_mean_reward", 0.0) for row in eval_rows]
            ax2.plot(ex, ey, color="#ff8d4d", linewidth=2.0, label="Eval reward")
            ax2.plot(ex, by, color="#7ad151", linewidth=1.6, linestyle="--", label="Best reward")
            ax2.legend(facecolor="#10151b", edgecolor="#33404d", labelcolor="#d9e4f2")
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

    def tensorboard_selected(self) -> None:
        if self.selected_run_dir is None:
            return
        subprocess.Popen([str(python_executable()), "-m", "tensorboard.main", "--logdir", str(self.selected_run_dir)], cwd=str(REPO_ROOT))

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

        self.run_list = tk.Listbox(browser, bg="#0b1015", fg="#d9e4f2", selectbackground="#2a8cff", height=22)
        self.run_list.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.run_list.bind("<<ListboxSelect>>", lambda _event: self.load_selected_run())
        browser.rowconfigure(3, weight=1)
        browser.columnconfigure(0, weight=1)

        ttk.Label(detail, textvariable=self.summary_var).grid(row=0, column=0, columnspan=2, sticky="w")
        self.tag_list = tk.Listbox(detail, bg="#0b1015", fg="#d9e4f2", selectbackground="#2a8cff", selectmode=tk.EXTENDED, width=32)
        self.tag_list.grid(row=1, column=0, sticky="nsw")
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

        figure = Figure(figsize=(8.2, 5.2), dpi=100, facecolor="#10151b")
        axes = [figure.add_subplot(len(tags), 1, idx + 1) for idx in range(len(tags))]
        if len(tags) == 1:
            axes = [axes[0]]
        stats_lines: list[str] = []

        for axis, tag in zip(axes, tags):
            axis.set_facecolor("#0b1015")
            axis.tick_params(colors="#d9e4f2")
            axis.title.set_color("#f4f8ff")
            axis.xaxis.label.set_color("#d9e4f2")
            axis.yaxis.label.set_color("#d9e4f2")

            points = self.scalar_series.get(tag, [])
            xs = [step for step, _ in points]
            ys = [value for _, value in points]
            axis.plot(xs, ys, color="#2a8cff", linewidth=1.8)
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
        )
        self.status_var.set("Building leaderboard...")
        threading.Thread(target=self._pump_logs, daemon=True).start()
        self.after(250, self._poll_process)

    def _pump_logs(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(line)

    def _poll_process(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)
        if self.process is None:
            return
        if self.process.poll() is None:
            self.after(500, self._poll_process)
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

        self.replay_list = tk.Listbox(browser, bg="#0b1015", fg="#d9e4f2", selectbackground="#2a8cff", height=24)
        self.replay_list.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self.replay_list.bind("<<ListboxSelect>>", lambda _event: self.load_selected_replay())
        browser.rowconfigure(3, weight=1)
        browser.columnconfigure(0, weight=1)

        button_bar = ttk.Frame(browser)
        button_bar.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(button_bar, text="Open Folder", command=self.open_root).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Play Replay", command=self.play_selected).pack(side="left", padx=(0, 8))
        ttk.Button(button_bar, text="Export Frames", command=self.export_selected).pack(side="left")

        ttk.Label(detail, textvariable=self.selected_path_var).grid(row=0, column=0, sticky="w")
        self.detail_text = self.text_box(detail, height=24)
        self.detail_text.grid(row=1, column=0, sticky="nsew")
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
        docs = self.section(0, 0, "Quick Access")
        tips = self.section(1, 0, "What To Edit")

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
        ttk.Label(tips, text=copy, justify="left").grid(row=0, column=0, sticky="w")


def main() -> None:
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
