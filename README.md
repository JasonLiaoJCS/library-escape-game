# Library Escape

`Library Escape` is the Python rewrite of the original C++/SDL2 stealth project in this repository. The new version keeps the same assets and map ideas, but moves the game to a continuous-time simulation with:

- `pygame-ce` rendering for human play
- a pure-Python core that can run headless
- `Gymnasium` support for training a single RL agent
- `PettingZoo` support for player-vs-enemy self-play
- YAML/JSON driven config so rewards, observations, and actions are easy to tune
- a desktop GUI for play, training, built-in TensorBoard charts, Elo leaderboards, checkpoint browsing, replay review, and result plotting

## Quick Start

Use Python `3.12`.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[rl,dev]
python -m library_escape.play.human_vs_ai
python -m library_escape.gui.app
python -m library_escape.replay.viewer --replay replays\\sample.ler.gz
```

Documentation:

- Quick operation guide: `docs/OPERATION_QUICKSTART.md`
- Full technical guide: `docs/OPERATION_GUIDE.md`

## Main Commands

```powershell
python -m library_escape.play.human_vs_ai
python -m library_escape.play.ai_vs_ai
python -m library_escape.train.train_enemy
python -m library_escape.train.train_player
python -m library_escape.train.train_selfplay
python -m library_escape.gui.app
pytest
```

## Config Files You Will Edit Most

- Rewards: `configs/rewards.yaml`
- Physics, map path, timers, action mode, observation mode: `configs/env.yaml`
- Training hyperparameters and checkpoint paths: `configs/training.yaml`
- Map layout, spawns, patrol path, escape zone: `configs/map.json`
- Observation feature construction: `library_escape/env/obs_builder.py`
- Action mapping and spaces: `library_escape/env/single_agent_env.py` and `library_escape/env/multi_agent_env.py`
- GUI launcher: `library_escape/gui/app.py`
- Live training ETA/progress callbacks: `library_escape/train/callbacks.py`

## Notes

- Human play uses continuous movement and smooth rendering.
- RL training uses fixed-step physics plus frame skip for stability.
- Training now includes reward shaping inspired by potential-based shaping, vector normalization, checkpoint/eval callbacks, live ETA logging, configurable opponent curriculum, and optional `MaskablePPO`.
- Self-play is league-style in spirit: agents continue training from previous rounds while sampling from historical opponents, and can switch between `league_ppo`, `league_maskable_ppo`, or an external `mappo_recipe` command.
- The project keeps the original image/audio/font assets already stored in this repository.
- The GUI can launch play mode, train enemy/player/self-play, inspect TensorBoard scalars inside the app, auto-build player/enemy Elo tables, browse `.ler.gz` replays, and export replay frames.
- If you only want to play and not train, `pip install -e .` is enough. RL libraries are installed by `.[rl]`.
