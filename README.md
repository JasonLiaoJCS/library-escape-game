# Library Escape

`Library Escape` 現在是完整的 Python 版 2D 潛行遊戲與 RL 訓練平台。原本 repo 內的 C++/SDL2 程式保留作為歷史參考，但目前的主要遊戲、環境封裝、訓練腳本與測試都已經改到 `library_escape/` 這套 Python 架構。

更完整的中文實作說明請看：

- `docs/DETAILED_IMPLEMENTATION_GUIDE.md`
- `docs/OPERATION_GUIDE.md`

## 這次完成了什麼

- 整個遊戲主體改成 Python，採用 `Pygame-CE`
- 遊戲更新改成「固定步長物理 + 可變渲染」，玩家與敵人都是連續移動
- 單智能體 RL 環境：`Gymnasium`
- 雙智能體 RL 環境：`PettingZoo ParallelEnv`
- PPO 訓練：`Stable-Baselines3`
- 人類 vs AI、AI vs AI、self-play 交替訓練都可跑
- reward 完全外部化到 YAML
- observation / action 配置集中化，方便調實驗
- 補了環境 API 測試、幾何測試與 headless smoke 路徑

## 專案結構

```text
library_escape/
├── core/             # 純遊戲邏輯：世界、碰撞、視野錐、導航
├── render/           # Pygame renderer 與 headless null view
├── input/            # 人類鍵盤輸入
├── env/              # Gymnasium / PettingZoo 封裝
├── rewards/          # Reward engine
├── agents/           # Rule-based / PPO controller
├── train/            # 訓練入口
├── play/             # 遊玩入口
└── scripts/          # ELO / 錄影工具

configs/
├── env.yaml          # 地圖、障礙、物理、obs/action 主要設定
├── rewards.yaml      # reward 權重
└── training.yaml     # PPO/self-play 超參數
```

## 安裝

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## 直接遊玩

人類對 AI 敵人：

```powershell
python -m library_escape.play.human_vs_ai
```

如果你已經有訓練好的敵人模型：

```powershell
python -m library_escape.play.human_vs_ai --enemy-model models\enemy\enemy_ppo.zip
```

AI 對 AI：

```powershell
python -m library_escape.play.ai_vs_ai
```

## 強化學習訓練

訓練敵人 PPO：

```powershell
python -m library_escape.train.train_enemy
```

交替 self-play：

```powershell
python -m library_escape.train.train_selfplay
```

ELO 粗估：

```powershell
python -m library_escape.scripts.eval_elo --episodes 20
```

錄影成 PNG frame sequence：

```powershell
python -m library_escape.scripts.record_video --output-dir recordings\demo_frames
```

## 你要調哪裡

Reward 權重：

- `configs/rewards.yaml`
- 例如 `enemy.catch_player`、`player.collect_note`、`player.seen_per_step`

Observation space 組成：

- `configs/env.yaml` 的 `observations:` 區塊
- 例如 `wall_ray_count`、`wall_ray_range`、`include_velocity`、`include_heading`
- 實際向量拼接邏輯在 [library_escape/env/obs_builder.py](library_escape/env/obs_builder.py)

Action space：

- `configs/env.yaml` 的 `actions.scheme`
- `discrete` = `Discrete(9)`，`continuous` = `Box(2,)`
- 動作解碼在 [library_escape/core/actions.py](library_escape/core/actions.py)

地圖、障礙、起點、出口、筆記位置：

- `configs/env.yaml` 的 `world:` 區塊

PPO / self-play 超參數：

- `configs/training.yaml`
- 單智能體看 `single_agent:`
- 交替 self-play 看 `multi_agent:`

## 測試

```powershell
pytest
```

這會檢查：

- 幾何碰撞
- 視野錐遮擋
- Gymnasium API
- PettingZoo Parallel API

## 目前設計取捨

- 多智能體訓練目前採「交替 best-response self-play」而不是直接 MAPPO。原因是這樣更容易維護、也比較穩，對這個 repo 目前的規模更實用。
- `PettingZoo` 環境已經做好，所以未來要換成 MAPPO、CleanRL 或 RLlib 都有明確接口。
- 原始 C++ 檔案沒有刪掉，方便和舊版對照；但新的主路徑已經是 Python。

## 主要使用的官方文件

- Gymnasium: https://gymnasium.farama.org/
- PettingZoo: https://pettingzoo.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Pygame-CE: https://pyga.me/docs/
