# Library Escape

`Library Escape` 是把這個 repo 原本的 **C++ / SDL2 潛行遊戲**，重寫成一套 **Python / pygame / RL** 版本的專案。

現在這個 repo 同時保留兩條線：

- **原版 C++ / SDL2 遊戲**
- **新版 Python 遊戲 + 強化學習訓練系統 + GUI**

如果你是第一次來這個專案，最重要的結論只有兩句：

1. **要玩新版、要訓練 AI、要用 GUI，請用 Python 版。**
2. **Python 版現在已經支援 Collection / Escape 兩種遊戲模式、玩家訓練、敵人訓練、self-play、replay、TensorBoard、Elo leaderboard。**

---

## 1. 這個專案到底在做什麼

這是一個以圖書館為場景的潛行 / 對抗遊戲。

玩家與敵人都在同一張地圖上移動。玩家需要一邊躲避敵人的視野，一邊完成目標；敵人則要巡邏、搜尋、阻止玩家達成目標。

這個 repo 的 Python 版除了能玩遊戲以外，也把整套遊戲包成可訓練的 RL 環境，因此你可以：

- 自己操作玩家玩遊戲
- 觀看 `AI vs AI`
- 單獨訓練敵人
- 單獨訓練玩家
- 讓玩家與敵人自我對抗、互相成長
- 用 GUI 操作 play / train / replay / TensorBoard / Elo leaderboard

---

## 2. 目前有哪些版本

### 原版：C++ / SDL2

原版還保留在 repo 裡，主要位置是：

- [`main.cpp`](./main.cpp)
- [`src/`](./src)
- [`include/`](./include)
- [`CMakeLists.txt`](./CMakeLists.txt)

### 新版：Python / pygame / RL

新版主要在：

- [`library_escape/`](./library_escape)
- [`configs/`](./configs)
- [`docs/`](./docs)
- [`tests/`](./tests)
- [`pyproject.toml`](./pyproject.toml)

如果你要：

- 開遊戲
- 開 GUI
- 訓練模型
- 看 replay
- 看 TensorBoard / leaderboard

都請使用 **Python 版**。

---

## 3. Python 版目前有什麼功能

- `pygame-ce` 視窗化遊戲
- 純 Python 世界模擬，可 headless 執行
- 固定物理步長與連續時間移動
- `Gymnasium` 單智能體環境
- `PettingZoo` 多智能體環境
- `Stable-Baselines3` / `sb3-contrib` / `MaskablePPO`
- 敵人訓練
- 玩家訓練
- self-play
- opponent curriculum
- GUI 訓練面板
- 內建 TensorBoard 曲線面板
- 自動 Elo leaderboard
- replay 錄製 / 回放 / 影格匯出
- 原 repo 內仍存在的背景音樂 / 偵測 / 成績音效

---

## 4. 兩種遊戲模式：Collection 與 Escape

這是目前最重要的概念。

### `Collection`

這是比較接近原本 C++ 玩法的模式。

特性：

- 核心是 **收集與計分**
- 玩家要在時間內盡量收集 `note` / `exam`
- 被敵人偵測到會吃到時間懲罰
- 沒有「跑到出口立即獲勝」這種主目標
- 比較像高壓力的潛行收集局

適合：

- 想玩比較接近原版 C++ 感覺
- 想做「收集效率 vs 潛行」型 RL

### `Escape`

這是比較明確的對抗 / 抓捕 / 逃脫模式。

特性：

- 玩家需要先完成必要目標，再往出口逃脫
- 敵人要阻止玩家達成目標或阻止玩家逃脫
- 這個模式更像對抗型 RL sandbox

適合：

- 想訓練更強的敵我對抗策略
- 想看 self-play 長出更有趣的行為

### 很重要的一個原則

**訓練時用哪個 `Game mode`，播放時通常也要回到同一個 `Game mode`。**

例如：

- 在 `Collection` 訓練出的玩家 checkpoint，最合理的播放方式是回到 `Collection`
- 在 `Escape` 訓練出的敵人 checkpoint，最合理的播放方式是回到 `Escape`

---

## 5. 先看最簡單的使用方式

### 最推薦入口：GUI

```powershell
python -m library_escape.gui.app
```

如果你只記一個指令，就記這個。

GUI 裡可以做的事：

- 開 `Human vs AI`
- 開 `AI vs AI`
- 選 `Collection` / `Escape`
- 選玩家模型 / 敵人模型
- 訓練敵人
- 訓練玩家
- 開始 self-play
- 看訓練 ETA
- 看 TensorBoard 圖
- 產生 Elo leaderboard
- 瀏覽 replay

### 只想直接玩

```powershell
python -m library_escape.play.human_vs_ai --game-mode collection
```

### 只想看 AI 對戰

```powershell
python -m library_escape.play.ai_vs_ai --game-mode escape
```

---

## 6. 安裝

### 需求

- Windows + PowerShell
- Python `3.12`

### 建議安裝流程

先切到專案根目錄：

```powershell
cd "C:\Users\User\Desktop\大四其他\library-escape-game"
```

建立專案內自己的虛擬環境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[rl,dev]"
```

如果你只想玩，不需要 RL 套件，可以改成：

```powershell
pip install -e .
```

### 驗證你有沒有用到正確的 Python

```powershell
python -c "import sys; print(sys.executable)"
```

你應該看到的是專案內的 `.venv`，例如：

```text
C:\Users\User\Desktop\大四其他\library-escape-game\.venv\Scripts\python.exe
```

不是家目錄底下的：

```text
C:\Users\User\.venv\Scripts\python.exe
```

---

## 7. 常見啟動指令

### 開 GUI

```powershell
python -m library_escape.gui.app
```

### 人類玩家對 AI

```powershell
python -m library_escape.play.human_vs_ai --game-mode collection
python -m library_escape.play.human_vs_ai --game-mode escape
```

### 看 AI vs AI

```powershell
python -m library_escape.play.ai_vs_ai --game-mode collection
python -m library_escape.play.ai_vs_ai --game-mode escape
```

### 指定 checkpoint 播放

```powershell
python -m library_escape.play.human_vs_ai --game-mode escape --enemy-model checkpoints\enemy\my_run\models\enemy_latest.zip
python -m library_escape.play.ai_vs_ai --game-mode collection --player-model checkpoints\player_single\my_run\models\player_latest.zip --enemy-model checkpoints\enemy\my_run\models\enemy_latest.zip
```

### replay

錄製：

```powershell
python -m library_escape.play.ai_vs_ai --game-mode escape --record-replay replays\demo_escape.ler.gz
```

回放：

```powershell
python -m library_escape.replay.viewer --replay replays\demo_escape.ler.gz
```

匯出影格：

```powershell
python -m library_escape.replay.viewer --replay replays\demo_escape.ler.gz --export-frames videos\frames\demo_escape
```

---

## 8. 怎麼玩這個遊戲

### `Human vs AI`

```powershell
python -m library_escape.play.human_vs_ai --game-mode collection
```

預設操作：

- `WASD`：移動
- `E`：長按收集
- `R`：重新開始
- `Esc`：離開

目前規則重點：

- 開場有一小段保護時間，不會第一瞬間就被偵測
- 被某個敵人看到時，只會讓**該敵人**短暫停住，不是整隊都停
- 同一次偵測只會扣一次秒數，不會持續重複扣
- 所有模式都不會瞬間拿書
- 同一張桌上若有多個道具，會優先選你實際比較靠近的道具

---

## 9. 怎麼訓練模型

你可以訓練三種東西：

- `enemy`
- `player`
- `selfplay`

而且每一種都可以指定：

- `--game-mode collection`
- `--game-mode escape`

### 9.1 訓練敵人

```powershell
python -m library_escape.train.train_enemy --game-mode collection --preset fast --run-name enemy_collection_v1
python -m library_escape.train.train_enemy --game-mode escape --preset balanced --run-name enemy_escape_v1
```

### 9.2 訓練玩家

```powershell
python -m library_escape.train.train_player --game-mode collection --preset fast --run-name player_collection_v1
python -m library_escape.train.train_player --game-mode escape --preset balanced --run-name player_escape_v1
```

### 9.3 訓練 self-play

```powershell
python -m library_escape.train.train_selfplay --game-mode collection --preset fast --run-name selfplay_collection_v1
python -m library_escape.train.train_selfplay --game-mode escape --preset balanced --run-name selfplay_escape_v1
```

### 9.4 常用訓練參數

敵人 / 玩家訓練：

- `--preset`
- `--game-mode`
- `--timesteps`
- `--n-envs`
- `--seed`
- `--device`
- `--resume`
- `--run-name`
- `--overrides-json`

self-play：

- `--preset`
- `--game-mode`
- `--rounds`
- `--timesteps-per-round`
- `--n-envs`
- `--seed`
- `--device`
- `--run-name`
- `--overrides-json`

### 9.5 訓練完的檔案在哪裡

常見輸出位置：

- 敵人：`checkpoints\enemy\<run-name>\...`
- 玩家：`checkpoints\player_single\<run-name>\...`
- self-play：`checkpoints\selfplay\<run-name>\...`

常見內容：

- `models\*_latest.zip`
- `progress.json`
- `evaluations.npz`
- TensorBoard log
- normalize stats

---

## 10. GUI 怎麼用

GUI 是目前最推薦的操作方式。

```powershell
python -m library_escape.gui.app
```

### Play 頁

你可以：

- 選 `Human vs AI` 或 `AI vs AI`
- 選 `Collection` 或 `Escape`
- 載入玩家 / 敵人 checkpoint
- 啟動遊戲

### Train 頁

你可以：

- 選 `enemy` / `player` / `selfplay`
- 選 `Collection` / `Escape`
- 選 `preset`
- 設定 timesteps / rounds / env 數量
- 開始訓練
- 看訓練進度與 ETA

### TensorBoard 頁

你可以直接在 GUI 裡看常見 scalar 曲線。

### Leaderboard 頁

你可以掃描 checkpoint 目錄，自動建立玩家 / 敵人的 Elo 排名。

### Replay 頁

你可以：

- 載入 `.ler.gz`
- 播放 replay
- 匯出 frames

---

## 11. 目前的主要訓練設計

目前訓練系統是「**訓練角色**」加上「**訓練規則模式**」。

也就是：

- 訓練敵人 in `Collection`
- 訓練敵人 in `Escape`
- 訓練玩家 in `Collection`
- 訓練玩家 in `Escape`
- self-play in `Collection`
- self-play in `Escape`

而不是只有一種通用 reward 硬套全部玩法。

### 訓練演算法

目前主要支援：

- `PPO`
- `MaskablePPO`
- league-style self-play
- opponent curriculum

### 為什麼這樣設計

因為 `Collection` 與 `Escape` 的遊戲目標本來就不同：

- `Collection` 比較重視收集效率、躲偵測、時間壓力
- `Escape` 比較重視完成必要目標、控制風險、逃脫或攔截

所以 reward、預設難度與部分觀測設計現在都是 **mode-aware**。

---

## 12. 你最常需要改的檔案

### 遊戲與環境設定

- [`configs/env.yaml`](./configs/env.yaml)
- [`configs/map.json`](./configs/map.json)
- [`library_escape/game_modes.py`](./library_escape/game_modes.py)

### reward

- [`configs/rewards_collection.yaml`](./configs/rewards_collection.yaml)
- [`configs/rewards_escape.yaml`](./configs/rewards_escape.yaml)
- [`library_escape/rewards/reward_fns.py`](./library_escape/rewards/reward_fns.py)

### observation / action

- [`library_escape/env/obs_builder.py`](./library_escape/env/obs_builder.py)
- [`library_escape/env/single_agent_env.py`](./library_escape/env/single_agent_env.py)
- [`library_escape/env/multi_agent_env.py`](./library_escape/env/multi_agent_env.py)
- [`library_escape/env/action_masking.py`](./library_escape/env/action_masking.py)

### 訓練

- [`configs/training.yaml`](./configs/training.yaml)
- [`library_escape/train/common.py`](./library_escape/train/common.py)
- [`library_escape/train/single_agent_runner.py`](./library_escape/train/single_agent_runner.py)
- [`library_escape/train/train_selfplay.py`](./library_escape/train/train_selfplay.py)

### GUI

- [`library_escape/gui/app.py`](./library_escape/gui/app.py)

---

## 13. 音效狀態

Python 版現在已經把 repo 裡現有的原版音效接回來：

- `audio/game_bgm.wav`
- `audio/detected_by_enemy.mp3`
- `audio/score_*.mp3`

目前已接回：

- 遊戲背景音樂
- 被敵人偵測音效
- 結算成績音效

另外，原本 C++ 版還引用了：

- `sounds/collect.wav`
- `sounds/powerup.wav`

但這兩個檔案目前不在 repo 裡，所以 Python 版保留了對應 hook，但目前會安全地靜音略過，不會導致崩潰。

---

## 14. 驗證與測試

完整測試：

```powershell
pytest -q
```

常用 smoke tests：

```powershell
python -m library_escape.play.human_vs_ai --hidden-window --max-seconds 1 --game-mode collection
python -m library_escape.play.ai_vs_ai --hidden-window --max-seconds 1 --game-mode escape
python -m library_escape.train.train_enemy --game-mode collection --preset fast --timesteps 64 --n-envs 1
python -m library_escape.train.train_player --game-mode escape --preset fast --timesteps 64 --n-envs 1
python -m library_escape.train.train_selfplay --game-mode escape --preset fast --rounds 1 --timesteps-per-round 64 --n-envs 1
```

---

## 15. 常見問題

### Q1. 為什麼 `pip install -e .[rl,dev]` 說目前資料夾不是 Python project？

因為你不在專案根目錄。

正確做法：

```powershell
cd "C:\Users\User\Desktop\大四其他\library-escape-game"
pip install -e ".[rl,dev]"
```

### Q2. 為什麼 PowerShell 前面顯示 `(.venv)`，但我還是跑到錯的 Python？

因為你可能同時有很多個 `.venv`。  
最可靠的檢查方式永遠是：

```powershell
python -c "import sys; print(sys.executable)"
```

### Q3. 為什麼我訓練完的 checkpoint 播起來怪怪的？

常見原因是：

- 你訓練於 `Collection`，卻拿去 `Escape` 播
- 或反過來

最穩妥的原則是：

**在哪個 `Game mode` 訓練，就先回那個 `Game mode` 播放。**

### Q4. 沒有載入 checkpoint 的 `AI vs AI` 是不是壞掉了？

不是。  
如果沒有載入 checkpoint，會改用內建 rule-based baseline，不是未訓練神經網路。

---

## 16. 更詳細的文件

如果你想看完整技術手冊，請看：

- [`docs/OPERATION_GUIDE.md`](./docs/OPERATION_GUIDE.md)

如果你只想看偏操作的快速手冊，請看：

- [`docs/OPERATION_QUICKSTART.md`](./docs/OPERATION_QUICKSTART.md)

---

## 17. 一組最推薦的新手流程

1. 安裝環境

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[rl,dev]"
```

2. 開 GUI

```powershell
python -m library_escape.gui.app
```

3. 先用 `Human vs AI` 玩一局 `Collection`

4. 再看 `AI vs AI` 的 `Escape`

5. 接著開 `Train`
   - 先訓練 `enemy`
   - 再訓練 `player`
   - 最後跑 `selfplay`

6. 訓練完之後，把 checkpoint 載回 `Play`

7. 用 replay / TensorBoard / leaderboard 看結果

---

## 18. License

本專案沿用 repo 目前的授權與素材使用脈絡；如需對外散布，請先自行確認原始素材與上游專案授權條件。
