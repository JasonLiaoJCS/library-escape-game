# Library Escape 完整操作與技術手冊

這份文件是目前這個 repo 的完整說明書。

如果你現在只想知道「怎麼操作」，先看：

- [`docs/OPERATION_QUICKSTART.md`](./OPERATION_QUICKSTART.md)

目標是回答下面這些問題：

- 舊版 C++ 遊戲在哪裡
- 新版 Python 遊戲在哪裡
- 我到底改了什麼
- 我要怎麼執行現在的遊戲
- 我要怎麼開始訓練 RL
- Action space / Observation space / Reward 要去哪裡改
- 訓練結果要去哪裡看
- GUI、TensorBoard、Leaderboard、Replay 要怎麼用

---

## 0. 先講最重要的結論
1. 保留原本 repo 裡面的 **C++ / SDL2 舊版專案**
2. 在同一個 repo 裡新增一套 **完整的 Python 重寫版**
3. 把玩法核心改成 **連續時間 / 固定物理步長**
4. 加入 **Gymnasium / PettingZoo / Stable-Baselines3 / Maskable PPO**
5. 加入 **GUI、ETA、TensorBoard 面板、Replay、Elo leaderboard**

所以現在這個 repo 同時有兩套東西：

- 舊版：C++ / SDL2
- 新版：Python / pygame / RL

如果你要玩、要訓練、要用新功能，**請用 Python 版**。

---

## 1. 專案現在的整體結構

### 1.1 舊版 C++ / SDL2 專案

舊版主要在這些位置：

- 入口：[`main.cpp`](../main.cpp)
- 建置：[`CMakeLists.txt`](../CMakeLists.txt)
- C++ 原始碼：[`src/`](../src)
- C++ 標頭：[`include/`](../include)
- SDL 相關：[`SDL_lib/`](../SDL_lib)
- 既有 debug build：[`cmake-build-debug/my_Final_project_game.exe`](../cmake-build-debug/my_Final_project_game.exe)

舊版主要檔案例如：

- [`src/engine.cpp`](../src/engine.cpp)
- [`src/playground.cpp`](../src/playground.cpp)
- [`src/enemy.cpp`](../src/enemy.cpp)
- [`src/vision.cpp`](../src/vision.cpp)
- [`src/menu.cpp`](../src/menu.cpp)
- [`include/engine.h`](../include/engine.h)
- [`include/enemy.h`](../include/enemy.h)
- [`include/vision.h`](../include/vision.h)

### 1.2 新版 Python 專案

新版主要在這些位置：

- Python package 根目錄：[`library_escape/`](../library_escape)
- 遊戲設定：[`configs/`](../configs)
- 文件：[`docs/`](../docs)
- 輔助腳本：[`scripts/`](../scripts)
- 測試：[`tests/`](../tests)
- Python 專案設定：[`pyproject.toml`](../pyproject.toml)

Python 版主要模組：

- 核心世界與物理：[`library_escape/core/`](../library_escape/core)
- RL 環境：[`library_escape/env/`](../library_escape/env)
- 獎勵：[`library_escape/rewards/`](../library_escape/rewards)
- 訓練：[`library_escape/train/`](../library_escape/train)
- GUI：[`library_escape/gui/`](../library_escape/gui)
- 播放模式：[`library_escape/play/`](../library_escape/play)
- Replay：[`library_escape/replay/`](../library_escape/replay)
- Elo / 評估：[`library_escape/eval/`](../library_escape/eval)

---

## 2. 我到底做了什麼

這次完成的事情可以分成 8 大塊。

### 2.1 把遊戲主體重寫成 Python

我新增了一套 Python 遊戲架構，核心不再依賴 SDL2 / C++。

關鍵檔案：

- 設定讀取：[`library_escape/config.py`](../library_escape/config.py)
- 核心世界：[`library_escape/core/world.py`](../library_escape/core/world.py)
- 物理：[`library_escape/core/physics.py`](../library_escape/core/physics.py)
- 行動映射：[`library_escape/core/actions.py`](../library_escape/core/actions.py)
- pygame 渲染：[`library_escape/render/pygame_view.py`](../library_escape/render/pygame_view.py)

### 2.2 把原本比較偏「離散刷新感」的遊戲流程改成連續時間

新版的玩法不是每次只做一個大刷新，而是：

- 用固定物理頻率 `physics_hz`
- 每次物理子步進用 `physics_dt = 1 / physics_hz`
- 視覺更新用 `render_fps`
- RL 為了穩定訓練，另外用 `rl_frame_skip`

換句話說，現在遊戲邏輯和畫面更新是分開的。

### 2.3 把地圖與設定改成 YAML / JSON

原本很多東西比較像硬編碼在 C++ 裡。

我把它拆成配置檔：

- 環境設定：[`configs/env.yaml`](../configs/env.yaml)
- 地圖：[`configs/map.json`](../configs/map.json)
- 獎勵：[`configs/rewards.yaml`](../configs/rewards.yaml)
- 訓練：[`configs/training.yaml`](../configs/training.yaml)

### 2.4 加入單智能體與多智能體 RL 環境

我做了兩層 RL 介面：

- Gymnasium 單智能體：[`library_escape/env/single_agent_env.py`](../library_escape/env/single_agent_env.py)
- PettingZoo 多智能體：[`library_escape/env/multi_agent_env.py`](../library_escape/env/multi_agent_env.py)

### 2.5 加入完整 RL 訓練流程

我加了：

- 敵人訓練
- 玩家訓練
- 自我對戰 self-play
- PPO
- Maskable PPO
- opponent curriculum
- VecNormalize
- checkpoint / eval / ETA / progress logging

關鍵檔案：

- 訓練共用工具：[`library_escape/train/common.py`](../library_escape/train/common.py)
- 單智能體訓練主流程：[`library_escape/train/single_agent_runner.py`](../library_escape/train/single_agent_runner.py)
- 敵人訓練入口：[`library_escape/train/train_enemy.py`](../library_escape/train/train_enemy.py)
- 玩家訓練入口：[`library_escape/train/train_player.py`](../library_escape/train/train_player.py)
- Self-play：[`library_escape/train/train_selfplay.py`](../library_escape/train/train_selfplay.py)
- ETA / progress callback：[`library_escape/train/callbacks.py`](../library_escape/train/callbacks.py)

### 2.6 加入 reward shaping

我沒有只做「事件式 reward」而已，而是加了：

- terminal reward
- dense reward / penalty
- potential-based shaping
- anti-exploit penalty
- zero-sum mixing

關鍵檔案：

- Reward config：[`configs/rewards.yaml`](../configs/rewards.yaml)
- Reward code：[`library_escape/rewards/reward_fns.py`](../library_escape/rewards/reward_fns.py)

### 2.7 加入 GUI

GUI 現在可以做這些事：

- 啟動遊戲
- 選人類 vs AI / AI vs AI
- 選 checkpoint
- 啟動敵人訓練 / 玩家訓練 / self-play
- 看 live ETA
- 看 built-in TensorBoard scalar 圖
- 建 Elo leaderboard
- 瀏覽 replay
- 匯出 replay frames

GUI 主檔：

- [`library_escape/gui/app.py`](../library_escape/gui/app.py)

### 2.8 加入 Replay / Leaderboard / TensorBoard 面板

新增功能：

- Replay 錄製
- Replay 回放
- Replay 匯出 frames
- 自動掃 checkpoint 做 Elo 排名
- GUI 內建 TensorBoard scalar 面板

關鍵檔案：

- Replay I/O：[`library_escape/replay/io.py`](../library_escape/replay/io.py)
- Replay viewer：[`library_escape/replay/viewer.py`](../library_escape/replay/viewer.py)
- Elo 核心：[`library_escape/eval/elo.py`](../library_escape/eval/elo.py)
- 模型掃描：[`library_escape/eval/registry.py`](../library_escape/eval/registry.py)
- Elo script：[`scripts/eval_elo.py`](../scripts/eval_elo.py)
- TensorBoard 讀取：[`library_escape/gui/tensorboard_data.py`](../library_escape/gui/tensorboard_data.py)

---

## 3. 舊版 C++ 是怎麼運作的

舊版入口在 [`main.cpp`](../main.cpp)。

核心 loop 很典型：

```cpp
while (game_is_running) {
    process_input();
    game_is_running = update();
    render();
}
```

也就是：

1. 收輸入
2. 更新遊戲
3. 畫畫面

這是一個典型 SDL 遊戲主迴圈。

舊版 build 設定在 [`CMakeLists.txt`](../CMakeLists.txt)，專案名稱是 `my_Final_project_game`。

這一輪我**沒有重新驗證舊版 C++ build**，所以你可以把它當成 legacy 保留版。

目前 repo 裡已經有一個現成 exe：

- [`cmake-build-debug/my_Final_project_game.exe`](../cmake-build-debug/my_Final_project_game.exe)

如果你真的要碰舊版 C++，請看：

- [`main.cpp`](../main.cpp)
- [`CMakeLists.txt`](../CMakeLists.txt)
- [`src/`](../src)
- [`include/`](../include)

---

## 4. 新版 Python 是怎麼運作的

### 4.1 核心設計

新版把遊戲拆成 4 層：

1. `core`
2. `render`
3. `play`
4. `train / env`

對應如下：

- 遊戲真實狀態在 [`library_escape/core/world.py`](../library_escape/core/world.py)
- 碰撞與幾何在 [`library_escape/core/physics.py`](../library_escape/core/physics.py)
- 畫面在 [`library_escape/render/pygame_view.py`](../library_escape/render/pygame_view.py)
- 人類遊玩 / AI 對戰入口在 [`library_escape/play/`](../library_escape/play)
- RL 環境與訓練在 [`library_escape/env/`](../library_escape/env) 和 [`library_escape/train/`](../library_escape/train)

### 4.2 這樣拆的好處

- 遊戲邏輯可以不開畫面直接跑
- 訓練時可以 headless
- 同一套世界模型可以拿來：
  - 人類遊玩
  - AI 對戰
  - 單智能體訓練
  - 多智能體訓練
  - Replay 回放

---

## 5. 我是怎麼把「離散時間感」改成「連續時間」的

### 5.1 不是把世界直接改成完全 variable timestep

我沒有讓物理直接跟著每一幀的 real-time `dt` 任意變化。

我做的是：

- **render** 用 real-time
- **physics** 用固定步長

這樣比較穩。

### 5.2 新版的時間參數在哪裡

在 [`configs/env.yaml`](../configs/env.yaml)：

```yaml
timing:
  render_fps: 60
  physics_hz: 120
  rl_frame_skip: 4
```

目前預設：

- `render_fps = 60`
- `physics_hz = 120`
- `rl_frame_skip = 4`

### 5.3 世界內部怎麼算

在 [`library_escape/core/world.py`](../library_escape/core/world.py)：

- `physics_dt = 1 / physics_hz`
- 每次 `world.step(...)` 可以做多個子步
- 真正碰撞與位移在 `_substep(...)`

### 5.4 播放模式怎麼跑

在 [`library_escape/play/human_vs_ai.py`](../library_escape/play/human_vs_ai.py) 和 [`library_escape/play/ai_vs_ai.py`](../library_escape/play/ai_vs_ai.py)：

- 用 `pygame.time.Clock()` 取得每幀 wall time
- 把 `frame_dt` 累積到 `accumulator`
- 當 `accumulator >= physics_dt` 時就多跑一次物理子步

也就是：

- 畫面可以 60 FPS
- 物理可以 120 Hz
- 玩家和敵人都能順暢連續移動

### 5.5 RL 為什麼又有 frame skip

RL 訓練時不需要每個物理子步都讓 agent 做一次決策，不然太細、太慢、太難學。

所以我保留了：

- 物理：高頻率
- 決策：每 `rl_frame_skip` 個物理子步做一次

這個設計在 stealth / pursuit / navigation 類遊戲很常見。

---

## 6. 現在我要執行哪一套

如果你要：

- 玩遊戲
- 看 GUI
- 訓練 RL
- 看 Replay
- 用 TensorBoard / Elo

請用 **Python 版**。

### 6.1 Python 版執行前準備

目標 Python 版本：

- `Python 3.12`

建立虛擬環境：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[rl,dev]
```

### 6.2 直接啟動 GUI

最推薦：

```powershell
python -m library_escape.gui.app
```

或：

```powershell
library-escape-gui
```

### 6.3 直接玩遊戲

人類對 AI：

```powershell
python -m library_escape.play.human_vs_ai
```

指定敵人 checkpoint：

```powershell
python -m library_escape.play.human_vs_ai --enemy-model checkpoints\enemy\your_run\models\enemy_latest.zip
```

AI 對 AI：

```powershell
python -m library_escape.play.ai_vs_ai
```

指定雙方模型：

```powershell
python -m library_escape.play.ai_vs_ai --player-model checkpoints\player_single\your_run\models\player_latest.zip --enemy-model checkpoints\enemy\your_run\models\enemy_latest.zip
```

### 6.4 錄 Replay

人類對 AI 時錄 replay：

```powershell
python -m library_escape.play.human_vs_ai --record-replay replays\my_session.ler.gz
```

AI 對 AI 時錄 replay：

```powershell
python -m library_escape.play.ai_vs_ai --record-replay replays\ai_match.ler.gz
```

### 6.5 播放 Replay

```powershell
python -m library_escape.replay.viewer --replay replays\my_session.ler.gz
```

匯出影格：

```powershell
python -m library_escape.replay.viewer --replay replays\my_session.ler.gz --export-frames videos\frames\my_session
```

### 6.6 跑測試

```powershell
pytest -q
```

---

## 7. GUI 怎麼用

GUI 主檔：

- [`library_escape/gui/app.py`](../library_escape/gui/app.py)

目前主要頁籤如下。

### 7.1 Play

用途：

- 開 `Human vs Rule Enemy`
- 開 `Human vs Enemy Checkpoint`
- 開 `AI vs AI`
- 指定 `player` / `enemy` checkpoint
- 開啟 replay 錄製

### 7.2 Train

用途：

- 訓練 `enemy`
- 訓練 `player`
- 訓練 `selfplay`
- 選 preset
- 改 seed / timesteps / rounds / n-envs / device
- 改 algorithm：
  - `ppo`
  - `maskable_ppo`
  - `league_ppo`
  - `league_maskable_ppo`
  - `mappo_recipe`
- 改 opponent curriculum 權重
- 看 live ETA / FPS / reward

### 7.3 Results

用途：

- 掃描舊 run
- 看 `training_summary.json`
- 看 training / eval 曲線
- 開 run 資料夾
- 啟動 AI vs AI
- 啟動 TensorBoard server

### 7.4 TensorBoard

用途：

- 直接讀 `tb/` 裡的 event files
- 在 GUI 內畫 scalar 曲線

### 7.5 Leaderboard

用途：

- 掃描 checkpoints
- 自動對戰
- 建 player / enemy 的 Elo 排名

### 7.6 Replay

用途：

- 掃描 `.ler.gz`
- 顯示 replay metadata
- 播放 replay
- 匯出 frames

### 7.7 Config

用途：

- 快速打開：
  - `env.yaml`
  - `rewards.yaml`
  - `training.yaml`
  - `map.json`
  - 本文件

---

## 8. Python 版重要資料夾總覽

### 8.1 `library_escape/core`

這是遊戲最核心的資料與規則。

重要檔案：

- [`actions.py`](../library_escape/core/actions.py)
- [`character.py`](../library_escape/core/character.py)
- [`collectible.py`](../library_escape/core/collectible.py)
- [`enemy.py`](../library_escape/core/enemy.py)
- [`events.py`](../library_escape/core/events.py)
- [`obstacle.py`](../library_escape/core/obstacle.py)
- [`physics.py`](../library_escape/core/physics.py)
- [`world.py`](../library_escape/core/world.py)

### 8.2 `library_escape/render`

畫面層。

重要檔案：

- [`pygame_view.py`](../library_escape/render/pygame_view.py)
- [`null_view.py`](../library_escape/render/null_view.py)

### 8.3 `library_escape/play`

遊戲入口。

重要檔案：

- [`human_vs_ai.py`](../library_escape/play/human_vs_ai.py)
- [`ai_vs_ai.py`](../library_escape/play/ai_vs_ai.py)

### 8.4 `library_escape/env`

RL 環境層。

重要檔案：

- [`single_agent_env.py`](../library_escape/env/single_agent_env.py)
- [`multi_agent_env.py`](../library_escape/env/multi_agent_env.py)
- [`obs_builder.py`](../library_escape/env/obs_builder.py)
- [`action_masking.py`](../library_escape/env/action_masking.py)

### 8.5 `library_escape/rewards`

Reward 計算。

重要檔案：

- [`reward_fns.py`](../library_escape/rewards/reward_fns.py)

### 8.6 `library_escape/train`

訓練主流程。

重要檔案：

- [`callbacks.py`](../library_escape/train/callbacks.py)
- [`common.py`](../library_escape/train/common.py)
- [`single_agent_runner.py`](../library_escape/train/single_agent_runner.py)
- [`train_enemy.py`](../library_escape/train/train_enemy.py)
- [`train_player.py`](../library_escape/train/train_player.py)
- [`train_selfplay.py`](../library_escape/train/train_selfplay.py)

### 8.7 `library_escape/agents`

推論 / 規則式對手。

重要檔案：

- [`rule_based_enemy.py`](../library_escape/agents/rule_based_enemy.py)
- [`rule_based_player.py`](../library_escape/agents/rule_based_player.py)
- [`ppo_agent.py`](../library_escape/agents/ppo_agent.py)
- [`opponent_pool.py`](../library_escape/agents/opponent_pool.py)

### 8.8 `library_escape/replay`

Replay 錄製與回放。

重要檔案：

- [`io.py`](../library_escape/replay/io.py)
- [`viewer.py`](../library_escape/replay/viewer.py)

### 8.9 `library_escape/eval`

評估與排行榜。

重要檔案：

- [`elo.py`](../library_escape/eval/elo.py)
- [`registry.py`](../library_escape/eval/registry.py)

---

## 9. Action Space 要去哪裡改

### 9.1 目前支援兩種 action mode

在 [`configs/env.yaml`](../configs/env.yaml)：

```yaml
action:
  type: discrete
  continuous_scale: 1.0
```

支援：

- `discrete`
- `continuous`

### 9.2 `discrete` 模式

目前是 9 個動作：

- `0`: 停止
- `1`: 上
- `2`: 右上
- `3`: 右
- `4`: 右下
- `5`: 下
- `6`: 左下
- `7`: 左
- `8`: 左上

定義位置：

- [`library_escape/core/actions.py`](../library_escape/core/actions.py)

### 9.3 `continuous` 模式

目前是 2D 向量：

- `x`
- `y`

最後會被 clamp 到最大長度。

同樣在：

- [`library_escape/core/actions.py`](../library_escape/core/actions.py)

### 9.4 RL 環境的 action_space 在哪裡宣告

- Gymnasium：[`library_escape/env/single_agent_env.py`](../library_escape/env/single_agent_env.py)
- PettingZoo：[`library_escape/env/multi_agent_env.py`](../library_escape/env/multi_agent_env.py)

### 9.5 什麼時候不能改成 continuous

如果你要用：

- `MaskablePPO`
- `league_maskable_ppo`

那就必須保持：

```yaml
action:
  type: discrete
```

因為 invalid action masking 是做在離散動作上的。

---

## 10. Observation Space 要去哪裡改

### 10.1 設定入口

在 [`configs/env.yaml`](../configs/env.yaml)：

```yaml
observation:
  type: vector
  partial_observability: true
  normalize: true
  wall_rays: 8
  player_enemy_rays: 3
  max_ray_distance: 8.0
```

### 10.2 實際組裝程式碼

在：

- [`library_escape/env/obs_builder.py`](../library_escape/env/obs_builder.py)

### 10.3 敵人 observation 目前包含什麼

`build_enemy_obs()` 目前包含：

1. enemy x
2. enemy y
3. enemy facing x
4. enemy facing y
5. relative player x
6. relative player y
7. player 是否可見
8. 剩餘 note 比例
9. 剩餘 exam 比例
10. 剩餘時間比例
11. `wall_rays` 個牆面 ray

預設 `wall_rays = 8`，所以敵人 observation 維度預設是：

- `10 + 8 = 18`

### 10.4 玩家 observation 目前包含什麼

`build_player_obs()` 目前包含：

1. player x
2. player y
3. player facing x
4. player facing y
5. relative enemy x
6. relative enemy y
7. enemy 是否可見
8. nearest target dx
9. nearest target dy
10. 剩餘時間比例
11. 是否已可逃脫
12. coffee timer 比例
13. enemy freeze timer 比例
14. `player_enemy_rays` 個扇形 ray
15. `wall_rays` 個牆面 ray

預設：

- `player_enemy_rays = 3`
- `wall_rays = 8`

所以玩家 observation 維度預設是：

- `13 + 3 + 8 = 24`

### 10.5 你想改 observation 時怎麼做

#### 情況 A：只想改參數數量

改：

- [`configs/env.yaml`](../configs/env.yaml)

例如：

- `wall_rays`
- `player_enemy_rays`
- `partial_observability`
- `max_ray_distance`

#### 情況 B：想加新 feature

改：

- [`library_escape/env/obs_builder.py`](../library_escape/env/obs_builder.py)

常見做法：

- 在 `build_player_obs()` 裡新增 feature
- 在 `build_enemy_obs()` 裡新增 feature
- 同步調整 `player_obs_dim()` / `enemy_obs_dim()`

---

## 11. Reward 系統是怎麼算的

Reward 設定檔：

- [`configs/rewards.yaml`](../configs/rewards.yaml)

Reward 程式碼：

- [`library_escape/rewards/reward_fns.py`](../library_escape/rewards/reward_fns.py)

### 11.1 目前 reward 的總體流程

每次 `env.step(...)` 之後，系統會：

1. 先收集本步事件 `events`
2. 讀取 `prev_metrics`
3. 讀取 `next_metrics`
4. 分別計算：
   - `player_reward`
   - `enemy_reward`
5. 再做：
   - anti-exploit penalty
   - zero-sum mixing
   - reward clipping

### 11.2 目前 enemy reward 權重

目前在 [`configs/rewards.yaml`](../configs/rewards.yaml)：

```yaml
enemy:
  catch_player: 140.0
  player_in_cone_per_step: 0.70
  deny_escape_bonus: 45.0
  time_penalty: -0.008
  wall_penalty: -0.12
  idle_penalty: -0.01
  stalemate: -10.0
```

enemy potential shaping 權重：

```yaml
enemy:
  potential:
    enabled: true
    capture_progress: 2.80
    visibility_lock: 0.75
    escape_pressure: 1.40
```

### 11.3 目前 player reward 權重

目前在 [`configs/rewards.yaml`](../configs/rewards.yaml)：

```yaml
player:
  collect_note: 28.0
  collect_exam: 42.0
  collect_coffee: 8.0
  collect_freeze: 10.0
  escape: 160.0
  seen_per_step: -0.35
  caught: -160.0
  time_bonus: 0.010
  wall_penalty: -0.12
  idle_penalty: -0.01
  stalemate: -10.0
```

player potential shaping 權重：

```yaml
player:
  potential:
    enabled: true
    objective_progress: 2.50
    target_navigation: 2.20
    escape_navigation: 3.20
    stealth_margin: 1.00
```

### 11.4 全域 reward 參數

```yaml
global:
  gamma: 0.99
  clip_range: 250.0
  zero_sum_mix: 0.10
```

以及 anti-exploit：

```yaml
anti_exploit:
  no_progress_penalty: -0.01
```

### 11.5 enemy reward 公式

目前 enemy reward 大致是：

```text
enemy_reward =
  visibility_scale * player_in_cone_per_step
  + enemy_wall_hits * wall_penalty
  + time_penalty
  + idle_penalty(if idle)
  + catch_player(if caught)
  + deny_escape_bonus(if timeout or stalemate)
  + stalemate(if stalemate)
  + potential_delta_enemy
```

其中：

```text
potential_delta_enemy = gamma * Phi_enemy(next) - Phi_enemy(prev)
```

而：

```text
Phi_enemy =
  capture_progress_weight * capture_progress
  + visibility_lock_weight * player_visible
  + escape_pressure_weight * escape_pressure
```

### 11.6 player reward 公式

目前 player reward 大致是：

```text
player_reward =
  note_count * collect_note
  + exam_count * collect_exam
  + coffee_count * collect_coffee
  + freeze_count * collect_freeze
  + visibility_scale * seen_per_step
  + player_wall_hits * wall_penalty
  + time_bonus
  + idle_penalty(if idle)
  + escape(if escaped)
  + caught(if caught)
  + stalemate(if stalemate)
  + potential_delta_player
```

其中：

```text
potential_delta_player = gamma * Phi_player(next) - Phi_player(prev)
```

而：

```text
Phi_player =
  objective_progress_weight * objective_progress
  + target_or_escape_navigation_weight * navigation_progress
  + stealth_margin_weight * stealth_margin
```

### 11.7 anti-exploit 怎麼算

如果這一步 `events.progress_made == false`，那麼：

```text
player_reward += no_progress_penalty
enemy_reward += no_progress_penalty
```

### 11.8 zero-sum mixing 怎麼算

如果 `zero_sum_mix > 0`，會做：

```text
player_adv = player_reward - enemy_reward
enemy_adv = enemy_reward - player_reward

player_reward = (1 - mix) * player_reward + mix * player_adv
enemy_reward = (1 - mix) * enemy_reward + mix * enemy_adv
```

### 11.9 reward clipping 怎麼算

最後如果 `clip_range > 0`：

```text
reward = clamp(reward, -clip_range, clip_range)
```

### 11.10 如果我想改 reward

#### 只改權重

改：

- [`configs/rewards.yaml`](../configs/rewards.yaml)

#### 改 reward 算法本身

改：

- [`library_escape/rewards/reward_fns.py`](../library_escape/rewards/reward_fns.py)

#### 改 shaping 所用的狀態量

改：

- [`library_escape/core/world.py`](../library_escape/core/world.py)

因為 `transition_metrics()` 是 shaping 依賴的核心輸入。

---

## 12. 環境參數要去哪裡改

主要改 [`configs/env.yaml`](../configs/env.yaml)。

### 12.1 timing

```yaml
timing:
  render_fps: 60
  physics_hz: 120
  rl_frame_skip: 4
  max_episode_seconds: 60.0
  stalemate_seconds: 8.0
```

你可以調：

- 畫面更新率
- 物理更新率
- RL frame skip
- 單局長度
- 僵局判定時間

### 12.2 world

```yaml
world:
  player_radius: 0.30
  enemy_radius: 0.30
  capture_radius: 0.55
  interaction_radius: 1.10
  player_speed: 4.20
  enemy_speed: 3.60
  coffee_speed_multiplier: 1.50
  coffee_duration_seconds: 15.0
  freeze_duration_seconds: 5.0
  require_all_notes_to_escape: true
  require_all_exams_to_escape: false
```

你可以調：

- 玩家 / 敵人大小
- 抓到判定距離
- 撿道具判定距離
- 玩家 / 敵人速度
- coffee / freeze 效果
- 逃脫條件

### 12.3 enemy

```yaml
enemy:
  vision_range: 6.0
  vision_angle_deg: 70.0
  chase_speed_multiplier: 1.10
  reaction_interval_seconds: 0.20
```

你可以調：

- 視野距離
- 視野角度
- 看見玩家時加速倍率

### 12.4 collectibles

```yaml
collectibles:
  notes: 6
  exams: 2
  coffee: 1
  freeze: 1
```

你可以直接改場上道具數量。

### 12.5 observation

前面第 10 節已經詳細說明。

### 12.6 action

前面第 9 節已經詳細說明。

### 12.7 randomization

```yaml
randomization:
  enabled: false
  min_agent_spawn_distance: 8.0
  player_speed_scale_range: [0.95, 1.05]
  enemy_speed_scale_range: [0.95, 1.08]
  vision_range_scale_range: [0.92, 1.08]
  vision_angle_jitter_deg: 8.0
```

這塊是為了訓練泛化能力。

你可以讓每局：

- 玩家出生位置不同
- 敵人出生位置不同
- 移速輕微變動
- 視野距離與角度輕微變動

### 12.8 ui

```yaml
ui:
  show_detection_meter: true
  show_last_seen_marker: true
  show_enemy_mode: true
  show_tactical_panel: true
```

這些只影響畫面顯示，不影響核心訓練。

---

## 13. 地圖要去哪裡改

地圖在：

- [`configs/map.json`](../configs/map.json)

你可以改：

- 障礙物
- 玩家出生點
- 敵人出生點
- 敵人 patrol waypoints
- collectibles spawn points
- escape zone

這是新版與舊版差異很大的地方之一：

原本比較偏硬編碼或散落在 C++ 類別裡的場景內容，現在統一拉到 JSON。

---

## 14. RL 環境是怎麼接起來的

### 14.1 單智能體環境

檔案：

- [`library_escape/env/single_agent_env.py`](../library_escape/env/single_agent_env.py)

用途：

- 訓練 `enemy`
- 訓練 `player`

流程：

1. `reset()` 建世界
2. `step(action)` 把 action 轉成速度向量
3. 對手由 scripted opponent 控制
4. `world.step(...)`
5. reward engine 計算 reward
6. 回傳 observation / reward / done / info

### 14.2 多智能體環境

檔案：

- [`library_escape/env/multi_agent_env.py`](../library_escape/env/multi_agent_env.py)

用途：

- self-play
- competitive multi-agent experiments

### 14.3 Action masking

檔案：

- [`library_escape/env/action_masking.py`](../library_escape/env/action_masking.py)

用途：

- 對離散 action 產生 valid mask
- 避免 agent 一直選會撞牆、完全不會動的無效動作
- `MaskablePPO` 會用這個 mask

---

## 15. RL 訓練系統是怎麼做的

### 15.1 單智能體訓練入口

- 敵人：[`library_escape/train/train_enemy.py`](../library_escape/train/train_enemy.py)
- 玩家：[`library_escape/train/train_player.py`](../library_escape/train/train_player.py)

這兩個檔其實很薄，真正主流程在：

- [`library_escape/train/single_agent_runner.py`](../library_escape/train/single_agent_runner.py)

### 15.2 Self-play 入口

- [`library_escape/train/train_selfplay.py`](../library_escape/train/train_selfplay.py)

### 15.3 訓練共用邏輯

在：

- [`library_escape/train/common.py`](../library_escape/train/common.py)

裡面處理了：

- algorithm 選擇
- preset 套用
- overrides deep merge
- opponent 建立
- vec env 建立
- VecNormalize 載入與儲存
- training summary 寫出
- model metadata 寫出

### 15.4 ETA 怎麼算

在：

- [`library_escape/train/callbacks.py`](../library_escape/train/callbacks.py)

做法：

1. 記錄訓練開始時間
2. 看目前 `num_timesteps`
3. 算 `fps = timesteps / elapsed_time`
4. 算剩餘步數
5. 算 `eta_seconds = remaining_steps / fps`
6. 寫到 `progress.json`

所以 ETA 是根據你這台機器的實際速度推估的。

### 15.5 訓練時會寫出哪些檔案

每個 run 會有：

- `models/`
- `monitor/`
- `tb/`
- `progress.json`
- `progress_history.jsonl`
- `eval_history.jsonl`
- `training_summary.json`

還可能包含：

- `vecnormalize.pkl`
- `*.obsnorm.npz`
- `*.meta.json`

---

## 16. 訓練演算法與參數要去哪裡改

訓練設定檔：

- [`configs/training.yaml`](../configs/training.yaml)

### 16.1 單智能體目前預設參數

```yaml
single_agent:
  algorithm: ppo
  total_timesteps: 300000
  n_envs: 8
  learning_rate: 0.00025
  n_steps: 1024
  batch_size: 256
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.02
  vf_coef: 0.5
```

### 16.2 Self-play 目前預設參數

```yaml
self_play:
  algorithm: league_ppo
  rounds: 4
  timesteps_per_round: 120000
  n_envs: 4
  learning_rate: 0.00025
  n_steps: 1024
  batch_size: 256
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.02
  vf_coef: 0.5
```

### 16.3 VecNormalize 參數

```yaml
vec_normalize:
  enabled: true
  norm_obs: true
  norm_reward: true
  clip_obs: 10.0
  clip_reward: 10.0
```

### 16.4 你最常需要改的訓練參數

- `algorithm`
- `total_timesteps`
- `timesteps_per_round`
- `n_envs`
- `learning_rate`
- `n_steps`
- `batch_size`
- `gamma`
- `gae_lambda`
- `clip_range`
- `ent_coef`
- `vf_coef`

### 16.5 current supported algorithms

單智能體：

- `ppo`
- `maskable_ppo`

self-play：

- `league_ppo`
- `league_maskable_ppo`
- `mappo_recipe`

### 16.6 `mappo_recipe` 是什麼

這裡要講清楚：

目前 repo 內**沒有原生實作完整 CTDE MAPPO trainer**。

我做的是：

- 在 GUI 和訓練系統裡保留 `mappo_recipe` 入口
- 允許你填一個外部 command
- 讓它把目前 run 的設定寫成 request JSON
- 再去呼叫外部 MAPPO runner

也就是說：

- `ppo` / `maskable_ppo` / `league_maskable_ppo`：是 repo 內直接可跑的
- `mappo_recipe`：是外部整合入口

設定位置：

- [`configs/training.yaml`](../configs/training.yaml)

---

## 17. Opponent Curriculum 怎麼調

### 17.1 單智能體 opponent curriculum

在 [`configs/training.yaml`](../configs/training.yaml)：

```yaml
single_agent:
  opponent_curriculum:
    use_history_pool: true
    max_history_pool: 8
    random_weight: 0.20
    heuristic_weight: 0.60
    history_weight: 0.20
```

意思：

- `random_weight`：隨機對手比例
- `heuristic_weight`：規則式對手比例
- `history_weight`：歷史 checkpoint 對手比例
- `max_history_pool`：最多保留多少歷史模型

### 17.2 Self-play curriculum

在 [`configs/training.yaml`](../configs/training.yaml)：

```yaml
self_play:
  opponent_curriculum:
    bootstrap_random_weight: 0.20
    bootstrap_heuristic_weight: 0.80
    latest_weight: 0.50
    historical_weight: 0.50
    max_history_pool: 6
```

意思：

- bootstrap 初期多用 heuristic/random 對手
- pool 裡最新模型與舊模型可以加權抽樣

### 17.3 程式碼在哪裡

- [`library_escape/agents/opponent_pool.py`](../library_escape/agents/opponent_pool.py)
- [`library_escape/train/common.py`](../library_escape/train/common.py)
- [`library_escape/train/train_selfplay.py`](../library_escape/train/train_selfplay.py)

---

## 18. 我今天要怎麼開始訓練

### 18.1 最簡單方法：用 GUI

```powershell
python -m library_escape.gui.app
```

然後：

1. 進 `Train`
2. 選 mode
3. 選 preset
4. 選 algorithm
5. 按 `Start Training`

### 18.2 CLI：訓練敵人

```powershell
python -m library_escape.train.train_enemy
```

快速測試版：

```powershell
python -m library_escape.train.train_enemy --preset fast
```

自訂 timesteps / envs：

```powershell
python -m library_escape.train.train_enemy --timesteps 300000 --n-envs 8
```

### 18.3 CLI：訓練玩家

```powershell
python -m library_escape.train.train_player
```

### 18.4 CLI：Self-play

```powershell
python -m library_escape.train.train_selfplay
```

自訂 rounds / timesteps-per-round：

```powershell
python -m library_escape.train.train_selfplay --rounds 4 --timesteps-per-round 120000
```

### 18.5 CLI：Maskable PPO

PowerShell 版本建議這樣寫：

```powershell
$json = '{"train":{"algorithm":"maskable_ppo"}}'
python -m library_escape.train.train_enemy --preset balanced --overrides-json $json
```

self-play 的 maskable：

```powershell
$json = '{"train":{"algorithm":"league_maskable_ppo"}}'
python -m library_escape.train.train_selfplay --preset balanced --overrides-json $json
```

### 18.6 CLI：外部 MAPPO recipe

最建議直接改 [`configs/training.yaml`](../configs/training.yaml) 或 GUI。

你需要填：

- `self_play.algorithm: mappo_recipe`
- `self_play.opponent_curriculum.mappo_recipe.external_command`

注意：

- 這個 command 需要你自己準備外部 MAPPO runner
- repo 會把 request JSON 寫出來再呼叫它

---

## 19. 訓練過程中我要注意哪些參數

### 19.1 `n_envs`

- 越大通常越快收資料
- 但吃 CPU / RAM

中端機建議先從：

- `1`
- `2`
- `4`

開始試。

### 19.2 `total_timesteps` / `timesteps_per_round`

- 太小：學不起來
- 太大：很久

建議：

- smoke test：先 `fast`
- 正式跑：再用 `balanced`

### 19.3 `learning_rate`

- 太高：不穩
- 太低：學太慢

### 19.4 `n_steps`

- 影響 rollout 長度
- 太短：更新雜訊大
- 太長：每輪更新更慢

### 19.5 `batch_size`

- 要和 `n_steps * n_envs` 相容
- 太大容易慢
- 太小容易 noisy

### 19.6 `ent_coef`

- 越大越鼓勵探索
- 太低可能早早收斂到糟糕策略

### 19.7 `clip_range`

- PPO 更新幅度控制

### 19.8 Reward 權重

如果出現：

- 玩家卡牆亂磨
- 敵人一直原地晃
- 玩家只會拖時間
- 敵人只會追最近不顧出口壓力

通常優先調：

- `wall_penalty`
- `idle_penalty`
- `no_progress_penalty`
- `capture_progress`
- `target_navigation`
- `escape_navigation`
- `escape_pressure`

---

## 20. 訓練結果要去哪裡看

### 20.1 run 目錄

目前預設輸出：

- 敵人：[`checkpoints/enemy/`](../checkpoints/enemy)
- 玩家：[`checkpoints/player_single/`](../checkpoints/player_single)
- self-play：[`checkpoints/selfplay/`](../checkpoints/selfplay)

### 20.2 一個 run 裡面通常會有

- `models/`
- `monitor/`
- `tb/`
- `progress.json`
- `progress_history.jsonl`
- `eval_history.jsonl`
- `training_summary.json`
- `vecnormalize.pkl`
- `*.obsnorm.npz`
- `*.meta.json`

### 20.3 `models/` 裡面會有什麼

常見：

- `enemy_latest.zip`
- `player_latest.zip`
- `best_model.zip`
- `enemy_round_01_latest.zip`
- `player_round_01_latest.zip`

### 20.4 TensorBoard

兩種看法：

#### 方法 A：GUI 的 TensorBoard 頁

直接看 scalar。

#### 方法 B：外開 TensorBoard server

```powershell
python -m tensorboard.main --logdir checkpoints
```

### 20.5 GUI Results 頁

可以直接：

- 選 run
- 看 summary
- 看 reward trend
- 看 eval reward

### 20.6 Leaderboard

GUI `Leaderboard` 頁可以自動建 Elo。

CLI 也可以：

```powershell
python scripts\eval_elo.py --root checkpoints --episodes 8 --output checkpoints\leaderboard.json
```

---

## 21. Replay / 錄影系統怎麼用

### 21.1 錄 replay

在 play mode 加：

```powershell
--record-replay replays\name.ler.gz
```

### 21.2 replay 格式

副檔名：

- `.ler.gz`

內容是：

- gzip 壓縮 JSON
- 含 metadata
- 含 env_config
- 含 map
- 含逐 frame world state

### 21.3 replay 目前可做什麼

- 回放
- 暫停
- 左右逐幀
- 重播
- 匯出 PNG frame sequence

### 21.4 影片輸出

目前是先匯出 frames。

如果你要真的合成 mp4，現在建議用外部 ffmpeg 再接。

---

## 22. Python 版的重要入口總表

### 22.1 遊戲入口

- [`library_escape/play/human_vs_ai.py`](../library_escape/play/human_vs_ai.py)
- [`library_escape/play/ai_vs_ai.py`](../library_escape/play/ai_vs_ai.py)

### 22.2 GUI 入口

- [`library_escape/gui/app.py`](../library_escape/gui/app.py)

### 22.3 Replay 入口

- [`library_escape/replay/viewer.py`](../library_escape/replay/viewer.py)

### 22.4 訓練入口

- [`library_escape/train/train_enemy.py`](../library_escape/train/train_enemy.py)
- [`library_escape/train/train_player.py`](../library_escape/train/train_player.py)
- [`library_escape/train/train_selfplay.py`](../library_escape/train/train_selfplay.py)

### 22.5 評估入口

- [`scripts/eval_elo.py`](../scripts/eval_elo.py)

---

## 23. 我如果只想「改玩法」要改哪裡

### 23.1 改速度 / 半徑 / 視野 / 道具數量

改：

- [`configs/env.yaml`](../configs/env.yaml)

### 23.2 改地圖

改：

- [`configs/map.json`](../configs/map.json)

### 23.3 改收集、逃脫、追逐的核心規則

改：

- [`library_escape/core/world.py`](../library_escape/core/world.py)

### 23.4 改碰撞

改：

- [`library_escape/core/physics.py`](../library_escape/core/physics.py)

### 23.5 改畫面 HUD / 視覺效果

改：

- [`library_escape/render/pygame_view.py`](../library_escape/render/pygame_view.py)

---

## 24. 我如果只想「改 RL」要改哪裡

### 24.1 改 reward 權重

- [`configs/rewards.yaml`](../configs/rewards.yaml)

### 24.2 改 reward 算法

- [`library_escape/rewards/reward_fns.py`](../library_escape/rewards/reward_fns.py)

### 24.3 改 observation

- [`configs/env.yaml`](../configs/env.yaml)
- [`library_escape/env/obs_builder.py`](../library_escape/env/obs_builder.py)

### 24.4 改 action

- [`configs/env.yaml`](../configs/env.yaml)
- [`library_escape/core/actions.py`](../library_escape/core/actions.py)
- [`library_escape/env/single_agent_env.py`](../library_escape/env/single_agent_env.py)
- [`library_escape/env/multi_agent_env.py`](../library_escape/env/multi_agent_env.py)

### 24.5 改 training hyperparameters

- [`configs/training.yaml`](../configs/training.yaml)

### 24.6 改 opponent curriculum

- [`configs/training.yaml`](../configs/training.yaml)
- [`library_escape/agents/opponent_pool.py`](../library_escape/agents/opponent_pool.py)

### 24.7 改 algorithm 對應邏輯

- [`library_escape/train/common.py`](../library_escape/train/common.py)

### 24.8 改 ETA / progress logging

- [`library_escape/train/callbacks.py`](../library_escape/train/callbacks.py)

---

## 25. 目前我已經驗證過哪些東西

我已經在這個 repo 實際驗證過：

- `pytest -q` 通過
- GUI 可正常建立
- `human_vs_ai` 可正常啟動
- `ai_vs_ai` 可正常啟動
- replay 錄製正常
- replay 匯出影格正常
- `train_enemy` 可正常跑
- `train_player` 可正常跑
- `train_selfplay` 可正常跑
- `maskable_ppo` 可正常訓練
- `league_maskable_ppo` 可正常 self-play
- built-in TensorBoard scalar 解析正常
- Elo leaderboard 建立正常

我**沒有在這一輪重新驗證**：

- 舊版 C++ build / 執行流程

所以如果你要我下一步再幫你，我最建議的是：

1. 以 Python 版為主
2. 把 C++ 版當 legacy 備份

---

## 26. 建議你的日常工作流程

如果你是要繼續開發這個專案，我建議用下面流程。

### 26.1 只想玩

```powershell
python -m library_escape.gui.app
```

然後用 GUI 的 `Play`。

### 26.2 想先做一輪快訓練

```powershell
python -m library_escape.train.train_enemy --preset fast
```

或直接 GUI 的 `Train`。

### 26.3 想調 reward

1. 改 [`configs/rewards.yaml`](../configs/rewards.yaml)
2. 跑 `fast` preset
3. 看 Results / TensorBoard
4. 再決定下一輪調整

### 26.4 想調 observation / action

1. 先改 [`configs/env.yaml`](../configs/env.yaml)
2. 再改 [`library_escape/env/obs_builder.py`](../library_escape/env/obs_builder.py) 或 [`library_escape/core/actions.py`](../library_escape/core/actions.py)
3. 跑 `pytest`
4. 再跑短訓練 smoke test

### 26.5 想做正式訓練

1. 先確認 `balanced` 或 `quality`
2. 調好 reward / env / curriculum
3. 開 GUI 看 ETA
4. 訓練完看：
   - Results
   - TensorBoard
   - Leaderboard
   - Replay

---

## 27. 最後總結

你現在可以把這個 repo 理解成：

- **舊版 C++/SDL2 遊戲還在**
- **新版 Python 版已經是主要工作版本**

新版已經包含：

- 連續時間遊戲核心
- Python 遊玩入口
- Gymnasium / PettingZoo RL 環境
- PPO / Maskable PPO
- Self-play
- Opponent curriculum
- Reward shaping
- GUI
- ETA
- TensorBoard 面板
- Elo leaderboard
- Replay / frame export

如果你之後要我再幫你，我最推薦你直接告訴我下面這種具體目標：

- 幫我把 reward 改得更 aggressive
- 幫我擴充 observation
- 幫我做真正內建 CTDE MAPPO trainer
- 幫我把 replay 匯出成 mp4
- 幫我加新的關卡 / 地圖

這樣我就可以直接沿著這份文件對應的位置繼續往下做。
