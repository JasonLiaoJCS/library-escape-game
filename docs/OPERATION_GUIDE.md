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
6. 把使用者模式整理成 **Collection / Escape 兩套 Game mode**
7. 把敵方壓力改成 **多敵人團隊巡邏**
8. 針對多人巡邏規則重做 **reward shaping / curriculum / Maskable PPO 預設**
9. 把原版 repo 內仍存在的 **背景音樂 / 偵測 / 成績音效** 接回 Python 播放流程

所以現在這個 repo 同時有兩套東西：

- 舊版：C++ / SDL2
- 新版：Python / pygame / RL

如果你要玩、要訓練、要用新功能，**請用 Python 版**。

補充：

- `human_vs_ai` / `ai_vs_ai` 現在正式用 `--game-mode collection` 與 `--game-mode escape`
- `Collection` 會把玩法拉回比較接近原本 C++ 的計分 / 躲偵測主線
- `Escape` 會保留目前 RL 訓練用的抓捕 / 逃脫對抗規則
- `Collection` 固定是 `5` 個敵人
- `Escape` 預設是 `2` 個敵人（`1` 主敵人 + `1` 支援巡邏敵人）
- `AI vs AI` 如果沒有載入 checkpoint，跑的是內建 script baseline，不是未訓練神經網路
- `Collection` 的人類玩法仍然需要長按收集
- 所有模式都加入了短暫收集延遲，不再有瞬間拿書
- reset 後會有短暫 startup grace，避免玩家在開場第一瞬間就被偵測
- 舊的 `classic` / `rl` 只保留成內部相容別名，不再是使用者主要名詞

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
- Collection 獎勵：[`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml)
- Escape 獎勵：[`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)
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

- Reward config：[`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml) / [`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)
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
- 在 `Train` 頁用 responsive 版面看進度與 logs
- 在 Windows 高 DPI 縮放下用比較清楚的字體顯示
- 中途停止訓練後，仍替該 run 生成可被 `Results` 讀取的 summary
- 一鍵把訓練 run 精簡成只保留最新可回播 checkpoint 的最小集合

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

先切到專案根目錄：

```powershell
cd "C:\Users\User\Desktop\大四其他\library-escape-game"
```

再建立虛擬環境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[rl,dev]"
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

### 6.3 先確認你用到的是正確的 Python

如果你同時在別的地方也有 `.venv`，請先執行：

```powershell
python -c "import sys; print(sys.executable)"
```

你應該看到：

```text
C:\Users\User\Desktop\大四其他\library-escape-game\.venv\Scripts\python.exe
```

如果你看到的是：

```text
C:\Users\User\.venv\Scripts\python.exe
```

那代表你啟用錯了虛擬環境。

### 6.4 直接玩遊戲

人類對 AI：

```powershell
python -m library_escape.play.human_vs_ai --game-mode collection
```

指定敵人 checkpoint：

```powershell
python -m library_escape.play.human_vs_ai --game-mode escape --enemy-model checkpoints\enemy\escape\your_run\models\enemy_latest.zip
```

AI 對 AI：

```powershell
python -m library_escape.play.ai_vs_ai --game-mode collection
```

指定雙方模型：

```powershell
python -m library_escape.play.ai_vs_ai --game-mode escape --player-model checkpoints\player_single\escape\your_run\models\player_latest.zip --enemy-model checkpoints\enemy\escape\your_run\models\enemy_latest.zip
```

### 6.5 錄 Replay

人類對 AI 時錄 replay：

```powershell
python -m library_escape.play.human_vs_ai --game-mode collection --record-replay replays\my_session.ler.gz
```

AI 對 AI 時錄 replay：

```powershell
python -m library_escape.play.ai_vs_ai --game-mode escape --record-replay replays\ai_match.ler.gz
```

### 6.6 播放 Replay

```powershell
python -m library_escape.replay.viewer --replay replays\my_session.ler.gz
```

匯出影格：

```powershell
python -m library_escape.replay.viewer --replay replays\my_session.ler.gz --export-frames videos\frames\my_session
```

### 6.7 跑測試

```powershell
pytest -q
```

### 6.8 你剛剛實際遇到過的安裝與啟動問題

#### 問題 A：`pip install -e .[rl,dev]` 裝錯資料夾

錯誤會長這樣：

```text
ERROR: file:///C:/Users/User does not appear to be a Python project
```

原因：

- `pip install -e .[rl,dev]` 裡的 `.` 代表目前資料夾
- 你如果當時人在 `C:\Users\User`
- pip 就會去安裝 `C:\Users\User`
- 但真正的 `pyproject.toml` 在專案根目錄，不在那裡

正確做法：

```powershell
cd "C:\Users\User\Desktop\大四其他\library-escape-game"
pip install -e ".[rl,dev]"
```

#### 問題 B：NumPy C-extension 載入失敗，看到 `cp314` 與 `_multiarray_umath`

錯誤會長這樣：

```text
ImportError: Importing the numpy C-extensions failed
...
_multiarray_umath.cp314-win_amd64.pyd
```

這代表：

- 你現在跑的是 Python 3.12
- 但你使用到的虛擬環境裡 NumPy 混進了 `cp314` 的編譯檔
- 也就是這個 venv 已經壞掉或裝混了

你那次實際上用到的是：

```text
C:\Users\User\.venv\Scripts\python.exe
```

不是專案內的：

```text
C:\Users\User\Desktop\大四其他\library-escape-game\.venv\Scripts\python.exe
```

修法：

```powershell
deactivate
cd "C:\Users\User\Desktop\大四其他\library-escape-game"
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m library_escape.gui.app
```

如果你懶得重新 activate，也可以直接指定專案內的 Python：

```powershell
cd "C:\Users\User\Desktop\大四其他\library-escape-game"
.\.venv\Scripts\python.exe -m library_escape.gui.app
```

#### 問題 C：Prompt 顯示 `(.venv)`，但其實不是專案內那個 venv

這是一個很常見的誤解。

PowerShell 左邊顯示：

```text
(.venv)
```

只代表你進了某個名叫 `.venv` 的虛擬環境。

但如果你同時有：

- `C:\Users\User\.venv`
- `C:\Users\User\Desktop\大四其他\library-escape-game\.venv`

它們顯示都一樣。

所以最可靠的檢查方式永遠是：

```powershell
python -c "import sys; print(sys.executable)"
```

#### 問題 D：如果家目錄那個 `.venv` 已經壞掉，要不要刪掉

如果你確認你不再需要 `C:\Users\User\.venv`，可以刪掉：

```powershell
Remove-Item -Recurse -Force C:\Users\User\.venv
```

這樣之後就比較不容易再誤用到它。

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
- 選 `Collection` 或 `Escape`
- 指定 `player` / `enemy` checkpoint
- 選擇固定 seed 或隨機 seed
- 選擇 deterministic 或 stochastic checkpoint 播放
- 開啟 replay 錄製
- 如果 checkpoint 的訓練模式和目前播放的 `Game mode` 不一致，GUI 會先警告

重點：

- `Play` 是手動發射頁
- 你自己決定載哪個 player checkpoint、哪個 enemy checkpoint、用哪個 `Game mode`
- 最適合做自由組合對戰、人工驗證、錄 replay
- `Seed` 現在預設留空；留空代表每次啟動都重新抽隨機 reset
- `Deterministic checkpoint playback` 現在預設開啟；開啟時 checkpoint 會用 argmax 播放，適合評估、展示、固定重現
- 如果你想觀察策略多樣性，再手動把 deterministic 關掉

### 7.2 Train

用途：

- 訓練 `enemy`
- 訓練 `player`
- 訓練 `selfplay`
- 先選 `Collection` 或 `Escape`
- 選 preset
- 改 seed / timesteps / rounds / n-envs / device
- 選擇是否從舊 checkpoint / 舊 self-play run 續訓
- 改 algorithm：
  - `ppo`
  - `maskable_ppo`
  - `league_ppo`
  - `league_maskable_ppo`
  - `mappo_recipe`
- 改 opponent curriculum 權重
- 看 live ETA / FPS / reward

主要按鈕：

- `Start Training`
- `Stop`
- `Compact Run`
- `Open Run Folder`
- `Open TensorBoard Server`

重點：

- `Train` 是訓練控制中心
- `Mode` 決定你訓練的是 `enemy` / `player` / `selfplay`
- `Game mode` 決定你是在 `Collection` 還是 `Escape` 規則下訓練
- `Run dir` 會直接顯示目前這個訓練 run 的實際輸出位置
- 右側監控區現在是 responsive 版面：上半部是進度摘要，下半部是 `Logs`
- `Logs` 使用 monospace 顯示，並支援水平 / 垂直捲動；視窗大小改變時不會再被固定壓成幾乎看不到的一條
- `Resume checkpoint (optional)`
  - 給 `enemy` / `player` 單訓用
  - 直接選舊的 `.zip` checkpoint，新的訓練會從那個 policy 接著練
  - 但如果那個 checkpoint 是 `11.10` 這輪 observation 改版前的舊模型，**不要直接續訓**；目前 `Collection` / `Escape` 兩個 mode 的 observation 維度都已經更新，建議從零重練或只接續同版本新 checkpoint
- `Resume self-play run (optional)`
  - 給 `selfplay` 用
  - 直接選舊的 self-play run 資料夾
  - 系統會從那個 run 的 `training_summary.json` 自動讀出 `final_enemy_model` 和 `final_player_model`
  - 新的 self-play run 會從那兩個最終 policy 接著訓練
- `Stop`
  - 會要求目前訓練結束
  - 如果 run 尚未寫出正式最終 summary，GUI 仍會補一份 `training_summary.json`
  - 所以該 run 仍然能在 `Results` 頁被掃描到
  - 這種 run 的 summary 會標記 `status = stopped_early`
- `Compact Run`
  - 會保留最新可回播 checkpoint 與必要 metadata
  - 額外 checkpoints、TensorBoard event、monitor CSV、歷史曲線 JSONL 會被刪掉
  - 目的不是保留完整訓練痕跡，而是保留「能回播、能在 Results 看摘要」的最小檔案集合

### 7.3 Results

用途：

- 掃描舊 run
- 看 `training_summary.json`
- 看 training / eval 曲線
- 開 run 資料夾
- 精簡 run
- 啟動 AI vs AI
- 啟動 TensorBoard server

重點：

- `Results` 不是手動選模型，而是先選一個已經存在的訓練 run
- 它會從該 run 的 `training_summary.json` 自動帶出模式、演算法、摘要與曲線
- `Play AI vs AI` 會自動用該 run 的最終模型啟動對戰
- 如果該 run 是中途按 `Stop` 停掉的，只要 GUI 已經補寫 summary，它一樣會出現在 `Results`
- `Compact Run` 可以直接從 `Results` 把這個 run 精簡成只保留最新 playable checkpoint 與必要 metadata

補充：

- 如果你選到的是 `enemy` 單訓 run，`Results -> Play AI vs AI` 會用這個 enemy checkpoint 對 rule-based player
- 如果你選到的是 `player` 單訓 run，會用這個 player checkpoint 對 rule-based enemy
- 如果你選到的是 `selfplay` run，會同時帶出該 run 的最終 player / enemy checkpoint

所以：

- 想看「這個 run 最終學成怎樣」：用 `Results`
- 想手動拼裝不同 run 的 player / enemy：用 `Play`

### 7.4 TensorBoard

用途：

- 直接讀 `tb/` 裡的 event files
- 在 GUI 內畫 scalar 曲線

重點：

- 這是 GUI 內建的 TensorBoard scalar 檢視器
- 不用另外開瀏覽器，也能直接挑 tag 看曲線
- 右側會顯示每條曲線的 `latest / min / max`
- 如果你想看更完整的 TensorBoard 頁面，還是可以從 `Train` 或 `Results` 開外部 TensorBoard server

### 7.5 Leaderboard

用途：

- 掃描 checkpoints
- 自動對戰
- 建 player / enemy 的 Elo 排名

重點：

- 這頁是拿來比較不同 checkpoint 強弱
- 它會把玩家模型和敵人模型配對對戰，再用 Elo 更新 rating
- 你可以調 `Episodes`、`Max players`、`Max enemies`、`K-factor`

### 7.6 Replay

用途：

- 掃描 `.ler.gz`
- 顯示 replay metadata
- 播放 replay
- 匯出 frames

重點：

- `Replay` 播的是已錄下來的對局，不是重新跑一次即時模擬
- 適合回顧 AI 行為、找 bug、展示成果、匯出畫面

### 7.7 Config

用途：

- 快速打開：
  - `env.yaml`
  - `rewards_collection.yaml`
  - `rewards_escape.yaml`
  - `training.yaml`
  - `map.json`
  - 本文件

重點：

- `Config` 不是在 GUI 裡直接編輯設定
- 它是「常用檔案快速入口」
- 當你想改 reward、環境、地圖、訓練 preset、文件時，這一頁會幫你直接打開對應檔案

### 7.8 `Play` 和 `Results -> Play AI vs AI` 到底差在哪裡

兩者最後都會開 `AI vs AI`，但差別是：

- `Play`
  - 你手動指定 checkpoint
  - 你手動指定 `Game mode`
  - 你可以自由混搭不同 run 的模型
  - 你可以順手錄 replay
- `Results -> Play AI vs AI`
  - 先選一個訓練 run
  - GUI 幫你自動讀該 run 的 `training_summary.json`
  - 自動帶出該 run 的最終模型與對應 `Game mode`
  - 預設用 deterministic playback；seed 則保持非固定，讓 reset 仍有隨機性
  - 更適合「我剛 train 完，想直接看這次 run 的最終成果」

### 7.9 為什麼新版 checkpoint 播放不一定每次一樣

這是刻意改的。

之前如果你每次重開都看到完全一樣的路線，通常是因為：

- checkpoint 播放被設成 `deterministic=True`
- GUI 又把 `seed` 固定死

現在新版改成：

- `Play` 頁的 `Seed` 預設留空
- `Deterministic checkpoint playback` 預設開啟
- `Results -> Play AI vs AI` 也不再幫你硬塞固定 seed

因此現在的預設行為是：

- 用 deterministic policy 播放，讓模型每步都選 argmax，適合正式評估
- 但 seed 仍預設留空，所以不同次啟動仍可能因 reset 隨機化而有不同對局

如果你想要：

- 策略多樣性：seed 留空、deterministic 關閉
- 穩定評估：seed 留空、deterministic 開啟
- 完全可重現：填固定 seed、deterministic 開啟

### 7.10 為什麼新版 `Escape` 觀戰比較順

之前播放 checkpoint 的時候，是每個 physics tick 都直接跑一次神經網路推論。

這會有兩個問題：

1. 播放時的決策頻率和訓練時不一致  
2. `Escape` self-play 模型在觀戰時比較容易卡頓

現在播放端改成：

- 畫面仍然維持連續渲染
- checkpoint policy 的決策頻率對齊 `rl_frame_skip`

這樣做之後：

- 觀戰更順
- 播放分布比較接近訓練分布
- 不會再那麼容易出現一幀一幀、離散式卡頓的觀感

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

### 9.1 目前 player / enemy 共用同一套 movement action space

在 [`configs/env.yaml`](../configs/env.yaml)：

```yaml
action:
  type: discrete
  continuous_scale: 1.0
```

支援：

- `discrete`
- `continuous`

目前不管是玩家還是敵人，底層控制都是「移動向量」。

- player RL agent 用這套 action
- enemy RL agent 用這套 action
- 規則式 AI 最後也會轉成同樣的移動向量

### 9.2 `discrete` 模式

目前最常用，也是 GUI / Maskable PPO / self-play 預設使用的模式。

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

實際映射就是：

```python
0: (0.0, 0.0)
1: (0.0, -1.0)
2: (1.0, -1.0)
3: (1.0, 0.0)
4: (1.0, 1.0)
5: (0.0, 1.0)
6: (-1.0, 1.0)
7: (-1.0, 0.0)
8: (-1.0, -1.0)
```

之後會再正規化成長度 1 的方向向量。

### 9.3 `continuous` 模式

目前是 2D 向量：

- `x`
- `y`

最後會被 clamp 到最大長度。

同樣在：

- [`library_escape/core/actions.py`](../library_escape/core/actions.py)

### 9.4 Maskable PPO 為什麼建議保持 discrete

如果你要用：

- `maskable_ppo`
- `league_maskable_ppo`

那就必須保持：

```yaml
action:
  type: discrete
```

因為 invalid action masking 是做在離散動作上的。

對應檔案：

- [`library_escape/env/action_masking.py`](../library_escape/env/action_masking.py)

目前 mask 的規則是：

- `0` 停止永遠有效
- 會立刻撞牆 / 邊界、導致幾乎不會動的方向會被 mask 掉
- 如果主敵人被凍結，敵人的 mask 只會保留 `0`

### 9.5 RL 環境的 `action_space` 在哪裡宣告

- Gymnasium：[`library_escape/env/single_agent_env.py`](../library_escape/env/single_agent_env.py)
- PettingZoo：[`library_escape/env/multi_agent_env.py`](../library_escape/env/multi_agent_env.py)

### 9.6 你想改 action 時怎麼做

#### 只想改模式或參數

改：

- [`configs/env.yaml`](../configs/env.yaml)

#### 想改 action 定義本身

改：

- [`library_escape/core/actions.py`](../library_escape/core/actions.py)

#### 想改 invalid action mask 規則

改：

- [`library_escape/env/action_masking.py`](../library_escape/env/action_masking.py)

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

重要：

- `configs/env.yaml` 這裡的 `partial_observability: true` 只是 base profile。
- 真正進入 [`library_escape/game_modes.py`](../library_escape/game_modes.py) 的 `Collection` / `Escape` preset 後，**目前兩個 mode 都會覆蓋成 `false`**。
- 也就是說，現在實際訓練與播放時，玩家與敵人都會直接拿到相對位置向量；這是這一輪為了打掉「不知道要去哪、原地擺爛」而刻意做的 observation 擴充。

### 10.2 實際組裝程式碼

在：

- [`library_escape/env/obs_builder.py`](../library_escape/env/obs_builder.py)

### 10.3 敵人 observation 目前包含什麼

`build_enemy_obs()` 現在是：

1. `enemy.x / world.width`
2. `enemy.y / world.height`
3. `enemy.facing_x`
4. `enemy.facing_y`
5. `relative_player_x / world.width`
6. `relative_player_y / world.height`
7. `primary_visible_flag`
8. `team_visible_flag`
9. `visible_enemy_ratio`
10. `nearest_ally_dx`
11. `nearest_ally_dy`
12. `remaining_note_ratio`
13. `remaining_exam_ratio`
14. `score_progress`
15. `objective_progress`
16. `time_remaining_ratio`
17. `can_player_escape_flag`
18. `player_in_escape_zone_flag`
19. `distance_agents / max_map_distance`
20. `distance_player_to_target / max_map_distance`
21. `distance_player_to_escape / max_map_distance`
22. `distance_enemy_to_escape / max_map_distance`
23. `exit_lead`
24. `collection_progress`
25. `team_detection_cooldown`
26. `primary_enemy_pause_fraction`
27. `primary_threat_margin`
28. `collection_mode_flag`
29. `escape_mode_flag`
30. `relative_escape_x`
31. `relative_escape_y`
32. `relative_player_goal_x`
33. `relative_player_goal_y`
34. `wall_rays[0]`
35. `wall_rays[1]`
36. `wall_rays[2]`
37. `wall_rays[3]`
38. `wall_rays[4]`
39. `wall_rays[5]`
40. `wall_rays[6]`
41. `wall_rays[7]`

幾個重要細節：

- 如果 `partial_observability = true` 且敵方隊伍目前沒看到玩家，`relative_player_x / y` 會被寫成 `0.0`
- 但目前 `Collection` / `Escape` preset 都把 `partial_observability` 覆蓋成 `false`，所以實際上這兩個 mode 會直接暴露相對位置
- `primary_visible_flag` 是「主敵人自己看沒看到玩家」
- `team_visible_flag` 是「整個敵方隊伍有沒有任何一人看到玩家」
- `visible_enemy_ratio` 是「目前看得到玩家的敵人數 / 全敵人數」
- `nearest_ally_dx / dy` 讓主敵人知道最近支援敵人的相對位置
- `score_progress` 與 `objective_progress` 分開放，讓 Collection 與 Escape 可以共用同一份 observation schema，但仍然學到不同成功指標
- `exit_lead` 能讓敵人知道目前自己隊伍相對玩家是不是更接近出口，對守出口與包抄很重要
- `collection_progress` / `team_detection_cooldown` / `primary_enemy_pause_fraction` 讓敵人知道玩家是不是正在收集，以及剛發生 spotting 後是否還在冷卻
- `collection_mode_flag / escape_mode_flag` 讓同一個模型結構能跨模式訓練，不需要整個 action/obs pipeline 換掉
- `relative_escape_x / y` 讓敵人直接知道出口在自己哪個方向，Escape 比較容易長出守出口與切線攔截
- `relative_player_goal_x / y` 讓敵人直接知道「玩家現在最該去的地方」在哪裡
  - Escape：通常是最近必收目標；一旦解鎖逃脫就切到出口中心
  - Collection：是 `current_collection_goal_collectible()` 選出的當前最高價值目標，不再只是死盯固定 required item
- `wall_rays` 是 360 度等角度打出去的障礙距離，已經除上 `max_ray_distance`

預設 `wall_rays = 8`，所以敵人 observation 維度預設是：

- `33 + 8 = 41`

### 10.4 玩家 observation 目前包含什麼

`build_player_obs()` 現在是：

1. `player.x / world.width`
2. `player.y / world.height`
3. `player.facing_x`
4. `player.facing_y`
5. `relative_nearest_enemy_x / world.width`
6. `relative_nearest_enemy_y / world.height`
7. `enemy_visible_to_player_flag`
8. `player_visible_primary_flag`
9. `visible_enemy_ratio`
10. `nearest_enemy_distance / max_map_distance`
11. `nearest_note_dx`
12. `nearest_note_dy`
13. `nearest_exam_dx`
14. `nearest_exam_dy`
15. `nearest_powerup_dx`
16. `nearest_powerup_dy`
17. `objective_progress`
18. `score_progress`
19. `time_remaining_ratio`
20. `can_player_escape_flag`
21. `player_in_escape_zone_flag`
22. `collection_progress`
23. `distance_player_to_target / max_map_distance`
24. `distance_player_to_escape / max_map_distance`
25. `exit_lead`
26. `coffee_timer_ratio`
27. `max_enemy_freeze_timer_ratio`
28. `support_enemy_ratio`
29. `team_detection_cooldown`
30. `collection_mode_flag`
31. `escape_mode_flag`
32. `relative_primary_enemy_x / world.width`
33. `relative_primary_enemy_y / world.height`
34. `relative_nearest_support_x / world.width`
35. `relative_nearest_support_y / world.height`
36. `relative_escape_x`
37. `relative_escape_y`
38. `front_fan_ray[0]`
39. `front_fan_ray[1]`
40. `front_fan_ray[2]`
41. `wall_rays[0]`
42. `wall_rays[1]`
43. `wall_rays[2]`
44. `wall_rays[3]`
45. `wall_rays[4]`
46. `wall_rays[5]`
47. `wall_rays[6]`
48. `wall_rays[7]`

幾個重要細節：

- `relative_nearest_enemy_x / y` 只看「最近的敵人」，不是所有敵人的平均
- 如果 `partial_observability = true` 且玩家視角下目前看不到敵人，這兩個相對座標會被寫成 `0.0`
- 但目前兩個 mode preset 都把 `partial_observability` 覆蓋成 `false`，所以實際上玩家會直接看到最近敵人、主敵人、最近支援敵人的相對方向
- `nearest_note / nearest_exam / nearest_powerup dx dy` 讓玩家能把主目標、加分道具與增益道具分開看
- `objective_progress` 與 `score_progress` 分開放，讓 Collection 重點落在壓分 / 拉分，Escape 重點落在完成前置條件後衝出口
- `collection_progress` 讓玩家知道自己是不是正在收集進度中
- `exit_lead` 幫助玩家判斷現在是不是比敵人更有機會先到出口
- `team_detection_cooldown` 幫助玩家知道剛被發現後是否仍在共享偵測冷卻
- `collection_mode_flag / escape_mode_flag` 讓同一份 observation schema 能在兩種 Game mode 都使用
- `relative_primary_enemy_x / y` 與 `relative_nearest_support_x / y` 是這一輪新增的策略資訊
  - 玩家不只知道「最近有人」，也知道 primary 從哪邊壓過來、最近支援敵人在哪邊封路
- `relative_escape_x / y` 讓玩家不必只靠 `distance_player_to_escape` 猜方向，而能直接規劃衝出口路線
- `front_fan_ray` 雖然參數名叫 `player_enemy_rays`，但目前實作其實是「玩家面朝方向前方扇形的距離 ray」，不是直接回傳敵人位置
- `wall_rays` 一樣是已標準化的障礙距離

預設：

- `player_enemy_rays = 3`
- `wall_rays = 8`

所以玩家 observation 維度預設是：

- `37 + 3 + 8 = 48`

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

Reward 程式碼：

- [`library_escape/rewards/reward_fns.py`](../library_escape/rewards/reward_fns.py)

Reward 設定檔現在是**依 Game mode 分開**的：

- Collection：[`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml)
- Escape：[`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)

下面這一整節講的是「強化學習環境的 reward」，不是畫面上的 HUD 分數。

### 11.1 先講最重要的：現在 reward 是跟 Game mode 綁定的

現在不是所有訓練都共用同一份 reward。

實際規則是：

- 你在 `Train` 或 CLI 選 `--game-mode collection`
  - 系統會自動用 `configs/rewards_collection.yaml`
- 你在 `Train` 或 CLI 選 `--game-mode escape`
  - 系統會自動用 `configs/rewards_escape.yaml`

這個 mapping 位置在：

- [`library_escape/game_modes.py`](../library_escape/game_modes.py)

也就是說：

- `enemy + collection`
  - 用 Collection reward
- `player + collection`
  - 用 Collection reward
- `selfplay + collection`
  - 用 Collection reward
- `enemy + escape`
  - 用 Escape reward
- `player + escape`
  - 用 Escape reward
- `selfplay + escape`
  - 用 Escape reward

### 11.2 reward 的總體計算流程

每次 `env.step(...)` 之後，系統會：

1. 收集這一步發生的事件 `events`
2. 取出前一狀態 `prev_metrics`
3. 取出下一狀態 `next_metrics`
4. 先算出 `enemy_reward_pre_mix` 與 `player_reward_pre_mix`
5. 如果完全沒有進展，兩邊都吃 `anti_exploit` 懲罰
6. 用 `zero_sum_mix` 做部分零和混合
7. 最後用 `clip_range` 做 clipping

visibility 相關不是直接用 raw 次數，而是先做：

```text
frame_scale = 1 / rl_frame_skip
```

所以 `primary_visible_steps`、`support_visible_steps` 這些都會先乘上 `frame_scale`，避免 `frame_skip` 不同時 reward 尺度飄掉。

### 11.3 共用公式是什麼

不管是 Collection 還是 Escape，底層 reward engine 都是同一支：

- [`library_escape/rewards/reward_fns.py`](../library_escape/rewards/reward_fns.py)

共用公式可以簡化理解成：

```text
enemy_reward =
  dense_enemy_terms
  + terminal_enemy_terms
  + potential_delta_enemy
  + anti_exploit_if_no_progress
  -> zero_sum_mix
  -> clip
```

```text
player_reward =
  dense_player_terms
  + terminal_player_terms
  + potential_delta_player
  + anti_exploit_if_no_progress
  -> zero_sum_mix
  -> clip
```

potential-based shaping 的共用形式是：

```text
potential_delta = gamma * Phi(next) - Phi(prev)
```

其中核心量來自 `world.transition_metrics()`，像是：

- `distance_agents`
- `visible_enemy_ratio`
- `objective_progress`
- `distance_player_to_target`
- `distance_player_to_escape`
- `distance_enemy_to_escape`
- `can_escape`
- `exit_lead`

### 11.4 enemy / player 的 potential 指標定義

#### enemy potential

```text
Phi_enemy =
  w_capture_pressure * capture_pressure
  + w_team_visibility * team_visibility
  + w_objective_denial * objective_denial
  + w_exit_guard * exit_guard
  + w_encirclement * encirclement
```

各項定義：

- `capture_pressure`
  - `1 - distance_agents / max_map_distance`
- `team_visibility`
  - `0.65 * player_visible_primary + 0.35 * visible_enemy_ratio`
- `objective_denial`
  - `1 - objective_progress`
- `exit_guard`
  - 只有在玩家已能逃脫，或目標進度至少 `75%` 時才啟用
- `encirclement`
  - `capture_pressure * visible_enemy_ratio`

#### player potential

```text
Phi_player =
  w_objective_progress * objective_progress
  + target_weight * target_navigation
  + w_threat_margin * threat_margin
  + w_exit_window * exit_window
```

各項定義：

- `objective_progress`
  - 目前主目標完成比例
- `target_navigation`
  - 還沒達成逃脫條件時，朝下一個目標前進的程度
- `escape_navigation`
  - 已達逃脫條件後，朝出口前進的程度
- `threat_margin`
  - 離敵人越遠越高，但會被主敵人視野與團隊可見比例壓低
- `exit_window`
  - 玩家比敵人更接近出口時更高

### 11.5 Collection 模式 reward 權重

來源：

- [`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml)

> **注意**：下面這個 YAML 區塊是 `2026-04-19` 重塑**之前**的舊版本，保留作歷史參考。
> 目前實際生效的 Collection reward 權重與設計理由請看 `11.10.6`。

```yaml
global:
  gamma: 0.99
  clip_range: 320.0
  zero_sum_mix: 0.18

enemy:
  catch_player: 0.0
  lose_on_escape: 0.0
  timeout_win: 0.0
  detection_event_bonus: 8.0
  primary_visible_per_step: 0.55
  support_visible_per_step: 0.22
  team_visible_ratio_per_step: 0.18
  player_collect_note_penalty: -16.0
  player_collect_exam_penalty: -26.0
  player_collect_powerup_penalty: -5.0
  player_objective_complete_penalty: -24.0
  timeout_score_denial_bonus: 42.0
  stalemate_score_denial_bonus: 28.0
  time_penalty: -0.002
  wall_penalty: -0.10
  idle_penalty: -0.010
  stalemate: -12.0
  potential:
    enabled: true
    capture_pressure: 1.10
    team_visibility: 1.55
    objective_denial: 3.10
    exit_guard: 0.0
    encirclement: 0.95

player:
  collect_note: 14.0
  collect_exam: 22.0
  collect_coffee: 4.0
  collect_freeze: 5.0
  objective_complete_bonus: 24.0
  detection_event_penalty: -8.0
  escape: 0.0
  caught: 0.0
  timeout_loss: 0.0
  timeout_score_progress_bonus: 42.0
  stalemate_score_progress_bonus: 28.0
  primary_seen_per_step: -0.40
  support_seen_per_step: -0.16
  multi_seen_penalty_per_step: -0.14
  time_penalty: -0.004
  wall_penalty: -0.10
  idle_penalty: -0.010
  stalemate: -12.0
  potential:
    enabled: true
    objective_progress: 4.80
    target_navigation: 2.10
    escape_navigation: 0.0
    threat_margin: 1.35
    exit_window: 0.0

anti_exploit:
  no_progress_penalty: -0.02
```

這份 reward 的設計意圖是：

- 玩家不是在追求「逃出去」
- 玩家是在追求「更高分、更有效率的收集、更少被看見」
- 敵人不是在追求 strict win/lose
- 敵人是在追求「壓分、拖慢收集、維持視野壓力」
- reward 主要對齊 `score_progress`，不是只看 required objective 是否完成

所以你會看到：

- `escape / caught / timeout` 在 Collection 都是 `0.0`
- `timeout_score_progress_bonus` / `timeout_score_denial_bonus` 這種 mode-specific shaping 比單純 terminal reward 更重要
- `objective_complete_bonus` / `player_objective_complete_penalty` 只是在所有主目標收完那一刻提供清楚事件訊號
- `collect_exam` 與 `player_collect_exam_penalty` 仍然高於 note，因為 exam 是高價值分數來源
- `exit_guard` / `escape_navigation` 在 Collection 都是 `0.0`

### 11.6 Escape 模式 reward 權重

來源：

- [`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)

> **注意**：下面這個 YAML 區塊是 `2026-04-19` 重塑**之前**的舊版本，保留作歷史參考。
> 目前實際生效的 Escape reward 權重與設計理由請看 `11.10.5`。

```yaml
global:
  gamma: 0.99
  clip_range: 320.0
  zero_sum_mix: 0.22

enemy:
  catch_player: 220.0
  lose_on_escape: -220.0
  timeout_win: 48.0
  detection_event_bonus: 0.0
  primary_visible_per_step: 0.50
  support_visible_per_step: 0.20
  team_visible_ratio_per_step: 0.16
  player_collect_note_penalty: -16.0
  player_collect_exam_penalty: -6.0
  player_collect_powerup_penalty: -5.0
  player_objective_complete_penalty: -30.0
  timeout_score_denial_bonus: 0.0
  stalemate_score_denial_bonus: 0.0
  time_penalty: -0.004
  wall_penalty: -0.08
  idle_penalty: -0.008
  stalemate: -24.0
  potential:
    enabled: true
    capture_pressure: 2.40
    team_visibility: 1.55
    objective_denial: 1.70
    exit_guard: 2.50
    encirclement: 1.75

player:
  collect_note: 14.0
  collect_exam: 6.0
  collect_coffee: 5.0
  collect_freeze: 7.0
  objective_complete_bonus: 32.0
  detection_event_penalty: 0.0
  escape: 220.0
  caught: -220.0
  timeout_loss: -48.0
  timeout_score_progress_bonus: 0.0
  stalemate_score_progress_bonus: 0.0
  primary_seen_per_step: -0.35
  support_seen_per_step: -0.14
  multi_seen_penalty_per_step: -0.12
  time_penalty: -0.004
  wall_penalty: -0.08
  idle_penalty: -0.008
  stalemate: -24.0
  potential:
    enabled: true
    objective_progress: 2.20
    target_navigation: 1.80
    escape_navigation: 4.00
    threat_margin: 1.00
    exit_window: 2.10

anti_exploit:
  no_progress_penalty: -0.02
```

這份 reward 的設計意圖是：

- 玩家在追求真實贏局
- 敵人在追求真實守成
- terminal outcome 非常重要
- 但同時保留 dense shaping，讓它們學得比較快
- `objective_complete_bonus` 與 `player_objective_complete_penalty` 會在「剛好完成逃脫前置條件」那一刻明確推一把
- `exit_guard` / `exit_window` 在 Escape 明顯比 Collection 更重，因為這個模式真的要學守出口與衝出口

所以你會看到：

- `escape = +220.0` / `caught = -220.0`
- `catch_player = +220.0` / `lose_on_escape = -220.0`
- `timeout_win = +48.0` / `timeout_loss = -48.0`
- `exit_guard` / `escape_navigation` 都是啟用的

### 11.7 anti-exploit / zero-sum / clipping 怎麼算

如果這一步 `events.progress_made == false`，那麼：

```text
player_reward += no_progress_penalty
enemy_reward += no_progress_penalty
```

然後會再做部分零和混合：

```text
player_adv = player_reward - enemy_reward
enemy_adv = enemy_reward - player_reward

player_reward = (1 - zero_sum_mix) * player_reward + zero_sum_mix * player_adv
enemy_reward = (1 - zero_sum_mix) * enemy_reward + zero_sum_mix * enemy_adv
```

最後如果有設定 `clip_range`：

```text
reward = clamp(reward, -clip_range, clip_range)
```

### 11.8 如果我想改 reward

#### 只改 Collection 權重

改：

- [`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml)

#### 只改 Escape 權重

改：

- [`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)

#### 改 reward 算法本身

改：

- [`library_escape/rewards/reward_fns.py`](../library_escape/rewards/reward_fns.py)

#### 改 shaping 所用的狀態量

改：

- [`library_escape/core/world.py`](../library_escape/core/world.py)

因為 `transition_metrics()` 是 shaping 依賴的核心輸入。

### 11.9 這一輪 reward / curriculum / observation 為什麼這樣重設

這一輪不是只憑直覺調數字，而是對齊了幾個 RL 常見做法：

- Potential-based reward shaping
  - dense shaping 盡量寫成 potential difference，避免直接亂加 reward 改掉最優策略。
  - 來源：https://wordpress.andrewng.org/index.php/publication/policy-invariance-under-reward-transformations-theory-and-application-to-reward-shaping/
- Self-play 與自動課程式對抗
  - `Escape` 需要追擊方與逃脫方一起變強，所以 opponent curriculum 與 self-play 參考了 OpenAI Hide-and-Seek 這種會因對手變強而逐漸長出新策略的做法。
  - 來源：https://openai.com/index/emergent-tool-use/
- 歷史對手池 / league 式訓練
  - 為了避免只 overfit 最新對手，self-play 會混合最新 checkpoint、歷史 checkpoint、heuristic baseline，這個精神接近 AlphaStar 的 league training。
  - 來源：https://www.nature.com/articles/s41586-019-1724-z
- PPO-family 在多人合作/對抗問題中的強 baseline 地位
  - 目前主力演算法保留在 `MaskablePPO`，是因為 PPO-family 在 cooperative / mixed multi-agent 任務上很常被當成實用基線；這個方向和 MAPPO 論文的實務結論一致。
  - 來源：https://arxiv.org/abs/2103.01955

這些來源最後落成了幾個實際設計決定：

- Collection 與 Escape 用兩份不同 reward 檔，而不是硬共用一套權重
- observation 同時提供 `objective_progress` 與 `score_progress`，讓同一個 obs schema 能跨模式，但 reward 仍然 mode-aware
- action space 先維持相同的離散 9 動作 + invalid-action mask，避免模式切換就讓 checkpoint 完全不相容
- `fast / balanced / quality` 用 mode-aware 的支援敵人數量 curriculum，讓 Escape 不會一開始就被多敵人壓死，同時又保留後期學包抄、守出口、協同圍捕的空間

這一輪平衡時，我另外做了一個 heuristic player vs rule-based enemy 的 10-seed smoke audit 當 sanity check：

- Collection：平均 `38.8` 分，平均 `4.1` note / `0.5` exam，全部在 `time_expired` 收束
- Escape：平均 `10.8` 分，`5` 次 `caught`、`5` 次 `time_expired`

這不是正式 benchmark，也不是訓練後的最終強度；它只是用來確認兩個模式目前不會一邊完全碾壓另一邊，並且 Escape 會同時出現「被抓到」與「拖到超時」兩種失敗型態，而不是單一崩壞模式。

### 11.10 2026-04-19 / 2026-04-20 reward / observation / patrol 大改版

這一節記錄 `2026-04-19` 到 `2026-04-20` 連續兩輪大改版。現在實際生效的不只 reward，還包括：

- `Escape` / `Collection` 兩個 mode 的 observation 擴充
- 玩家當前目標選擇邏輯
- 支援敵人的 patrol 壓力邏輯
- 舊 checkpoint 相容性保護

前面 `11.5` / `11.6` 兩個小節的 YAML 區塊保留作歷史版本參考；目前實際生效的內容請以本節 (`11.10`) 與程式碼 / config 檔為準。

#### 11.10.1 為什麼要重塑：兩個 mode 都出現病態行為

在這輪重塑前，`Escape` 與 `Collection` 都會出現非常類似的崩壞：

- 主敵人只會原地甩頭、自旋，沒有真正追擊。
- 支援敵人常卡在很小的範圍做固定路徑循環，沒有對玩家目標區施壓。
- 玩家會左右抖動、原地站住、或只在目標旁邊猶豫，不真的去收集或往出口推進。
- 雙方 reward 看起來還在跳，但實際行為沒有「想贏」。

`Escape` 的壞相是自旋、拖時間、守不到出口；`Collection` 的壞相是蹭 visibility、壓不到得分路線、玩家不主動拉分。

#### 11.10.2 這次 reward engine 真正改了什麼

這一輪不是只調幾個權重，而是把 reward engine 的主訊號換掉：

- 保留 `enemy.chase_progress_per_unit` / `enemy.search_move_per_unit`
- 保留 `player.goal_progress_per_unit` / `player.evade_progress_per_unit`
- 新增 `enemy.objective_guard_progress_per_unit`
  - primary 敵人縮短自己到「玩家當前目標」的距離才有 reward
  - 用來打掉「守自己小圈圈」但完全沒壓到玩家路線的壞策略
- 新增 `enemy.exit_guard_progress_per_unit`
  - 只在 `can_player_escape()` 之後生效
  - 敵人縮短自己到出口區的距離才有 reward，學會切出口而不是繞錯邊
- `_quiet_step()` 不再因為 `visible_steps > 0` 就免罰
  - 現在即使看得到玩家，原地不動 / 小範圍來回震盪仍然會吃 anti-stall / anti-oscillation penalty
- `zero_sum_mix` 現在兩個 mode 都是 `0.0`
  - 這一輪觀察到只要保留 adversarial mix，還是容易收斂到 mutual-freeze
- potential shaping 現在兩個 mode 都全部關閉
  - reward 改成依賴更直接、可解釋的 dense signal，而不是讓多組 potential 彼此打架

#### 11.10.3 observation 與目標選擇也一起重做

這一輪最重要的不是只有 reward，還包含 observation / target-selection：

- `Collection` / `Escape` 目前都在 [`library_escape/game_modes.py`](../library_escape/game_modes.py) 內把 `partial_observability` 覆蓋成 `false`
- 敵人 observation 新增：
  - `relative_escape_x / y`
  - `relative_player_goal_x / y`
- 玩家 observation 新增：
  - `relative_primary_enemy_x / y`
  - `relative_nearest_support_x / y`
  - `relative_escape_x / y`
- [`library_escape/core/world.py`](../library_escape/core/world.py) 新增 `current_player_goal_position()`
  - Escape：先指向目前必收目標；解鎖後改指向出口
  - Collection：改成 `current_collection_goal_collectible()`，會依道具價值 / 距離 / coffee / freeze 情況選出當前最值得搶的目標
- `distance_player_to_target` 也跟著改成對這個「動態目標」量測，不再只是舊版固定 target

這些改動的目的很直接：讓 agent 不用再靠模糊的 visibility shaping 猜「現在該去哪」，而是直接學會壓目標、守路線、衝出口。

#### 11.10.4 世界腳本與 baseline 也同步改革

病態行為裡有一部分不是 RL 自己學壞，而是 baseline / support patrol 本來就太死：

- [`library_escape/core/world.py`](../library_escape/core/world.py) 的 support enemy patrol 現在兩個 mode 都會偏向 `current_player_goal_position()`
  - Collection：偏向玩家當前得分目標，必要時再往玩家位置補一點壓力
  - Escape：前期往玩家必收目標壓，後期逐漸往出口線收
- [`library_escape/agents/rule_based_enemy.py`](../library_escape/agents/rule_based_enemy.py) 的 Collection primary baseline 現在也會朝玩家當前得分目標移動，不再只是守死 waypoint
- [`library_escape/train/common.py`](../library_escape/train/common.py) 的 history opponent pool 現在會先檢查 `obsnorm` 維度，只混入 observation 維度相容的舊模型，避免訓練中途 shape mismatch 直接炸掉

這一輪**沒有**改掉你在意的核心玩法規則：

- 拿東西規則沒改
- `Collection` 的長按收集規則沒改
- `Escape` 的逃脫條件沒改
- GUI 功能沒改
- 轉頭限速功能目前仍然維持關閉，沒有偷偷改壞實際物理規則

#### 11.10.5 目前實際生效的 Escape 權重

檔案：[`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)

設計原則：

- `catch_player` / `escape` 是絕對主訊號，`±300`
- `timeout` 與 `stalemate` 明顯比真抓到 / 真逃掉差
- visibility reward 幾乎只剩「弱確認」，不再是主要訓練目標
- primary 應該壓玩家目標路線與出口，而不是蹭支援敵人視野

```yaml
global:
  gamma: 0.99
  clip_range: 320.0
  zero_sum_mix: 0.0

enemy:
  catch_player: 300.0
  lose_on_escape: -300.0
  timeout_win: 20.0
  stalemate: -80.0
  detection_event_bonus: 0.0
  player_collect_note_penalty: -15.0
  player_collect_exam_penalty: -3.0
  player_collect_powerup_penalty: -3.0
  player_objective_complete_penalty: -80.0
  primary_visible_per_step: 0.05
  support_visible_per_step: 0.0
  team_visible_ratio_per_step: 0.0
  chase_progress_per_unit: 1.50
  objective_guard_progress_per_unit: 0.85
  exit_guard_progress_per_unit: 0.95
  search_move_per_unit: 0.25
  time_penalty: -0.02
  wall_penalty: -0.15
  idle_penalty: -0.40
  timeout_score_denial_bonus: 0.0
  stalemate_score_denial_bonus: 0.0
  potential:
    enabled: false
    use_primary_distance: true
    capture_pressure: 0.0
    team_visibility: 0.0
    objective_denial: 0.0
    exit_guard: 0.0
    encirclement: 0.0

player:
  escape: 300.0
  caught: -300.0
  timeout_loss: -30.0
  stalemate: -80.0
  collect_note: 35.0
  collect_exam: 8.0
  collect_coffee: 6.0
  collect_freeze: 10.0
  objective_complete_bonus: 80.0
  detection_event_penalty: 0.0
  primary_seen_per_step: -0.05
  support_seen_per_step: 0.0
  multi_seen_penalty_per_step: 0.0
  goal_progress_per_unit: 1.40
  evade_progress_per_unit: 0.60
  time_penalty: -0.01
  wall_penalty: -0.15
  idle_penalty: -0.30
  timeout_score_progress_bonus: 0.0
  stalemate_score_progress_bonus: 0.0
  potential:
    enabled: false
    objective_progress: 0.0
    target_navigation: 0.0
    escape_navigation: 0.0
    threat_margin: 0.0
    exit_window: 0.0

anti_exploit:
  no_progress_penalty: -0.10
  stationary_threshold: 0.05
  oscillation_path_threshold: 0.08
  oscillation_net_threshold: 0.04
  player_stationary_penalty: -0.35
  enemy_stationary_penalty: -0.45
  player_oscillation_penalty: -0.50
  enemy_oscillation_penalty: -0.60
```

#### 11.10.6 目前實際生效的 Collection 權重

檔案：[`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml)

設計原則：

- Collection 沒有真正 terminal capture / escape，所以把 `timeout_score_*_bonus` 當成真正的終局訊號
- 玩家主目標回到「有效拉分」，敵人主目標回到「有效壓分」
- 敵人 dense reward 比 Escape 更重視 `objective_guard_progress_per_unit`，因為這個 mode 的核心是壓玩家得分路線

```yaml
global:
  gamma: 0.99
  clip_range: 320.0
  zero_sum_mix: 0.0

enemy:
  catch_player: 0.0
  lose_on_escape: 0.0
  timeout_win: 0.0
  stalemate: -80.0
  timeout_score_denial_bonus: 150.0
  stalemate_score_denial_bonus: 120.0
  detection_event_bonus: 10.0
  player_collect_note_penalty: -18.0
  player_collect_exam_penalty: -34.0
  player_collect_powerup_penalty: -5.0
  player_objective_complete_penalty: -60.0
  primary_visible_per_step: 0.05
  support_visible_per_step: 0.0
  team_visible_ratio_per_step: 0.0
  chase_progress_per_unit: 0.85
  objective_guard_progress_per_unit: 1.10
  exit_guard_progress_per_unit: 0.0
  search_move_per_unit: 0.22
  time_penalty: -0.01
  wall_penalty: -0.15
  idle_penalty: -0.35
  potential:
    enabled: false
    use_primary_distance: true
    capture_pressure: 0.0
    team_visibility: 0.0
    objective_denial: 0.0
    exit_guard: 0.0
    encirclement: 0.0

player:
  escape: 0.0
  caught: 0.0
  timeout_loss: 0.0
  stalemate: -80.0
  timeout_score_progress_bonus: 150.0
  stalemate_score_progress_bonus: 120.0
  collect_note: 24.0
  collect_exam: 42.0
  collect_coffee: 6.0
  collect_freeze: 8.0
  objective_complete_bonus: 55.0
  detection_event_penalty: -10.0
  primary_seen_per_step: -0.04
  support_seen_per_step: 0.0
  multi_seen_penalty_per_step: 0.0
  goal_progress_per_unit: 1.30
  evade_progress_per_unit: 0.30
  time_penalty: -0.015
  wall_penalty: -0.15
  idle_penalty: -0.28
  potential:
    enabled: false
    objective_progress: 0.0
    target_navigation: 0.0
    escape_navigation: 0.0
    threat_margin: 0.0
    exit_window: 0.0

anti_exploit:
  no_progress_penalty: -0.08
  stationary_threshold: 0.05
  oscillation_path_threshold: 0.08
  oscillation_net_threshold: 0.04
  player_stationary_penalty: -0.30
  enemy_stationary_penalty: -0.36
  player_oscillation_penalty: -0.42
  enemy_oscillation_penalty: -0.48
```

#### 11.10.7 背後的 RL / 系統設計依據

這次不是只「把 reward 調大」而已，而是對齊幾個更穩的設計原則：

- Ng / Harada / Russell 的 PBRS 理論：如果 potential shaping 太多、太重，dense total 會壓過 terminal，policy 反而學成「不要結束局」
- OpenAI Hide-and-Seek / DeepMind Capture the Flag：獎勵要鼓勵物理上真的靠近目標、切路線、拉開距離，而不是只 farm 抽象 visibility
- 在這個專案的實測裡，`zero_sum_mix` 即使只留一點，也仍然容易把雙方推回互相拖住的平衡，所以目前直接降到 `0.0`
- observation 直接補上出口向量、主敵人 / 支援敵人向量、玩家當前目標向量，比繼續堆 visibility shaping 更能讓 policy 長出策略

#### 11.10.8 舊 checkpoint 要重練嗎

一句話：**兩個 mode 都建議從零重練，而且舊 checkpoint 不能直接跟這版混用。**

原因：

- 這一輪不只 reward 變，observation 維度也變了
  - enemy obs：`37 -> 41`
  - player obs：`42 -> 48`
- 只要是舊 observation revision 的 checkpoint，就不該直接拿來：
  - 續訓單訓
  - 當 self-play 對手池
  - 跟新版 checkpoint 混播做公平比較
- [`library_escape/train/common.py`](../library_escape/train/common.py) 現在已經會自動跳過 `obsnorm` 維度不相容的舊 history model，避免訓練時 shape mismatch 直接崩掉
- 但如果你手動從 GUI 或 CLI 指向很舊的 `.zip`，還是建議你自己當成不相容版本，不要硬接

最安全的做法：

- `Escape` 重新訓練
- `Collection` 重新訓練
- 舊 run 當歷史對照，不當續訓起點

---

## 12. 環境參數要去哪裡改

主要改 [`configs/env.yaml`](../configs/env.yaml)。

注意：

- `configs/env.yaml` 是 base profile
- 真正進入 `Collection` 或 `Escape` 時，還會再經過 [`library_escape/game_modes.py`](../library_escape/game_modes.py) 的 mode-specific 覆蓋
- 所以如果你想改「兩個模式共用的底層物理 / 觀測 / action / randomization」，優先改 `env.yaml`
- 如果你想改「Collection 預設多難」或「Escape 預設多難」，優先改 `game_modes.py`

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
  enemy_speed: 3.70
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
  chase_speed_multiplier: 1.14
  reaction_interval_seconds: 0.20
  max_turn_rate_deg_per_sec: 0.0
```

你可以調：

- 視野距離
- 視野角度
- 看見玩家時加速倍率
- `max_turn_rate_deg_per_sec`

目前補充：

- `max_turn_rate_deg_per_sec` 目前 base config 與 `Collection` / `Escape` mode preset 都維持 `0.0`
- 也就是說，這一輪沒有把「敵人轉頭速度限制」正式納入遊戲規則；相關程式支援還在，但預設關閉，避免偷偷改壞玩法

### 12.4 enemy_team

```yaml
enemy_team:
  support_count: 3
  support_speed_scale: 0.98
  support_vision_range_scale: 0.95
  support_vision_angle_scale: 0.94
  shared_last_seen: true
  shared_detection_cooldown_seconds: 1.00
```

這一段控制：

- 場上總共有幾個支援敵人
- 支援敵人的速度與視野要比主敵人弱多少
- 敵方是否共享最後目擊位置
- 經典規則下多敵人共同看到玩家時，扣秒的共享冷卻

### 12.5 collectibles

```yaml
collectibles:
  notes: 6
  exams: 2
  coffee: 1
  freeze: 1
```

你可以直接改場上道具數量。

### 12.6 observation

前面第 10 節已經詳細說明。

目前補充：

- base `env.yaml` 還是寫 `partial_observability: true`
- 但現在 `Collection` / `Escape` mode preset 都會覆蓋成 `false`
- 如果你之後想回到比較 stealth / partial-observable 的版本，請優先改 [`library_escape/game_modes.py`](../library_escape/game_modes.py)，不要只改 base YAML

### 12.7 action

前面第 9 節已經詳細說明。

### 12.8 randomization

```yaml
randomization:
  enabled: false
  min_agent_spawn_distance: 8.0
  player_speed_scale_range: [0.93, 1.07]
  enemy_speed_scale_range: [0.93, 1.10]
  vision_range_scale_range: [0.88, 1.12]
  vision_angle_jitter_deg: 10.0
  support_count_range: [2, 4]
```

這塊就是 `domain randomization`。

意思是：

- 訓練時不要永遠只在一個固定世界裡背板
- 而是每局從一個參數分布中重新抽樣
- 讓 policy 學到的是「在一整段環境變動範圍內都能 work 的策略」

這塊是為了訓練泛化能力，也是強化學習實作裡很常見的做法。

你可以讓每局：

- 玩家出生位置不同
- 敵人出生位置不同
- 移速輕微變動
- 視野距離與角度輕微變動
- 支援敵人數量輕微變動

這個專案的訓練 preset 也會在 `fast / balanced / quality` 對兩個 `Game mode` 分別開不同程度的 randomization，所以 `Escape` 和 `Collection` 不會永遠只練同一個開局。

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

如果你是在 GUI 中途按 `Stop`：

- GUI 仍會盡量補寫 `training_summary.json`
- 所以這個 run 不一定要完整 train 完，`Results` 也能先看得到

如果你之後按了 GUI 的 `Compact Run`：

- 會保留 `training_summary.json`
- 會保留最新可回播 checkpoint
- 也會保留該 checkpoint 需要的 `*.meta.json`、`*.obsnorm.npz`、`vecnormalize.pkl`
- 但 `tb/`、`monitor/`、額外 checkpoints、`progress_history.jsonl`、`eval_history.jsonl` 等大檔可能會被刪掉

也就是說：

- compact 之後適合「保留最終可用模型 + 最小紀錄」
- 不適合拿來保留完整訓練歷史細節

---

## 16. 訓練演算法與參數要去哪裡改

訓練設定檔：

- [`configs/training.yaml`](../configs/training.yaml)

### 16.1 單智能體目前預設參數

```yaml
single_agent:
  algorithm: maskable_ppo
  total_timesteps: 360000
  n_envs: 8
  learning_rate: 0.0002
  n_steps: 1024
  batch_size: 256
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.18
  ent_coef: 0.015
  vf_coef: 0.5
```

### 16.2 Self-play 目前預設參數

```yaml
self_play:
  algorithm: league_maskable_ppo
  rounds: 5
  timesteps_per_round: 140000
  n_envs: 4
  learning_rate: 0.0002
  n_steps: 1024
  batch_size: 256
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.18
  ent_coef: 0.015
  vf_coef: 0.5
```

### 16.3 VecNormalize 參數

```yaml
vec_normalize:
  enabled: true
  norm_obs: true
  norm_reward: true
  clip_obs: 10.0
  clip_reward: 15.0
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
    max_history_pool: 12
    random_weight: 0.15
    heuristic_weight: 0.45
    history_weight: 0.40
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
    bootstrap_random_weight: 0.15
    bootstrap_heuristic_weight: 0.85
    latest_weight: 0.35
    historical_weight: 0.65
    max_history_pool: 8
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
3. 選 game mode
4. 選 preset
5. 選 algorithm
6. 按 `Start Training`

補充：

- 如果你中途想停，可以按 `Stop`
- 如果這次 run 很失敗、不想留一堆大檔，可以按 `Compact Run`

### 18.2 CLI：訓練敵人

```powershell
python -m library_escape.train.train_enemy --game-mode escape
```

快速測試版：

```powershell
python -m library_escape.train.train_enemy --game-mode escape --preset fast
```

自訂 timesteps / envs：

```powershell
python -m library_escape.train.train_enemy --game-mode escape --timesteps 300000 --n-envs 8
```

### 18.3 CLI：訓練玩家

```powershell
python -m library_escape.train.train_player --game-mode collection
```

### 18.4 CLI：Self-play

```powershell
python -m library_escape.train.train_selfplay --game-mode escape
```

自訂 rounds / timesteps-per-round：

```powershell
python -m library_escape.train.train_selfplay --game-mode escape --rounds 4 --timesteps-per-round 120000
```

### 18.5 CLI：Maskable PPO

PowerShell 版本建議這樣寫：

```powershell
$json = '{"train":{"algorithm":"maskable_ppo"}}'
python -m library_escape.train.train_enemy --game-mode escape --preset balanced --overrides-json $json
```

self-play 的 maskable：

```powershell
$json = '{"train":{"algorithm":"league_maskable_ppo"}}'
python -m library_escape.train.train_selfplay --game-mode collection --preset balanced --overrides-json $json
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
- 長時間徹夜跑：用 `overnight`

`overnight` preset 的設計目標是：

- 給中端機長時間 self-play 用
- opponent pool 比 `balanced` 更大
- latest / historical 對手比例更接近 league 訓練
- 支援敵人數量 randomization 更強
- entropy 會隨訓練逐步下降，前期保留探索、後期收斂成更穩定的競技策略

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
- `capture_pressure`
- `objective_denial`
- `exit_guard`
- `target_navigation`
- `escape_navigation`
- `threat_margin`
- `exit_window`

---

## 20. 訓練結果要去哪裡看

### 20.1 run 目錄

目前預設輸出：

- 敵人：[`checkpoints/enemy/`](../checkpoints/enemy)
- 玩家：[`checkpoints/player_single/`](../checkpoints/player_single)
- self-play：[`checkpoints/selfplay/`](../checkpoints/selfplay)

目前正式使用中的 run 目錄規則是：

- 敵人單獨訓練：`checkpoints/enemy/<game_mode>/<run_name>/`
- 玩家單獨訓練：`checkpoints/player_single/<game_mode>/<run_name>/`
- self-play：`checkpoints/selfplay/<game_mode>/<run_name>/`

其中 `<game_mode>` 只有：

- `collection`
- `escape`

所以你之後找檔案時，請直接照這 6 種路徑看：

- `checkpoints/enemy/collection/<run_name>/`
- `checkpoints/enemy/escape/<run_name>/`
- `checkpoints/player_single/collection/<run_name>/`
- `checkpoints/player_single/escape/<run_name>/`
- `checkpoints/selfplay/collection/<run_name>/`
- `checkpoints/selfplay/escape/<run_name>/`

### 20.1.1 目前整理後的 checkpoint 規則

我已經把舊版殘留輸出與 smoke 測試輸出搬去 archive：

- [`checkpoints/_archive/`](../checkpoints/_archive)

這代表：

- `checkpoints/enemy/`
- `checkpoints/player_single/`
- `checkpoints/selfplay/`

現在應該只拿來放**目前這版正式結構**的 run。

GUI 的：

- `Results`
- `TensorBoard`
- `Leaderboard`

也都已經改成會忽略 `checkpoints/_archive/`，所以你平常不需要去 archive 裡找新訓練結果。

### 20.1.2 最常見的實際例子

如果你 train 的是：

- `enemy + escape + run_name = enemy_escape_test_01`

那 run 目錄就是：

```text
checkpoints/enemy/escape/enemy_escape_test_01/
```

如果你 train 的是：

- `player + collection + run_name = player_collection_test_01`

那 run 目錄就是：

```text
checkpoints/player_single/collection/player_collection_test_01/
```

如果你 train 的是：

- `selfplay + escape + run_name = selfplay_escape_long_01`

那 run 目錄就是：

```text
checkpoints/selfplay/escape/selfplay_escape_long_01/
```

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

### 20.2.1 哪些檔案是你最常需要看的

如果你只是想快速定位：

- 看目前進度 / ETA / FPS / mean reward：
  - `progress.json`
- 看整段訓練歷史：
  - `progress_history.jsonl`
- 看 evaluation 歷史：
  - `eval_history.jsonl`
- 看 TensorBoard event：
  - `tb/`
- 看 SB3 monitor：
  - `monitor/`
- 看這次 run 的完整摘要：
  - `training_summary.json`

### 20.2.2 GUI 裡對應到哪裡

在 GUI：

- `Train`
  - 目前正在跑的 run 路徑會顯示在 `Run dir`
  - 按 `Open Run Folder` 可直接打開
- `Results`
  - 會直接掃描這些 run 目錄裡的 `training_summary.json`
- `TensorBoard`
  - 會直接掃描 run 目錄裡的 `tb/`

### 20.3 `models/` 裡面會有什麼

常見：

- `enemy_latest.zip`
- `player_latest.zip`
- `best_model.zip`
- `enemy_round_01_latest.zip`
- `player_round_01_latest.zip`

### 20.3.1 各種訓練類型最重要的模型檔在哪裡

#### 敵人單獨訓練

最常用的是：

```text
checkpoints/enemy/<game_mode>/<run_name>/models/enemy_latest.zip
```

#### 玩家單獨訓練

最常用的是：

```text
checkpoints/player_single/<game_mode>/<run_name>/models/player_latest.zip
```

#### self-play

self-play 不是只產生一個單一 `models/` 資料夾，而是會分回合存在：

```text
checkpoints/selfplay/<game_mode>/<run_name>/
  enemy/round_01/models/...
  enemy/round_02/models/...
  player/round_01/models/...
  player/round_02/models/...
```

最終應該用哪個檔，最穩妥的方式是看：

- `training_summary.json`

裡面的：

- `final_player_model`
- `final_enemy_model`

這兩個欄位會告訴你最後建議拿去播放 / 評估的模型路徑。

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

- Collection：[`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml)
- Escape：[`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)

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
python -m library_escape.train.train_enemy --game-mode escape --preset fast
```

或直接 GUI 的 `Train`。

### 26.3 想調 reward

1. Collection 想調分數壓力就改 [`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml)
2. Escape 想調勝負對抗就改 [`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)
3. 跑 `fast` preset
4. 看 Results / TensorBoard
5. 再決定下一輪調整

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

## 27. 現在遊戲規則、勝負條件、分數與道具效果

### 27.1 現在正式有兩個 Game mode

現在 Python 版遊戲的使用者規則分成：

- `Collection`
  - 比較接近原本 C++ 主線的收集 / 計分 / 躲偵測
- `Escape`
  - 比較偏抓捕 / 逃脫 / 明確勝負

規則切換入口：

- [`library_escape/play/presets.py`](../library_escape/play/presets.py)
- [`library_escape/game_modes.py`](../library_escape/game_modes.py)

補充：

- 舊的 `classic` 只是 `Collection` 的相容別名
- 舊的 `rl` 只是 `Escape` 的相容別名
- 使用者現在應該只看 `Collection / Escape`

### 27.2 Collection 規則現在是什麼

Collection 目前的重點規則是：

- 敵人固定 `5` 人
  - `1` 個主敵人
  - `4` 個支援巡邏敵人
- 遊戲時間預設 `60` 秒
- 玩家主要目標是盡量收集桌上的 `note` / `exam`
- 沒有真正的「逃出口即獲勝」條件
- `can_player_escape()` 在 Collection 會直接回傳 `False`
- 所以 code-level outcome 不會出現 `escaped`
- 敵人看到玩家時：
  - 直接扣玩家剩餘時間 `8` 秒
  - 只有這次真正觸發偵測事件的那個敵人會進入 `1.2` 秒 detection pause
  - 其他敵人不會一起被定住
  - 同一次 spotting episode 不會一直重複刷新 pause timer
  - 團隊共享偵測冷卻 `1` 秒，避免同一瞬間或短時間內被重複狂扣秒
- Collection 不會啟用「看到就追擊」的 Escape 式追逐規則
- `coffee` 在 Collection 的效果是加快收集速度
  - `coffee_collection_multiplier = 1.5`
- `freeze` 會讓全部敵人凍結 `5` 秒
- reset 後有短暫開場保護
  - `Collection` 預設 `1.0` 秒
  - `Escape` 預設 `0.75` 秒
  - 這段時間敵人不會偵測玩家，避免一出生第一瞬間就被扣秒
- 收集不是瞬間完成
  - 需要 `0.85` 秒收集時間
  - 人類玩家要按住 `E`
  - AI baseline 也會停住並持續收集

從一般遊玩視角來看，Collection 比較像：

- 玩家想辦法在時間被扣光前拿到更高分
- 敵人想辦法多次看到玩家，拖慢收集並消耗時間

### 27.3 Collection 到底怎麼算贏

這裡要特別講清楚：

- Collection 不是嚴格的二元勝負模式
- 它比較像「計分生存挑戰」
- code-level 的回合結束通常是：
  - `time_expired`
  - 或 `stalemate`

所以：

- 對玩家來說
  - 一般意義上的「玩得好」是：
    - 在時間結束前收更多 `note` / `exam`
    - 拿到更高 `score_value`
    - 少被敵人看到
- 對敵人來說
  - 一般意義上的「防守成功」是：
    - 讓玩家時間快速被扣光
    - 壓低玩家最終分數

換句話說，Collection 沒有單獨的 `player_win = true` 或 `enemy_win = true` 旗標；它是用時間與分數來表現成敗。

### 27.4 Escape 規則現在是什麼

Escape 目前的重點規則是：

- 預設敵人 `2` 人
  - `1` 個主敵人
  - `1` 個支援巡邏敵人
- 遊戲時間預設 `60` 秒
- 玩家要先完成主目標，才能逃脫
- 預設主目標是：
  - `note` 必須全部收完
  - `exam` 預設不是必須，但可以在 [`configs/env.yaml`](../configs/env.yaml) 裡打開
- 玩家達成主目標後，還要真正走進出口區域，才算逃脫成功
- 任一敵人只要與玩家距離小於等於 `capture_radius = 0.55`，就算抓到玩家
- `coffee` 在 Escape 的效果是加快移動速度
  - `coffee_speed_multiplier = 1.50`
  - 持續 `15` 秒
- `freeze` 會讓全部敵人凍結 `5` 秒
- 收集一樣不是瞬間完成
  - 不需要按 `E`
  - 但仍然必須在互動範圍內停留到 `0.85` 秒進度滿，才算真的收集成功
- reset 後有短暫開場保護
  - `Escape` 預設 `0.75` 秒
  - 這段時間敵人不會立刻偵測或追擊玩家
- Escape 會啟用「看到玩家就追擊」規則

### 27.5 Escape 模式怎麼算玩家贏、敵人贏

#### 玩家贏

滿足下面兩件事：

1. 已完成逃脫前置條件
   - 預設是所有 `note` 收完
   - 如果你把 `require_all_exams_to_escape = true` 打開，那就還要把 `exam` 也收完
2. 玩家真的走進 `escape_zone`

這時：

- `events.player_escaped = True`
- `world.outcome = "escaped"`
- player reward 會拿到 `escape = +220.0`
- enemy reward 會吃到 `lose_on_escape = -220.0`

#### 敵人贏

最直接的情況是：

- 任一敵人碰到玩家到抓捕半徑內

這時：

- `events.player_caught = True`
- `world.outcome = "caught"`
- enemy reward 會拿到 `catch_player = +220.0`
- player reward 會吃到 `caught = -220.0`

此外在 Escape 訓練視角裡，下面兩種也可以視為「玩家失敗、敵方達成防守」：

- `time_expired`
- `stalemate`

其中 `time_expired` 目前對 reward 的影響是：

- enemy: `timeout_win = +48.0`
- player: `timeout_loss = -48.0`

### 27.6 訓練時到底在學什麼

這件事情要同時看：

- `Train mode`
  - `enemy` / `player` / `selfplay`
- `Game mode`
  - `collection` / `escape`

例如：

- `enemy + collection`
  - 練的是 Collection 規則下的敵人，目標是更會壓分、更會維持視野壓力
- `player + collection`
  - 練的是 Collection 規則下的玩家，目標是更會偷收集、更會走位保分
- `enemy + escape`
  - 練的是 Escape 規則下的敵人，目標是更會追擊、包夾、守出口
- `player + escape`
  - 練的是 Escape 規則下的玩家，目標是更會完成主目標並成功逃脫
- `selfplay + collection`
  - 雙方都在學 Collection 規則下的得分與壓分對抗
- `selfplay + escape`
  - 雙方都在學 Escape 規則下的圍捕與逃脫對抗

### 27.7 訓練完之後應該去哪個模式播放

原則很簡單：

- Collection 訓練出的 checkpoint，就放回 Collection 播放
- Escape 訓練出的 checkpoint，就放回 Escape 播放

目前 checkpoint 目錄也會直接分開：

- `checkpoints/enemy/collection/...`
- `checkpoints/enemy/escape/...`
- `checkpoints/player_single/collection/...`
- `checkpoints/player_single/escape/...`
- `checkpoints/selfplay/collection/...`
- `checkpoints/selfplay/escape/...`

GUI 也會把 `game_mode` 寫進 `training_summary.json` 與 model metadata，播放時如果模式不一致，會先警告你。

### 27.8 分數怎麼算

畫面上的 `score_value` 現在是：

```text
score_value = note_count * 8 + exam_count * 12
```

對應程式：

- [`library_escape/core/world.py`](../library_escape/core/world.py)

也就是：

- 每拿到一個 `note`，分數加 `8`
- 每拿到一個 `exam`，分數加 `12`
- `coffee` 和 `freeze` 不會增加 `score_value`

補充：

- `world.score` 內部會記錄各種 collectible 的收集數量
- 但 HUD 上真正顯示的加權總分，目前只看 `note` 與 `exam`

### 27.9 道具效果現在是什麼

#### `note`

- 主目標道具
- 會增加 `score_value`
- 在 Escape 規則裡通常是逃脫必要條件

#### `exam`

- 次主目標道具
- 會增加 `score_value`
- Escape 規則裡預設不是逃脫必要條件
- 但可以在 [`configs/env.yaml`](../configs/env.yaml) 把 `require_all_exams_to_escape` 改成 `true`

#### `coffee`

- `Collection`
  - 提升收集速度
- `Escape`
  - 提升移動速度

#### `freeze`

- 讓所有敵人進入凍結狀態
- 目前持續 `5` 秒

### 27.10 地圖與生成規則現在是什麼

目前預設不是程序化生成地圖，而是固定讀：

- [`configs/map.json`](../configs/map.json)

裡面的內容包括：

- player spawn
- 主敵人 spawn
- 支援敵人 spawn
- 障礙物 / 桌子 / 書櫃
- collectible spawn 點
- escape zone

也就是說：

- 地圖布局本身預設是固定的
- RL 訓練主要隨機化的是速度、視野、支援敵人數量等參數
- 不是每回合重生一張完全不同的新地圖

---

## 28. 最後總結

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
