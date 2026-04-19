# Library Escape 操作版快速手冊

這份文件是給「我現在只想知道怎麼操作」的人看的。

如果你目前不想先理解完整架構，只想知道：

- 怎麼開遊戲
- 怎麼選模式
- 怎麼訓練敵人
- 怎麼訓練玩家
- 怎麼讓兩邊自己互相訓練
- 怎麼看訓練結果
- 我要改哪些參數

那你先看這份就夠了。

完整技術版文件在：

- [`docs/OPERATION_GUIDE.md`](./OPERATION_GUIDE.md)

如果你現在最想知道的是「目前這版遊戲到底怎麼算贏、怎麼算分、強化學習 reward 怎麼給」，可以直接跳到：

- 本文的 `## 22. 目前遊戲規則 / 勝負條件 / 強化學習規格速查`

---

## 1. 先記住這個最簡單的用法

### 1.1 最推薦的入口

直接開 GUI：

```powershell
python -m library_escape.gui.app
```

如果你只記一個指令，就記這個。

因為 GUI 裡已經可以做：

- 開遊戲
- 選模式
- 選模型
- 開始訓練
- 看 ETA
- 看結果
- 看 TensorBoard
- 建 Elo leaderboard
- 看 replay

補充：

- 現在 GUI 和 CLI 都統一用兩個使用者模式：`Collection` 與 `Escape`
- `Collection` 比較接近原本 C++ 的計分 / 躲偵測玩法，固定 `5` 個敵人
- `Escape` 是明確的逃脫對抗玩法，預設 `2` 個敵人（`1` 個主敵人 + `1` 個支援巡邏敵人）
- 你在 `Train` 裡面訓練時，一定要先選 `Game mode`
- 訓練好的 checkpoint，通常應該放回**相同的 Game mode** 裡播放
- 如果同一張桌上有多個道具，現在會優先選你實際比較靠近的那個，不會再固定先拿書
- 現在所有模式都不會瞬間拿書
- `Collection` 的人類玩法要長按 `E`
- `Escape` 與 `AI vs AI` 會在道具旁停留到進度滿才完成收集
- 開局會有很短的保護時間，避免你一出生第一瞬間就被偵測到
- Python 版已經把原 repo 內仍存在的背景音樂 / 偵測音效 / 結算成績音效接回來

---

## 2. 第一次使用的完整準備

### 2.1 建立 Python 環境

請使用 `Python 3.12`。

先切到專案根目錄：

```powershell
cd "C:\Users\User\Desktop\大四其他\library-escape-game"
```

再建立專案內自己的 `.venv`：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[rl,dev]"
```

### 2.2 驗證有沒有安裝好

先確認你現在真的用到的是**專案內的** Python，不是家目錄裡其他同名 `.venv`：

```powershell
python -c "import sys; print(sys.executable)"
```

你應該看到像這樣的路徑：

```text
C:\Users\User\Desktop\大四其他\library-escape-game\.venv\Scripts\python.exe
```

不是這種：

```text
C:\Users\User\.venv\Scripts\python.exe
```

確認無誤後再啟動 GUI：

```powershell
python -m library_escape.gui.app
```

如果 GUI 可以打開，代表基本上就能用了。

### 2.3 你剛剛遇到過的兩個常見問題

#### 問題 A：`pip install -e .[rl,dev]` 說目前資料夾不是 Python project

錯誤長這樣：

```text
ERROR: file:///C:/Users/User does not appear to be a Python project
```

原因：

- 你在 `C:\Users\User` 執行了安裝
- 但 `pyproject.toml` 不在那裡
- `.` 代表目前資料夾，所以 pip 裝錯地方了

正確做法：

```powershell
cd "C:\Users\User\Desktop\大四其他\library-escape-game"
pip install -e ".[rl,dev]"
```

#### 問題 B：啟動 GUI 時 NumPy 爆掉，出現 `cp314` / `numpy._core._multiarray_umath`

你剛剛遇到的是這種錯誤：

```text
ImportError: Importing the numpy C-extensions failed
...
_multiarray_umath.cp314-win_amd64.pyd
```

原因：

- 你啟用的是錯的 `.venv`
- 你實際用到的是 `C:\Users\User\.venv`
- 那個 venv 裡的 NumPy 安裝壞掉了，混進了 `cp314` 檔案
- 但你現在跑的是 Python 3.12

最直接修法：

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

如果你想徹底避免之後再踩到，可以把家目錄那個壞掉的 venv 刪掉：

```powershell
Remove-Item -Recurse -Force C:\Users\User\.venv
```

### 2.4 一個很重要的觀念

PowerShell prompt 前面出現：

```text
(.venv)
```

**不代表你一定在用正確的 `.venv`**。

因為你可能同時有：

- `C:\Users\User\.venv`
- `C:\Users\User\Desktop\大四其他\library-escape-game\.venv`

它們名字都叫 `.venv`，prompt 看起來會一樣。

所以最可靠的檢查方式永遠是：

```powershell
python -c "import sys; print(sys.executable)"
```

---

## 3. 先分清楚兩個不同概念：Game Mode 與 Train Mode

這是整個專案最容易混淆的地方。

你要先把它拆成兩層來理解：

- `Game mode`
  - 這是在說「這一局遊戲的規則是什麼」
- `Train mode`
  - 這是在說「這一次訓練，到底是訓練敵人、訓練玩家，還是雙方輪流訓練」

### 3.1 Game mode A：`Collection`

你可以把它理解成：

- 比較接近原本 C++ 的收集 / 計分玩法
- 玩家目標是：在時間結束前盡量收集 `note` / `exam`
- 沒有真正的「跑到出口就贏」條件
- 敵人目標是：看到玩家、扣時間、把玩家最終分數壓低
- 固定 `5` 個敵人

這個模式訓練出來的策略，通常會偏向：

- 偷路線
- 卡視角
- 搶高分桌位
- 壓收集節奏
- 防守玩家的分數成長

### 3.2 Game mode B：`Escape`

你可以把它理解成：

- 明確的逃脫對抗玩法
- 玩家目標是：先收完必要目標，再走到出口
- 敵人目標是：抓到玩家，或至少拖到時間結束
- 預設 `2` 個敵人（`1` 主敵人 + `1` 支援巡邏敵人）
- 訓練 preset 會再逐步把支援敵人數量往上拉，讓後期 self-play 仍然能學到包夾與守出口

這個模式訓練出來的策略，通常會偏向：

- 抄近路
- 包夾
- 守出口
- 圍堵
- 目標路線預判

### 3.3 Train mode A：訓練敵人 `enemy`

意思是：

- 玩家先用規則式 controller
- 敵人是 RL 模型
- 你是在練「敵人怎麼變強」

### 3.4 Train mode B：訓練玩家 `player`

意思是：

- 敵人先用規則式 controller
- 玩家是 RL 模型
- 你是在練「玩家怎麼變強」

### 3.5 Train mode C：`selfplay`

意思是：

- 兩邊都訓練
- 會交替訓練 enemy / player
- 你是在練整個對抗系統一起升級

### 3.6 最重要的實際規則

你在 `Train` 頁面訓練時，一定會同時決定兩件事：

1. `Mode`
   - `enemy` / `player` / `selfplay`
2. `Game mode`
   - `collection` / `escape`

例如：

- `Mode = enemy` + `Game mode = collection`
  - 代表你在訓練「Collection 規則下的敵人」
- `Mode = player` + `Game mode = escape`
  - 代表你在訓練「Escape 規則下的玩家」
- `Mode = selfplay` + `Game mode = escape`
  - 代表你在訓練「Escape 規則下的玩家與敵人交替對抗」

### 3.7 訓練完之後要放到哪裡播放

原則非常簡單：

- **Collection 訓練出的 checkpoint，就放回 Collection 模式播放**
- **Escape 訓練出的 checkpoint，就放回 Escape 模式播放**

例如：

- 你用 `--game-mode collection` 訓練敵人
  - 之後播放也應該用 `--game-mode collection`
- 你用 `--game-mode escape` 做 self-play
  - 之後觀戰也應該用 `--game-mode escape`

GUI 現在也會把 `Game mode` 寫進訓練摘要與 checkpoint metadata，播放時如果你選錯模式，會先警告你。

---

## 4. 我要怎麼開遊戲

有兩種方式。

### 4.1 方法 A：用 GUI

```powershell
python -m library_escape.gui.app
```

然後進入 `Play` 分頁。

你會先看到兩個最重要的欄位：

- `Mode`
- `Game mode`

### 4.2 方法 B：直接用指令

#### 人類玩，Collection 模式，敵人是規則式 AI

```powershell
python -m library_escape.play.human_vs_ai --game-mode collection
```

#### 人類玩，Escape 模式，敵人是你訓練好的模型

```powershell
python -m library_escape.play.human_vs_ai --game-mode escape --enemy-model checkpoints\enemy\escape\你的_run_name\models\enemy_latest.zip
```

#### AI 對 AI，Collection 模式

```powershell
python -m library_escape.play.ai_vs_ai --game-mode collection
```

#### AI 對 AI，Escape 模式，指定雙方模型

```powershell
python -m library_escape.play.ai_vs_ai --game-mode escape --player-model checkpoints\player_single\escape\你的_run_name\models\player_latest.zip --enemy-model checkpoints\enemy\escape\你的_run_name\models\enemy_latest.zip
```

---

## 5. 開遊戲之後我要怎麼選模式

### 5.1 GUI 裡怎麼選

進 `Play` 分頁後：

1. 先選 `Mode`
   - `Human vs Rule Enemy`
   - `Human vs Enemy Checkpoint`
   - `AI vs AI`
2. 再選 `Game mode`
   - `collection`
   - `escape`
3. 如果要用模型，就在右邊選 `.zip`
4. 按 `Start Game`

### 5.2 每個 `Mode` 代表什麼

#### `Human vs Rule Enemy`

適合：

- 先測試遊戲有沒有正常
- 不用模型也可以玩

#### `Human vs Enemy Checkpoint`

適合：

- 用你訓練好的敵人模型來跟人類玩

#### `AI vs AI`

適合：

- 不載模型時，用 script baseline 快速檢查遊戲流程
- 看兩個模型對戰
- 測試訓練成果

### 5.3 每個 `Game mode` 代表什麼

#### `collection`

適合：

- 想玩接近原本 C++ 的收集 / 計分感
- 想訓練搶分、隱匿、干擾收集

#### `escape`

適合：

- 想看明確的抓捕 vs 逃脫
- 想訓練守出口、圍捕、抄路線、路徑規劃

補充：

- 如果你沒有指定 `player` / `enemy` checkpoint，`AI vs AI` 跑的是內建 heuristic / patrol controller
- 它應該會巡邏、繞障礙、收集，不應該整批卡在牆邊發呆
- 真正要看 RL 策略，請載入訓練後的 `.zip` checkpoint

---

## 6. 我只是想先玩，不想訓練

### 最簡單步驟

1. 開 GUI

```powershell
python -m library_escape.gui.app
```

2. 去 `Play`
3. 選 `Human vs Rule Enemy`
4. 按 `Start Game`

這是最簡單上手方式。

---

## 7. 我要怎麼開始訓練模型

你有兩條路：

- GUI 路線
- CLI 指令路線

如果你不熟，**先用 GUI**。

---

## 8. GUI 路線：怎麼訓練

### 8.1 開 GUI

```powershell
python -m library_escape.gui.app
```

### 8.2 進 `Train` 分頁

這裡你會看到幾個重要欄位：

- `Mode`
- `Game mode`
- `Preset`
- `Run name`
- `Timesteps / round`
- `Rounds`
- `Vector envs`
- `Seed`
- `Device`
- `Algorithm`

介面補充：

- `Train` 頁現在是可伸縮的 responsive 版面，不是固定死的比例
- 右側上半部是進度摘要，下半部是 `Logs`
- `Logs` 區有水平 / 垂直捲軸，視窗縮放時不會再被壓成幾乎看不到的一條
- Windows 高 DPI 縮放下，GUI 會先啟用 DPI awareness，150% 之類的縮放比例文字會比以前清楚

### 8.3 `Mode` 要選什麼

- `enemy`：訓練敵人
- `player`：訓練玩家
- `selfplay`：雙方輪流訓練

### 8.4 `Game mode` 要選什麼

- `collection`
  - 訓練計分 / 收集 / 壓分玩法
- `escape`
  - 訓練抓捕 / 逃脫 / 守出口玩法

新手建議：

1. 如果你想先看最明確的勝負，先用 `escape`
2. 如果你想保留原版收集感，也要另外跑一組 `collection`

### 8.5 `Preset` 要選什麼

- `fast`：快速測試
- `balanced`：一般推薦
- `quality`：更久、更重、更強
- `overnight`：專門給長時間 self-play / 長時間單邊訓練的版本，預設已經加大 opponent pool、加強 randomization，並把探索強度隨訓練逐步降下來

新手建議：

1. 先用 `fast`
2. 確認流程都會跑
3. 再改 `balanced`
4. 真正要睡前長跑，再用 `overnight`

### 8.6 `Algorithm` 要選什麼

如果 `Mode = enemy` 或 `player`：

- `ppo`
- `maskable_ppo`

如果 `Mode = selfplay`：

- `league_ppo`
- `league_maskable_ppo`
- `mappo_recipe`

新手建議：

- 單邊訓練直接用 `maskable_ppo`
- self-play 直接用 `league_maskable_ppo`
- 只有你想做特殊對照實驗時，再手動改回 `ppo` 或 `league_ppo`

### 8.7 `Run name` 要填什麼

建議填一個你看得懂的名字，例如：

- `enemy_escape_test_01`
- `player_collection_fast_01`
- `selfplay_escape_balanced_v1`

### 8.8 `Timesteps / round`

如果是：

- `enemy` / `player`：這就是總訓練 timesteps
- `selfplay`：這是一回合的 timesteps

### 8.9 `Rounds`

只有 self-play 會用到。

例如：

- `Rounds = 4`

表示：

1. 訓練 enemy 第 1 輪
2. 訓練 player 第 1 輪
3. 訓練 enemy 第 2 輪
4. 訓練 player 第 2 輪
5. 一直做下去

### 8.10 `Vector envs`

這是同時開幾個環境來收資料。

建議：

- 中端機：`1`、`2`、`4`

如果你不確定，先從：

```text
1
```

開始。

### 8.11 `Device`

- `auto`
- `cpu`
- `cuda`

如果你不確定，先用：

```text
auto
```

---

## 9. GUI 路線：訓練敵人

### 9.1 操作步驟

1. 開 GUI
2. 進 `Train`
3. `Mode` 選 `enemy`
4. `Game mode` 選 `escape`
5. `Preset` 選 `fast`
6. `Algorithm` 選 `maskable_ppo`
7. `Run name` 填 `enemy_escape_test_01`
8. 按 `Start Training`

### 9.2 你會看到什麼

右邊會顯示：

- `Run dir`
- `Phase`
- `Progress`
- `Elapsed`
- `ETA`
- `FPS`
- `Mean reward`

下方 `Logs` 會顯示訓練輸出。

補充：

- `Logs` 現在是 monospace 顯示，長輸出可以水平捲動
- 右側中間的分隔條可以拖動，所以你想多看 logs 時，可以直接把 logs 區拉高

### 9.3 訓練結束後模型在哪裡

通常在：

```text
checkpoints/enemy/escape/enemy_escape_test_01/models/enemy_latest.zip
```

---

## 10. GUI 路線：訓練玩家

### 10.1 操作步驟

1. 開 GUI
2. 進 `Train`
3. `Mode` 選 `player`
4. `Game mode` 選 `collection` 或 `escape`
5. `Preset` 選 `fast`
6. `Algorithm` 選 `maskable_ppo`
7. `Run name` 填 `player_escape_test_01`
8. 按 `Start Training`

### 10.2 訓練結果在哪裡

通常在：

```text
checkpoints/player_single/escape/player_escape_test_01/models/player_latest.zip
```

---

## 11. GUI 路線：讓他們自己互相訓練

這就是 self-play。

### 11.1 操作步驟

1. 開 GUI
2. 進 `Train`
3. `Mode` 選 `selfplay`
4. `Game mode` 選 `escape`
5. `Preset` 選 `fast`
6. `Algorithm` 選 `league_maskable_ppo`
7. `Run name` 填 `selfplay_escape_test_01`
8. `Rounds` 填 `2` 或 `4`
9. 按 `Start Training`

### 11.2 這個模式內部會做什麼

它不是兩邊同時一步到位一起學，而是交替：

1. 先訓練敵人
2. 再訓練玩家
3. 再訓練敵人
4. 再訓練玩家

### 11.3 訓練結果在哪裡

通常在：

```text
checkpoints/selfplay/escape/selfplay_escape_test_01/
```

裡面會分成：

- `enemy/round_01/...`
- `player/round_01/...`
- `enemy/round_02/...`
- `player/round_02/...`

最終模型通常會記在：

- `training_summary.json`

你也可以直接去 GUI 的 `Results` 頁選這個 run。

### 11.4 以後我訓練完到底要去哪裡找檔案

先記這個總規則：

- 敵人單獨訓練：`checkpoints/enemy/<game_mode>/<run_name>/`
- 玩家單獨訓練：`checkpoints/player_single/<game_mode>/<run_name>/`
- self-play：`checkpoints/selfplay/<game_mode>/<run_name>/`

其中 `<game_mode>` 只有兩種：

- `collection`
- `escape`

所以你之後最常看的實際位置就是：

```text
checkpoints/enemy/collection/<run_name>/
checkpoints/enemy/escape/<run_name>/
checkpoints/player_single/collection/<run_name>/
checkpoints/player_single/escape/<run_name>/
checkpoints/selfplay/collection/<run_name>/
checkpoints/selfplay/escape/<run_name>/
```

如果你想找「最重要的模型檔」，直接看：

- 敵人單訓：`models/enemy_latest.zip`
- 玩家單訓：`models/player_latest.zip`
- self-play：看 `training_summary.json` 裡記錄的 `final_player_model` / `final_enemy_model`

self-play 的 run 裡通常會長這樣：

```text
checkpoints/selfplay/escape/<run_name>/
  enemy/round_01/models/...
  enemy/round_02/models/...
  player/round_01/models/...
  player/round_02/models/...
  progress.json
  progress_history.jsonl
  eval_history.jsonl
  training_summary.json
```

如果你想找「訓練過程的 logs」，最常看的是：

- `progress.json`
  - 現在這一刻的進度、ETA、FPS、mean reward
- `progress_history.jsonl`
  - 每次進度更新的歷史紀錄
- `eval_history.jsonl`
  - 每次 evaluation 的歷史紀錄
- `monitor/`
  - SB3 monitor 輸出
- `tb/`
  - TensorBoard event 檔

如果你是用 GUI 半途按 `Stop`：

- GUI 仍會補寫一份可被 `Results` 讀懂的 `training_summary.json`
- 所以就算沒有完整 train 完，這個 run 也還是會出現在 `Results`
- 這種 run 的 summary 會標示 `status = stopped_early`

如果你後來按了 GUI 的 `Compact Run`：

- 會保留最新可回播的 checkpoint
- 也會保留必要的 `*.meta.json`、`*.obsnorm.npz`、`vecnormalize.pkl`、`training_summary.json`
- 額外 checkpoints、`tb/`、`monitor/`、`progress_history.jsonl`、`eval_history.jsonl` 這些大檔會被刪掉
- 所以 compact 之後，你還是可以在 `Results` 看摘要，也還是可以拿保留下來的最新模型回播

你現在也可以把 `checkpoints/_archive/` 當成「舊資料倉庫」來理解：

- 以前整理前的舊 run、smoke run、舊版殘留資料都搬去那裡了
- GUI 的 `Results` / `TensorBoard` / `Leaderboard` 現在會忽略它
- 所以你平常**不要去 `_archive` 找新訓練結果**

最簡單的做法其實是：

1. 在 GUI 的 `Train` 開始訓練
2. 訓練結束後按 `Open Run Folder`
3. 或去 GUI 的 `Results` 頁直接選你的 run

### 11.5 GUI 各分頁到底差在哪裡

這是很多人第一次用時最容易混淆的地方。

先記一句最重要的：

- `Play` 是**手動選模型、手動選模式、手動發射遊戲**
- `Results` 是**先選一個已存在的訓練 run，然後讓 GUI 幫你帶出那個 run 對應的模型與模式**

也就是說：

- 如果你已經知道自己要拿哪兩個 checkpoint 打，去 `Play`
- 如果你想從「某次訓練結果」直接接著看曲線、看摘要、按一下就播放，去 `Results`

#### `Play` 分頁

用途：

- 開遊戲
- 手動指定 `player` / `enemy` checkpoint
- 手動選 `Collection` / `Escape`
- 手動決定要不要錄 replay

你會看到的主要欄位：

- `Mode`
  - `Human vs Rule Enemy`
  - `Human vs Enemy Checkpoint`
  - `AI vs AI`
- `Game mode`
  - `collection`
  - `escape`
- `Seed`
- `Deterministic checkpoint playback`
- `Record replay`
- `Enemy checkpoint`
- `Player checkpoint`

主要按鈕：

- `Start Game`
  - 依你現在畫面上填的內容直接開遊戲
- `Open Replays`
  - 打開 `replays/`
- `Open Checkpoints`
  - 打開 `checkpoints/`

這一頁的特性是：

- 你可以任意組合 checkpoint
- 你可以用不同 run 訓練出來的 player / enemy 直接互打
- 你也可以一邊載 player checkpoint、一邊讓 enemy 留空，這樣它就會退回 rule-based baseline
- `Seed` 留空時，每次啟動都會重新抽一個隨機 reset；只有你真的想重現完全同一場，才手動填 seed
- `Deterministic checkpoint playback` 現在預設是開的
- 開啟時，checkpoint 每一步都走 argmax 動作，也就是永遠選目前機率最高的動作，這適合評估、展示、固定重現
- 關閉時，checkpoint 才會改成 stochastic policy 播放，從動作分布中抽樣，用來觀察策略多樣性

#### `Results` 分頁

用途：

- 看你以前 train 過的 run
- 看那個 run 的摘要與曲線
- 從那個 run 直接評估 / 播放 / 開 TensorBoard

你會看到：

- 左邊：run 清單
- 右邊：這個 run 的 `training_summary.json` 內容與圖表

主要按鈕：

- `Refresh`
  - 重新掃描 run
- `Open Folder`
  - 打開你選到的 run 資料夾
- `Compact Run`
  - 幫你把這個 run 精簡成「保留最新可回播 checkpoint + 必要 metadata」的最小集合
- `Evaluate`
  - 用這個 run 的最終模型做一次 quick evaluation
- `Play AI vs AI`
  - 用這個 run 對應的最終模型直接開一場 `AI vs AI`
- `TensorBoard Server`
  - 直接替這個 run 開 TensorBoard server

##### `Results -> Play AI vs AI` 和 `Play` 分頁自己手動選 checkpoint 有什麼差別？

兩者最後都是在開 `AI vs AI`，但差別在於「模型和模式是誰幫你決定」。

`Results -> Play AI vs AI`：

- 模型是從你選的 run 的 `training_summary.json` 自動帶出來
- `Game mode` 也是從那個 run 的摘要自動帶出來
- 預設會用隨機 seed 加 deterministic checkpoint playback
- 適合：「我剛 train 完一個 run，我現在就想看它到底學成怎樣」

`Play` 分頁手動選 checkpoint：

- 你自己決定 player model 是哪個
- 你自己決定 enemy model 是哪個
- 你自己決定 `Game mode`
- 適合：
  - 比較不同 run 的模型
  - 交叉測試不同來源的 player / enemy
  - 想錄 replay
  - 想做很自由的對戰組合

##### 再更具體地講

如果你在 `Results` 選的是：

- 一個 `enemy` 單訓 run

那 `Play AI vs AI` 其實會：

- 用這個敵人模型
- 玩家那邊留空，所以回退成 heuristic / baseline player

如果你選的是：

- 一個 `player` 單訓 run

那它會：

- 用這個玩家模型
- 敵人那邊留空，所以回退成 rule-based enemy

如果你選的是：

- 一個 `selfplay` run

那它會：

- 同時載入這個 run 最終的 player model
- 同時載入這個 run 最終的 enemy model

所以：

- 想看「這個 run 最終成果」：用 `Results`
- 想自由拼裝對戰：用 `Play`

#### 明天想接著今天的 policy 繼續練，怎麼做

現在 GUI `Train` 頁已經支援續訓。

- 如果 `Mode = enemy` 或 `player`
  - 用 `Resume checkpoint (optional)`
  - 直接選昨天的 `.zip`
  - 例如：
    - `checkpoints/enemy/escape/<run_name>/models/enemy_latest.zip`
    - `checkpoints/player_single/escape/<run_name>/models/player_latest.zip`
- 如果 `Mode = selfplay`
  - 用 `Resume self-play run (optional)`
  - 直接選昨天整個 self-play run 資料夾
  - 例如：
    - `checkpoints/selfplay/escape/selfplay_escape_overnight_01/`

`selfplay` 續訓時，系統會自動讀那個 run 的：

- `training_summary.json`
- `final_enemy_model`
- `final_player_model`

然後從那兩個最終 policy 接著往下做新一輪 self-play。

#### deterministic=True / deterministic=False 到底差在哪裡

這個專案現在採用的是下面這個標準定義：

- `deterministic=False`
  - 訓練時用
  - 從動作機率分布中抽樣
  - 目的是保留探索能力
  - 例：機率 `[0.6, 0.3, 0.1]`，就有 `60% / 30% / 10%` 機率各選一個
- `deterministic=True`
  - 評估 / 展示 / 固定重播時用
  - 直接取 argmax
  - 目的是拿到最穩定、最可重現的表現
  - 例：同樣的分布 `[0.6, 0.3, 0.1]`，永遠選第 `0` 個動作

現在 GUI 的預設是：

- `Play` 頁的 `Seed` 預設留空
- `Play` 頁的 `Deterministic checkpoint playback` 預設開啟
- `Results -> Play AI vs AI` 也預設用 deterministic 播放
- 只有你主動把 deterministic 關掉，才會切回 stochastic

所以現在：

- 想看最穩定、最像正式評估的結果：保持 deterministic 開啟
- 想看模型的多樣化策略：不要填 seed，並把 deterministic 關掉
- 想做完全重現的 debug：填固定 seed，保持 deterministic 開啟

#### 什麼是 Domain Randomization，這個專案現在怎麼用

`Domain randomization` 的意思是：訓練時不要永遠在完全一樣的地圖條件、視野、移速、出生點下學習，而是每局都對一部分環境參數做隨機擾動，讓 policy 學到的是「一整個分布」下都能 work 的策略，而不是背板。

這個專案現在在訓練 preset 裡已經會開：

- 玩家 / 敵人出生點隨機化
- 玩家 / 敵人移速微擾
- 視野距離微擾
- 視野角度抖動
- `Escape` / `Collection` 的支援敵人數量擾動

你會在這些地方改到：

- 基礎範圍：[`configs/env.yaml`](../configs/env.yaml)
- 不同訓練 preset 何時啟用：[`configs/training.yaml`](../configs/training.yaml)

如果你想要更強泛化、不要每次都背固定開局，優先保持：

- `Preset = balanced` 或 `quality`
- `randomization.enabled = true`

#### 為什麼 `Escape` 觀戰現在比較順，不會那麼像一幀一幀卡住

新版播放端不再每個 physics tick 都重新跑一次神經網路，而是改成：

- 渲染仍然每幀更新
- policy 決策頻率則跟訓練時的 `rl_frame_skip` 對齊

這樣做的好處是：

- 播放分布和訓練分布比較一致
- `Escape` self-play checkpoint 觀戰時比較不容易卡頓
- 畫面仍然是連續渲染，不是退回舊版那種一跳一跳的離散刷新感

#### `Train` 分頁

用途：

- 開訓練
- 看 live progress
- 看 ETA
- 看 logs

你會填的主要欄位：

- `Mode`
  - `enemy`
  - `player`
  - `selfplay`
- `Game mode`
  - `collection`
  - `escape`
- `Preset`
- `Run name`
- `Timesteps / round`
- `Rounds (selfplay)`
- `Vector envs`
- `Seed`
- `Device`

右邊 `Algorithms And Curriculum` 是進階區：

- `Algorithm`
- single-agent opponent curriculum 權重
- self-play bootstrap / latest / historical 權重
- `MAPPO recipe command`

主要按鈕：

- `Start Training`
  - 依照目前欄位開始訓練
- `Stop`
  - 中止目前訓練
  - 如果這次 run 還沒正式寫出最終 summary，GUI 仍會補一份 `training_summary.json`
  - 所以這個 run 還是能直接出現在 `Results`
- `Compact Run`
  - 刪掉額外 checkpoints、TensorBoard、monitor、歷史曲線大檔
  - 只留下最新可回播 checkpoint 與必要 metadata
- `Open Run Folder`
  - 打開目前正在訓練的 run 目錄
- `Open TensorBoard Server`
  - 替目前這個 run 開 TensorBoard server

畫面補充：

- `Train` 右側現在會把進度摘要和 `Logs` 分開
- `Logs` 會保留比較大的預設高度，並支援水平 / 垂直捲動
- 視窗比例改變時，版面會跟著重排，不是只對某一種螢幕比例可用

#### `TensorBoard` 分頁

用途：

- 不用另外開瀏覽器，也能直接在 GUI 裡看 TensorBoard scalar

它做的事是：

- 掃描 run 目錄裡 `tb/` 下面的 event files
- 把 scalar tag 列出來
- 讓你一次選最多幾條曲線來畫
- 顯示每條曲線的 `latest / min / max`

它比較像：

- 「內建版 TensorBoard 曲線瀏覽器」

如果你想看得更完整，還是可以用：

- `Train` 頁的 `Open TensorBoard Server`
- 或 `Results` 頁的 `TensorBoard Server`

#### `Leaderboard` 分頁

用途：

- 自動替模型做 Elo 排名

你要填的主要欄位：

- `Checkpoint root`
- `Output JSON`
- `Episodes`
- `Max players`
- `Max enemies`
- `K-factor`

主要按鈕：

- `Build Leaderboard`
  - 掃描 root 下的模型並自動配對對戰
- `Load Existing`
  - 載入以前存好的 leaderboard JSON
- `Open Output`
  - 打開 leaderboard JSON

這頁的意義是：

- 如果你手上已經有很多 player / enemy checkpoint
- 想知道誰比較強
- 想用 Elo 方式做相對排名

那就用這頁。

#### `Replay` 分頁

用途：

- 看你之前錄下來的 `.ler.gz` 對局檔

它可以做的事：

- 掃描 replay 檔
- 顯示 replay metadata
- `Play Replay`
- `Export Frames`

這裡的 `Replay` 不是重新跑一場模擬，而是：

- 播放你以前錄下來的對局紀錄

所以它很適合拿來：

- 回顧某次 AI 對戰
- 檢查模型是不是做了奇怪的決策
- 匯出畫面做展示

#### `Config` 分頁

用途：

- 快速打開最常改的設定檔和說明書

它不是拿來直接在 GUI 裡改參數，而是：

- 幫你一鍵打開常用檔案

像是：

- `Quick Start Guide`
- `Collection Rewards`
- `Escape Rewards`
- `Environment`
- `Training`
- `Map`
- `Game Modes Helper`
- `Operation Guide`
- `Project Root`

如果你現在的需求是：

- 「我要改 reward」
  - 去 `Config -> Collection Rewards / Escape Rewards`
- 「我要改 observation / world 參數」
  - 去 `Config -> Environment`
- 「我要改訓練 preset / algorithm」
  - 去 `Config -> Training`

---

## 12. CLI 路線：如果你想用指令操作

這一節就是「每件事情要怎麼執行」。

### 12.1 開 GUI

```powershell
python -m library_escape.gui.app
```

### 12.2 開遊戲：人類對規則式敵人

```powershell
python -m library_escape.play.human_vs_ai --game-mode collection
```

### 12.3 開遊戲：人類對訓練好的敵人

```powershell
python -m library_escape.play.human_vs_ai --game-mode escape --enemy-model checkpoints\enemy\escape\enemy_escape_test_01\models\enemy_latest.zip
```

### 12.4 開遊戲：AI 對 AI

```powershell
python -m library_escape.play.ai_vs_ai --game-mode escape --player-model checkpoints\player_single\escape\player_escape_test_01\models\player_latest.zip --enemy-model checkpoints\enemy\escape\enemy_escape_test_01\models\enemy_latest.zip
```

### 12.5 訓練敵人

```powershell
python -m library_escape.train.train_enemy --game-mode escape
```

### 12.6 訓練敵人，用 fast preset

```powershell
python -m library_escape.train.train_enemy --game-mode escape --preset fast
```

### 12.7 訓練敵人，自訂 timesteps

```powershell
python -m library_escape.train.train_enemy --game-mode escape --timesteps 300000 --n-envs 4 --run-name enemy_escape_test_01
```

### 12.8 訓練玩家

```powershell
python -m library_escape.train.train_player --game-mode collection --preset fast --run-name player_collection_test_01
```

### 12.9 Self-play

```powershell
python -m library_escape.train.train_selfplay --game-mode escape --preset fast --rounds 2 --timesteps-per-round 60000 --run-name selfplay_escape_test_01
```

### 12.10 訓練敵人，用 Maskable PPO

PowerShell 建議這樣寫：

```powershell
$json = '{"train":{"algorithm":"maskable_ppo"}}'
python -m library_escape.train.train_enemy --game-mode escape --preset balanced --run-name enemy_escape_mask_01 --overrides-json $json
```

### 12.11 Self-play，用 league Maskable PPO

```powershell
$json = '{"train":{"algorithm":"league_maskable_ppo"}}'
python -m library_escape.train.train_selfplay --game-mode collection --preset balanced --run-name selfplay_collection_mask_01 --overrides-json $json
```

### 12.12 replay 錄製

```powershell
python -m library_escape.play.ai_vs_ai --game-mode escape --record-replay replays\match_01.ler.gz
```

### 12.13 replay 播放

```powershell
python -m library_escape.replay.viewer --replay replays\match_01.ler.gz
```

### 12.14 建 leaderboard

```powershell
python scripts\eval_elo.py --root checkpoints --episodes 8 --output checkpoints\leaderboard.json
```

---

## 13. 如果我要調參數，我要先知道哪幾種

先把參數分成 4 類。

### 13.1 遊戲規則參數

位置：

- [`configs/env.yaml`](../configs/env.yaml)

例如：

- 玩家速度
- 敵人速度
- 視野距離
- 視野角度
- 遊戲時間
- action mode
- observation mode

### 13.2 獎勵參數

位置：

- Collection 模式：[`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml)
- Escape 模式：[`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)

例如：

- 抓到人幾分
- 成功逃脫幾分
- 撿 note 幾分
- 被看到每步扣多少
- 撞牆扣多少

### 13.3 訓練參數

位置：

- [`configs/training.yaml`](../configs/training.yaml)

例如：

- `algorithm`
- `learning_rate`
- `n_steps`
- `batch_size`
- `gamma`
- `ent_coef`
- `vf_coef`
- `n_envs`

### 13.4 地圖參數

位置：

- [`configs/map.json`](../configs/map.json)

例如：

- 玩家出生點
- 敵人出生點
- 障礙物
- collectable 位置
- 出口位置

---

## 14. 我最常要調哪些參數

如果你現在沒有概念，先只看這些。

### 14.1 只想讓遊戲速度感不同

改：

- `configs/env.yaml`

重點欄位：

```yaml
world:
  player_speed: 4.00
  enemy_speed: 3.65
enemy:
  vision_range: 5.50
  vision_angle_deg: 52.0
enemy_team:
  support_count: 4
timing:
  rl_frame_skip: 4
```

### 14.2 只想讓敵人更積極學抓人

改：

- `configs/rewards_escape.yaml`

重點欄位：

```yaml
enemy:
  catch_player: 220.0
  primary_visible_per_step: 0.50
  support_visible_per_step: 0.20
  team_visible_ratio_per_step: 0.16
  potential:
    capture_pressure: 2.40
    team_visibility: 1.55
    exit_guard: 2.50
```

### 14.3 只想讓玩家更積極學逃脫

改：

- `configs/rewards_escape.yaml`

重點欄位：

```yaml
player:
  escape: 220.0
  collect_note: 14.0
  primary_seen_per_step: -0.35
  support_seen_per_step: -0.14
  potential:
    objective_progress: 2.20
    target_navigation: 1.80
    escape_navigation: 4.00
    threat_margin: 1.00
```

### 14.4 只想讓 Collection 模式更偏重分數與隱匿

改：

- `configs/rewards_collection.yaml`

重點欄位：

```yaml
player:
  collect_note: 14.0
  collect_exam: 22.0
  primary_seen_per_step: -0.40
enemy:
  player_collect_note_penalty: -16.0
  player_collect_exam_penalty: -26.0
  primary_visible_per_step: 0.55
```

### 14.5 只想讓訓練更快

改：

- `configs/training.yaml`

先從這幾個下手：

```yaml
total_timesteps
timesteps_per_round
n_envs
n_steps
batch_size
```

最簡單做法：

- 先選 `fast`

### 14.5 只想讓訓練更穩

可以先試：

- `ppo -> maskable_ppo`
- `league_ppo -> league_maskable_ppo`
- 降低 `learning_rate`
- 適度提高 `n_steps`

---

## 15. 我應該怎麼挑訓練方法

### 15.1 如果你只是想先讓敵人變強

選：

- `Mode = enemy`

適合：

- 先把敵人練好
- 人類之後拿來玩

### 15.2 如果你只是想先讓玩家變強

選：

- `Mode = player`

適合：

- 先看玩家會不會逃

### 15.3 如果你想讓兩邊都變強

選：

- `Mode = selfplay`

適合：

- 長期訓練
- 建立比較有對抗性的模型

### 15.4 如果你不知道要選哪個

建議順序：

1. `enemy + fast`
2. `player + fast`
3. `selfplay + fast`
4. 全部正常後再用 `balanced`

---

## 16. 我訓練過程中要看哪裡

### 16.1 GUI 裡看

在 `Train` 頁看：

- `Progress`
- `ETA`
- `FPS`
- `Mean reward`

### 16.2 訓練後看

在 `Results` 頁看：

- summary
- reward trend
- eval reward

在 `TensorBoard` 頁看：

- scalar 曲線

在 `Leaderboard` 頁看：

- Elo 排名

### 16.3 CLI 版輸出檔

每個 run 裡最重要的幾個檔案：

- `training_summary.json`
- `progress.json`
- `progress_history.jsonl`
- `eval_history.jsonl`
- `models/*.zip`

如果你只是想最快找到東西，直接照下面找：

- `enemy + collection`
  - `checkpoints/enemy/collection/<run_name>/`
- `enemy + escape`
  - `checkpoints/enemy/escape/<run_name>/`
- `player + collection`
  - `checkpoints/player_single/collection/<run_name>/`
- `player + escape`
  - `checkpoints/player_single/escape/<run_name>/`
- `selfplay + collection`
  - `checkpoints/selfplay/collection/<run_name>/`
- `selfplay + escape`
  - `checkpoints/selfplay/escape/<run_name>/`

---

## 17. 我訓練完之後怎麼拿去玩

### 17.1 GUI 方法

1. 開 GUI
2. 去 `Play`
3. 先選對應的 `Game mode`
4. 如果要人類對敵人模型：
   - 選 `Human vs Enemy Checkpoint`
   - 載入 `enemy_latest.zip`
5. 如果要 AI 對 AI：
   - 選 `AI vs AI`
   - 載入 `player_latest.zip`
   - 載入 `enemy_latest.zip`
6. 按 `Start Game`

### 17.2 CLI 方法

#### 載入敵人模型給人類對戰

```powershell
python -m library_escape.play.human_vs_ai --game-mode escape --enemy-model checkpoints\enemy\escape\enemy_escape_test_01\models\enemy_latest.zip
```

#### 載入玩家模型與敵人模型做 AI 對戰

```powershell
python -m library_escape.play.ai_vs_ai --game-mode escape --player-model checkpoints\player_single\escape\player_escape_test_01\models\player_latest.zip --enemy-model checkpoints\enemy\escape\enemy_escape_test_01\models\enemy_latest.zip
```

---

## 18. 如果我今天想要一個最實用的建議流程

### 18.1 先做 smoke test

```powershell
python -m library_escape.gui.app
```

在 GUI 裡：

1. Play 一次
2. 訓練 enemy 一次，選 `fast`
3. 訓練 player 一次，選 `fast`
4. 跑 self-play 一次，選 `fast`

### 18.2 確認沒問題後

再做：

- `Preset = balanced`

### 18.3 如果你目標是做比較強的模型

建議順序：

1. 先調 reward
2. 再調 observation
3. 再調 training hyperparameters
4. 最後再跑比較長的 self-play

---

## 19. 最常用的指令總表

### 開 GUI

```powershell
python -m library_escape.gui.app
```

### 玩遊戲

```powershell
python -m library_escape.play.human_vs_ai --game-mode collection
```

### AI 對 AI

```powershell
python -m library_escape.play.ai_vs_ai --game-mode escape
```

### 訓練敵人

```powershell
python -m library_escape.train.train_enemy --game-mode escape --preset fast --run-name enemy_escape_test_01
```

### 訓練玩家

```powershell
python -m library_escape.train.train_player --game-mode collection --preset fast --run-name player_collection_test_01
```

### Self-play

```powershell
python -m library_escape.train.train_selfplay --game-mode escape --preset fast --rounds 2 --timesteps-per-round 60000 --run-name selfplay_escape_test_01
```

### Replay 播放

```powershell
python -m library_escape.replay.viewer --replay replays\match_01.ler.gz
```

### Leaderboard

```powershell
python scripts\eval_elo.py --root checkpoints --episodes 8 --output checkpoints\leaderboard.json
```

---

## 20. 如果你現在真的很沒概念，直接照這份做

### 20.1 先開 GUI

```powershell
python -m library_escape.gui.app
```

### 20.2 先試玩

在 `Play`：

1. 選 `Human vs Rule Enemy`
2. `Game mode = collection`
3. 按 `Start Game`

### 20.3 再練一個敵人

在 `Train`：

1. `Mode = enemy`
2. `Game mode = escape`
3. `Preset = fast`
4. `Algorithm = maskable_ppo`
5. `Run name = enemy_escape_test_01`
6. 按 `Start Training`

### 20.4 再拿這個敵人來玩

在 `Play`：

1. `Mode = Human vs Enemy Checkpoint`
2. `Game mode = escape`
3. 選 `checkpoints/enemy/escape/enemy_escape_test_01/models/enemy_latest.zip`
4. 按 `Start Game`

### 20.5 再做 self-play

在 `Train`：

1. `Mode = selfplay`
2. `Game mode = escape`
3. `Preset = fast`
4. `Algorithm = league_maskable_ppo`
5. `Run name = selfplay_escape_test_01`
6. `Rounds = 2`
7. 按 `Start Training`

### 20.6 再去看結果

在：

- `Results`
- `TensorBoard`
- `Leaderboard`
- `Replay`

這樣你就會對整個系統有感覺了。

---

## 21. 你之後最常打開的檔案

如果你只是操作，最常開這幾個：

- [`docs/OPERATION_QUICKSTART.md`](./OPERATION_QUICKSTART.md)
- [`configs/env.yaml`](../configs/env.yaml)
- [`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml)
- [`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)
- [`configs/training.yaml`](../configs/training.yaml)
- [`library_escape/game_modes.py`](../library_escape/game_modes.py)

如果你要進一步研究細節，再看：

- [`docs/OPERATION_GUIDE.md`](./OPERATION_GUIDE.md)

---

## 22. 目前遊戲規則 / 勝負條件 / 強化學習規格速查

### 22.1 現在有兩個 Game mode

- `Collection`
  - 收集 / 計分 / 躲偵測
  - 比較接近原本 C++ 的遊玩感
  - 固定 `5` 個敵人
  - 沒有真正的出口勝利條件
- `Escape`
  - 先完成目標，再逃到出口
  - 用來做最清楚的抓捕 vs 逃脫訓練
  - 預設 `2` 個敵人（`1` 主敵人 + `1` 支援敵人）
  - 有真正的玩家勝利 / 敵人勝利

### 22.2 Collection 模式怎麼算玩得好

Collection 不是嚴格的二元勝負模式。

你可以把它理解成：

- 玩家目標：在時間結束前盡量收 `note` / `exam`，把分數拉高
- 敵人目標：盡量看到玩家、扣時間、壓低玩家最終分數

也就是：

- Collection 沒有真正的 `escaped` 勝利條件
- 比較像單人挑戰 / 計分模式
- 被看到時，只會有真正觸發偵測的那一個敵人短暫停住，不會整隊一起停格

### 22.3 Escape 模式怎麼算玩家贏、敵人贏

玩家贏：

1. 先完成逃脫前置條件
   - 預設是所有 `note` 收完
   - `exam` 預設不是必要，但可以在 [`configs/env.yaml`](../configs/env.yaml) 改
2. 再真的走進出口區域

敵人贏：

- 任一敵人進到抓捕半徑內抓到玩家

另外在訓練視角裡：

- `time_expired`
- `stalemate`

也都代表玩家這一局沒有成功完成目標。

### 22.4 訓練時到底在練什麼

這件事情一定要一起看 `Train mode` 與 `Game mode`。

例如：

- `enemy + collection`
  - 在練「Collection 規則下的敵人怎麼更會壓分」
- `player + collection`
  - 在練「Collection 規則下的玩家怎麼更會偷收集、走位、保分」
- `enemy + escape`
  - 在練「Escape 規則下的敵人怎麼更會追擊、包夾、守出口」
- `player + escape`
  - 在練「Escape 規則下的玩家怎麼更會拿目標、躲追擊、成功逃脫」
- `selfplay + collection`
  - 雙方在 Collection 規則下互相拉高收集 vs 壓分策略
- `selfplay + escape`
  - 雙方在 Escape 規則下互相拉高圍捕 vs 逃脫策略

### 22.5 訓練完之後要去哪個模式播放

原則就是：

- `Collection` 訓練出的 checkpoint，就回 `Collection` 播放
- `Escape` 訓練出的 checkpoint，就回 `Escape` 播放

實際上：

- 敵人模型會放在 `checkpoints/enemy/<game_mode>/...`
- 玩家模型會放在 `checkpoints/player_single/<game_mode>/...`
- self-play 會放在 `checkpoints/selfplay/<game_mode>/...`

### 22.6 分數怎麼算

目前 HUD 上的分數公式是：

```text
score_value = note_count * 8 + exam_count * 12
```

也就是：

- `note` 一個 `+8`
- `exam` 一個 `+12`
- `coffee` / `freeze` 不加 HUD 總分

### 22.7 道具效果是什麼

- `note`
  - 主目標道具
- `exam`
  - 次主目標道具
- `coffee`
  - `Collection` 會加快收集速度
  - `Escape` 會加快移動速度
- `freeze`
  - 會把所有敵人凍住一段時間

### 22.8 收集規則現在是什麼

- 現在所有模式都不是瞬間撿起來
- 收集時間預設是 `0.85` 秒
- `Collection`
  - 要按住 `E`
- `Escape`
  - 不用按 `E`
  - 但也要停在互動範圍內直到進度條滿
- 如果同一張桌上有多個道具
  - 會優先拿你實際比較靠近的那一個

### 22.9 強化學習的 action / observation / reward 要去哪裡看

如果你想直接查完整規格，請看完整技術手冊：

- Action space：[`docs/OPERATION_GUIDE.md`](./OPERATION_GUIDE.md) 的 `## 9`
- Observation space：[`docs/OPERATION_GUIDE.md`](./OPERATION_GUIDE.md) 的 `## 10`
- Reward 權重與公式：[`docs/OPERATION_GUIDE.md`](./OPERATION_GUIDE.md) 的 `## 11`
- 遊戲規則 / 勝負條件 / 分數 / 道具效果：[`docs/OPERATION_GUIDE.md`](./OPERATION_GUIDE.md) 的 `## 27`

### 22.10 我要改這些東西時，最重要的檔案是哪幾個

- 遊戲規則 / 速度 / 視野 / 敵人數：[`configs/env.yaml`](../configs/env.yaml)
- Collection reward 權重：[`configs/rewards_collection.yaml`](../configs/rewards_collection.yaml)
- Escape reward 權重：[`configs/rewards_escape.yaml`](../configs/rewards_escape.yaml)
- Observation feature 組裝：[`library_escape/env/obs_builder.py`](../library_escape/env/obs_builder.py)
- Action 定義：[`library_escape/core/actions.py`](../library_escape/core/actions.py)
- 互動遊玩規則 preset：[`library_escape/play/presets.py`](../library_escape/play/presets.py)
- 使用者模式映射：[`library_escape/game_modes.py`](../library_escape/game_modes.py)

### 22.11 2026-04-19 reward 大改版：實際生效的權重與新項目

`2026-04-19` 這天把兩個模式的 reward 做了大刀闊斧的重塑，原因是訓練一整夜後出現「主敵人原地自旋、玩家被看到就左右抖動、雙方死循環」這類病態行為。

**新增了四個 reward 項目（在 `configs/rewards_*.yaml` 可以直接調）：**

- `enemy.chase_progress_per_unit`
  - 敵人真的縮短與玩家的距離，就給正 reward
- `enemy.search_move_per_unit`
  - 看不到玩家但有移動時，給小額 reward，打破「原地自旋」的 local optimum
- `player.goal_progress_per_unit`
  - 玩家靠近當前目標（最近要收的書 / 解鎖後改成逃脫區）就給正 reward
- `player.evade_progress_per_unit`
  - 玩家被 primary 看到時，拉開距離就給 reward

**另外加了一個旗標：**

- `enemy.potential.use_primary_distance` 預設 `true`
  - 把 `capture_pressure` 從「團隊最近敵人距離」改成「主敵人距離」
  - 這樣 primary 不能再靠近支援敵人就蹭 reward

**權重整體改動方向：**

- Terminal reward 從 `±240` 提到 `±300`，`timeout_win` 從 `60` 降到 `30`
  - 讓抓到/逃到明顯勝過拖時間
- Potential 權重總和砍到約 `1/5`
  - 讓 dense shaping 不再壓過 terminal
- `idle_penalty` 從 `-0.016` 提到 `-0.10`（約 `6x`）
  - 站著不動永遠是壞策略
- `zero_sum_mix` 從 `0.30` 降到 `0.10`
  - 避免雙方 minmax 收斂成「互相不動」
- 反震盪懲罰（`_quiet_step`）拿掉「玩家被看到時不計」的例外
  - 使用者觀察到的死循環，就是在對視時抖動，那個 case 現在會照常罰

**完整診斷、完整新 YAML、完整設計理由**都寫在詳細手冊：

- [`docs/OPERATION_GUIDE.md`](./OPERATION_GUIDE.md) 的 `## 11.10`

**舊 checkpoint 建議從零重練**，因為舊 policy 已經卡在「原地自旋 / 對視抖動」這兩個 local optima，續訓要走出來很慢。
