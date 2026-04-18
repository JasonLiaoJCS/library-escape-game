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

---

## 2. 第一次使用的完整準備

### 2.1 建立 Python 環境

請使用 `Python 3.12`。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[rl,dev]
```

### 2.2 驗證有沒有安裝好

```powershell
python -m library_escape.gui.app
```

如果 GUI 可以打開，代表基本上就能用了。

---

## 3. 你現在要怎麼理解這個專案

你可以先把這個系統想成只有 3 種訓練模式。

### 3.1 模式 A：訓練敵人

意思是：

- 玩家先用「規則式 / script」控制
- 敵人是 RL 模型
- 目標是讓敵人學會更會抓人

### 3.2 模式 B：訓練玩家

意思是：

- 敵人先用「規則式 / script」控制
- 玩家是 RL 模型
- 目標是讓玩家學會更會收集、躲避、逃脫

### 3.3 模式 C：Self-play

意思是：

- 兩邊都訓練
- 先訓練敵人一輪
- 再訓練玩家一輪
- 再繼續交替

你可以把它理解成：

- 訓練敵人 = 單邊訓練
- 訓練玩家 = 單邊訓練
- self-play = 雙邊輪流訓練

---

## 4. 我要怎麼開遊戲

有兩種方式。

### 4.1 方法 A：用 GUI

```powershell
python -m library_escape.gui.app
```

然後進入 `Play` 分頁。

你會看到模式選單。

目前主要模式：

- `Human vs Rule Enemy`
- `Human vs Enemy Checkpoint`
- `AI vs AI`

### 4.2 方法 B：直接用指令

#### 人類玩，敵人是規則式 AI

```powershell
python -m library_escape.play.human_vs_ai
```

#### 人類玩，敵人是你訓練好的模型

```powershell
python -m library_escape.play.human_vs_ai --enemy-model checkpoints\enemy\你的_run_name\models\enemy_latest.zip
```

#### AI 對 AI

```powershell
python -m library_escape.play.ai_vs_ai
```

#### AI 對 AI，指定雙方模型

```powershell
python -m library_escape.play.ai_vs_ai --player-model checkpoints\player_single\你的_run_name\models\player_latest.zip --enemy-model checkpoints\enemy\你的_run_name\models\enemy_latest.zip
```

---

## 5. 開遊戲之後我要怎麼選模式

### 5.1 GUI 裡怎麼選

進 `Play` 分頁後：

1. 找到 `Mode`
2. 從下拉選單選：
   - `Human vs Rule Enemy`
   - `Human vs Enemy Checkpoint`
   - `AI vs AI`
3. 如果要用模型，就在右邊選 `.zip`
4. 按 `Start Game`

### 5.2 每個模式代表什麼

#### `Human vs Rule Enemy`

適合：

- 先測試遊戲有沒有正常
- 不用模型也可以玩

#### `Human vs Enemy Checkpoint`

適合：

- 用你訓練好的敵人模型來跟人類玩

#### `AI vs AI`

適合：

- 看兩個模型對戰
- 測試訓練成果

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
- `Preset`
- `Run name`
- `Timesteps / round`
- `Rounds`
- `Vector envs`
- `Seed`
- `Device`
- `Algorithm`

### 8.3 `Mode` 要選什麼

- `enemy`：訓練敵人
- `player`：訓練玩家
- `selfplay`：雙方輪流訓練

### 8.4 `Preset` 要選什麼

- `fast`：快速測試
- `balanced`：一般推薦
- `quality`：更久、更重、更強

新手建議：

1. 先用 `fast`
2. 確認流程都會跑
3. 再改 `balanced`

### 8.5 `Algorithm` 要選什麼

如果 `Mode = enemy` 或 `player`：

- `ppo`
- `maskable_ppo`

如果 `Mode = selfplay`：

- `league_ppo`
- `league_maskable_ppo`
- `mappo_recipe`

新手建議：

- 單邊訓練先用 `ppo`
- 如果你要更穩的離散動作控制，可試 `maskable_ppo`
- self-play 先用 `league_ppo`

### 8.6 `Run name` 要填什麼

建議填一個你看得懂的名字，例如：

- `enemy_test_01`
- `player_fast_01`
- `selfplay_balanced_v1`

### 8.7 `Timesteps / round`

如果是：

- `enemy` / `player`：這就是總訓練 timesteps
- `selfplay`：這是一回合的 timesteps

### 8.8 `Rounds`

只有 self-play 會用到。

例如：

- `Rounds = 4`

表示：

1. 訓練 enemy 第 1 輪
2. 訓練 player 第 1 輪
3. 訓練 enemy 第 2 輪
4. 訓練 player 第 2 輪
5. 一直做下去

### 8.9 `Vector envs`

這是同時開幾個環境來收資料。

建議：

- 中端機：`1`、`2`、`4`

如果你不確定，先從：

```text
1
```

開始。

### 8.10 `Device`

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
4. `Preset` 選 `fast`
5. `Algorithm` 選 `ppo`
6. `Run name` 填 `enemy_test_01`
7. 按 `Start Training`

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

### 9.3 訓練結束後模型在哪裡

通常在：

```text
checkpoints/enemy/enemy_test_01/models/enemy_latest.zip
```

---

## 10. GUI 路線：訓練玩家

### 10.1 操作步驟

1. 開 GUI
2. 進 `Train`
3. `Mode` 選 `player`
4. `Preset` 選 `fast`
5. `Algorithm` 選 `ppo`
6. `Run name` 填 `player_test_01`
7. 按 `Start Training`

### 10.2 訓練結果在哪裡

通常在：

```text
checkpoints/player_single/player_test_01/models/player_latest.zip
```

---

## 11. GUI 路線：讓他們自己互相訓練

這就是 self-play。

### 11.1 操作步驟

1. 開 GUI
2. 進 `Train`
3. `Mode` 選 `selfplay`
4. `Preset` 選 `fast`
5. `Algorithm` 選 `league_ppo`
6. `Run name` 填 `selfplay_test_01`
7. `Rounds` 填 `2` 或 `4`
8. 按 `Start Training`

### 11.2 這個模式內部會做什麼

它不是兩邊同時一步到位一起學，而是交替：

1. 先訓練敵人
2. 再訓練玩家
3. 再訓練敵人
4. 再訓練玩家

### 11.3 訓練結果在哪裡

通常在：

```text
checkpoints/selfplay/selfplay_test_01/
```

裡面會分成：

- `enemy/round_01/...`
- `player/round_01/...`
- `enemy/round_02/...`
- `player/round_02/...`

最終模型通常會記在：

- `training_summary.json`

你也可以直接去 GUI 的 `Results` 頁選這個 run。

---

## 12. CLI 路線：如果你想用指令操作

這一節就是「每件事情要怎麼執行」。

### 12.1 開 GUI

```powershell
python -m library_escape.gui.app
```

### 12.2 開遊戲：人類對規則式敵人

```powershell
python -m library_escape.play.human_vs_ai
```

### 12.3 開遊戲：人類對訓練好的敵人

```powershell
python -m library_escape.play.human_vs_ai --enemy-model checkpoints\enemy\enemy_test_01\models\enemy_latest.zip
```

### 12.4 開遊戲：AI 對 AI

```powershell
python -m library_escape.play.ai_vs_ai --player-model checkpoints\player_single\player_test_01\models\player_latest.zip --enemy-model checkpoints\enemy\enemy_test_01\models\enemy_latest.zip
```

### 12.5 訓練敵人

```powershell
python -m library_escape.train.train_enemy
```

### 12.6 訓練敵人，用 fast preset

```powershell
python -m library_escape.train.train_enemy --preset fast
```

### 12.7 訓練敵人，自訂 timesteps

```powershell
python -m library_escape.train.train_enemy --timesteps 300000 --n-envs 4 --run-name enemy_test_01
```

### 12.8 訓練玩家

```powershell
python -m library_escape.train.train_player --preset fast --run-name player_test_01
```

### 12.9 Self-play

```powershell
python -m library_escape.train.train_selfplay --preset fast --rounds 2 --timesteps-per-round 60000 --run-name selfplay_test_01
```

### 12.10 訓練敵人，用 Maskable PPO

PowerShell 建議這樣寫：

```powershell
$json = '{"train":{"algorithm":"maskable_ppo"}}'
python -m library_escape.train.train_enemy --preset balanced --run-name enemy_mask_01 --overrides-json $json
```

### 12.11 Self-play，用 league Maskable PPO

```powershell
$json = '{"train":{"algorithm":"league_maskable_ppo"}}'
python -m library_escape.train.train_selfplay --preset balanced --run-name selfplay_mask_01 --overrides-json $json
```

### 12.12 replay 錄製

```powershell
python -m library_escape.play.ai_vs_ai --record-replay replays\match_01.ler.gz
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

- [`configs/rewards.yaml`](../configs/rewards.yaml)

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
  player_speed: 4.20
  enemy_speed: 3.60
enemy:
  vision_range: 6.0
  vision_angle_deg: 70.0
timing:
  rl_frame_skip: 4
```

### 14.2 只想讓敵人更積極學抓人

改：

- `configs/rewards.yaml`

重點欄位：

```yaml
enemy:
  catch_player: 140.0
  player_in_cone_per_step: 0.70
  potential:
    capture_progress: 2.80
    visibility_lock: 0.75
    escape_pressure: 1.40
```

### 14.3 只想讓玩家更積極學逃脫

改：

- `configs/rewards.yaml`

重點欄位：

```yaml
player:
  escape: 160.0
  collect_note: 28.0
  seen_per_step: -0.35
  potential:
    objective_progress: 2.50
    target_navigation: 2.20
    escape_navigation: 3.20
```

### 14.4 只想讓訓練更快

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

---

## 17. 我訓練完之後怎麼拿去玩

### 17.1 GUI 方法

1. 開 GUI
2. 去 `Play`
3. 如果要人類對敵人模型：
   - 選 `Human vs Enemy Checkpoint`
   - 載入 `enemy_latest.zip`
4. 如果要 AI 對 AI：
   - 選 `AI vs AI`
   - 載入 `player_latest.zip`
   - 載入 `enemy_latest.zip`
5. 按 `Start Game`

### 17.2 CLI 方法

#### 載入敵人模型給人類對戰

```powershell
python -m library_escape.play.human_vs_ai --enemy-model checkpoints\enemy\enemy_test_01\models\enemy_latest.zip
```

#### 載入玩家模型與敵人模型做 AI 對戰

```powershell
python -m library_escape.play.ai_vs_ai --player-model checkpoints\player_single\player_test_01\models\player_latest.zip --enemy-model checkpoints\enemy\enemy_test_01\models\enemy_latest.zip
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
python -m library_escape.play.human_vs_ai
```

### AI 對 AI

```powershell
python -m library_escape.play.ai_vs_ai
```

### 訓練敵人

```powershell
python -m library_escape.train.train_enemy --preset fast --run-name enemy_test_01
```

### 訓練玩家

```powershell
python -m library_escape.train.train_player --preset fast --run-name player_test_01
```

### Self-play

```powershell
python -m library_escape.train.train_selfplay --preset fast --rounds 2 --timesteps-per-round 60000 --run-name selfplay_test_01
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
2. 按 `Start Game`

### 20.3 再練一個敵人

在 `Train`：

1. `Mode = enemy`
2. `Preset = fast`
3. `Algorithm = ppo`
4. `Run name = enemy_test_01`
5. 按 `Start Training`

### 20.4 再拿這個敵人來玩

在 `Play`：

1. `Mode = Human vs Enemy Checkpoint`
2. 選 `checkpoints/enemy/enemy_test_01/models/enemy_latest.zip`
3. 按 `Start Game`

### 20.5 再做 self-play

在 `Train`：

1. `Mode = selfplay`
2. `Preset = fast`
3. `Algorithm = league_ppo`
4. `Run name = selfplay_test_01`
5. `Rounds = 2`
6. 按 `Start Training`

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
- [`configs/rewards.yaml`](../configs/rewards.yaml)
- [`configs/training.yaml`](../configs/training.yaml)

如果你要進一步研究細節，再看：

- [`docs/OPERATION_GUIDE.md`](./OPERATION_GUIDE.md)
