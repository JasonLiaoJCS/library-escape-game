# Library Escape 操作版說明書

## 1. 這份文件是給誰看的

這份文件只回答一件事：

- 如果你今天不想先理解整個架構，只想直接操作這個專案，你要怎麼做？

我會直接告訴你：

1. 怎麼安裝
2. 怎麼開遊戲
3. 怎麼選模式
4. 怎麼訓練敵人
5. 怎麼做 AI 自己互相訓練
6. 哪些參數最常改
7. 每一件事對應的指令是什麼

---

## 2. 先記住一個重點

### 2.1 現在的模式選擇不是在遊戲 UI 裡選

目前這個版本的模式切換方式是：

- **用不同的 Python 指令啟動不同模式**

也就是說，現在不是先開一個大選單再選模式，而是：

- 人類玩遊戲：跑一個指令
- AI 對 AI：跑另一個指令
- 訓練敵人：跑另一個指令
- self-play：再跑另一個指令

所以你之後只要記住：

- **模式 = 不同入口指令**

---

## 3. 第一次安裝

### 3.1 進入專案資料夾

```powershell
Set-Location 'C:\Users\User\Desktop\大四其他\library-escape-game'
```

### 3.2 建立虛擬環境

```powershell
python -m venv .venv
```

### 3.3 啟用虛擬環境

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3.4 安裝依賴

```powershell
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

### 3.5 之後每次打開專案時

你通常只需要做這兩步：

```powershell
Set-Location 'C:\Users\User\Desktop\大四其他\library-escape-game'
.\.venv\Scripts\Activate.ps1
```

---

## 4. 最常用的四個指令

### 4.1 人類玩家 vs AI 敵人

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai
```

### 4.2 AI vs AI 對戰

```powershell
.\.venv\Scripts\python -m library_escape.play.ai_vs_ai --render
```

### 4.3 訓練敵人模型

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy
```

### 4.4 AI 自己互相訓練

```powershell
.\.venv\Scripts\python -m library_escape.train.train_selfplay
```

---

## 5. 我要怎麼開遊戲

### 5.1 開啟一般遊戲

如果你想自己操作玩家，讓 AI 當敵人：

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai
```

### 5.2 操作方式

- `W A S D` 或方向鍵：移動
- `ESC`：離開

### 5.3 如果你想用已經訓練好的敵人模型

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai --enemy-model models\enemy\enemy_ppo.zip
```

### 5.4 如果你只是想測一下會不會跑，不想開視窗

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai --headless --max-steps 200
```

---

## 6. 我要怎麼選模式

目前主要有四種操作模式。

### 模式 A：人類玩家 vs AI 敵人

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai
```

用途：

- 你自己玩
- 測試敵人 AI 強不強

### 模式 B：AI 玩家 vs AI 敵人

```powershell
.\.venv\Scripts\python -m library_escape.play.ai_vs_ai --render
```

用途：

- 看兩個 AI 自己對打
- 觀察策略表現

### 模式 C：訓練敵人模型

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy
```

用途：

- 專門訓練敵人
- 玩家端用 scripted baseline 當對手

### 模式 D：self-play 交替訓練

```powershell
.\.venv\Scripts\python -m library_escape.train.train_selfplay
```

用途：

- 先訓練 player
- 再訓練 enemy
- 雙方輪流變強

---

## 7. 我要怎麼訓練敵人模型

### 7.1 最基本做法

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy
```

這個腳本做的事情是：

1. 建立單智能體 Gymnasium 環境
2. 讓敵人當訓練對象
3. 讓玩家使用 scripted baseline / random baseline 當對手
4. 用 PPO 訓練敵人
5. 把模型存起來

### 7.2 訓練結果會存去哪裡

預設會存到：

- `models/enemy/enemy_ppo.zip`

TensorBoard log 會存到：

- `models/logs/enemy/`

### 7.2.1 現在終端會自動顯示還剩多久

訓練時你會看到像這樣的行：

```text
[ETA] Enemy PPO | phase 8192/50000 ( 16.4%) | phase_eta 02:45 | overall 8192/50000 ( 16.4%) | overall_eta 02:45
```

你主要看兩個欄位：

- `phase_eta`：目前這一段訓練還剩多久
- `overall_eta`：整體還剩多久

### 7.3 如果你想指定訓練步數

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy --timesteps 200000
```

### 7.4 如果你想指定輸出檔名

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy --timesteps 200000 --output models\enemy\enemy_v2
```

實際輸出檔會是：

- `models\enemy\enemy_v2.zip`

### 7.5 如果你想改 frame skip

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy --frame-skip 6
```

---

## 8. 我要怎麼讓 AI 自己互相訓練

### 8.1 最基本做法

```powershell
.\.venv\Scripts\python -m library_escape.train.train_selfplay
```

### 8.1.1 如果你想直接看到 AI 在訓練中的畫面

```powershell
.\.venv\Scripts\python -m library_escape.train.train_selfplay --render
```

如果你想先用比較短的訓練確認畫面有正常出來：

```powershell
.\.venv\Scripts\python -m library_escape.train.train_selfplay --rounds 1 --timesteps 5000 --render
```

注意：

- `--render` 會大幅降低訓練速度
- 它會在 player 訓練階段與 enemy 訓練階段都顯示視窗
- 如果你只是想快速訓練，不建議開 `--render`

### 8.2 它實際上怎麼做

目前的 self-play 不是完整 MAPPO，而是：

1. 訓練 `player`
2. 把 `player` checkpoint 存進 pool
3. 訓練 `enemy`
4. 把 `enemy` checkpoint 存進 pool
5. 下一輪從 pool 裡抽對手再訓練

也就是：

- **交替式 self-play**
- **歷史對手池**

### 8.2.1 現在 self-play 也會自動顯示剩餘時間

你會看到像這樣的行：

```text
[SELFPLAY] Round 2/6 | phase=enemy | global 60000/240000
[ETA] Round 2/6 enemy | phase 4096/20000 ( 20.5%) | phase_eta 01:02 | overall 64096/240000 ( 26.7%) | overall_eta 09:12
```

你可以這樣看：

- `Round 2/6`：目前第幾輪
- `phase=enemy`：現在在訓練敵人
- `phase_eta`：這個 phase 還剩多久
- `overall_eta`：整個 self-play 全部還剩多久

### 8.3 如果你想指定訓練輪數和每輪步數

```powershell
.\.venv\Scripts\python -m library_escape.train.train_selfplay --rounds 10 --timesteps 50000
```

### 8.4 self-play 輸出會存去哪裡

- `models/player_pool/`
- `models/enemy_pool/`

### 8.5 如果你只是想看兩個 AI 對打，不一定要看訓練中

```powershell
.\.venv\Scripts\python -m library_escape.play.ai_vs_ai --render
```

如果你想指定已訓練好的模型：

```powershell
.\.venv\Scripts\python -m library_escape.play.ai_vs_ai --player-model models\player_pool\player_round_1.zip --enemy-model models\enemy_pool\enemy_round_1.zip --render
```

---

## 9. 我要怎麼讓「我自己」去訓練敵人

如果你說的「我自己訓練敵人」是指：

- 我自己當玩家
- 然後想讓敵人根據我玩的方式學

那目前**還沒有做成人類對戰資料回放訓練**。

目前已經有的是：

- 你可以自己玩：`human_vs_ai`
- 敵人可以用 PPO 訓練
- 但訓練時對手是 scripted / random / self-play policy

也就是說目前的訓練方式是：

- **強化學習自我探索**
- **不是 imitation learning 或 human demonstration**

如果你今天要「讓敵人學會打你」，目前比較實際的做法是：

1. 先訓練敵人模型
2. 用訓練好的模型跟你對戰
3. 覺得太弱就再增加 timesteps 或調 reward

---

## 10. 訓練方法到底是什麼

### 10.1 敵人單獨訓練

方法：

- `PPO`
- 使用 `Stable-Baselines3`
- 環境是 `Gymnasium`

訓練對象：

- `enemy`

對手來源：

- `RuleBasedPlayerController`
- `RandomController`
- 依照 `opponent_mix` 混合

### 10.2 雙方互訓

方法：

- 交替 PPO
- 歷史對手池

目前不是：

- 完整 MAPPO
- 完整 PSRO
- 完整 league training

但已經足夠做：

- 自我訓練
- AI vs AI
- 基本策略共演化

---

## 11. 我最常要改哪些參數

你平常幾乎只要記住三個檔案：

1. `configs/env.yaml`
2. `configs/rewards.yaml`
3. `configs/training.yaml`

---

## 12. 我要改地圖、角色速度、視野、物理

改這個檔案：

- [configs/env.yaml](/C:/Users/User/Desktop/大四其他/library-escape-game/configs/env.yaml)

### 12.1 最常改的欄位

```yaml
world:
  width: 1200
  height: 600
  max_time: 90.0
  physics_dt: 0.0083333333
```

用途：

- `width` / `height`：地圖大小
- `max_time`：每局最長時間
- `physics_dt`：物理步長，越小越細

### 12.2 玩家參數

```yaml
player:
  radius: 16.0
  speed: 220.0
  vision_range: 260.0
  vision_angle_deg: 180.0
```

### 12.3 敵人參數

```yaml
enemy:
  radius: 18.0
  speed: 205.0
  vision_range: 285.0
  vision_angle_deg: 78.0
  catch_radius: 10.0
```

### 12.4 巡邏路徑

```yaml
enemy:
  patrol_points:
    - [1040.0, 90.0]
    - [930.0, 90.0]
    - [930.0, 250.0]
```

### 12.5 筆記與障礙物

```yaml
notes:
  - [180.0, 120.0]
  - [340.0, 500.0]

obstacles:
  - {type: bookshelf, x: 220.0, y: 80.0, w: 80.0, h: 160.0}
  - {type: table, x: 120.0, y: 260.0, w: 100.0, h: 60.0}
```

---

## 13. 我要改 Action Space

改這個檔案：

- [configs/env.yaml](/C:/Users/User/Desktop/大四其他/library-escape-game/configs/env.yaml)

### 13.1 離散動作

```yaml
actions:
  scheme: discrete
```

代表：

- `Discrete(9)`

### 13.2 連續動作

```yaml
actions:
  scheme: continuous
```

代表：

- `Box(shape=(2,))`

### 13.3 如果你想改 action 解碼邏輯

改：

- [library_escape/core/actions.py](/C:/Users/User/Desktop/大四其他/library-escape-game/library_escape/core/actions.py)

---

## 14. 我要改 Observation Space

### 14.1 先改設定版

改：

- [configs/env.yaml](/C:/Users/User/Desktop/大四其他/library-escape-game/configs/env.yaml)

例如：

```yaml
observations:
  include_velocity: true
  include_heading: true
  include_last_seen_age: true
  wall_ray_count: 8
  wall_ray_range: 220.0
  memory_seconds: 1.5
```

### 14.2 如果你想直接改 observation 組成

改：

- [library_escape/env/obs_builder.py](/C:/Users/User/Desktop/大四其他/library-escape-game/library_escape/env/obs_builder.py)

最重要的函式：

- `_player_obs()`
- `_enemy_obs()`

---

## 15. 我要改 Reward 權重

改這個檔案：

- [configs/rewards.yaml](/C:/Users/User/Desktop/大四其他/library-escape-game/configs/rewards.yaml)

### 15.1 目前敵人 reward

```yaml
enemy:
  catch_player: 100.0
  player_in_cone_per_step: 1.0
  distance_shaping: 0.04
  time_penalty: -0.01
  wall_penalty: -0.1
  stalemate: -5.0
```

### 15.2 目前玩家 reward

```yaml
player:
  collect_note: 20.0
  escape: 100.0
  seen_per_step: -0.5
  caught: -100.0
  time_bonus: 0.01
  wall_penalty: -0.1
```

### 15.3 如果你想改 reward 演算法本身

改：

- [library_escape/rewards/reward_fns.py](/C:/Users/User/Desktop/大四其他/library-escape-game/library_escape/rewards/reward_fns.py)

---

## 16. 我要改訓練參數

改這個檔案：

- [configs/training.yaml](/C:/Users/User/Desktop/大四其他/library-escape-game/configs/training.yaml)

### 16.1 敵人單獨訓練常改這些

```yaml
single_agent:
  total_timesteps: 50000
  n_envs: 2
  vector_env: dummy
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  n_steps: 512
  batch_size: 256
  ent_coef: 0.01
  clip_range: 0.2
  vf_coef: 0.5
  net_arch: [128, 128]
```

### 16.2 self-play 常改這些

```yaml
multi_agent:
  rounds: 6
  timesteps_per_round: 20000
  pool_size: 4
  player_net_arch: [128, 128]
  enemy_net_arch: [128, 128]
```

### 16.3 shared 常改這些

```yaml
shared:
  seed: 42
  frame_skip: 4
  device: auto
```

---

## 17. 你如果不知道怎麼調，先照這幾個方向改

### 17.1 我想讓敵人更強

做法：

1. 增加訓練步數
2. 增加 `enemy.catch_player`
3. 增加 `enemy.distance_shaping`
4. 降低 `enemy.time_penalty`

範例：

```yaml
enemy:
  catch_player: 120.0
  distance_shaping: 0.06
  time_penalty: -0.005
```

然後重新訓練：

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy --timesteps 300000
```

### 17.2 我想讓玩家更有逃脫動機

做法：

- 增加 `player.escape`
- 增加 `player.collect_note`
- 降低 `player.caught`

### 17.3 我想讓策略更平滑

做法：

- 把 `actions.scheme` 改成 `continuous`

```yaml
actions:
  scheme: continuous
```

### 17.4 我想讓學習更快

做法：

- 增加 `n_envs`
- 把 `vector_env` 改成 `subproc`
- 增加 `frame_skip`

例如：

```yaml
single_agent:
  n_envs: 4
  vector_env: subproc

shared:
  frame_skip: 6
```

---

## 18. 訓練完之後我要怎麼看成果

### 18.1 最直接的方法：拿模型回去玩

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai --enemy-model models\enemy\enemy_ppo.zip
```

### 18.2 看 TensorBoard

```powershell
.\.venv\Scripts\tensorboard --logdir models\logs\enemy
```

然後開瀏覽器：

```text
http://localhost:6006
```

### 18.3 AI vs AI 看對戰結果

```powershell
.\.venv\Scripts\python -m library_escape.play.ai_vs_ai --render
```

### 18.4 跑 ELO 粗估

```powershell
.\.venv\Scripts\python -m library_escape.scripts.eval_elo --episodes 20
```

### 18.5 輸出對戰影格

```powershell
.\.venv\Scripts\python -m library_escape.scripts.record_video --output-dir recordings\demo_frames
```

---

## 19. 最簡單的推薦操作流程

### 流程 1：我只想先玩

```powershell
Set-Location 'C:\Users\User\Desktop\大四其他\library-escape-game'
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python -m library_escape.play.human_vs_ai
```

### 流程 2：我想先訓練一個敵人

```powershell
Set-Location 'C:\Users\User\Desktop\大四其他\library-escape-game'
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python -m library_escape.train.train_enemy --timesteps 200000
.\.venv\Scripts\python -m library_escape.play.human_vs_ai --enemy-model models\enemy\enemy_ppo.zip
```

### 流程 3：我想做 AI 自己互訓

```powershell
Set-Location 'C:\Users\User\Desktop\大四其他\library-escape-game'
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python -m library_escape.train.train_selfplay --rounds 10 --timesteps 50000
.\.venv\Scripts\python -m library_escape.play.ai_vs_ai --render
```

### 流程 3-1：我想一邊訓練一邊看畫面

```powershell
Set-Location 'C:\Users\User\Desktop\大四其他\library-escape-game'
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python -m library_escape.train.train_selfplay --rounds 1 --timesteps 5000 --render
```

### 流程 4：我想做 reward 實驗

1. 改 `configs/rewards.yaml`
2. 重新訓練
3. 測試模型

範例命令：

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy --timesteps 150000 --output models\enemy\reward_test_01
.\.venv\Scripts\python -m library_escape.play.human_vs_ai --enemy-model models\enemy\reward_test_01.zip
```

---

## 20. 哪幾個檔案你一定要知道

如果你真的只想操作，請至少記住這幾個：

### 遊戲入口

- [library_escape/play/human_vs_ai.py](/C:/Users/User/Desktop/大四其他/library-escape-game/library_escape/play/human_vs_ai.py)
- [library_escape/play/ai_vs_ai.py](/C:/Users/User/Desktop/大四其他/library-escape-game/library_escape/play/ai_vs_ai.py)

### 訓練入口

- [library_escape/train/train_enemy.py](/C:/Users/User/Desktop/大四其他/library-escape-game/library_escape/train/train_enemy.py)
- [library_escape/train/train_selfplay.py](/C:/Users/User/Desktop/大四其他/library-escape-game/library_escape/train/train_selfplay.py)

### 主要參數

- [configs/env.yaml](/C:/Users/User/Desktop/大四其他/library-escape-game/configs/env.yaml)
- [configs/rewards.yaml](/C:/Users/User/Desktop/大四其他/library-escape-game/configs/rewards.yaml)
- [configs/training.yaml](/C:/Users/User/Desktop/大四其他/library-escape-game/configs/training.yaml)

---

## 21. 你現在如果完全不知道從哪裡開始

我建議你照這個順序：

1. 先開遊戲

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai
```

2. 再做一次短訓練

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy --timesteps 50000
```

3. 拿訓練後的敵人回去玩

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai --enemy-model models\enemy\enemy_ppo.zip
```

4. 如果你想進一步玩 MARL，再做 self-play

```powershell
.\.venv\Scripts\python -m library_escape.train.train_selfplay --rounds 6 --timesteps 20000
```

---

## 22. 一句話版

如果你只想記一句話：

- **開遊戲：`human_vs_ai`**
- **訓練敵人：`train_enemy`**
- **AI 自己互訓：`train_selfplay`**
- **改設定：去 `configs/`**
