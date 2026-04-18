# Library Escape 詳細實作說明書

## 1. 這份文件是做什麼的

這份文件會完整說明目前這個 repo 裡：

1. 原本的 C++ 版檔案在哪裡
2. 我新做的 Python 版檔案在哪裡
3. 我到底改了哪些東西
4. 怎麼執行 C++ 舊版
5. 怎麼執行 Python 新版
6. 連續時間是怎麼做的
7. 強化學習功能是怎麼接進來的
8. reward / observation / action / 環境 / 訓練參數要去哪裡調
9. 訓練後的模型和紀錄會存到哪裡

這份說明會盡量避免只講抽象概念，而是直接指出「檔案位置」、「命令」、「參數名稱」、「公式」。

---

## 2. 先講最重要的結論

### 2.1 不是「Python 改成 C」

這次做的方向是：

- **把舊的 C++/SDL2 專案保留下來**
- **另外建立一套新的 Python 版遊戲主體**
- **在 Python 版上加入連續時間與強化學習訓練功能**

也就是說，正確描述應該是：

- **原本是 C++**
- **現在新增了 Python 版主系統**

不是把 Python 改成 C，也不是把舊 C++ 檔案全部刪掉。

### 2.2 現在 repo 裡其實有兩套東西

| 版本 | 狀態 | 主要用途 |
|---|---|---|
| **舊 C++/SDL2 版** | 保留 | 歷史版本、素材與原始結構參考 |
| **新 Python/Pygame/RL 版** | 現在主版本 | 真正可訓練、可擴充、可做 RL 的版本 |

### 2.3 哪個版本是現在主要應該使用的

你現在應該主要使用的是：

- `library_escape/`
- `configs/`
- `tests/`
- `pyproject.toml`

這些是新的 Python 版。

---

## 3. 原本的 C++ 檔案在哪裡

原本的 C++ 版主要在下面這些位置：

```text
main.cpp
CMakeLists.txt
include/
src/
SDL_lib/
imgs/
fonts/
```

### 3.1 舊 C++ 版的重要檔案

| 路徑 | 用途 |
|---|---|
| `main.cpp` | 程式進入點 |
| `CMakeLists.txt` | CMake 建置設定 |
| `src/engine.cpp` | 主遊戲迴圈與頁面切換 |
| `src/playground.cpp` | 舊版遊戲場景與玩家移動 |
| `src/menu.cpp` | 選單 |
| `src/settings.cpp` | 設定頁 |
| `src/character.cpp` | 舊版角色控制 |
| `src/obstacle.cpp` `src/table.cpp` `src/bookshelf.cpp` | 障礙物 |
| `include/*.h` | 對應標頭檔 |
| `SDL_lib/` | SDL2 相關依賴 |

### 3.2 舊 C++ 版目前的真實狀態

這件事我必須很坦白講：

- repo 裡的舊 C++ 版**不是完全對得上 proposal 裡描述的完整功能**
- 某些檔案是空的，例如：
  - `src/enemy.cpp`
  - `src/collectible.cpp`
  - `src/GM.cpp`
  - 對應的一些 header 也是空的
- `CMakeLists.txt` 實際上也沒有把那些空檔編進主程式

所以：

- **舊 C++ 版目前比較像是殘留的早期版本**
- **不是你應該繼續往上疊 RL 功能的最佳基礎**

這也是我為什麼改採「保留 C++，但重新建立 Python 主架構」。

### 3.3 如果你還是想執行舊 C++ 版

在 PowerShell 裡：

```powershell
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

如果建置成功，執行檔通常會在：

- `build/Release/`
- 或 `build/`

但請注意：

- 舊版功能不完整
- 不含這次新增的 RL 架構
- 不含這次新增的連續時間設計
- 不應再當作未來主開發版本

---

## 4. 新的 Python 版檔案在哪裡

新的 Python 主版本主要在：

```text
pyproject.toml
library_escape/
configs/
tests/
docs/
```

### 4.1 新 Python 版的目錄結構

```text
library_escape/
├── core/
├── render/
├── input/
├── env/
├── rewards/
├── agents/
├── train/
├── play/
└── scripts/

configs/
├── env.yaml
├── rewards.yaml
└── training.yaml

tests/
docs/
pyproject.toml
```

### 4.2 這些目錄各做什麼

| 路徑 | 用途 |
|---|---|
| `library_escape/core/` | 純遊戲邏輯：世界、碰撞、視野錐、導航、動作解碼 |
| `library_escape/render/` | Pygame 繪圖與 headless null view |
| `library_escape/input/` | 鍵盤輸入轉動作 |
| `library_escape/env/` | Gymnasium 與 PettingZoo 環境 |
| `library_escape/rewards/` | reward 計算 |
| `library_escape/agents/` | rule-based baseline 與 PPO policy 載入 |
| `library_escape/train/` | 訓練入口 |
| `library_escape/play/` | 遊玩入口 |
| `library_escape/scripts/` | ELO 評估、錄影等工具 |
| `configs/` | 所有可調參數 |
| `tests/` | 測試 |

---

## 5. 我這次到底做了什麼

### 5.1 大方向

我這次不是只修幾個 bug，而是做了以下幾件大事：

1. **建立一套新的 Python 專案架構**
2. **把遊戲核心改成連續座標與固定物理步長**
3. **把 render 與遊戲邏輯解耦**
4. **把遊戲包成 Gymnasium 單智能體環境**
5. **把遊戲包成 PettingZoo 多智能體環境**
6. **加入 reward 外部化設定**
7. **加入 PPO 訓練入口**
8. **加入 self-play 交替訓練入口**
9. **加入 baseline 對手**
10. **加入測試與 smoke 驗證**

### 5.2 實際新增/修改的重要檔案

#### 新增

- `pyproject.toml`
- `library_escape/...` 全部新 Python 架構
- `configs/env.yaml`
- `configs/rewards.yaml`
- `configs/training.yaml`
- `tests/test_physics.py`
- `tests/test_cone.py`
- `tests/test_env_api.py`
- `docs/DETAILED_IMPLEMENTATION_GUIDE.md`

#### 修改

- `README.md`
- `.gitignore`

### 5.3 重要的誠實說明

這次 Python 版不是逐行把所有 C++ 檔案翻譯過去，而是：

- 參考 proposal 的目標架構
- 參考現有素材與地圖元素
- 參考 README 與舊 C++ 檔案中仍然存在的規則
- 再重新建立一套更適合 RL 的版本

原因是舊 C++ repo 本身就不完整，直接照翻沒有意義。

---

## 6. 新 Python 版的核心設計

### 6.1 核心世界類別

核心世界在：

- `library_escape/core/world.py`

這個 `LibraryWorld` 負責：

- 地圖大小
- 玩家與敵人的狀態
- 筆記
- 出口
- 障礙物
- 剩餘時間
- 視野錐判定
- 抓到 / 逃脫 / 超時等終局條件

### 6.2 世界資料是從哪裡來的

主要從：

- `configs/env.yaml`

讀進來。

這個檔案定義：

- 世界寬高
- `physics_dt`
- 玩家起點與速度
- 敵人起點與速度
- 視野範圍與角度
- 出口位置
- 筆記位置
- 障礙物位置

### 6.3 角色座標不再是舊版那種單純格點移動

舊版 C++ 的 `Character` 基本上是：

- `gridX`, `gridY`
- 按一次鍵就移動一格

新的 Python 版改成：

- `position = [x, y]` 浮點數座標
- `velocity = [vx, vy]`
- `facing = 角度`

這樣就能做：

- 連續移動
- 連續轉向
- frame-skip
- RL 的固定物理步長

---

## 7. 我是怎麼把離散時間改成連續時間的

這一段非常重要。

### 7.1 舊版概念

舊版比較接近：

- 一個主迴圈
- 按鍵事件發生時改位置
- render 跟 update 綁在一起
- 玩家像「一格一格」地移

### 7.2 新版概念

新版改成：

- **固定物理步長**
- **可變 render 頻率**
- **玩家與敵人都用速度向量連續移動**

### 7.3 固定物理步長在哪裡設

在：

- `configs/env.yaml`

預設值：

```yaml
world:
  physics_dt: 0.0083333333
```

這大約等於：

- `1 / 120`
- 也就是 **120 Hz 物理更新**

### 7.4 連續時間的實作流程

在 `library_escape/core/world.py` 的 `tick()` 裡：

1. 把玩家動作和敵人動作轉成方向向量
2. 用 `direction * max_speed` 算出速度
3. 交給 `move_circle()` 做位置更新
4. 檢查碰撞
5. 檢查視野錐
6. 檢查筆記收集
7. 檢查是否抓到玩家
8. 檢查是否全部收集完並逃脫
9. 更新時間與 episode 狀態

### 7.5 位置更新怎麼做

在：

- `library_escape/core/physics.py`

主要函式：

- `move_circle()`

做法是：

1. `delta = velocity * dt`
2. 先更新 x
3. 檢查 x 軸碰撞
4. 再更新 y
5. 檢查 y 軸碰撞

這是簡單、穩定、容易 debug 的 2D 軸向分離碰撞方式。

### 7.6 為什麼這樣就叫連續時間

因為現在角色不是按鍵就跳格，而是：

- 每個物理步都依速度前進一小段
- 位置是浮點數
- render 只是顯示目前位置

所以角色看起來會是「順跑」而不是「一格一格刷新」。

### 7.7 render 與邏輯怎麼解耦

在：

- `library_escape/play/human_vs_ai.py`

我用了 accumulator 的概念：

1. 現實時間一直累積到 `accumulator`
2. 只要 `accumulator >= fixed_dt`
3. 就反覆呼叫 `world.tick(...)`
4. render 另外跑自己的刷新節奏

這樣的好處是：

- render FPS 波動時，物理仍穩定
- RL 訓練時可以不 render
- 同一個策略比較不受 FPS 改變影響

---

## 8. 新 Python 版要怎麼執行

### 8.1 第一次安裝

在 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

如果 PowerShell 不讓你 activate，可以直接用：

```powershell
.\.venv\Scripts\python -m pip install -e .[dev]
```

### 8.2 執行 Python 人類對 AI 遊戲

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai
```

### 8.3 執行 Python AI 對 AI

```powershell
.\.venv\Scripts\python -m library_escape.play.ai_vs_ai --render
```

### 8.4 只做 headless smoke test

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai --headless --max-steps 200
```

### 8.5 如果你有訓練好的敵人模型

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai --enemy-model models\enemy\enemy_ppo.zip
```

### 8.6 控制方式

目前人類遊玩時：

- `W A S D` 或方向鍵移動
- `ESC` 離開

---

## 9. Python 版的重要入口在哪裡

### 遊玩入口

| 路徑 | 用途 |
|---|---|
| `library_escape/play/human_vs_ai.py` | 人類玩家 vs AI 敵人 |
| `library_escape/play/ai_vs_ai.py` | AI 玩家 vs AI 敵人 |

### 訓練入口

| 路徑 | 用途 |
|---|---|
| `library_escape/train/train_enemy.py` | 訓練敵人 PPO |
| `library_escape/train/train_selfplay.py` | 交替 self-play |

### 評估與工具

| 路徑 | 用途 |
|---|---|
| `library_escape/scripts/eval_elo.py` | 對戰 ELO 粗估 |
| `library_escape/scripts/record_video.py` | 輸出錄影 frame |

---

## 10. 強化學習部分我加了哪些東西

### 10.1 單智能體環境

檔案：

- `library_escape/env/single_agent_env.py`

類別：

- `LibraryEscapeEnv`

用途：

- 把遊戲包成標準 Gymnasium 環境
- 可設定訓練角色是 `player` 或 `enemy`

API：

- `reset() -> observation, info`
- `step(action) -> observation, reward, terminated, truncated, info`

### 10.2 多智能體環境

檔案：

- `library_escape/env/multi_agent_env.py`

類別：

- `LibraryEscapeMAEnv`

用途：

- 把遊戲包成 PettingZoo `ParallelEnv`
- 讓玩家與敵人同時輸入動作

agent 名稱：

- `player_0`
- `enemy_0`

### 10.3 reward engine

檔案：

- `library_escape/rewards/reward_fns.py`

功能：

- 從 YAML 讀 reward 權重
- 把多個 physics substep 的 metrics 合併
- 算出 player reward 與 enemy reward

### 10.4 observation builder

檔案：

- `library_escape/env/obs_builder.py`

功能：

- 決定 observation vector 內容
- 定義 observation space
- 定義 action space

### 10.5 baseline 對手

檔案：

- `library_escape/agents/rule_based_enemy.py`

包含：

- `RuleBasedEnemyController`
- `RuleBasedPlayerController`
- `RandomController`
- `MixtureController`

這些基線對手讓 PPO 在沒有真人資料的情況下也能先學。

### 10.6 PPO 模型包裝

檔案：

- `library_escape/agents/ppo_agent.py`

用途：

- 建立 PPO 模型
- 載入已訓練模型
- 讓已訓練 policy 可以當作 controller 使用

---

## 11. Action Space 在哪裡調

### 11.1 設定檔位置

在：

- `configs/env.yaml`

這一段：

```yaml
actions:
  scheme: discrete
  continuous_clip: 1.0
```

### 11.2 目前支援兩種

#### 模式 A：離散動作

```yaml
actions:
  scheme: discrete
```

代表：

- `Discrete(9)`

9 個動作是：

1. 停止
2. 上
3. 右上
4. 右
5. 右下
6. 下
7. 左下
8. 左
9. 左上

對應邏輯在：

- `library_escape/core/constants.py`
- `library_escape/core/actions.py`

#### 模式 B：連續動作

```yaml
actions:
  scheme: continuous
```

代表：

- `Box(low=-1.0, high=1.0, shape=(2,))`

也就是：

- 一個 2 維向量 `[ax, ay]`
- 再由系統轉成移動方向

### 11.3 真正把 action 解碼成方向的地方

在：

- `library_escape/core/actions.py`

函式：

- `action_to_direction(action, scheme)`

做法：

- `discrete`：用編號查表
- `continuous`：把 `[x, y]` clip 到 `[-1, 1]`

---

## 12. Observation Space 在哪裡調

### 12.1 設定檔位置

在：

- `configs/env.yaml`

這一段：

```yaml
observations:
  normalize: true
  include_velocity: true
  include_heading: true
  include_last_seen_age: true
  player_view_range: 320.0
  player_view_angle_deg: 200.0
  wall_ray_count: 8
  wall_ray_range: 220.0
  memory_seconds: 1.5
```

### 12.2 真正拼 observation vector 的地方

在：

- `library_escape/env/obs_builder.py`

主要函式：

- `_player_obs()`
- `_enemy_obs()`
- `observation_space()`

### 12.3 玩家 observation 目前包含什麼

預設 `player` observation 包含：

1. 玩家位置 `(x, y)`
2. 玩家速度 `(vx, vy)`，如果 `include_velocity=true`
3. 玩家朝向 `(cos(theta), sin(theta))`，如果 `include_heading=true`
4. 敵人相對位置 `(dx, dy)`，但只有玩家看得到敵人時才有值
5. 敵人是否可見 `0/1`
6. 最近一張 note 的相對位置
7. 出口相對位置
8. 剩餘 note 比例
9. 剩餘時間比例
10. 牆面 raycast 距離特徵

預設維度大致是：

- `23 維`

但如果你關掉某些 feature，維度會跟著改。

### 12.4 敵人 observation 目前包含什麼

預設 `enemy` observation 包含：

1. 敵人位置 `(x, y)`
2. 敵人速度 `(vx, vy)`
3. 敵人朝向 `(cos(theta), sin(theta))`
4. 玩家相對位置 `(dx, dy)`，只有看得到時才有值
5. 玩家是否在視野內 `0/1`
6. 上次看到玩家的位置相對量
7. 上次看到玩家距今多久
8. 剩餘 note 比例
9. 剩餘時間比例
10. 牆面 raycast 距離特徵

預設維度大致是：

- `22 維`

### 12.5 如果你要大改 observation

你有兩層可以改：

#### 只改參數，不改結構

改：

- `configs/env.yaml`

例如：

- `wall_ray_count`
- `wall_ray_range`
- `include_velocity`
- `include_heading`
- `memory_seconds`

#### 改 observation 的實際組成邏輯

改：

- `library_escape/env/obs_builder.py`

---

## 13. Reward 在哪裡調

### 13.1 reward 設定檔

在：

- `configs/rewards.yaml`

目前預設值：

```yaml
enemy:
  catch_player: 100.0
  player_in_cone_per_step: 1.0
  distance_shaping: 0.04
  time_penalty: -0.01
  wall_penalty: -0.1
  stalemate: -5.0

player:
  collect_note: 20.0
  escape: 100.0
  seen_per_step: -0.5
  caught: -100.0
  time_bonus: 0.01
  wall_penalty: -0.1
```

### 13.2 reward 的計算程式在哪裡

在：

- `library_escape/rewards/reward_fns.py`

### 13.3 enemy reward 公式

敵人的單次 RL step reward 是：

```text
R_enemy
= catch_player * I(caught)
+ player_in_cone_per_step * visible_steps
+ distance_shaping * distance_delta
+ time_penalty * steps_taken
+ wall_penalty * enemy_wall_hits
+ stalemate * I(stalemate)
```

其中：

- `I(caught)`：如果抓到玩家就是 1，否則 0
- `visible_steps`：這個 RL step 內部，玩家被看到的 physics substep 次數
- `distance_delta = 舊距離 - 新距離`
  - 如果敵人更接近玩家，這個值是正的
  - 如果敵人離玩家更遠，這個值是負的
- `steps_taken`：這次 `env.step()` 實際包了幾個 physics substep
- `enemy_wall_hits`：敵人撞牆次數

### 13.4 player reward 公式

玩家的單次 RL step reward 是：

```text
R_player
= collect_note * note_collected
+ escape * I(escaped)
+ seen_per_step * visible_steps
+ caught * I(caught)
+ time_bonus * steps_taken
+ wall_penalty * player_wall_hits
```

其中：

- `note_collected`：這次 step 收到幾張筆記
- `I(escaped)`：成功逃脫為 1
- `visible_steps`：被敵人視野看到的內部 substep 次數
- `I(caught)`：被抓到為 1
- `player_wall_hits`：玩家撞牆次數

### 13.5 reward 是在哪一層加總的

在 `LibraryEscapeEnv.step()` / `LibraryEscapeMAEnv.step()` 裡：

1. 先做多個內部 `world.tick()`
2. 每個 tick 回傳一個 `StepMetrics`
3. `RewardEngine.combine()` 把多個 tick 的 metrics 加總
4. 再用 `RewardEngine.single_agent_reward()` 或 `multi_agent_rewards()` 算最終 reward

### 13.6 如果我要換 reward 演算法怎麼做

#### 只改權重

改：

- `configs/rewards.yaml`

#### 改 reward 結構本身

改：

- `library_escape/rewards/reward_fns.py`

如果你要新增新的 shaping 項目，通常流程是：

1. 在 `StepMetrics` 加欄位
2. 在 `world.tick()` 裡更新這個欄位
3. 在 `reward_fns.py` 裡把它乘上權重
4. 在 `configs/rewards.yaml` 補上對應係數

---

## 14. 環境本身要去哪裡調

### 14.1 地圖、筆記、出口、障礙

改：

- `configs/env.yaml`

關鍵區塊：

- `world.width`
- `world.height`
- `world.note_radius`
- `world.notes`
- `world.exit_zone`
- `world.obstacles`

### 14.2 玩家與敵人的移動速度、視野與碰撞

也是改：

- `configs/env.yaml`

例如：

```yaml
player:
  radius: 16.0
  speed: 220.0
  vision_range: 260.0
  vision_angle_deg: 180.0

enemy:
  radius: 18.0
  speed: 205.0
  vision_range: 285.0
  vision_angle_deg: 78.0
  catch_radius: 10.0
```

### 14.3 巡邏路徑

改：

- `configs/env.yaml`

的：

```yaml
enemy:
  patrol_points:
```

這些點會被 `RuleBasedEnemyController` 用來當巡邏路徑。

### 14.4 如果你要改更深層的遊戲規則

改：

- `library_escape/core/world.py`
- `library_escape/core/physics.py`
- `library_escape/core/navigation.py`

---

## 15. Gymnasium 與 PettingZoo 環境的入口

### 15.1 Gymnasium 單智能體

檔案：

- `library_escape/env/single_agent_env.py`

類別：

- `LibraryEscapeEnv`

建立方式範例：

```python
from library_escape.env.single_agent_env import LibraryEscapeEnv

env = LibraryEscapeEnv(role="enemy", frame_skip=4)
obs, info = env.reset(seed=42)
```

### 15.2 PettingZoo 多智能體

檔案：

- `library_escape/env/multi_agent_env.py`

類別：

- `LibraryEscapeMAEnv`

建立方式範例：

```python
from library_escape.env.multi_agent_env import LibraryEscapeMAEnv

env = LibraryEscapeMAEnv(frame_skip=4)
obs, infos = env.reset(seed=42)
```

---

## 16. frame_skip 是什麼，去哪裡調

### 16.1 定義

`frame_skip` 的意思是：

- 一次 RL `env.step(action)`
- 內部不只做 1 次 physics tick
- 而是連做 `frame_skip` 次

### 16.2 預設值

在：

- `configs/training.yaml`

```yaml
shared:
  frame_skip: 4
```

### 16.3 這代表什麼

如果：

- `physics_dt = 1 / 120`
- `frame_skip = 4`

那麼 RL agent 控制頻率約等於：

- `120 / 4 = 30 Hz`

### 16.4 為什麼要這樣做

好處：

- 加速訓練
- 降低 render 負擔
- 讓策略更接近高層決策而不是每個超細小 timestep 都改指令

---

## 17. 怎麼開始訓練

### 17.1 訓練敵人 PPO

命令：

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy
```

這會：

1. 載入 `configs/training.yaml`
2. 建立 `LibraryEscapeEnv(role="enemy")`
3. 用 scripted player 當對手
4. 用 PPO 訓練敵人
5. 把模型存到預設位置

### 17.2 訓練敵人的輸出位置

預設模型位置：

- `models/enemy/enemy_ppo.zip`

tensorboard log：

- `models/logs/enemy/`

### 17.3 指定 timesteps

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy --timesteps 200000
```

### 17.4 指定輸出檔名

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy --output models\enemy\enemy_v2
```

實際會存成：

- `models\enemy\enemy_v2.zip`

### 17.5 交替 self-play 訓練

```powershell
.\.venv\Scripts\python -m library_escape.train.train_selfplay
```

這個腳本會：

1. 先訓練 player policy
2. 再訓練 enemy policy
3. 把新 checkpoint 加入 pool
4. 下一輪從 pool 抽對手
5. 重複多輪

### 17.6 self-play 的輸出位置

- 玩家池：`models/player_pool/`
- 敵人池：`models/enemy_pool/`

### 17.7 現在訓練時終端會自動顯示 ETA

單模型訓練 `train_enemy.py` 會顯示：

- 目前 phase 進度
- 目前 phase 還剩多久
- overall 還剩多久

self-play `train_selfplay.py` 會另外顯示：

- 目前第幾輪
- 現在是 `player` phase 還是 `enemy` phase
- 當前 phase ETA
- 整體 self-play ETA

---

## 18. 訓練參數要去哪裡調

全部在：

- `configs/training.yaml`

### 18.1 shared 區塊

```yaml
shared:
  seed: 42
  frame_skip: 4
  deterministic_eval: true
  device: auto
```

用途：

- `seed`：隨機種子
- `frame_skip`：RL step 包幾個 physics step
- `deterministic_eval`：評估時是否 deterministic
- `device`：`cpu` / `cuda` / `auto`

### 18.2 single_agent 區塊

```yaml
single_agent:
  algorithm: ppo
  role: enemy
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
  checkpoint_dir: models/enemy
  log_dir: models/logs/enemy
  opponent_mix:
    rule_based: 0.75
    random: 0.25
```

用途：

- `total_timesteps`：總訓練步數
- `n_envs`：同時開幾個 env
- `vector_env`：`dummy` 或 `subproc`
- `learning_rate`、`gamma`、`gae_lambda`、`ent_coef` 等：PPO 超參數
- `net_arch`：MLP 網路層大小
- `opponent_mix`：對手類型比例

### 18.3 multi_agent 區塊

```yaml
multi_agent:
  rounds: 6
  timesteps_per_round: 20000
  pool_size: 4
  player_checkpoint_dir: models/player_pool
  enemy_checkpoint_dir: models/enemy_pool
  player_net_arch: [128, 128]
  enemy_net_arch: [128, 128]
  symmetric_reward: false
  reward_shaping: true
```

用途：

- `rounds`：總共做幾輪交替訓練
- `timesteps_per_round`：每輪訓練多久
- `pool_size`：保留多少歷史 checkpoint
- `player_net_arch` / `enemy_net_arch`：雙方網路結構

### 18.4 目前實作上的注意事項

proposal 裡提到的 `MAPPO` / `RLlib` / 更完整的 league training，目前我**沒有直接實作成完整版**。

現在實作的是：

- **Gymnasium 單智能體 PPO**
- **PettingZoo 多智能體環境**
- **交替式 self-play checkpoint pool**

也就是說：

- 多智能體訓練能力已經有基礎
- 但目前主腳本是較簡單、較穩定、較容易維護的交替 PPO，而不是完整 MAPPO pipeline

這一點我必須講清楚。

---

## 19. 訓練過程怎麼看成果

### 19.1 TensorBoard

訓練敵人時，我有把 log 寫到：

- `models/logs/enemy/`

啟動方式：

```powershell
.\.venv\Scripts\tensorboard --logdir models\logs\enemy
```

然後打開瀏覽器看：

- `http://localhost:6006`

### 19.2 直接拿模型回去玩

例如：

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai --enemy-model models\enemy\enemy_ppo.zip
```

### 19.3 AI 對 AI 檢查

```powershell
.\.venv\Scripts\python -m library_escape.play.ai_vs_ai --player-model models\player_pool\player_round_1.zip --enemy-model models\enemy_pool\enemy_round_1.zip --render
```

### 19.4 ELO 粗估

```powershell
.\.venv\Scripts\python -m library_escape.scripts.eval_elo --episodes 20
```

### 19.5 錄影

```powershell
.\.venv\Scripts\python -m library_escape.scripts.record_video --output-dir recordings\demo_frames
```

---

## 20. 我目前已經跑過的驗證

我已經實際執行過以下項目：

1. `python -m compileall library_escape`
2. `pytest`
3. Gymnasium `check_env`
4. PettingZoo `parallel_api_test`
5. 人類對 AI headless smoke test
6. AI 對 AI smoke test
7. PPO 短步數 smoke training
8. `train_enemy.py` smoke training
9. `train_selfplay.py` smoke training
10. `eval_elo.py` smoke run
11. `record_video.py` smoke run

### 20.1 目前測試檔

| 檔案 | 用途 |
|---|---|
| `tests/test_physics.py` | 碰撞 |
| `tests/test_cone.py` | 視野遮擋 |
| `tests/test_env_api.py` | Gymnasium / PettingZoo API |

### 20.2 已產生的 smoke 輸出

目前本機上有一些我測試留下來的檔案：

- `models/smoke/*.zip`
- `models/player_pool/*.zip`
- `models/enemy_pool/*.zip`
- `models/logs/enemy/*`
- `recordings/test_frames/*`

它們是測試產物，不是必須保留，但可以拿來參考。

---

## 21. 你如果要改東西，最常改的地方整理

### 我要改地圖

改：

- `configs/env.yaml`

### 我要改角色速度

改：

- `configs/env.yaml`

### 我要改視野範圍

改：

- `configs/env.yaml`

### 我要改 action space

改：

- `configs/env.yaml`
- `library_escape/core/actions.py`
- `library_escape/env/obs_builder.py`

### 我要改 observation space

改：

- `configs/env.yaml`
- `library_escape/env/obs_builder.py`

### 我要改 reward 權重

改：

- `configs/rewards.yaml`

### 我要改 reward 算法

改：

- `library_escape/rewards/reward_fns.py`
- 必要時 `library_escape/core/world.py`

### 我要改訓練超參數

改：

- `configs/training.yaml`

### 我要改 baseline 對手邏輯

改：

- `library_escape/agents/rule_based_enemy.py`

### 我要改 render 樣子

改：

- `library_escape/render/pygame_view.py`

---

## 22. 目前這個版本的限制

### 22.1 已完成的部分

- Python 化
- 連續移動
- 固定物理步長
- Gymnasium env
- PettingZoo env
- PPO 訓練敵人
- 交替 self-play 基礎版
- reward / env / training 配置外部化

### 22.2 尚未做成完整 proposal 終局版的部分

- 還沒做成完整 MAPPO 訓練管線
- 還沒做成更複雜的 league / PSRO
- 還沒做像素輸入 CNN 版本
- 還沒做正式 UI 內建切換 checkpoint 難度
- 還沒做多關卡地圖編輯器整合

### 22.3 這不代表不能用

目前這版已經可以：

- 玩
- 訓練
- 測
- 改 reward
- 改 observation / action
- 做 AI vs AI
- 做 headless RL

所以它已經是一個可用的 RL 遊戲平台雛形，而不只是 demo。

---

## 23. 我建議你接下來怎麼用

### 路線 A：先確認遊戲可玩

```powershell
.\.venv\Scripts\python -m library_escape.play.human_vs_ai
```

### 路線 B：先訓練一個敵人模型

```powershell
.\.venv\Scripts\python -m library_escape.train.train_enemy --timesteps 200000
```

### 路線 C：先改 reward 做實驗

1. 改 `configs/rewards.yaml`
2. 重新訓練
3. 用 TensorBoard 與遊玩測試比較差異

### 路線 D：先改 observation / action 做 ablation study

1. 改 `configs/env.yaml`
2. 必要時改 `library_escape/env/obs_builder.py`
3. 重新訓練

---

## 24. 一句話總結

現在這個 repo 的狀態是：

- **舊 C++ 還在，但只是 legacy**
- **新的 Python 版才是現在真正可擴充的主版本**
- **連續時間、RL 環境、PPO 訓練、self-play 基礎版都已經接起來了**
- **所有你最常要調的東西，都已經集中到 `configs/` 與少數幾個核心 Python 檔案**
