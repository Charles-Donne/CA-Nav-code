# rollout() 函数详细说明文档

## 📋 目录
1. [函数概览](#函数概览)
2. [执行流程](#执行流程)
3. [状态变量说明](#状态变量说明)
4. [子任务管理机制](#子任务管理机制)
5. [异常恢复策略](#异常恢复策略)
6. [关键模块调用](#关键模块调用)
7. [示例场景](#示例场景)

---

## 函数概览

### 功能
`rollout()` 是 VLN (Vision-and-Language Navigation) 任务的**核心执行函数**，负责处理从初始化到完成的整个导航过程。

### 输入输出
```python
def rollout(self) -> None:
    """
    输入: self (包含环境、模型、配置等)
    输出: None (结果通过 self._calculate_metric() 记录到 self.state_eps)
    """
```

### 处理流程概览
```
┌─────────────────────┐
│  1. 初始化地图      │ _maps_initialization()
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│  2. 环视 360°       │ _look_around()
│     建立初始地图    │ 12 步 × 30° = 360°
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│  3. 主导航循环      │ for step in range(12, 500):
│     步数 12-500     │   ├─ 更新轨迹
│                     │   ├─ 检查约束
│                     │   ├─ 切换子任务
│                     │   ├─ 执行动作
│                     │   ├─ 更新地图
│                     │   └─ 规划下一步
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│  4. 计算评估指标    │ _calculate_metric()
└─────────────────────┘
```

---

## 执行流程

### 阶段 1: 初始化与环视建图

#### 1.1 地图初始化
```python
self._maps_initialization()
```
**执行内容:**
- `envs.reset()` - 重置 Habitat 环境
- `_process_llm_reply()` - 解析 LLM 指令分解结果
  - `sub_instructions` - 子指令列表
  - `sub_constraints` - 每个子指令的约束条件
  - `destination` - 最终目标
- `mapping_module.init_map_and_pose()` - 初始化语义地图
  - `full_map`: (1, N+4, 480, 480)
  - `local_map`: (1, N+4, 240, 240)
  - 智能体位于地图中心 (12m, 12m)

#### 1.2 环视建图
```python
full_pose, obs, dones, infos = self._look_around()
```
**执行内容:**
- 循环 12 次，每次左转 30° (总计 360°)
- 每次转向后:
  1. 语义分割 (GroundedSAM)
  2. 点云生成 + 坐标变换
  3. 3D 体素投影 + 高度压缩
  4. 多帧融合 (max 运算)
  5. 更新全局地图

### 阶段 2: 主导航循环 (步数 12-500)

主循环执行 15 个关键步骤：

#### 2.1 打印当前状态
```python
print(f"episode:{self.current_episode_id}, step:{step}")
print(f"instr: {self.instruction}")
print(f"sub_instr_{current_idx}: {self.sub_instructions[current_idx]}")
constraint_steps += 1
```

#### 2.2 更新位置和轨迹
```python
position = full_pose[0][:2] * 100 / self.resolution  # 米 → 像素
trajectory_points.append((y, x))  # 保留最近 2 个点
direction_points.append(np.array([x, y]))  # 保留最近 5 个点
```

#### 2.3 计算历史地图
```python
history_map = self.history_module(trajectory_points, step, episode_id)
```
**用途:** 避免原地徘徊，降低已访问区域的价值

#### 2.4 方向约束处理
```python
if "direction constraint" in all_constraint_types:
    direction_map = self.direction_module(
        current_position, last_five_position, heading,
        direction, step, episode_id
    )
```
**支持的方向:**
- `"turn left"` - 屏蔽右侧和正前方
- `"turn right"` - 屏蔽左侧和正前方
- `"go straight"` - 屏蔽左右两侧

#### 2.5 约束检查
```python
check = self.constraints_monitor(
    current_constraint, obs[0], 
    current_detections, classes, 
    current_pose, start_check_pose
)
```
**约束类型:**
- `direction constraint` - 检查转向角度
- `landmark constraint` - 检查是否看到目标
- `distance constraint` - 检查与目标的距离

#### 2.6 子任务切换
```python
if start_to_wait and (constraint_steps >= min_constraint_steps):
    current_idx = self.constraints_check.index(False)
    # 更新目标类别和约束
    constraint_steps = 0
    start_to_wait = False
```

#### 2.7-2.15 其他步骤
- 更新导航目标
- 价值图为空检测
- 执行动作
- 检查 episode 结束
- 更新语义地图
- 处理导航地图
- 碰撞检测
- 更新碰撞地图
- 计算价值图 (BLIP)
- 路径规划 (FMM)

---

## 状态变量说明

### 轨迹追踪
| 变量 | 类型 | 用途 | 更新频率 |
|------|------|------|----------|
| `trajectory_points` | `List[(y,x)]` | 最近 2 个位置点，用于 HistoryMap | 每步 |
| `direction_points` | `List[array]` | 最近 5 个位置点，用于 DirectionMap | 每步 |

### 约束管理
| 变量 | 类型 | 初始值 | 说明 |
|------|------|--------|------|
| `constraint_steps` | `int` | 0 | 当前子任务已执行步数 |
| `start_to_wait` | `bool` | False | 约束满足后的等待标志 |
| `search_destination` | `bool` | False | 是否到达最后一个子任务 |

### 异常恢复
| 变量 | 阈值 | 触发条件 | 恢复策略 |
|------|------|----------|----------|
| `collided` | ≥30 | 位移 < 0.2m/步 | 重新规划 (replan=True) |
| `empty_value_map` | ≥5 | 价值图 ≤24×24像素 | 重新环视 360° |

### 方向约束
| 变量 | 类型 | 说明 |
|------|------|------|
| `direction_map` | `ndarray (480,480)` | 方向约束掩码，全1=无限制 |
| `direction_map_exist` | `bool` | 是否已计算，避免重复 |

### 位姿追踪
| 变量 | 类型 | 用途 |
|------|------|------|
| `last_pose` | `array [x,y,θ]` | 上一步位姿，计算位移 |
| `current_pose` | `array [x,y,θ]` | 当前位姿，检测卡住 |
| `start_check_pose` | `array [x,y,θ]` | 开始检查方向约束时的位姿 |

---

## 子任务管理机制

### 状态机设计
```
   [EXECUTING]
        │
        │ 约束满足 OR 超过 max_constraint_steps
        ↓
    [WAITING]
  (start_to_wait=True)
        │
        │ 达到 min_constraint_steps
        ↓
  [SWITCH_TASK]
  (切换到下一个子任务)
        │
        ↓
   [EXECUTING]
```

### 切换逻辑
```python
# 进入等待状态
if (sum(check) == len(check) or 
    constraint_steps >= self.max_constraint_steps):
    start_to_wait = True
    self.constraints_check[current_idx] = True

# 切换子任务
if start_to_wait and (constraint_steps >= self.min_constraint_steps):
    if False in self.constraints_check:
        current_idx = self.constraints_check.index(False)
        # 更新目标和约束
        constraint_steps = 0
        start_to_wait = False
```

### 为什么需要 min/max 步数？
- **MIN_CONSTRAINT_STEPS** (默认 5-10步): 避免子任务切换过快，确保每个子任务执行足够时间
- **MAX_CONSTRAINT_STEPS** (默认 50-100步): 避免卡在某个子任务，强制切换下一个

---

## 异常恢复策略

### 1. 碰撞卡住检测
```python
if displacement < 0.2 * 100 / self.resolution:  # < 0.2m/步
    collided += 1
    if collided >= 30:
        replan = True  # 触发重新规划
```
**恢复机制:**
- 将 `replan=True` 传递给 `policy`
- Policy 会清空缓存，重新计算 FMM 路径

### 2. 价值图为空
```python
if np.sum(value_map[1].astype(bool)) <= 24**2:  # ≤ 576 像素
    empty_value_map += 1
    if empty_value_map >= 5:
        _look_around()  # 重新环视 360°
```
**原因分析:**
- 目标不在当前视野内
- 目标被遮挡或未被检测到
- 语义分割失败

**恢复机制:**
- 重新环视 360° 建立完整地图
- 重置 `empty_value_map` 和 `constraint_steps`

### 3. 超时保护
```python
for step in range(12, 500):  # 最多 488 步
    if dones[0]:
        break
```
**触发条件:**
- 达到 episode 最大步数 (默认 500)

---

## 关键模块调用

### 语义分割 (GroundedSAM)
```python
batch_obs = self._batch_obs(obs)  # 包含语义分割
  → _preprocess_obs(obs)
    → _get_sem_pred(rgb)
      → segment_module.segment(rgb, classes)
        返回: masks (N, 480, 640)
```

### 地图更新 (Semantic_Mapping)
```python
self.mapping_module(batch_obs, poses)
full_map, full_pose, one_step_map = self.mapping_module.update_map(...)
```
**流程:**
1. 点云生成: Depth → (120,160,3) 3D点
2. 坐标变换: 相机系 → 智能体系 → 世界系
3. 体素投影: 点云+语义 → (N+1,100,100,80) 3D体素
4. 高度压缩: 沿z轴sum → (N+1,100,100) 2D
5. 位姿变换: agent_view → local_map
6. 多帧融合: max(历史, 当前)

### 价值图计算 (BLIP + ValueMap)
```python
blip_value = self.value_map_module.get_blip_value(rgb, destination)
value_map = self.value_map_module(
    step, full_map, floor, one_step_floor, 
    collision_map, blip_value, ...
)
```
**融合信息:**
- BLIP 语义相似度 (160×160)
- 语义通道 (目标类别掩码)
- 探索奖励 (新探索区域)
- 碰撞惩罚 (碰撞区域价值为0)

### 路径规划 (FMM)



```python
self._action = self.policy(
    value_map[1] * history_map,  # 融合约束
    collision_map, full_map, floor, traversible,
    full_pose, frontiers, detected_classes,
    destination_class, classes, search_destination,
    one_step_map, current_detections, episode_id, replan, step
)
```
**返回:**
```python
{"action": 0}  # STOP
{"action": 1}  # MOVE_FORWARD
{"action": 2}  # TURN_LEFT (30°)
{"action": 3}  # TURN_RIGHT (30°)
```

---

## 示例场景

### 场景 1: 简单指令
**指令:** "Go to the kitchen."

**执行流程:**
```
LLM 解析:
  sub_instructions: ["Go to the kitchen."]
  sub_constraints: {"0": [("landmark constraint", "kitchen")]}
  destination: "kitchen"

第 1 个子任务:
  目标: kitchen
  约束: 看到 kitchen
  
导航过程:
  1-12步: 环视 360° 建图
  13步: 价值图指向 kitchen 方向
  14-15步: 前进 → 看到 kitchen
  16步: constraints_monitor 检测到 kitchen
  17步: 约束满足，进入 WAITING
  22步: 达到 min_constraint_steps
  23步: 切换子任务 (已完成所有子任务)
  24步: 继续前进到 kitchen
  ...
  50步: 到达目标，调用 STOP
```

### 场景 2: 复杂指令
**指令:** "Walk towards the dining table, turn left, and enter the bedroom."

**执行流程:**
```
LLM 解析:
  sub_instructions: [
    "Walk towards the dining table",
    "Turn left",
    "Enter the bedroom"
  ]
  sub_constraints: {
    "0": [("landmark constraint", "dining table")],
    "1": [("direction constraint", "turn left")],
    "2": [("landmark constraint", "bedroom")]
  }

第 1 个子任务 (步数 12-40):
  目标: dining table
  约束: 看到 dining table
  30步: 检测到 dining table → WAITING
  35步: 切换下一子任务

第 2 个子任务 (步数 35-55):
  目标: (保持 dining table)
  约束: 转向角度 > 60° 向左
  40步: 开始左转
  50步: 累计转向 75° → 约束满足
  55步: 切换下一子任务

第 3 个子任务 (步数 55-120):
  目标: bedroom
  约束: 看到 bedroom
  110步: 检测到 bedroom → WAITING
  115步: 所有子任务完成
  120步: 到达 bedroom，调用 STOP
```

### 场景 3: 异常恢复
**指令:** "Go to the bathroom."

**执行流程:**
```
13-40步: 正常导航，朝向 bathroom 前进

41步: 被家具卡住
42-70步: 连续 30 步位移 < 0.2m
  collided 计数器: 1 → 2 → ... → 30

71步: 触发 replan=True
  policy 重新规划路径
  尝试绕过障碍物

72-90步: 成功绕过，继续前进
  collided 重置为 0

100步: 价值图为空 (bathroom 不在视野内)
101-105步: 连续 5 次 empty_value_map++

106步: 触发重新环视
  _look_around() 执行 12 步 360° 扫描
  重新建立完整地图

118步: 重新检测到 bathroom
119-150步: 继续导航到目标
```

---

## 调试技巧

### 1. 查看当前状态
```python
print(f"step:{step}, idx:{current_idx}, constraint_steps:{constraint_steps}")
print(f"constraints:{current_constraint}, check:{check}")
print(f"collided:{collided}, empty_value_map:{empty_value_map}")
```

### 2. 可视化地图
```python
# 在 mapping.py 中启用可视化
config.MAP.VISUALIZE = True
config.MAP.PRINT_IMAGES = True
```

### 3. 手动控制
```python
config.KEYBOARD_CONTROL = 1
# 使用 w/a/d 键手动控制智能体
```

### 4. 记录轨迹
```python
# 保存位姿历史
pose_history.append(full_pose[0].copy())
# 保存到文件
np.save(f"trajectory_{episode_id}.npy", pose_history)
```

---

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_EPISODE_STEPS` | 500 | episode 最大步数 |
| `MIN_CONSTRAINT_STEPS` | 5 | 子任务最小执行步数 |
| `MAX_CONSTRAINT_STEPS` | 100 | 子任务最大执行步数 |
| `CENTER_RESET_STEPS` | 25 | 地图自动居中间隔 |
| `MAP_SIZE_CM` | 2400 | 地图物理尺寸 (24m) |
| `MAP_RESOLUTION` | 5 | 地图分辨率 (5cm/pixel) |

---

**最后更新:** 2025-11-08
