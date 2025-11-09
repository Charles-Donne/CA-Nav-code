# CA-Nav 代码运行流程精要文档

> **项目定位**: 零样本视觉语言导航（VLN）评估系统  
> **核心思想**: LLM 指令分解 + 语义地图 + 价值引导 + 约束监控  
> **更新时间**: 2025-11-07

---

## 📋 目录
- [1. 系统架构概览](#1-系统架构概览)
- [2. 多进程并行评估流程](#2-多进程并行评估流程)
- [3. 单 Episode 执行流程](#3-单-episode-执行流程)
- [4. 核心模块说明](#4-核心模块说明)
- [5. 关键数据流](#5-关键数据流)
- [6. 评估指标](#6-评估指标)
- [7. 快速调试指南](#7-快速调试指南)

---

## 1. 系统架构概览

### 1.1 整体架构
```bash
┌─────────────────────────────────────────────────────┐
│                   多进程评估系统                      │
│  ┌───────────┐  ┌───────────┐       ┌───────────┐  │
│  │ Worker 0  │  │ Worker 1  │  ...  │ Worker 15 │  │
│  │  (GPU 0)  │  │  (GPU 1)  │       │  (GPU 7)  │  │
│  └─────┬─────┘  └─────┬─────┘       └─────┬─────┘  │
│        │              │                    │       │
│        └──────────────┴────────────────────┘       │
│                       ↓                            │
│              结果汇总 & 指标计算                      │
└─────────────────────────────────────────────────────┘
```

### 1.2 关键文件
| 文件 | 作用 | 行数参考 |
|------|------|---------|
| `main.sh` | 启动脚本 | - |
| `run_mp.py` | 多进程管理器 | 170 行 |
| `ZS_Evaluator_mp.py` | 评估器核心 | 816 行 |
| `exp1.yaml` | 配置文件 | - |

---

## 2. 多进程并行评估流程

### 2.1 启动流程（run_mp.py）
```python
main.sh 执行
    ↓
① 解析命令行参数
   --exp_name exp_1
   --nprocesses 16
   --exp-config exp1.yaml
    ↓
② 加载配置 & 数据集分片
   episode_ids = [1, 2, ..., 1000]
   split_episode_ids = [
       [1, 17, 33, ...],    # Worker 0 (每隔16取一个)
       [2, 18, 34, ...],    # Worker 1
       ...
       [16, 32, 48, ...]    # Worker 15
   ]
    ↓
③ 创建进程配置
   for i in range(16):
       config.local_rank = i
       config.TORCH_GPU_ID = i % 8    # GPU 循环分配
       config.EPISODES_ALLOWED = split_episode_ids[i]
    ↓
④ 启动进程池
   Pool(16).map(worker, configs)
    ↓
⑤ 并行执行 & 结果汇总
   worker → trainer.eval() → stats_ep_ckpt_*.json
   汇总 → stats_ckpt_val_unseen.json
```

### 2.2 GPU 分配策略
```
进程 ID    GPU ID    处理 Episodes
  0    →    0    →   [1, 17, 33, 49, ...]
  1    →    1    →   [2, 18, 34, 50, ...]
  ...
  7    →    7    →   [8, 24, 40, 56, ...]
  8    →    0    →   [9, 25, 41, 57, ...]  ← 循环复用
  ...
  15   →    7    →   [16, 32, 48, 64, ...]
```

---

## 3. 单 Episode 执行流程

### 3.1 核心流程（trainer.eval() → rollout()）
```
┌─────────────────────────────────────────┐
│ Episode 开始                             │
└─────────────────────────────────────────┘
    ↓
【阶段 1】初始化 (0-12 步)
    ├─ envs.reset() → 获取初始观察
    ├─ _process_llm_reply() → 解析指令
    │   ├─ instruction: "走到厨房，左转到客厅，找沙发"
    │   ├─ sub_instructions: ["走到厨房", "左转", "找沙发"]
    │   ├─ sub_constraints: {"0": [["object", "kitchen"]], ...}
    │   └─ destination: "kitchen"
    ├─ _look_around() → 环视 360°
    │   └─ 执行 12 次 TURN_LEFT (12×30° = 360°)
    │       └─ 每步更新语义地图 + 价值图
    └─ 初始动作规划
    ↓
【阶段 2】主导航循环 (13-500 步)
    for step in range(12, 500):
        ├─ 更新轨迹 & 历史地图
        │
        ├─ 约束检查
        │   ├─ 检测物体: GroundedSAM("kitchen")
        │   ├─ 检查方向: heading 变化
        │   └─ 满足条件 → 切换子任务
        │
        ├─ 执行动作
        │   └─ envs.step(action)
        │
        ├─ 更新地图
        │   ├─ RGB + Depth → GroundedSAM
        │   ├─ 语义分割 → 语义地图
        │   └─ 处理可穿越区域 & 边界
        │
        ├─ 计算价值图
        │   ├─ BLIP(RGB, "kitchen") → 0.75
        │   └─ ValueMap × HistoryMap
        │
        └─ 规划下一步
            └─ FMM(value_map) → action
    ↓
【阶段 3】结束 & 指标计算
    ├─ 对比真实轨迹
    ├─ 计算 Success / SPL / NDTW
    └─ 保存到 state_eps[ep_id]
```

### 3.2 子任务切换机制
```python
约束检查循环:
    current_constraint = [["object", "kitchen"]]
    ↓
    check = constraints_monitor(obs, "kitchen")
    ↓
    if check == [True]:  # 检测到 kitchen
        constraints_check[0] = True
        constraint_steps = 0
        ↓
        切换到下一个子任务
        current_constraint = [["direction", "left"]]
        destination = "living room"
```

---

## 4. 核心模块说明

### 4.1 关键模块表
| 模块 | 功能 | 输入 | 输出 | 位置 |
|------|------|------|------|------|
| **GroundedSAM** | 开放词汇目标检测 | RGB + classes | masks, labels | `semantic_prediction.py` |
| **Semantic_Mapping** | 构建语义占据地图 | RGB+Depth+Pose | full_map | `mapping.py` |
| **ValueMap** | 目标价值分布 | map + BLIP | value_map | `value_map.py` |
| **HistoryMap** | 访问惩罚 | trajectory | history_map | `history_map.py` |
| **DirectionMap** | 方向约束掩码 | trajectory + direction | direction_map | `direction_map.py` |
| **FusionMapPolicy** | FMM 路径规划 | value_map + maps | action | `Policy.py` |
| **ConstraintsMonitor** | 约束检查 | constraint + obs | [True/False] | `constraints.py` |

### 4.2 数据维度
```python
# 图像
RGB:    (480, 640, 3)
Depth:  (480, 640, 1)

# 地图 (分辨率 5cm/pixel)
map_shape: (480, 480)  # 2400cm = 24m
full_map: (4+N, 480, 480)
    ├─ [0]: 障碍物
    ├─ [1]: 已探索区域
    ├─ [2]: 当前位置
    ├─ [3]: 已访问
    └─ [4:]: N个类别的语义通道 (动态)

# 价值图
value_map: (2, 480, 480)
    ├─ [0]: 原始价值
    └─ [1]: 处理后价值 (融合历史/方向)
```

---

## 5. 关键数据流

### 5.1 观察处理流程
```
obs (dict)
    ├─ rgb: (480, 640, 3)
    ├─ depth: (480, 640, 1)
    └─ sensor_pose: (x, y, heading)
    ↓
_batch_obs()
    ├─ _concat_obs(): RGB + Depth → (4, 480, 640)
    ├─ _preprocess_state()
    │   ├─ _get_sem_pred()
    │   │   ├─ GroundedSAM(rgb, ["kitchen", "floor"])
    │   │   │   → masks (K, 480, 640), labels
    │   │   └─ _process_masks()
    │   │       → final_masks (N, 480, 640)  # N = len(detected_classes)
    │   ├─ _preprocess_depth()
    │   │   → depth (480, 640)
    │   └─ 下采样 (4x) → (160, 160)
    └─ batch: (1, 4+N, 160, 160)
    ↓
mapping_module(batch, poses)
    └─ full_map: (4+N, 480, 480)
```

### 5.2 动作规划流程
```
full_map (4+N, 480, 480)
    ↓
_process_map()
    ├─ 提取: obstacles, explored, objects
    ├─ 计算: traversible, floor, frontiers
    └─ 形态学处理 (闭运算)
    ↓
value_map_module()
    ├─ BLIP(RGB, "kitchen") → 0.75
    ├─ 投影到地图 → blip_map
    ├─ 结合语义通道 → semantic_value
    └─ 融合 → value_map (2, 480, 480)
    ↓
history_map × value_map
    ↓
policy(value_map, collision_map, ...)
    ├─ FMM 扩散 → distance_map
    ├─ 找最优路径点
    └─ 转换为动作: {0: STOP, 1: FORWARD, 2: LEFT, 3: RIGHT}
```

---

## 6. 评估指标

### 6.1 指标定义
```python
# Success Rate (SR)
success = 1.0 if distance_to_goal <= 3.0 else 0.0

# Oracle Success (OS)
oracle_success = 1.0 if any(distances <= 3.0) else 0.0

# Success weighted by Path Length (SPL)
spl = success × gt_length / max(gt_length, path_length)

# Normalized Dynamic Time Warping (NDTW)
dtw_dist = fastdtw(pred_path, gt_path)
ndtw = exp(-dtw_dist / (len(gt_path) × 3.0))

# Success weighted by NDTW (SDTW)
sdtw = ndtw × success
```

### 6.2 指标计算位置
```python
# 单个 episode
_calculate_metric(infos)  # ZS_Evaluator_mp.py:129

# 汇总结果
run_mp.py:129-147
```

---

## 7. 快速调试指南

### 7.1 关键打印位置
```python
# Episode 信息
print(f"current episode id: {self.current_episode_id}")  # Line 483

# LLM 解析
print(f"first destination: {self.destination}")  # Line 391

# 检测类别
print("current step detected classes: ", labels)  # Line 228

# 约束检查
print(current_constraint, check)  # Line 666

# 子任务切换
print(f"sub_instr_{current_idx}: {self.sub_instructions[current_idx]}")  # Line 683
```

### 7.2 常见断点
```python
# 1. 初始化完成
ZS_Evaluator_mp.py:783  # trainer.eval() 开始

# 2. Episode 开始
ZS_Evaluator_mp.py:570  # rollout() 开始

# 3. 环视完成
ZS_Evaluator_mp.py:543  # _look_around() 返回

# 4. 约束检查
ZS_Evaluator_mp.py:665  # check = constraints_monitor()

# 5. 动作规划
ZS_Evaluator_mp.py:768  # self._action = policy()
```

### 7.3 可视化检查
```python
# 地图可视化
config.MAP.VISUALIZE = True  # exp1.yaml

# 保存位置
self.mapping_module.visualize(
    step, 
    episode_id, 
    config.MAP.RESULTS_DIR
)
```

---

## 8. 配置快速参考

### 8.1 关键配置（exp1.yaml）
```yaml
# 进程配置
NUM_ENVIRONMENTS: 1        # 每个进程的环境数
GPU_NUMBERS: 1            # GPU 总数

# 地图配置
MAP:
  MAP_SIZE_CM: 2400       # 24m × 24m
  MAP_RESOLUTION: 5       # 5cm/pixel
  VISUALIZE: False        # 是否可视化

# 评估配置
EVAL:
  MIN_CONSTRAINT_STEPS: 10   # 最小约束步数
  MAX_CONSTRAINT_STEPS: 25   # 最大约束步数
  VALUE_THRESHOLD: 0.30      # 价值图阈值
```

### 8.2 环境变量
```bash
# main.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAGNUM_LOG=quiet
HABITAT_SIM_LOG=quiet
```

---

## 9. 性能优化要点

### 9.1 内存优化
- ✅ 动态通道掩码 (避免固定大类别数)
- ✅ 单步地图清零 (避免累积)
- ✅ 批处理 padding (支持不同通道数)

### 9.2 速度优化
- ✅ 多进程并行 (16 进程 × 8 GPU)
- ✅ GPU 复用 (每个 GPU 运行 2 个进程)
- ✅ FMM 快速行进算法

### 9.3 鲁棒性设计
- ✅ 碰撞检测 (30 步卡住 → 重新规划)
- ✅ 价值图为空 (5 次 → 重新环视)
- ✅ 约束超时 (最大步数 → 强制切换)

---

## 10. 故障排查

### 10.1 常见问题
| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| CUDA OOM | GPU 内存不足 | 减少 nprocesses 或关闭可视化 |
| 进程卡住 | Episode 太长 | 检查 MAX_EPISODE_STEPS |
| 结果文件缺失 | 进程崩溃 | 查看 collision_stuck.txt |
| 指标异常 | 真实轨迹缺失 | 检查 GT_PATH 配置 |

### 10.2 日志位置
```
data/logs/running_log/exp_1_log.txt       # 运行日志
data/checkpoints/exp_1/*_collision_stuck.txt  # 碰撞日志
data/logs/eval_results/exp_1/             # 可视化结果
```

---

## 📌 快速查找

### 按功能查找代码
- **多进程管理**: `run_mp.py:20-147`
- **Episode 循环**: `ZS_Evaluator_mp.py:783-810`
- **主导航循环**: `ZS_Evaluator_mp.py:633-770`
- **约束检查**: `ZS_Evaluator_mp.py:654-697`
- **地图更新**: `ZS_Evaluator_mp.py:728-735`
- **价值图计算**: `ZS_Evaluator_mp.py:761-764`
- **动作规划**: `ZS_Evaluator_mp.py:767-772`

### 按模块查找
- **GroundedSAM**: `vlnce_baselines/map/semantic_prediction.py`
- **Semantic_Mapping**: `vlnce_baselines/map/mapping.py`
- **ValueMap**: `vlnce_baselines/map/value_map.py`
- **FusionMapPolicy**: `vlnce_baselines/models/Policy.py`
- **ConstraintsMonitor**: `vlnce_baselines/common/constraints.py`

---

## 🎯 核心要点总结

1. **架构**: 多进程并行 + 数据分片 + 结果汇总
2. **流程**: 环视初始化 → 约束驱动导航 → 指标计算
3. **关键**: LLM 指令分解 + 语义地图 + 价值引导 + FMM 规划
4. **创新**: 开放词汇检测 + 动态约束监控 + 子任务自动切换

---

**最后更新**: 2025-11-07  
**维护者**: CA-Nav Team  
**相关论文**: [待补充]



每一步循环 (step 12 → 500):
  ↓
1. 更新轨迹点 trajectory_points, direction_points
    ↓
2. 计算 history_map (避免重复访问)
    ↓
3. 检查约束 → 更新 direction_map
    ↓
4. 约束满足 → constraint_steps 达标 → 切换子任务
    ↓
5. 检测卡住/价值图空 → 触发重规划/重新环视
    ↓
6. 执行动作 → 更新地图
    ↓
7. 计算新的价值图 (融合 history_map * direction_map)
    ↓
8. FMM规划下一步动作

