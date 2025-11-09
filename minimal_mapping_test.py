"""
最小化建图验证程序

功能：
1. 初始化一个 Habitat 环境
2. 环视 360° (12 步 × 30°)
3. 保存所有地图数据

用法：
python minimal_mapping_test.py --exp-config vlnce_baselines/config/exp1.yaml
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from habitat import Config
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.config.default import get_config
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.utils.constant import base_classes, map_channels
from vlnce_baselines.utils.map_utils import *

# 图像处理库
import cv2
from skimage.morphology import remove_small_objects, binary_closing, disk


class MinimalMappingTest:
    """最小化建图测试"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = get_device(config.TORCH_GPU_ID)
        torch.cuda.set_device(self.device)
        
        # 地图配置
        self.resolution = config.MAP.MAP_RESOLUTION
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution)
        
        # 创建输出目录
        self.output_dir = "data/minimal_mapping_test"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/maps", exist_ok=True)
        os.makedirs(f"{self.output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{self.output_dir}/depth", exist_ok=True)
        
        print(f"[INFO] 输出目录: {self.output_dir}")
        print(f"[INFO] 地图尺寸: {self.map_shape}")
        print(f"[INFO] 地图分辨率: {self.resolution} cm/pixel")
        
    def initialize_environment(self):
        """初始化环境"""
        print("\n[STEP 1] 初始化 Habitat 环境...")
        
        # 构建环境
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        
        # 重置环境
        obs = self.envs.reset()
        self.episode_id = self.envs.current_episodes()[0].episode_id
        
        print(f"[INFO] Episode ID: {self.episode_id}")
        print(f"[INFO] 场景: {self.envs.current_episodes()[0].scene_id}")
        
        return obs[0]
    
    def initialize_modules(self):
        """初始化建图模块"""
        print("\n[STEP 2] 初始化建图模块...")
        
        # 语义分割模块
        self.segment_module = GroundedSAM(self.config, self.device)
        print("[INFO] GroundedSAM 初始化完成")
        
        # 语义地图模块
        # 注意：mapping_module 内部维护两套地图：
        #   • full_map (480×480): 全局地图，对应 24m×24m 物理空间
        #   • local_map (240×240): 局部地图，以智能体为中心的 12m×12m 活动窗口
        # 观察数据先投影到 local_map，再写回 full_map 对应区域
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        print("[INFO] Semantic_Mapping 初始化完成")
        # 注意：full_w, full_h 等属性要在 init_map_and_pose() 调用后才会初始化
        
        # 检测类别
        self.detected_classes = OrderedSet()
        self.classes = base_classes.copy()  # ["floor", "wall", "door", ...]
        
    def preprocess_observation(self, obs):
        """预处理观察：语义分割 + 深度预处理"""
        # 提取 RGB 和 Depth
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        
        # ============ 深度预处理（与 ZS_Evaluator_mp 一致）============
        # 1. 移除通道维度
        depth = depth[:, :, 0] * 1
        
        # 2. 填充缺失深度值（用该列的最大值填充）
        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()
        
        # 3. 将过远的像素设为无效
        mask2 = depth > 0.99
        depth[mask2] = 0.
        
        # 4. 将无效像素设为视野范围（100米）
        mask1 = depth == 0
        depth[mask1] = 100.0
        
        # 5. 归一化到厘米单位（关键步骤！）
        min_depth = 0.5  # 从 zs_vlnce_task.yaml: DEPTH_SENSOR.MIN_DEPTH
        max_depth = 5.0  # 从 zs_vlnce_task.yaml: DEPTH_SENSOR.MAX_DEPTH
        depth = min_depth * 100.0 + depth * max_depth * 100.0
        # 转换: [0, 1] → [50cm, 550cm]
        
        # 6. 恢复通道维度
        depth = depth[:, :, np.newaxis]
        # ============================================================
        
        # 语义分割
        masks, labels, annotated_image, detections = \
            self.segment_module.segment(rgb[:,:,::-1], classes=self.classes)
        
        print(f"[INFO] 检测到类别: {labels}")
        
        # 处理标签
        class_names = []
        for label in labels:
            class_name = " ".join(label.split(' ')[:-1])
            class_names.append(class_name)
            self.detected_classes.add(class_name)
        
        # 处理掩码
        if masks.shape != (0,):
            from collections import defaultdict
            same_label_indexs = defaultdict(list)
            for idx, item in enumerate(class_names):
                same_label_indexs[item].append(idx)
            
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)
            
            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), 480, 640))
        
        # 合并 RGB + Depth + Semantic
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)  # (4, 480, 640)
        sem_masks = final_masks.transpose(1, 2, 0)  # (480, 640, N)
        state = np.concatenate((state[:3], state[3:4], sem_masks.transpose(2,0,1)), axis=0)  # (4+N, 480, 640)
        
        # 不需要 resize，直接使用原始尺寸（与配置文件中的 FRAME_WIDTH/HEIGHT 一致）
        # state 已经是 (4+N, 480, 640)，符合 mapping 模块的预期
        
        return state, rgb, depth, annotated_image
    
    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int=3) -> tuple:
        """处理语义地图，提取导航相关信息（参考 ZS_Evaluator_mp.py）
        
        Args:
            step: 当前步数
            full_map: (N+4, H, W) 语义地图
            kernel_size: 形态学操作的核大小
            
        Returns:
            traversible: 可穿越区域
            floor: 地板区域
            frontiers: 边界区域（探索边缘）
        """
        # 区分可导航和不可导航的类别
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        full_map = remove_small_objects(full_map.astype(bool), min_size=64)
        
        # 提取地图通道
        obstacles = full_map[0, ...].astype(bool)  # 障碍物
        explored_area = full_map[1, ...].astype(bool)  # 已探索区域
        objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool) if len(not_navigable_index) > 0 else np.zeros_like(obstacles)
        
        # 形态学处理（闭运算，填充小孔）
        footprint = disk(kernel_size)  # 新版 scikit-image 使用 footprint 替代 selem
        obstacles_closed = binary_closing(obstacles, footprint=footprint)
        objects_closed = binary_closing(objects, footprint=footprint)
        navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index]) if len(navigable_index) > 0 else np.zeros_like(obstacles)
        navigable = np.logical_and(navigable, np.logical_not(objects))
        navigable_closed = binary_closing(navigable, footprint=footprint)
        
        # 计算不可穿越区域
        untraversible = np.logical_or(objects_closed, obstacles_closed)
        untraversible[navigable_closed == 1] = 0
        untraversible = remove_small_objects(untraversible, min_size=64)
        untraversible = binary_closing(untraversible, footprint=disk(3))
        traversible = np.logical_not(untraversible)

        # 计算地板区域
        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=footprint)
        traversible = np.logical_or(floor, traversible)
        
        # 计算边界（探索边缘）
        explored_area = binary_closing(explored_area, footprint=footprint)
        contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
        frontiers = np.logical_and(floor, image)
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

        return traversible, floor, frontiers.astype(np.uint8)
    
    def _visualize_semantic_map(self, full_map, full_pose, lmb):
        """生成彩色语义地图可视化（完全按照原程序 mapping.py._visualize 的逻辑）
        
        颜色映射：
        - 0: 黑色（未探索）
        - 1: 深灰（障碍物）
        - 2: 浅灰（已探索但无语义）
        - 3: 蓝色（智能体轨迹，这里用当前位置）
        - 4+: 彩色（各类检测到的物体）
        """
        from vlnce_baselines.utils.constant import color_palette, legend_color_palette
        
        # 提取语义通道（4以后是物体类别）
        # semantic_map: 每个像素的值是该像素最可能的类别 ID
        semantic_map = full_map[4:].argmax(0)  # (480, 480)
        obstacle_map = full_map[0]
        explored_map = full_map[1]
        
        # 语义ID偏移：0保留给未探索，1保留给障碍物，2保留给已探索空区域，3保留给智能体
        # 所以物体类别从5开始
        semantic_map = semantic_map + 5
        
        # 获取特殊区域的 mask
        not_cat_id = len(self.detected_classes) + 5  # 最后一个是"未分类"
        not_cat_mask = (semantic_map == not_cat_id)
        obstacle_mask = np.rint(obstacle_map) == 1
        explored_mask = np.rint(explored_map) == 1
        
        # 未分类区域设为0（黑色）
        semantic_map[not_cat_mask] = 0
        
        # 已探索但无物体的区域设为2（浅灰）
        free_mask = np.logical_and(not_cat_mask, explored_mask)
        semantic_map[free_mask] = 2
        
        # 障碍物设为1（深灰）
        obstacle_mask = np.logical_and(not_cat_mask, obstacle_mask)
        semantic_map[obstacle_mask] = 1
        
        # 当前位置设为3（蓝色）
        pose_r = int(full_pose[1] * 100 / self.resolution)
        pose_c = int(full_pose[0] * 100 / self.resolution)
        # 画一个小圆圈
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                if dr*dr + dc*dc <= 9:  # 半径3的圆
                    r, c = pose_r + dr, pose_c + dc
                    if 0 <= r < semantic_map.shape[0] and 0 <= c < semantic_map.shape[1]:
                        semantic_map[r, c] = 3
        
        # 生成调色板
        color_pal = [int(x * 255.) for x in color_palette]
        
        # 创建调色板模式的图像
        sem_map_vis = Image.new("P", (semantic_map.shape[1], semantic_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(semantic_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        
        # 翻转（使坐标系与Habitat一致）
        sem_map_vis = np.flipud(sem_map_vis)
        
        # 保存语义地图
        plt.imsave(f"{self.output_dir}/semantic_map.png", sem_map_vis)
        print(f"[INFO] 保存彩色语义地图: {self.output_dir}/semantic_map.png")
        
        # 创建带图例的完整可视化
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(sem_map_vis)
        ax.set_title(f'Semantic Map - {len(self.detected_classes)} classes detected', fontsize=14)
        ax.axis('off')
        
        # 添加图例
        legend_text = "🎨 颜色图例:\n"
        legend_text += "  • 黑色: 未探索\n"
        legend_text += "  • 深灰: 障碍物\n"
        legend_text += "  • 浅灰: 已探索空区域\n"
        legend_text += "  • 蓝色: 智能体位置\n"
        legend_text += "\n🏷️ 检测到的物体:\n"
        for i, cls in enumerate(self.detected_classes):
            legend_text += f"  {i+1}. {cls}\n"
        
        ax.text(1.02, 0.5, legend_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/semantic_map_with_legend.png", dpi=150, bbox_inches='tight')
        print(f"[INFO] 保存语义地图（带图例）: {self.output_dir}/semantic_map_with_legend.png")
        plt.close()
    
    def _save_all_map_types(self, maps_history):
        """保存所有类型的地图（障碍物、探索区域、地板、可穿越、边界、语义）"""
        print("\n[INFO] 保存所有地图类型...")
        
        final_map = maps_history[-1]['full_map'][0]
        final_floor = maps_history[-1]['floor']
        final_traversible = maps_history[-1]['traversible']
        final_frontiers = maps_history[-1]['frontiers']
        
        # 创建子目录
        os.makedirs(f"{self.output_dir}/map_types", exist_ok=True)
        
        # 1. 障碍物地图
        plt.figure(figsize=(8, 8))
        plt.imshow(final_map[0], cmap='Reds', vmin=0, vmax=1)
        plt.title('Obstacle Map (Channel 0)')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(f"{self.output_dir}/map_types/obstacles.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. 探索区域地图
        plt.figure(figsize=(8, 8))
        plt.imshow(final_map[1], cmap='Blues', vmin=0, vmax=1)
        plt.title('Explored Area Map (Channel 1)')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(f"{self.output_dir}/map_types/explored.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. 地板地图（后处理）
        plt.figure(figsize=(8, 8))
        plt.imshow(final_floor, cmap='Greens')
        plt.title('Floor Map (Processed)')
        plt.axis('off')
        plt.savefig(f"{self.output_dir}/map_types/floor.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. 可穿越区域
        plt.figure(figsize=(8, 8))
        plt.imshow(final_traversible, cmap='YlGn')
        plt.title('Traversible Area')
        plt.axis('off')
        plt.savefig(f"{self.output_dir}/map_types/traversible.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. 边界地图（探索边缘）
        plt.figure(figsize=(8, 8))
        plt.imshow(final_frontiers, cmap='Oranges')
        plt.title('Frontiers (Exploration Boundaries)')
        plt.axis('off')
        plt.savefig(f"{self.output_dir}/map_types/frontiers.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 6. 各个物体类别的地图
        for i, cls_name in enumerate(self.detected_classes):
            channel_idx = 4 + i  # 语义通道从4开始
            if channel_idx < final_map.shape[0]:
                plt.figure(figsize=(8, 8))
                plt.imshow(final_map[channel_idx], cmap='viridis')
                plt.title(f'Semantic Channel: {cls_name}')
                plt.colorbar()
                plt.axis('off')
                safe_name = cls_name.replace(' ', '_').replace('/', '_')
                plt.savefig(f"{self.output_dir}/map_types/semantic_{i:02d}_{safe_name}.png", 
                           dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"[INFO] 所有地图类型已保存到: {self.output_dir}/map_types/")
    
    def _visualize_comprehensive_summary(self, maps_history):
        """生成综合总结可视化"""
        final_map = maps_history[-1]['full_map'][0]
        final_pose = maps_history[-1]['full_pose'][0]
        final_floor = maps_history[-1]['floor']
        final_traversible = maps_history[-1]['traversible']
        final_frontiers = maps_history[-1]['frontiers']
        
        if hasattr(self.mapping_module, 'lmb'):
            lmb = self.mapping_module.lmb[0].astype(int)
        else:
            lmb = None
    
    def look_around_and_map(self):
        """环视 360° 并建图 - 完全按照 ZS_Evaluator_mp._look_around() 的逻辑"""
        print("\n[STEP 3] 环视 360° 建图...")
        
        # 初始化地图
        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        print(f"[INFO] 地图初始化完成，检测类别数: {len(self.detected_classes)}")
        print(f"[INFO] 全局地图尺寸: {self.mapping_module.full_w} × {self.mapping_module.full_h} pixels")
        print(f"[INFO] 局部地图尺寸: {self.mapping_module.local_w} × {self.mapping_module.local_h} pixels")
        
        maps_history = []
        rgb_history = []
        depth_history = []
        
        # 累积的地板和可穿越区域
        accumulated_floor = np.zeros(self.map_shape)
        accumulated_traversible = np.zeros(self.map_shape)
        
        # ========== 完全按照原始程序的流程 ==========
        for step in range(12):
            print(f"\n[STEP 3.{step+1}] 左转 30° (总计 {(step+1)*30}°)...")
            
            # ===== 1. 执行左转动作 =====
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            if dones[0]:
                print("[WARNING] Episode 提前结束")
                break
            
            # ===== 2. 预处理观察（使用原来的方法）=====
            state, rgb, depth, annotated_rgb = self.preprocess_observation(obs[0])
            
            # 保存 RGB 和 Depth
            rgb_history.append(rgb)
            depth_history.append(depth)
            
            # ===== 打印深度信息 =====
            depth_values = depth[:,:,0]
            print(f"[DEBUG] 深度图统计（预处理后）:")
            print(f"  • 形状: {depth_values.shape}")
            print(f"  • Min: {depth_values.min():.2f}cm, Max: {depth_values.max():.2f}cm")
            print(f"  • Mean: {depth_values.mean():.2f}cm, Median: {np.median(depth_values):.2f}cm")
            print(f"  • 有效像素（< 1000cm）: {np.count_nonzero(depth_values < 1000)} / {depth_values.size}")
            print(f"  • 直方图: 50-100cm: {np.sum((depth_values >= 50) & (depth_values < 100))}, "
                  f"100-200cm: {np.sum((depth_values >= 100) & (depth_values < 200))}, "
                  f"200-400cm: {np.sum((depth_values >= 200) & (depth_values < 400))}, "
                  f"400-550cm: {np.sum((depth_values >= 400) & (depth_values < 550))}, "
                  f">1000cm(invalid): {np.sum(depth_values >= 1000)}")
            
            # 保存可视化
            plt.imsave(f"{self.output_dir}/rgb/step_{step:02d}.png", rgb)
            plt.imsave(f"{self.output_dir}/rgb/step_{step:02d}_annotated.png", annotated_rgb)
            plt.imsave(f"{self.output_dir}/depth/step_{step:02d}.png", depth[:,:,0], cmap='viridis')
            
            # ===== 3. 批处理观察 =====
            # 关键：padding 到最大通道数（模拟动态通道数）
            batch_obs = torch.from_numpy(state[None, ...]).float().to(self.device)
            
            # ===== 打印输入统计 =====
            print(f"[DEBUG] 输入 batch_obs:")
            print(f"  • 形状: {batch_obs.shape}")
            print(f"  • RGB通道 (0-2) - Min: {batch_obs[0,0:3].min():.2f}, Max: {batch_obs[0,0:3].max():.2f}")
            print(f"  • Depth通道 (3) - Min: {batch_obs[0,3].min():.2f}, Max: {batch_obs[0,3].max():.2f}, NonZero: {torch.count_nonzero(batch_obs[0,3])}")
            if batch_obs.shape[1] > 4:
                print(f"  • Semantic通道 (4+) - 通道数: {batch_obs.shape[1] - 4}, NonZero总计: {torch.count_nonzero(batch_obs[0,4:])}")
            
            # ===== 4. 获取位姿变化 =====
            poses = torch.from_numpy(np.array([obs[0]['sensor_pose']])).float().to(self.device)
            
            # ===== 5. 调用 mapping_module 前向传播（核心建图）=====
            self.mapping_module(batch_obs, poses)
            
            # ===== 6. 更新全局地图 =====
            full_map, full_pose, one_step_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.episode_id)
            
            # ===== 7. 清空单步地图 =====
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            
            # ===== 8. 处理导航地图 =====
            traversible, floor, frontiers = self._process_map(step, full_map[0])
            accumulated_floor = np.logical_or(accumulated_floor, floor)
            accumulated_traversible = traversible
            
            # ===== 9. 打印调试信息 =====
            print(f"[INFO] 位姿: [{full_pose[0,0]:.2f}, {full_pose[0,1]:.2f}, {full_pose[0,2]:.2f}]")
            print(f"[INFO] 地图形状: {full_map.shape}")
            print(f"[INFO] 地板像素: {np.sum(floor)}, 可穿越像素: {np.sum(traversible)}")
            print(f"[DEBUG] 障碍物通道 - Max: {full_map[0,0].max():.4f}, Sum: {full_map[0,0].sum():.1f}, NonZero: {np.count_nonzero(full_map[0,0])}")
            print(f"[DEBUG] 探索区域通道 - Max: {full_map[0,1].max():.4f}, Sum: {full_map[0,1].sum():.1f}, NonZero: {np.count_nonzero(full_map[0,1])}")
            
            # ===== 10. 保存地图（包含后处理结果）=====
            maps_history.append({
                'full_map': full_map.copy(),
                'full_pose': full_pose.copy(),
                'one_step_map': one_step_map.copy(),
                'floor': floor.copy(),
                'traversible': traversible.copy(),
                'frontiers': frontiers.copy(),
            })
        
        return maps_history, rgb_history, depth_history
    
    def save_maps(self, maps_history):
        """保存地图数据和可视化"""
        print("\n[STEP 4] 保存地图...")
        
        # 保存原始数据
        np.save(f"{self.output_dir}/maps_history.npy", maps_history)
        print(f"[INFO] 保存地图历史: {self.output_dir}/maps_history.npy")
        
        # 保存最终地图
        final_map = maps_history[-1]['full_map'][0]  # (N+4, 480, 480) - 全局地图
        final_pose = maps_history[-1]['full_pose'][0]  # (3,) - 全局坐标 [x, y, θ]
        final_floor = maps_history[-1]['floor']  # (480, 480) - 处理后的地板
        final_traversible = maps_history[-1]['traversible']  # (480, 480) - 可穿越区域
        
        # 获取局部地图信息（如果mapping_module有的话）
        if hasattr(self.mapping_module, 'lmb'):
            lmb = self.mapping_module.lmb[0].astype(int)  # [gx1, gx2, gy1, gy2]
            print(f"[INFO] 局部地图边界: x=[{lmb[0]}, {lmb[1]}], y=[{lmb[2]}, {lmb[3]}]")
        else:
            lmb = None
        
        # ========== 新增：生成彩色语义地图（与原程序一致）==========
        self._visualize_semantic_map(final_map, final_pose, lmb)
        
        # ========== 保存所有地图类型 ==========
        self._save_all_map_types(maps_history)
        
        # ========== 原有的综合可视化 ==========
        self._visualize_comprehensive_summary(maps_history)
        
        # 保存动画 (地图演化过程)
        self._save_evolution_animation(maps_history)
        
        # 打印统计信息
        self._print_final_statistics(maps_history)
    
    def _save_evolution_animation(self, maps_history):
        """保存地图演化动画"""
        print("[INFO] 生成地图演化动画...")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for i, map_data in enumerate(maps_history):
            ax.clear()
            m = map_data['full_map'][0]
            floor = map_data['floor']
            map_h, map_w = m.shape[1], m.shape[2]
            
            composite = np.zeros((map_h, map_w, 3))
            composite[:, :, 0] = m[0]  # 障碍物
            composite[:, :, 1] = floor  # 地板（处理后）
            composite[:, :, 2] = m[2]  # 当前位置
            
            ax.imshow(composite)
            ax.set_title(f'Step {i+1}/12 - Rotation {(i+1)*30}° - Floor pixels: {np.sum(floor)}')
            ax.axis('off')
            
            plt.savefig(f"{self.output_dir}/maps/map_step_{i:02d}.png", dpi=100)
        
        plt.close('all')
        print(f"[INFO] 保存地图演化: {self.output_dir}/maps/map_step_*.png")
    
    def _print_final_statistics(self, maps_history):
        """打印最终统计信息"""
        final_map = maps_history[-1]['full_map'][0]
        final_pose = maps_history[-1]['full_pose'][0]
        final_floor = maps_history[-1]['floor']
        final_traversible = maps_history[-1]['traversible']
        
        if hasattr(self.mapping_module, 'lmb'):
            lmb = self.mapping_module.lmb[0].astype(int)
        else:
            lmb = None
        
        pose_c = int(final_pose[0] * 100 / self.resolution)
        pose_r = int(final_pose[1] * 100 / self.resolution)
        
        print("\n" + "="*60)
        print("📊 建图统计信息")
        print("="*60)
        print(f"Episode ID: {self.episode_id}")
        print(f"检测到的类别数: {len(self.detected_classes)}")
        print(f"类别列表: {list(self.detected_classes)}")
        print()
        print("🔍 原始地图数据范围:")
        print(f"  • 障碍物通道 - Min: {final_map[0].min():.4f}, Max: {final_map[0].max():.4f}, Sum: {final_map[0].sum():.1f}")
        print(f"  • 探索区域通道 - Min: {final_map[1].min():.4f}, Max: {final_map[1].max():.4f}, Sum: {final_map[1].sum():.1f}")
        print(f"  • 当前位置通道 - Min: {final_map[2].min():.4f}, Max: {final_map[2].max():.4f}, Sum: {final_map[2].sum():.1f}")
        print()
        print("📍 坐标系统:")
        print(f"  • 全局地图尺寸: {final_map.shape[1:]} pixels = ({final_map.shape[1]*self.resolution/100:.1f}m × {final_map.shape[2]*self.resolution/100:.1f}m)")
        if lmb is not None:
            local_w = lmb[1] - lmb[0]
            local_h = lmb[3] - lmb[2]
            print(f"  • 局部地图尺寸: ({local_w} × {local_h}) pixels = ({local_w*self.resolution/100:.1f}m × {local_h*self.resolution/100:.1f}m)")
            print(f"  • 局部地图边界: x=[{lmb[0]}, {lmb[1]}], y=[{lmb[2]}, {lmb[3]}]")
        print(f"  • 分辨率: {self.resolution} cm/pixel")
        print()
        print("📌 最终位姿 (全局坐标):")
        print(f"  • x = {final_pose[0]:.2f} m (像素: {pose_c})")
        print(f"  • y = {final_pose[1]:.2f} m (像素: {pose_r})")
        print(f"  • θ = {final_pose[2]:.2f} rad ({np.degrees(final_pose[2]):.1f}°)")
        print()
        print("🗺️ 地图覆盖:")
        total_pixels = final_map.shape[1] * final_map.shape[2]
        print(f"  • 已探索像素数: {np.sum(final_map[1] > 0):,} ({np.sum(final_map[1] > 0) / total_pixels * 100:.1f}%)")
        print(f"  • 障碍物像素数: {np.sum(final_map[0] > 0):,}")
        print(f"  • 地板像素数（处理后）: {np.sum(final_floor > 0):,}")
        print(f"  • 可穿越像素数: {np.sum(final_traversible > 0):,}")
        print("="*60)
    
    def _visualize_comprehensive_summary(self, maps_history):
        """生成综合总结可视化"""
        final_map = maps_history[-1]['full_map'][0]
        final_pose = maps_history[-1]['full_pose'][0]
        final_floor = maps_history[-1]['floor']
        final_traversible = maps_history[-1]['traversible']
        
        if hasattr(self.mapping_module, 'lmb'):
            lmb = self.mapping_module.lmb[0].astype(int)
        else:
            lmb = None
        
        pose_r = int(final_pose[1] * 100 / self.resolution)
        pose_c = int(final_pose[0] * 100 / self.resolution)
        
        # 创建综合可视化
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # ===== 第一行：原始通道 =====
        
        # 通道 0: 障碍物
        axes[0, 0].imshow(final_map[0], cmap='gray', vmin=0, vmax=1)
        axes[0, 0].set_title(f'Channel 0: Obstacles (Full Map)\nMax: {final_map[0].max():.3f}')
        axes[0, 0].axis('off')
        
        # 通道 1: 已探索区域
        axes[0, 1].imshow(final_map[1], cmap='Blues', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Channel 1: Explored Area (Full Map)\nMax: {final_map[1].max():.3f}, Sum: {final_map[1].sum():.0f}')
        axes[0, 1].axis('off')
        
        # 通道 2: 当前位置
        axes[0, 2].imshow(final_map[2], cmap='Reds', vmin=0, vmax=1)
        axes[0, 2].set_title(f'Channel 2: Current Location (Full Map)\nMax: {final_map[2].max():.3f}')
        pose_r = int(final_pose[1] * 100 / self.resolution)  # y -> row
        pose_c = int(final_pose[0] * 100 / self.resolution)  # x -> col
        axes[0, 2].plot(pose_c, pose_r, 'r*', markersize=20)
        axes[0, 2].text(pose_c, pose_r-20, f'({final_pose[0]:.1f}m, {final_pose[1]:.1f}m)', 
                       color='red', fontsize=8, ha='center')
        axes[0, 2].axis('off')
        
        # ===== 第二行：后处理结果 =====
        
        # 处理后的地板 (关键！)
        axes[1, 0].imshow(final_floor, cmap='YlGn')
        axes[1, 0].set_title('Processed Floor (after morphology)')
        axes[1, 0].axis('off')
        
        # 可穿越区域
        axes[1, 1].imshow(final_traversible, cmap='Greens')
        axes[1, 1].set_title('Traversible Area')
        axes[1, 1].axis('off')
        
        # 边界区域
        if 'frontiers' in maps_history[-1]:
            axes[1, 2].imshow(maps_history[-1]['frontiers'], cmap='Oranges')
            axes[1, 2].set_title('Frontiers (Exploration Boundary)')
        else:
            axes[1, 2].axis('off')
        axes[1, 2].axis('off')
        
        # ===== 第三行：局部地图与综合视图 =====
        
        # 局部地图区域（如果有）
        if lmb is not None:
            local_region = np.zeros_like(final_map[0])
            local_region[lmb[0]:lmb[1], lmb[2]:lmb[3]] = 1
            axes[2, 0].imshow(local_region, cmap='Purples', alpha=0.3)
            axes[2, 0].imshow(final_map[0], cmap='gray', alpha=0.7)
            axes[2, 0].plot(pose_c, pose_r, 'r*', markersize=20)
            # 画出局部地图边界
            rect = plt.Rectangle((lmb[2], lmb[0]), lmb[3]-lmb[2], lmb[1]-lmb[0], 
                                fill=False, edgecolor='red', linewidth=2)
            axes[2, 0].add_patch(rect)
            axes[2, 0].set_title('Local Map Region (240×240) in Full Map')
            axes[2, 0].axis('off')
        else:
            axes[2, 0].axis('off')
        
        # 综合地图 (障碍物 + 地板 + 位置)
        map_h, map_w = final_map.shape[1], final_map.shape[2]
        composite = np.zeros((map_h, map_w, 3))
        composite[:, :, 0] = final_map[0]  # 红色：障碍物
        composite[:, :, 1] = final_floor  # 绿色：地板
        composite[:, :, 2] = final_map[2]  # 蓝色：当前位置
        axes[2, 1].imshow(composite)
        axes[2, 1].plot(pose_c, pose_r, 'w*', markersize=20)
        axes[2, 1].set_title('Composite Map (R:Obstacle, G:Floor, B:Pose)')
        axes[2, 1].axis('off')
        
        # 显示坐标系统信息
        map_size_cm = self.config.MAP.MAP_SIZE_CM
        info_text = f"""
📍 坐标系统:
• 全局地图: {final_map.shape[1]}×{final_map.shape[2]} ({map_size_cm/100:.1f}m×{map_size_cm/100:.1f}m)
• 局部地图: {lmb[1]-lmb[0] if lmb is not None else 'N/A'}×{lmb[3]-lmb[2] if lmb is not None else 'N/A'} pixels
• 分辨率: {self.resolution} cm/pixel

📌 当前位姿 (全局坐标):
• x = {final_pose[0]:.2f} m
• y = {final_pose[1]:.2f} m
• θ = {final_pose[2]:.2f} rad

🗺️ 统计信息:
• 探索: {np.sum(final_map[1] > 0)} pixels
• 障碍: {np.sum(final_map[0] > 0)} pixels
• 地板: {np.sum(final_floor > 0)} pixels
• 可穿越: {np.sum(final_traversible > 0)} pixels
        """
        axes[2, 2].text(0.1, 0.5, info_text.strip(), 
                       fontsize=10, family='monospace',
                       verticalalignment='center')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/final_map.png", dpi=150)
        print(f"[INFO] 保存综合地图可视化: {self.output_dir}/final_map.png")
        plt.close()
    
    def run(self):
        """运行完整测试"""
        try:
            # 1. 初始化环境
            obs = self.initialize_environment()
            
            # 2. 初始化模块
            self.initialize_modules()
            
            # 3. 环视建图
            maps_history, rgb_history, depth_history = self.look_around_and_map()
            
            # 4. 保存结果
            self.save_maps(maps_history)
            
            print("\n[SUCCESS] 测试完成！")
            print(f"[INFO] 查看结果: {self.output_dir}/")
            
        except Exception as e:
            print(f"\n[ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理
            if hasattr(self, 'envs'):
                self.envs.close()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="最小化建图验证程序")
    parser.add_argument(
        "--exp-config",
        type=str,
        default="vlnce_baselines/config/exp1.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="修改配置选项"
    )
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args.exp_config, args.opts)
    
    # 修改配置（单环境、单 GPU）
    config.defrost()
    config.NUM_ENVIRONMENTS = 1
    config.TORCH_GPU_ID = 0
    config.SIMULATOR_GPU_IDS = [0]
    config.MAP.VISUALIZE = False
    config.MAP.PRINT_IMAGES = False
    config.freeze()
    
    # 运行测试
    tester = MinimalMappingTest(config)
    tester.run()


if __name__ == "__main__":
    main()
