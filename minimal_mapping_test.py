"""
最小化建图验证程序

功能：
1. 初始化一个 Habitat 环境
2. 环视 360° (12 步 × 30°)
3. 保存全局地图和局部地图的演化过程

用法：
python minimal_mapping_test.py --exp-config vlnce_baselines/config/exp1.yaml
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from habitat import Config 
from habitat import make_dataset, Env
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
    
    def __init__(self, config: Config, episode_index: int = None):
        self.config = config
        self.device = get_device(config.TORCH_GPU_ID)
        torch.cuda.set_device(self.device)
        
        # 地图配置
        self.resolution = config.MAP.MAP_RESOLUTION
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution)
        
        # 创建输出目录（稍后根据 episode_id 创建子目录）
        self.output_dir = "data/minimal_mapping_test"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Episode 索引（替代 instruction_id）
        self.episode_index = episode_index
        self.env = None  # 单个环境（不是 VectorEnv）
        self.episode_output_dir = None  # 根据 episode_id 创建的子目录
        
        print(f"[INFO] 输出目录: {self.output_dir}")
        print(f"[INFO] 地图尺寸: {self.map_shape}")
        print(f"[INFO] 地图分辨率: {self.resolution} cm/pixel")
        
    def initialize_environment(self):
        """初始化环境（通过 EPISODES_ALLOWED 配置已经过滤）"""
        print("\n[STEP 1] 初始化 Habitat 环境...")
        
        # 加载数据集（已通过 config.TASK_CONFIG.DATASET.EPISODES_ALLOWED 过滤）
        print("Loading dataset...")
        dataset = make_dataset(
            id_dataset=self.config.TASK_CONFIG.DATASET.TYPE,
            config=self.config.TASK_CONFIG.DATASET
        )
        print(f"✓ Dataset loaded ({len(dataset.episodes)} episodes)")
        
        # 调试信息
        if len(dataset.episodes) > 0:
            print(f"[DEBUG] 数据集类型: {self.config.TASK_CONFIG.DATASET.TYPE}")
            print(f"[DEBUG] Split: {self.config.TASK_CONFIG.DATASET.SPLIT}")
            print(f"[DEBUG] Episode ID: {dataset.episodes[0].episode_id}")
        
        # 初始化环境
        try:
            self.env = Env(self.config.TASK_CONFIG, dataset)
            print(f"✓ 环境初始化完成")
        except Exception as e:
            print(f"✗ 环境初始化失败: {e}")
            raise
        
        # 重置环境获取初始观察
        obs = self.env.reset()
        
        # 获取 episode 信息
        self.episode_id = self.env.current_episode.episode_id
        self.scene_id = self.env.current_episode.scene_id.split('/')[-1].split('.')[0]
        
        # 获取 instruction
        if hasattr(obs, 'get') and 'instruction' in obs:
            if isinstance(obs['instruction'], dict) and 'text' in obs['instruction']:
                self.instruction_text = obs['instruction']['text']
            else:
                self.instruction_text = str(obs['instruction'])
        elif hasattr(self.env.current_episode, 'instruction'):
            if hasattr(self.env.current_episode.instruction, 'instruction_text'):
                self.instruction_text = self.env.current_episode.instruction.instruction_text
            elif hasattr(self.env.current_episode.instruction, 'text'):
                self.instruction_text = self.env.current_episode.instruction.text
            else:
                self.instruction_text = str(self.env.current_episode.instruction)
        else:
            self.instruction_text = "No instruction available"
        
        print(f"[INFO] Episode ID: {self.episode_id}")
        print(f"[INFO] 场景: {self.scene_id}")
        print(f"[INFO] Instruction: {self.instruction_text[:100]}..." if len(self.instruction_text) > 100 else f"[INFO] Instruction: {self.instruction_text}")
        
        # 根据 episode_id 创建输出目录
        self.episode_output_dir = os.path.join(self.output_dir, f"episode_{self.episode_id}")
        os.makedirs(self.episode_output_dir, exist_ok=True)
        os.makedirs(f"{self.episode_output_dir}/rgb", exist_ok=True)
        os.makedirs(f"{self.episode_output_dir}/depth", exist_ok=True)
        os.makedirs(f"{self.episode_output_dir}/semantic", exist_ok=True)
        os.makedirs(f"{self.episode_output_dir}/maps", exist_ok=True)
        print(f"[INFO] Episode 输出目录: {self.episode_output_dir}")
        
        return obs
    
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
    
    def _save_observation_images(self, step: int, rgb: np.ndarray, depth: np.ndarray, annotated_rgb: np.ndarray):
        """保存每一步的观察图像
        
        Args:
            step: 当前步数
            rgb: RGB 图像 (H, W, 3)
            depth: Depth 图像 (H, W, 1)
            annotated_rgb: 标注后的 RGB 图像
        """
        # 保存原始 RGB
        rgb_img = Image.fromarray(rgb)
        rgb_img.save(f"{self.episode_output_dir}/rgb/step_{step:02d}.png")
        
        # 保存标注后的 RGB（语义分割可视化）
        if annotated_rgb is not None:
            annotated_img = Image.fromarray(annotated_rgb)
            annotated_img.save(f"{self.episode_output_dir}/semantic/step_{step:02d}_annotated.png")
        
        # 保存 Depth（归一化到 0-255 用于可视化）
        depth_normalized = depth[:, :, 0]  # 去掉通道维度
        depth_normalized = np.clip(depth_normalized, 50, 550)  # 限制到 50-550cm
        depth_normalized = ((depth_normalized - 50) / 500 * 255).astype(np.uint8)
        depth_img = Image.fromarray(depth_normalized)
        depth_img.save(f"{self.episode_output_dir}/depth/step_{step:02d}.png")
        
        # 保存带颜色映射的深度图
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_colored_img = Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
        depth_colored_img.save(f"{self.episode_output_dir}/depth/step_{step:02d}_colored.png")
    
    def look_around_and_map(self):
        """环视 360° 并建图 - 12次旋转，每次30°"""
        print("\n[STEP 3] 环视 360° 建图 (12 × 30°)...")
        
        # 初始化地图
        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        print(f"[INFO] 地图初始化完成，检测类别数: {len(self.detected_classes)}")
        print(f"[INFO] 全局地图尺寸: {self.mapping_module.full_w} × {self.mapping_module.full_h} pixels")
        print(f"[INFO] 局部地图尺寸: {self.mapping_module.local_w} × {self.mapping_module.local_h} pixels")
        
        maps_history = []
        
        # ========== 12次旋转，每次30° ==========
        for step in range(12):
            print(f"\n[STEP 3.{step+1}] 左转 30° (总计 {(step+1)*30}°)...")
            
            # ===== 1. 执行左转动作 =====
            actions = {"action": HabitatSimActions.TURN_LEFT}
            obs = self.env.step(actions)
            
            # ===== 2. 预处理观察 =====
            state, rgb, depth, annotated_rgb = self.preprocess_observation(obs)
            
            # ===== 2.1 保存 RGB 和 Depth 图像 =====
            self._save_observation_images(step, rgb, depth, annotated_rgb)
            
            # ===== 3. 批处理观察 =====
            batch_obs = torch.from_numpy(state[None, ...]).float().to(self.device)
            
            # ===== 4. 获取位姿 =====
            poses = torch.from_numpy(np.array([obs['sensor_pose']])).float().to(self.device)
            
            # ===== 5. 映射模块前向传播 =====
            self.mapping_module(batch_obs, poses)
            
            # ===== 6. 更新全局地图 =====
            full_map, full_pose, one_step_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.episode_id)
            
            # ===== 7. 清空单步地图 =====
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            
            # ===== 8. 处理导航地图 =====
            traversible, floor, frontiers = self._process_map(step, full_map[0])
            
            # ===== 9. 打印调试信息 =====
            print(f"[INFO] 位姿: [{full_pose[0,0]:.2f}, {full_pose[0,1]:.2f}, {full_pose[0,2]:.2f}]")
            print(f"[DEBUG] 障碍物: {np.count_nonzero(full_map[0,0])}, 探索区域: {np.count_nonzero(full_map[0,1])}")
            
            # ===== 10. 保存地图 =====
            maps_history.append({
                'full_map': full_map.copy(),
                'full_pose': full_pose.copy(),
                'floor': floor.copy(),
                'traversible': traversible.copy(),
            })
        
        return maps_history
    
    def save_maps(self, maps_history):
        """保存地图数据和可视化（简化版）"""
        print("\n[STEP 4] 保存地图...")
        
        # 保存原始数据
        np.save(f"{self.episode_output_dir}/maps_history.npy", maps_history)
        print(f"[INFO] 保存地图历史: {self.episode_output_dir}/maps_history.npy")
        
        final_map = maps_history[-1]['full_map'][0]
        final_pose = maps_history[-1]['full_pose'][0]
        final_floor = maps_history[-1]['floor']
        
        # 获取局部地图边界
        if hasattr(self.mapping_module, 'lmb'):
            lmb = self.mapping_module.lmb[0].astype(int)
        else:
            lmb = None
        
        # 生成地图演化动画（全局地图 + 局部地图）
        self._save_map_evolution(maps_history)
        
        # 打印统计信息
        self._print_final_statistics(maps_history)
    
    def _draw_arrow(self, img, center, angle_rad, length=15, color=(0, 0, 139), thickness=3):
        """在地图上画箭头表示智能体朝向
        
        Args:
            img: numpy array (H, W, 3)
            center: (x, y) 中心位置
            angle_rad: 朝向角度（弧度）
            length: 箭头长度
            color: RGB颜色 (深蓝色 默认)
            thickness: 线条粗细
        """
        cx, cy = center
        # 计算箭头端点
        dx = int(length * np.cos(angle_rad))
        dy = int(length * np.sin(angle_rad))
        end_x, end_y = cx + dx, cy - dy  # 注意y轴翻转
        
        # 画箭头主干
        cv2.arrowedLine(img, (int(cx), int(cy)), (end_x, end_y), 
                       color, thickness, tipLength=0.4)
        return img
    
    def _create_colored_map(self, obstacles, floor, pose, map_title="Map"):
        """创建自定义配色的地图
        
        配色方案:
        - 白色(255,255,255): 未探索区域
        - 浅蓝色(173,216,230): 地面 (LightBlue)
        - 深红色(139,0,0): 障碍物 (DarkRed)
        - 深蓝色(0,0,139): 智能体箭头 (DarkBlue)
        """
        h, w = obstacles.shape
        # 初始化为白色背景
        colored_map = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # 地面 - 浅蓝色
        floor_mask = floor > 0
        colored_map[floor_mask] = [173, 216, 230]  # LightBlue (RGB)
        
        # 障碍物 - 深红色
        obstacle_mask = obstacles > 0.1
        colored_map[obstacle_mask] = [139, 0, 0]  # DarkRed (RGB)
        
        # 画智能体箭头
        pose_x = int(pose[0] * 100 / self.resolution)  # x -> col
        pose_y = int(pose[1] * 100 / self.resolution)  # y -> row
        angle = pose[2]  # 朝向角度（弧度）
        
        colored_map = self._draw_arrow(colored_map, (pose_x, pose_y), angle, 
                                       length=20, color=(0, 0, 139), thickness=4)
        
        return colored_map
    
    def _save_map_evolution(self, maps_history):
        """保存地图演化过程（全局地图 + 局部地图）"""
        print(f"[INFO] 生成地图演化动画 ({len(maps_history)}帧)...")
        
        # 创建子目录
        os.makedirs(f"{self.episode_output_dir}/maps/global", exist_ok=True)
        os.makedirs(f"{self.episode_output_dir}/maps/local", exist_ok=True)
        os.makedirs(f"{self.episode_output_dir}/maps/combined", exist_ok=True)
        
        for i, map_data in enumerate(maps_history):
            full_map = map_data['full_map'][0]
            full_pose = map_data['full_pose'][0]
            floor = map_data['floor']
            
            # 获取局部地图边界
            if hasattr(self.mapping_module, 'lmb'):
                lmb = self.mapping_module.lmb[0].astype(int)
            else:
                lmb = None
            
            # ===== 1. 生成全局地图（单独） =====
            global_colored = self._create_colored_map(
                full_map[0], floor, full_pose, "Global Map"
            )
            
            # 保存全局地图（带红框标注）
            fig_global = plt.figure(figsize=(10, 10))
            ax_global = fig_global.add_subplot(111)
            ax_global.imshow(global_colored)
            ax_global.set_title(f'Global Map - Step {i+1}/{len(maps_history)} (Rotation {(i+1)*30}°)', fontsize=14)
            ax_global.axis('off')
            
            if lmb is not None:
                from matplotlib.patches import Rectangle
                rect = Rectangle((lmb[2], lmb[0]), lmb[3]-lmb[2], lmb[1]-lmb[0],
                               fill=False, edgecolor='red', linewidth=2, linestyle='--')
                ax_global.add_patch(rect)
            
            plt.tight_layout()
            plt.savefig(f"{self.episode_output_dir}/maps/global/global_step_{i:02d}.png", dpi=100, bbox_inches='tight')
            plt.close(fig_global)
            
            # ===== 2. 生成局部地图（单独） =====
            if lmb is not None:
                local_obstacles = full_map[0, lmb[0]:lmb[1], lmb[2]:lmb[3]]
                local_floor = floor[lmb[0]:lmb[1], lmb[2]:lmb[3]]
                
                # 计算局部坐标系中的位姿
                local_pose = full_pose.copy()
                local_pose[0] = full_pose[0] - lmb[2] * self.resolution / 100
                local_pose[1] = full_pose[1] - lmb[0] * self.resolution / 100
                
                local_colored = self._create_colored_map(
                    local_obstacles, local_floor, local_pose, "Local Map"
                )
                
                fig_local = plt.figure(figsize=(8, 8))
                ax_local = fig_local.add_subplot(111)
                ax_local.imshow(local_colored)
                ax_local.set_title(f'Local Map (12m×12m) - Step {i+1}/{len(maps_history)}', fontsize=14)
                ax_local.axis('off')
                plt.tight_layout()
                plt.savefig(f"{self.episode_output_dir}/maps/local/local_step_{i:02d}.png", dpi=100, bbox_inches='tight')
                plt.close(fig_local)
            
            # ===== 3. 生成组合图（全局+局部） =====
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            ax1.imshow(global_colored)
            ax1.set_title(f'Global Map - Step {i+1}/{len(maps_history)} (Rotation {(i+1)*30}°)', fontsize=14)
            ax1.axis('off')
            
            if lmb is not None:
                rect = Rectangle((lmb[2], lmb[0]), lmb[3]-lmb[2], lmb[1]-lmb[0],
                               fill=False, edgecolor='red', linewidth=2, linestyle='--')
                ax1.add_patch(rect)
                
                ax2.imshow(local_colored)
                ax2.set_title(f'Local Map (12m×12m) - Step {i+1}/{len(maps_history)}', fontsize=14)
            else:
                ax2.text(0.5, 0.5, 'No Local Map', ha='center', va='center')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{self.episode_output_dir}/maps/combined/combined_step_{i:02d}.png", dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"[INFO] 保存地图演化:")
        print(f"  • 全局地图: {self.episode_output_dir}/maps/global/global_step_00~{len(maps_history)-1:02d}.png")
        print(f"  • 局部地图: {self.episode_output_dir}/maps/local/local_step_00~{len(maps_history)-1:02d}.png")
        print(f"  • 组合图: {self.episode_output_dir}/maps/combined/combined_step_00~{len(maps_history)-1:02d}.png")
    
    def _print_final_statistics(self, maps_history):
        """打印最终统计信息"""
        final_map = maps_history[-1]['full_map'][0]
        final_pose = maps_history[-1]['full_pose'][0]
        final_floor = maps_history[-1]['floor']
        
        print("\n" + "="*60)
        print("📊 建图统计信息")
        print("="*60)
        print(f"Episode ID: {self.episode_id}")
        print(f"检测到的类别数: {len(self.detected_classes)}")
        print(f"类别列表: {list(self.detected_classes)}")
        print()
        print("�️ 地图覆盖:")
        total_pixels = final_map.shape[1] * final_map.shape[2]
        print(f"  • 全局地图: {final_map.shape[1]}×{final_map.shape[2]} pixels ({final_map.shape[1]*self.resolution/100:.1f}m × {final_map.shape[2]*self.resolution/100:.1f}m)")
        print(f"  • 已探索: {np.sum(final_map[1] > 0):,} pixels ({np.sum(final_map[1] > 0) / total_pixels * 100:.1f}%)")
        print(f"  • 障碍物: {np.sum(final_map[0] > 0):,} pixels")
        print(f"  • 地板: {np.sum(final_floor > 0):,} pixels")
        print()
        print("📌 最终位姿:")
        print(f"  • x = {final_pose[0]:.2f} m")
        print(f"  • y = {final_pose[1]:.2f} m")
        print(f"  • θ = {final_pose[2]:.2f} rad ({np.degrees(final_pose[2]):.1f}°)")
        print("="*60)
    
    def run(self):
        """运行完整测试"""
        try:
            # 1. 初始化环境
            obs = self.initialize_environment()
            
            # 2. 初始化模块
            self.initialize_modules()
            
            # 3. 环视建图
            maps_history = self.look_around_and_map()
            
            # 4. 保存结果
            self.save_maps(maps_history)
            
            print("\n[SUCCESS] 测试完成！")
            print(f"[INFO] 查看结果: {self.episode_output_dir}/")
            print(f"\n📁 输出目录结构:")
            print(f"  {self.episode_output_dir}/")
            print(f"  ├── rgb/              (RGB 图像，每步一张)")
            print(f"  ├── depth/            (深度图像，灰度 + 彩色)")
            print(f"  ├── semantic/         (语义分割标注)")
            print(f"  ├── maps/")
            print(f"  │   ├── global/       (全局地图演化)")
            print(f"  │   ├── local/        (局部地图演化)")
            print(f"  │   └── combined/     (全局+局部组合)")
            print(f"  └── maps_history.npy  (原始地图数据)")
            
        except Exception as e:
            print(f"\n[ERROR] 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 清理
            if hasattr(self, 'env') and self.env:
                self.env.close()


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
        "--episode-id",
        type=int,
        default=None,
        help="指定 Episode ID (例如: 701, 389, 等)"
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
    
    # 如果指定了 episode_id，覆盖 EPISODES_ALLOWED
    if args.episode_id is not None:
        print(f"\n[INFO] 指定 Episode ID: {args.episode_id}")
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = [args.episode_id]
    else:
        # 使用默认配置（default.py 中已设置为 None，加载所有 episodes）
        print(f"\n[INFO] 使用配置中的 EPISODES_ALLOWED: {config.TASK_CONFIG.DATASET.EPISODES_ALLOWED}")
    
    config.freeze()
    
    # 运行测试（不再需要传递 episode_index）
    tester = MinimalMappingTest(config, episode_index=None)
    tester.run()


if __name__ == "__main__":
    main()
