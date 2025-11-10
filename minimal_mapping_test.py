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
        
        # 输出目录
        self.output_dir = "data/minimal_mapping_test"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.episode_index = episode_index
        self.env = None
        self.episode_output_dir = None
        
        # 箭头配置（可自定义）
        self.agent_icon_path = None  # 设置为图标路径（如 "assets/agent_arrow.png"）使用贴图
        self.use_icon = False  # 是否使用图标（False=三角形箭头，True=图标贴图）
        
    def initialize_environment(self):
        """初始化环境（通过 EPISODES_ALLOWED 配置已经过滤）"""
        print("\n=== 初始化环境 ===")
        
        # 加载数据集
        dataset = make_dataset(
            id_dataset=self.config.TASK_CONFIG.DATASET.TYPE,
            config=self.config.TASK_CONFIG.DATASET
        )
        print(f"✓ 加载 {len(dataset.episodes)} 个 episodes")
        
        # 初始化环境
        self.env = Env(self.config.TASK_CONFIG, dataset)
        obs = self.env.reset()
        
        # 获取 episode 信息
        self.episode_id = self.env.current_episode.episode_id
        self.scene_id = self.env.current_episode.scene_id.split('/')[-1].split('.')[0]
        
        # 获取 instruction
        if hasattr(obs, 'get') and 'instruction' in obs:
            self.instruction_text = obs['instruction'].get('text', str(obs['instruction']))
        elif hasattr(self.env.current_episode, 'instruction'):
            self.instruction_text = getattr(self.env.current_episode.instruction, 'instruction_text', 
                                           getattr(self.env.current_episode.instruction, 'text', 
                                                  str(self.env.current_episode.instruction)))
        else:
            self.instruction_text = "No instruction"
        
        # 创建 episode 输出目录
        self.episode_output_dir = os.path.join(self.output_dir, f"episode_{self.episode_id}")
        for subdir in ['rgb', 'depth', 'semantic', 'maps']:
            os.makedirs(f"{self.episode_output_dir}/{subdir}", exist_ok=True)
        
        print(f"Episode {self.episode_id} | Scene: {self.scene_id}")
        print(f"Instruction: {self.instruction_text[:80]}{'...' if len(self.instruction_text) > 80 else ''}")
        
        return obs
    
    def initialize_modules(self):
        """初始化建图模块"""
        print("\n=== 初始化建图模块 ===")
        
        # 语义分割模块
        self.segment_module = GroundedSAM(self.config, self.device)
        
        # 语义地图模块
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        
        # 检测类别
        self.detected_classes = OrderedSet()
        self.classes = base_classes.copy()
        
        print("✓ GroundedSAM & Semantic_Mapping 初始化完成")
        
    def preprocess_observation(self, obs):
        """预处理观察：语义分割 + 深度预处理"""
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth'][:, :, 0] * 1
        
        # 深度预处理
        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()
        
        depth[depth > 0.99] = 0.
        depth[depth == 0] = 100.0
        depth = 0.5 * 100.0 + depth * 5.0 * 100.0  # 转换到厘米: [50, 550]
        depth = depth[:, :, np.newaxis]
        
        # 语义分割
        masks, labels, annotated_image, detections = \
            self.segment_module.segment(rgb[:,:,::-1], classes=self.classes)
        
        # 处理标签
        class_names = [" ".join(label.split(' ')[:-1]) for label in labels]
        for name in class_names:
            self.detected_classes.add(name)
        
        # 处理掩码
        if masks.shape != (0,):
            from collections import defaultdict
            same_label_indexs = defaultdict(list)
            for idx, name in enumerate(class_names):
                same_label_indexs[name].append(idx)
            
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)
            
            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), 480, 640))
        
        # 合并 RGB + Depth + Semantic
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1)
        sem_masks = final_masks.transpose(1, 2, 0)
        state = np.concatenate((state[:3], state[3:4], sem_masks.transpose(2,0,1)), axis=0)
        
        return state, rgb, depth, annotated_image
    
    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int=3) -> tuple:
        """处理语义地图，提取导航相关信息"""
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        full_map = remove_small_objects(full_map.astype(bool), min_size=64)
        
        obstacles = full_map[0, ...].astype(bool)
        explored_area = full_map[1, ...].astype(bool)
        objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool) if len(not_navigable_index) > 0 else np.zeros_like(obstacles)
        
        footprint = disk(kernel_size)
        obstacles_closed = binary_closing(obstacles, footprint=footprint)
        objects_closed = binary_closing(objects, footprint=footprint)
        navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index]) if len(navigable_index) > 0 else np.zeros_like(obstacles)
        navigable = np.logical_and(navigable, np.logical_not(objects))
        navigable_closed = binary_closing(navigable, footprint=footprint)
        
        untraversible = np.logical_or(objects_closed, obstacles_closed)
        untraversible[navigable_closed == 1] = 0
        untraversible = remove_small_objects(untraversible, min_size=64)
        untraversible = binary_closing(untraversible, footprint=disk(3))
        traversible = np.logical_not(untraversible)

        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = binary_closing(floor, footprint=footprint)
        traversible = np.logical_or(floor, traversible)
        
        explored_area = binary_closing(explored_area, footprint=footprint)
        contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
        frontiers = np.logical_and(floor, image)
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

        return traversible, floor, frontiers.astype(np.uint8)
    
    def _save_observation_images(self, step: int, rgb: np.ndarray, depth: np.ndarray, annotated_rgb: np.ndarray):
        """保存每一步的观察图像"""
        # 保存 RGB（PNG无损格式）
        Image.fromarray(rgb).save(f"{self.episode_output_dir}/rgb/step_{step:02d}.png")
        
        # 保存语义分割标注
        if annotated_rgb is not None:
            Image.fromarray(annotated_rgb).save(f"{self.episode_output_dir}/semantic/step_{step:02d}.png")
        
        # 保存深度图（灰度 + 彩色）
        depth_normalized = np.clip(depth[:, :, 0], 50, 550)
        depth_normalized = ((depth_normalized - 50) / 500 * 255).astype(np.uint8)
        Image.fromarray(depth_normalized).save(f"{self.episode_output_dir}/depth/step_{step:02d}.png")
        
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        Image.fromarray(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)).save(
            f"{self.episode_output_dir}/depth/step_{step:02d}_color.png"
        )
    
    def look_around_and_map(self):
        """环视 360° 并建图 - 12次旋转，每次30°"""
        print("\n=== 环视建图 (12 × 30°) ===")
        
        # 初始化地图
        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        
        maps_history = []
        
        for step in range(12):
            # 执行左转
            obs = self.env.step({"action": HabitatSimActions.TURN_LEFT})
            
            # 预处理 & 保存图像
            state, rgb, depth, annotated_rgb = self.preprocess_observation(obs)
            self._save_observation_images(step, rgb, depth, annotated_rgb)
            
            # 建图
            batch_obs = torch.from_numpy(state[None, ...]).float().to(self.device)
            poses = torch.from_numpy(np.array([obs['sensor_pose']])).float().to(self.device)
            
            self.mapping_module(batch_obs, poses)
            full_map, full_pose, _ = self.mapping_module.update_map(step, self.detected_classes, self.episode_id)
            
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            
            traversible, floor, frontiers = self._process_map(step, full_map[0])
            
            # 保存地图状态
            maps_history.append({
                'full_map': full_map.copy(),
                'full_pose': full_pose.copy(),
                'floor': floor.copy(),
                'traversible': traversible.copy(),
            })
            
            if (step + 1) % 3 == 0:
                print(f"✓ Step {step+1}/12 完成")
        
        return maps_history
    
    def save_maps(self, maps_history):
        """保存地图数据和可视化"""
        print("\n=== 保存地图 ===")
        
        # 保存原始数据
        np.save(f"{self.episode_output_dir}/maps_history.npy", maps_history)
        
        # 生成地图演化动画
        self._save_map_evolution(maps_history)
        
        # 打印统计信息
        self._print_final_statistics(maps_history)
    
    def _draw_arrow(self, img, center, angle_rad, length=30, color=(0, 0, 255), thickness=2):
        """在地图上画箭头表示智能体朝向（使用三角形）
        
        Args:
            img: numpy array (H, W, 3)
            center: (x, y) 中心位置
            angle_rad: 朝向角度（弧度）
            length: 箭头长度
            color: RGB颜色 (红色 默认)
            thickness: 线条粗细
        """
        cx, cy = int(center[0]), int(center[1])
        
        # 计算箭头三个顶点（等腰三角形）
        # 注意：地图坐标系 y轴向下，需要翻转
        tip_x = cx + int(length * np.cos(angle_rad))
        tip_y = cy - int(length * np.sin(angle_rad))  # y轴翻转
        
        # 箭头底部两个顶点（夹角120度）
        base_angle1 = angle_rad + np.pi * 2.5 / 3  # 150度
        base_angle2 = angle_rad - np.pi * 2.5 / 3  # -150度
        base_length = length * 0.5
        
        base1_x = cx + int(base_length * np.cos(base_angle1))
        base1_y = cy - int(base_length * np.sin(base_angle1))
        
        base2_x = cx + int(base_length * np.cos(base_angle2))
        base2_y = cy - int(base_length * np.sin(base_angle2))
        
        # 绘制实心三角形
        triangle = np.array([[tip_x, tip_y], [base1_x, base1_y], [base2_x, base2_y]], dtype=np.int32)
        cv2.fillPoly(img, [triangle], color)
        
        # 绘制边框使其更清晰
        cv2.polylines(img, [triangle], True, (0, 0, 0), thickness=1)
        
        # 在中心画一个圆表示智能体位置
        cv2.circle(img, (cx, cy), radius=int(length*0.3), color=color, thickness=-1)
        cv2.circle(img, (cx, cy), radius=int(length*0.3), color=(0, 0, 0), thickness=1)
        
        return img
    
    def _draw_arrow_with_icon(self, img, center, angle_rad, icon_path=None, scale=1.0):
        """使用图标贴图表示智能体（可选方法）
        
        Args:
            img: numpy array (H, W, 3)
            center: (x, y) 中心位置
            angle_rad: 朝向角度（弧度）
            icon_path: 图标文件路径（PNG格式，建议透明背景）
            scale: 缩放比例
        """
        cx, cy = int(center[0]), int(center[1])
        
        # 如果没有提供图标，使用默认箭头
        if icon_path is None or not os.path.exists(icon_path):
            return self._draw_arrow(img, center, angle_rad)
        
        try:
            # 加载图标
            icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)  # 保留alpha通道
            if icon is None:
                return self._draw_arrow(img, center, angle_rad)
            
            # 缩放图标
            h, w = icon.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            icon = cv2.resize(icon, (new_w, new_h))
            
            # 旋转图标（角度转换：弧度 -> 度数，y轴翻转）
            angle_deg = -np.degrees(angle_rad)  # 负号因为y轴翻转
            M = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle_deg, 1.0)
            icon_rotated = cv2.warpAffine(icon, M, (new_w, new_h), 
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0, 0))
            
            # 计算粘贴位置
            y1, y2 = cy - new_h//2, cy + new_h//2
            x1, x2 = cx - new_w//2, cx + new_w//2
            
            # 边界检查
            if y1 < 0 or x1 < 0 or y2 > img.shape[0] or x2 > img.shape[1]:
                return self._draw_arrow(img, center, angle_rad)
            
            # 处理透明通道（如果有）
            if icon_rotated.shape[2] == 4:
                alpha = icon_rotated[:, :, 3] / 255.0
                for c in range(3):
                    img[y1:y2, x1:x2, c] = (
                        alpha * icon_rotated[:, :, c] + 
                        (1 - alpha) * img[y1:y2, x1:x2, c]
                    ).astype(np.uint8)
            else:
                img[y1:y2, x1:x2] = icon_rotated[:, :, :3]
            
            return img
        
        except Exception as e:
            print(f"[WARNING] 加载图标失败: {e}, 使用默认箭头")
            return self._draw_arrow(img, center, angle_rad)
    
    def _create_colored_map(self, obstacles, floor, pose, map_title="Map"):
        """创建自定义配色的地图
        
        配色方案:
        - 白色(255,255,255): 未探索区域
        - 浅蓝色(173,216,230): 地面 (LightBlue)
        - 深红色(139,0,0): 障碍物 (DarkRed)
        - 红色三角形: 智能体朝向
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
        
        # 画智能体箭头（红色三角形或图标）
        pose_x = int(pose[0] * 100 / self.resolution)  # x -> col
        pose_y = int(pose[1] * 100 / self.resolution)  # y -> row
        angle = pose[2]  # 朝向角度（弧度）
        
        if self.use_icon and self.agent_icon_path:
            colored_map = self._draw_arrow_with_icon(colored_map, (pose_x, pose_y), angle, 
                                                     icon_path=self.agent_icon_path, scale=0.5)
        else:
            colored_map = self._draw_arrow(colored_map, (pose_x, pose_y), angle, 
                                           length=25, color=(255, 0, 0), thickness=2)
        
        return colored_map
    
    def _save_map_evolution(self, maps_history):
        """保存地图演化过程"""
        for subdir in ['global', 'local', 'combined']:
            os.makedirs(f"{self.episode_output_dir}/maps/{subdir}", exist_ok=True)
        
        lmb = self.mapping_module.lmb[0].astype(int) if hasattr(self.mapping_module, 'lmb') else None
        
        for i, map_data in enumerate(maps_history):
            full_map = map_data['full_map'][0]
            full_pose = map_data['full_pose'][0]
            floor = map_data['floor']
            
            # 全局地图
            global_colored = self._create_colored_map(full_map[0], floor, full_pose)
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            ax.imshow(global_colored)
            ax.set_title(f'Global Map - Step {i+1}/12 ({(i+1)*30}°)', fontsize=14)
            ax.axis('off')
            
            if lmb is not None:
                from matplotlib.patches import Rectangle
                rect = Rectangle((lmb[2], lmb[0]), lmb[3]-lmb[2], lmb[1]-lmb[0],
                               fill=False, edgecolor='red', linewidth=2, linestyle='--')
                ax.add_patch(rect)
            
            plt.tight_layout()
            plt.savefig(f"{self.episode_output_dir}/maps/global/step_{i:02d}.png", dpi=100, bbox_inches='tight')
            plt.close()
            
            # 局部地图
            if lmb is not None:
                local_obstacles = full_map[0, lmb[0]:lmb[1], lmb[2]:lmb[3]]
                local_floor = floor[lmb[0]:lmb[1], lmb[2]:lmb[3]]
                
                local_pose = full_pose.copy()
                local_pose[0] = full_pose[0] - lmb[2] * self.resolution / 100
                local_pose[1] = full_pose[1] - lmb[0] * self.resolution / 100
                
                local_colored = self._create_colored_map(local_obstacles, local_floor, local_pose)
                
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                ax.imshow(local_colored)
                ax.set_title(f'Local Map - Step {i+1}/12', fontsize=14)
                ax.axis('off')
                plt.tight_layout()
                plt.savefig(f"{self.episode_output_dir}/maps/local/step_{i:02d}.png", dpi=100, bbox_inches='tight')
                plt.close()
                
                # 组合图
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                ax1.imshow(global_colored)
                ax1.set_title(f'Global ({(i+1)*30}°)', fontsize=14)
                ax1.axis('off')
                rect = Rectangle((lmb[2], lmb[0]), lmb[3]-lmb[2], lmb[1]-lmb[0],
                               fill=False, edgecolor='red', linewidth=2, linestyle='--')
                ax1.add_patch(rect)
                
                ax2.imshow(local_colored)
                ax2.set_title('Local (12m×12m)', fontsize=14)
                ax2.axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{self.episode_output_dir}/maps/combined/step_{i:02d}.png", dpi=100, bbox_inches='tight')
                plt.close()
        
        print(f"✓ 保存 {len(maps_history)} 帧地图")
    
    def _print_final_statistics(self, maps_history):
        """打印最终统计信息"""
        final_map = maps_history[-1]['full_map'][0]
        final_pose = maps_history[-1]['full_pose'][0]
        final_floor = maps_history[-1]['floor']
        
        total_pixels = final_map.shape[1] * final_map.shape[2]
        explored_pixels = np.sum(final_map[1] > 0)
        
        print(f"\n{'='*50}")
        print(f"Episode {self.episode_id} | Scene: {self.scene_id}")
        print(f"检测类别 ({len(self.detected_classes)}): {list(self.detected_classes)[:5]}...")
        print(f"探索率: {explored_pixels / total_pixels * 100:.1f}%")
        print(f"最终位姿: x={final_pose[0]:.1f}m, y={final_pose[1]:.1f}m, θ={np.degrees(final_pose[2]):.0f}°")
        print(f"{'='*50}")
    
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
