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
from vlnce_baselines.utils.constant import base_classes


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
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        print("[INFO] Semantic_Mapping 初始化完成")
        
        # 检测类别
        self.detected_classes = OrderedSet()
        self.classes = base_classes.copy()  # ["floor", "wall", "door", ...]
        
    def preprocess_observation(self, obs):
        """预处理观察：语义分割"""
        # 提取 RGB 和 Depth
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        
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
    
    def look_around_and_map(self):
        """环视 360° 并建图"""
        print("\n[STEP 3] 环视 360° 建图...")
        
        # 初始化地图
        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        print(f"[INFO] 地图初始化完成，检测类别数: {len(self.detected_classes)}")
        
        maps_history = []
        rgb_history = []
        depth_history = []
        
        for step in range(12):
            print(f"\n[STEP 3.{step+1}] 左转 30° (总计 {(step+1)*30}°)...")
            
            # 执行左转动作
            actions = [{"action": HabitatSimActions.TURN_LEFT}]
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            obs = obs[0]
            
            if dones[0]:
                print("[WARNING] Episode 提前结束")
                break
            
            # 预处理观察
            state, rgb, depth, annotated_rgb = self.preprocess_observation(obs)
            
            # 保存 RGB 和 Depth
            rgb_history.append(rgb)
            depth_history.append(depth)
            
            # 保存可视化
            plt.imsave(f"{self.output_dir}/rgb/step_{step:02d}.png", rgb)
            plt.imsave(f"{self.output_dir}/rgb/step_{step:02d}_annotated.png", annotated_rgb)
            plt.imsave(f"{self.output_dir}/depth/step_{step:02d}.png", depth[:,:,0], cmap='viridis')
            
            # 准备 batch
            batch_obs = torch.from_numpy(state[None, ...]).float().to(self.device)  # (1, 4+N, 120, 160)
            
            # 获取位姿变化
            sensor_pose = obs['sensor_pose']  # [Δx, Δy, Δθ]
            poses = torch.tensor([sensor_pose]).float().to(self.device)
            
            # 更新地图
            self.mapping_module(batch_obs, poses)
            full_map, full_pose, one_step_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.episode_id)
            
            # 保存地图
            maps_history.append({
                'full_map': full_map.copy(),
                'full_pose': full_pose.copy(),
                'one_step_map': one_step_map.copy()
            })
            
            # 清空单步地图
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            
            print(f"[INFO] 位姿: [{full_pose[0,0]:.2f}, {full_pose[0,1]:.2f}, {full_pose[0,2]:.2f}]")
            print(f"[INFO] 地图形状: {full_map.shape}")
        
        return maps_history, rgb_history, depth_history
    
    def save_maps(self, maps_history):
        """保存地图数据和可视化"""
        print("\n[STEP 4] 保存地图...")
        
        # 保存原始数据
        np.save(f"{self.output_dir}/maps_history.npy", maps_history)
        print(f"[INFO] 保存地图历史: {self.output_dir}/maps_history.npy")
        
        # 保存最终地图
        final_map = maps_history[-1]['full_map'][0]  # (N+4, 480, 480)
        final_pose = maps_history[-1]['full_pose'][0]  # (3,)
        
        # 可视化不同通道
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 通道 0: 障碍物
        axes[0, 0].imshow(final_map[0], cmap='gray')
        axes[0, 0].set_title('Channel 0: Obstacles')
        axes[0, 0].axis('off')
        
        # 通道 1: 已探索区域
        axes[0, 1].imshow(final_map[1], cmap='Blues')
        axes[0, 1].set_title('Channel 1: Explored Area')
        axes[0, 1].axis('off')
        
        # 通道 2: 当前位置
        axes[0, 2].imshow(final_map[2], cmap='Reds')
        axes[0, 2].set_title('Channel 2: Current Location')
        axes[0, 2].plot(final_pose[1]*100/self.resolution, 
                       final_pose[0]*100/self.resolution, 
                       'r*', markersize=20)
        axes[0, 2].axis('off')
        
        # 通道 3: 历史轨迹
        axes[1, 0].imshow(final_map[3], cmap='Greens')
        axes[1, 0].set_title('Channel 3: Past Locations')
        axes[1, 0].axis('off')
        
        # 语义类别 (如果有)
        if len(self.detected_classes) > 0:
            # 找到 floor 类别
            floor_idx = None
            for i, cls in enumerate(self.detected_classes):
                if 'floor' in cls.lower():
                    floor_idx = i + 4  # 前4个是固定通道
                    break
            
            if floor_idx is not None and floor_idx < final_map.shape[0]:
                axes[1, 1].imshow(final_map[floor_idx], cmap='YlOrBr')
                axes[1, 1].set_title(f'Channel {floor_idx}: Floor')
                axes[1, 1].axis('off')
        
        # 综合地图
        composite = np.zeros((480, 480, 3))
        composite[:, :, 0] = final_map[0]  # 红色：障碍物
        composite[:, :, 1] = final_map[1]  # 绿色：已探索
        composite[:, :, 2] = final_map[2]  # 蓝色：当前位置
        axes[1, 2].imshow(composite)
        axes[1, 2].set_title('Composite Map')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/final_map.png", dpi=150)
        print(f"[INFO] 保存最终地图可视化: {self.output_dir}/final_map.png")
        
        # 保存动画 (地图演化过程)
        print("[INFO] 生成地图演化动画...")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for i, map_data in enumerate(maps_history):
            ax.clear()
            composite = np.zeros((480, 480, 3))
            m = map_data['full_map'][0]
            composite[:, :, 0] = m[0]  # 障碍物
            composite[:, :, 1] = m[1]  # 已探索
            composite[:, :, 2] = m[2]  # 当前位置
            
            ax.imshow(composite)
            ax.set_title(f'Step {i+1}/12 - Rotation {(i+1)*30}°')
            ax.axis('off')
            
            plt.savefig(f"{self.output_dir}/maps/map_step_{i:02d}.png", dpi=100)
        
        plt.close('all')
        print(f"[INFO] 保存地图演化: {self.output_dir}/maps/map_step_*.png")
        
        # 打印统计信息
        print("\n" + "="*50)
        print("建图统计信息")
        print("="*50)
        print(f"Episode ID: {self.episode_id}")
        print(f"检测到的类别数: {len(self.detected_classes)}")
        print(f"类别列表: {list(self.detected_classes)}")
        print(f"最终位姿: [{final_pose[0]:.2f}, {final_pose[1]:.2f}, {final_pose[2]:.2f}]")
        print(f"地图尺寸: {final_map.shape}")
        print(f"已探索像素数: {np.sum(final_map[1] > 0)}")
        print(f"障碍物像素数: {np.sum(final_map[0] > 0)}")
        print("="*50)
    
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
