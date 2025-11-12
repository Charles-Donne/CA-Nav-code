import os
import pdb
import queue
import copy
import gzip
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
from fastdtw import fastdtw
from typing import List, Any, Dict
from collections import defaultdict
from skimage.morphology import binary_closing
import inspect

import torch
from torch import Tensor
from torchvision import transforms

from habitat import Config, logger
from habitat_extensions.measures import NDTW
from habitat.core.simulator import Observations
from habitat_baselines.common.base_trainer import BaseTrainer
from habitat_baselines.common.environments import get_env_class
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.map.value_map import ValueMap
from vlnce_baselines.map.history_map import HistoryMap
from vlnce_baselines.map.direction_map import DirectionMap
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.map.mapping import Semantic_Mapping
from vlnce_baselines.models.Policy import FusionMapPolicy
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import gather_list_and_concat, get_device
from vlnce_baselines.map.semantic_prediction import GroundedSAM
from vlnce_baselines.common.constraints import ConstraintsMonitor
from vlnce_baselines.utils.constant import base_classes, map_channels

from pyinstrument import Profiler
import warnings
warnings.filterwarnings('ignore')


# 兼容不同版本的 scikit-image
def _binary_closing_compat(image, footprint):
    """兼容 scikit-image 旧版本的 binary_closing 调用"""
    sig = inspect.signature(binary_closing)
    if 'footprint' in sig.parameters:
        return binary_closing(image, footprint=footprint)
    else:
        # 旧版本使用 selem 参数
        return binary_closing(image, selem=footprint)


@baseline_registry.register_trainer(name="ZS-Evaluator-mp")
class ZeroShotVlnEvaluatorMP(BaseTrainer):
    """零样本视觉语言导航评估器（多进程版本）
    
    功能：在多个GPU上并行评估VLN任务
    核心流程：环境初始化 → 环视探索 → 导航执行 → 指标计算
    """
    def __init__(self, config: Config, segment_module=None, mapping_module=None) -> None:
        super().__init__()
        
        # GPU 设备配置
        self.device = get_device(config.TORCH_GPU_ID)
        torch.cuda.set_device(self.device)
        self.config = config
        
        # 地图相关配置
        self.map_args = config.MAP
        self.visualize = config.MAP.VISUALIZE  # 是否可视化
        self.resolution = config.MAP.MAP_RESOLUTION  # 地图分辨率（cm/pixel）
        self.keyboard_control = config.KEYBOARD_CONTROL  # 是否手动控制
        self.width = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH  # RGB 图像宽度
        self.height = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT  # RGB 图像高度
        self.max_step = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS  # 每个 episode 最大步数
        self.map_shape = (config.MAP.MAP_SIZE_CM // self.resolution,
                          config.MAP.MAP_SIZE_CM // self.resolution)  # 地图尺寸
        
        # 图像预处理
        self.trans = transforms.Compose([transforms.ToPILImage(), 
                                         transforms.Resize(
                                             (self.map_args.FRAME_HEIGHT, self.map_args.FRAME_WIDTH), 
                                             interpolation=Image.NEAREST)
                                        ])
        
        # 状态变量初始化
        self.classes = []  # 当前要检测的类别列表
        self.current_episode_id = None  # 当前 episode ID
        self.current_detections = None  # 当前检测结果
        self.map_channels = map_channels  # 地图通道数（障碍物、探索区域等）
        
        # 各种地图初始化（全零）
        self.floor = np.zeros(self.map_shape)  # 地板地图
        self.one_step_floor = np.zeros(self.map_shape)  # 当前步新探索的地板
        self.frontiers = np.zeros(self.map_shape)  # 边界地图（探索边缘）
        self.traversible = np.zeros(self.map_shape)  # 可穿越区域
        self.collision_map = np.zeros(self.map_shape)  # 碰撞地图
        self.visited = np.zeros(self.map_shape)  # 已访问区域
        
        # 约束相关配置
        self.base_classes = copy.deepcopy(base_classes)  # 基础类别（如 floor, wall 等）
        self.min_constraint_steps = config.EVAL.MIN_CONSTRAINT_STEPS  # 最小约束步数
        self.max_constraint_steps = config.EVAL.MAX_CONSTRAINT_STEPS  # 最大约束步数
    
    def _set_eval_config(self) -> None:
        """设置评估配置（主要是进程和设备信息）"""
        print("set eval configs")
        self.config.defrost()
        self.config.MAP.DEVICE = self.config.TORCH_GPU_ID
        self.config.MAP.HFOV = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HFOV
        self.config.MAP.AGENT_HEIGHT = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        self.config.MAP.NUM_ENVIRONMENTS = self.config.NUM_ENVIRONMENTS
        self.config.MAP.RESULTS_DIR = self.config.RESULTS_DIR
        self.world_size = self.config.world_size  # 总进程数
        self.local_rank = self.config.local_rank  # 当前进程编号
        self.config.freeze()
        
    def _init_envs(self) -> None:
        """初始化 Habitat 仿真环境"""
        print("start to initialize environments")

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED,  # 只加载分配给该进程的 episodes
        )
        print(f"local rank: {self.local_rank}, num of episodes: {self.envs.number_of_episodes}")
        self.detected_classes = OrderedSet()  # 记录已检测到的所有类别（去重）
        print("initializing environments finished!")
        
    def _collect_val_traj(self) -> None:
        """加载真实轨迹数据（用于计算 NDTW 等指标）"""
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(split=split)) as f:
            gt_data = json.load(f)

        self.gt_data = gt_data
        
    def _calculate_metric(self, infos: List):
        """计算评估指标：Success, SPL, NDTW, SDTW 等"""
        curr_eps = self.envs.current_episodes()
        info = infos[0]
        ep_id = curr_eps[0].episode_id
        
        # 获取真实路径和预测路径
        gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
        pred_path = np.array(info['position']['position'])
        distances = np.array(info['position']['distance'])  # 每步到目标的距离
        gt_length = distances[0]  # 起点到终点的直线距离
        
        # 计算 DTW 距离
        dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
        
        metric = {}
        metric['steps_taken'] = info['steps_taken']
        metric['distance_to_goal'] = distances[-1]  # 最终距离目标的距离
        metric['success'] = 1. if distances[-1] <= 3. else 0.  # Success: 距离 ≤ 3 米
        metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.  # Oracle Success: 任意时刻距离 ≤ 3 米
        metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
        
        # SPL (Success weighted by Path Length)
        metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
        
        # NDTW (Normalized Dynamic Time Warping)
        metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
        
        # SDTW (Success weighted by NDTW)
        metric['sdtw'] = metric['ndtw'] * metric['success']
        
        self.state_eps[ep_id] = metric
        print(self.state_eps[ep_id])
        
    def _initialize_policy(self) -> None:
        """初始化所有策略模块：语义分割、地图构建、价值估计、路径规划等"""
        print("start to initialize policy")
        
        # 语义分割模块（GroundedSAM：开放词汇目标检测）
        self.segment_module = GroundedSAM(self.config, self.device)
        
        # 语义地图构建模块
        self.mapping_module = Semantic_Mapping(self.config.MAP).to(self.device)
        self.mapping_module.eval()
        
        # 价值地图模块（计算每个位置到目标的价值）
        self.value_map_module = ValueMap(self.config, self.mapping_module.map_shape, self.device)
        
        # 历史地图模块（记录已访问区域，避免重复探索）
        self.history_module = HistoryMap(self.config, self.mapping_module.map_shape)
        
        # 方向地图模块（处理方向约束，如"左转"）
        self.direction_module = DirectionMap(self.config, self.mapping_module.map_shape)
        
        # 路径规划策略（FMM：Fast Marching Method）
        self.policy = FusionMapPolicy(self.config, self.mapping_module.map_shape[0])
        self.policy.reset()
        
        # 约束监控模块（检查是否满足子任务约束）
        self.constraints_monitor = ConstraintsMonitor(self.config, self.device)
        
    def _concat_obs(self, obs: Observations) -> np.ndarray:
        """合并 RGB 和 Depth 观察为一个状态"""
        rgb = obs['rgb'].astype(np.uint8)
        depth = obs['depth']
        state = np.concatenate((rgb, depth), axis=2).transpose(2, 0, 1) # (h, w, c)->(c, h, w)
        
        return state
    
    def _preprocess_state(self, state: np.ndarray) -> np.ndarray:
        """预处理状态：语义分割 + 深度处理 + 下采样"""
        state = state.transpose(1, 2, 0)
        rgb = state[:, :, :3].astype(np.uint8) #[3, h, w]
        rgb = rgb[:,:,::-1] # RGB to BGR（OpenCV 格式）
        depth = state[:, :, 3:4] #[1, h, w]
        min_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        max_depth = self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        env_frame_width = self.config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
        
        # 语义分割预测（GroundedSAM）
        sem_seg_pred = self._get_sem_pred(rgb) #[num_detected_classes, h, w]
        
        # 深度预处理（归一化、去噪）
        depth = self._preprocess_depth(depth, min_depth, max_depth) #[1, h, w]
        
        # 下采样因子（640 / 160 = 4）
        ds = env_frame_width // self.map_args.FRAME_WIDTH # ds = 4
        if ds != 1:
            rgb = np.asarray(self.trans(rgb.astype(np.uint8))) # resize
            depth = depth[ds // 2::ds, ds // 2::ds] # down scaling start from 2, step=4
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2) # recover depth.shape to (height, width, 1)
        state = np.concatenate((rgb, depth, sem_seg_pred),axis=2).transpose(2, 0, 1) # (4+num_detected_classes, h, w)
        
        return state
        
    def _get_sem_pred(self, rgb: np.ndarray) -> np.ndarray:
        """使用 GroundedSAM 进行语义分割
        
        返回：
            masks: [num_detected_classes, h, w] 每个类别的掩码
        """
        masks, labels, annotated_images, self.current_detections = \
            self.segment_module.segment(rgb, classes=self.classes)
        self.mapping_module.rgb_vis = annotated_images
        assert len(masks) == len(labels), f"The number of masks not equal to the number of labels!"
        print("current step detected classes: ", labels)  # 例如: ["kitchen counter 0.69", "floor 0.37"]
        
        # 处理标签（去掉置信度分数）
        class_names = self._process_labels(labels)
        
        # 处理掩码（合并相同类别）
        masks = self._process_masks(masks, class_names)
        
        return masks.transpose(1, 2, 0)
    
    def _process_labels(self, labels: List[str]) -> List:
        """处理标签：去除置信度分数，记录到已检测类别集合"""
        class_names = []
        for label in labels:
            # "kitchen counter 0.69" -> "kitchen counter"
            class_name = " ".join(label.split(' ')[:-1])
            class_names.append(class_name)
            self.detected_classes.add(class_name)  # 添加到已检测类别（自动去重）
        
        return class_names
        
    def _process_masks(self, masks: np.ndarray, labels: List[str]):
        """处理掩码：合并相同类别的掩码，构建动态通道的掩码张量
        
        由于是开放词汇语义映射，需要维护一个动态通道的掩码张量。
        将所有相同类别的掩码合并为一个通道。
        
        Args:
            masks: shape (c, h, w)，每个实例一个通道
            labels: 对应的标签列表
            
        Returns:
            final_masks: shape (len(detected_classes), h, w)
        """
        if masks.shape != (0,):
            # 按类别分组
            same_label_indexs = defaultdict(list)
            for idx, item in enumerate(labels):
                same_label_indexs[item].append(idx) #dict {class name: [idx]}
            
            # 合并同类掩码
            combined_mask = np.zeros((len(same_label_indexs), *masks.shape[1:]))
            for i, indexs in enumerate(same_label_indexs.values()):
                combined_mask[i] = np.sum(masks[indexs, ...], axis=0)
            
            # 找到每个类别在 detected_classes 中的索引
            idx = [self.detected_classes.index(label) for label in same_label_indexs.keys()]
            
            # 构建最终掩码（维度 = 所有已检测类别数）
            final_masks = np.zeros((len(self.detected_classes), *masks.shape[1:]))
            final_masks[idx, ...] = combined_mask
        else:
            final_masks = np.zeros((len(self.detected_classes), self.height, self.width))
        
        return final_masks
    
    def _preprocess_depth(self, depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """预处理深度图：处理缺失值、去除异常值、归一化"""
        depth = depth[:, :, 0] * 1

        # 填充缺失深度值
        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        # 将过远的像素设为无效
        mask2 = depth > 0.99
        depth[mask2] = 0.

        # 将无效像素设为视野范围（100米）
        mask1 = depth == 0
        depth[mask1] = 100.0
        
        # 归一化到厘米单位
        depth = min_depth * 100.0 + depth * max_depth * 100.0
        
        return depth
    
    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        """预处理观察：合并 + 预处理"""
        concated_obs = self._concat_obs(obs)
        state = self._preprocess_state(concated_obs)
        
        return state # state.shape=(c,h,w)
    
    def _batch_obs(self, n_obs: List[Observations]) -> Tensor:
        """批处理观察（支持动态通道数，padding 到最大通道数）"""
        n_states = [self._preprocess_obs(obs) for obs in n_obs]
        max_channels = max([len(state) for state in n_states])
        batch = np.stack([np.pad(state, 
                [(0, max_channels - state.shape[0]), 
                 (0, 0), 
                 (0, 0)], 
                mode='constant') 
         for state in n_states], axis=0)
        
        # 确保返回 float32 类型，避免 depth_utils 中的类型不匹配问题
        return torch.from_numpy(batch).float().to(self.device)
    
    def _random_policy(self):
        """随机策略（用于测试）"""
        action = np.random.choice([
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ])
        
        return {"action": action}

    def _process_classes(self, base_class: List, target_class: List) -> List:
        """处理类别列表：移除重复的目标类别，然后添加到末尾"""
        for item in target_class:
            if item in base_class:
                base_class.remove(item)
        base_class.extend(target_class)
        
        return base_class
    
    def _check_destination(self, current_idx: int, sub_constraints: dict, llm_destination: str, decisions: dict) -> str:
        for idx in range(current_idx, len(sub_constraints)):
                constraints = sub_constraints[str(idx)]
                landmarks = decisions[str(idx)]["landmarks"]
                for constraint in constraints:
                    if constraint[0] == "direction constraint":
                        continue
                    else:
                        landmark = constraint[1]
                        for item in landmarks:
                            print(landmark, item)
                            if landmark in item:
                                choice = item[1]
                            else:
                                continue
                            print(choice, choice != "move away")
                            if choice != "move away":
                                return constraint[1]
                            else:
                                break
        else:
            return llm_destination
    
    def _process_llm_reply(self, obs: Observations):
        """解析 LLM 生成的指令分解结果
        
        LLM 输出包含：
        - sub-instructions: 子指令列表
        - state-constraints: 每个子指令的约束条件
        - decisions: 每个子指令的决策 landmarks
        - destination: 最终目标
        """
        def _get_first_destination(sub_constraints: dict, llm_destination: str) -> str:
            """获取第一个目标（第一个非方向约束的 landmark）"""
            for constraints in sub_constraints.values():
                for constraint in constraints:
                    if constraint[0] != "direction constraint":
                        return constraint[1]
            else:
                return llm_destination
        
        
        self.llm_reply = obs['llm_reply']
        self.instruction = obs['instruction']['text']  # 原始指令
        self.sub_instructions = self.llm_reply['sub-instructions']  # 子指令列表
        self.sub_constraints = self.llm_reply['state-constraints']  # 约束条件
        self.decisions = self.llm_reply['decisions']  # 决策 landmarks
        self.destination = _get_first_destination(self.sub_constraints, self.llm_reply['destination'])  # 当前目标
        print("!!!!!!!!!!!!!!! first destination: ", self.destination)
        
        self.last_destination = self.destination  # 上一个目标
        first_landmarks = self.decisions['0']['landmarks']  # 第一个子任务的 landmarks
        self.destination_class = [item[0] for item in first_landmarks]  # 目标类别列表
        self.classes = self._process_classes(self.base_classes, self.destination_class)  # 更新检测类别
        self.constraints_check = [False] * len(self.sub_constraints)  # 约束检查状态（未完成）
    
    
    def _process_one_step_floor(self, one_step_full_map: np.ndarray, kernel_size: int=3) -> np.ndarray:
        """处理当前步新探索的地板区域"""
        navigable_index = process_navigable_classes(self.detected_classes)
        not_navigable_index = [i for i in range(len(self.detected_classes)) if i not in navigable_index]
        one_step_full_map = remove_small_objects(one_step_full_map.astype(bool), min_size=64)
        
        obstacles = one_step_full_map[0, ...].astype(bool)
        explored_area = one_step_full_map[1, ...].astype(bool)
        objects = np.sum(one_step_full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)
        navigable = np.logical_or.reduce(one_step_full_map[map_channels:, ...][navigable_index])
        navigable = np.logical_and(navigable, np.logical_not(objects))
        
        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = _binary_closing_compat(floor, disk(kernel_size))
        
        return floor
        
    def _process_map(self, step: int, full_map: np.ndarray, kernel_size: int=3) -> tuple:
        """处理语义地图，提取导航相关信息
        
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
        objects = np.sum(full_map[map_channels:, ...][not_navigable_index], axis=0).astype(bool)  # 不可导航物体
        
        # 形态学处理（闭运算，填充小孔）
        footprint = disk(kernel_size)
        obstacles_closed = _binary_closing_compat(obstacles, footprint)
        objects_closed = _binary_closing_compat(objects, footprint)
        navigable = np.logical_or.reduce(full_map[map_channels:, ...][navigable_index])
        navigable = np.logical_and(navigable, np.logical_not(objects))
        navigable_closed = _binary_closing_compat(navigable, footprint)
        
        # 计算不可穿越区域
        untraversible = np.logical_or(objects_closed, obstacles_closed)
        untraversible[navigable_closed == 1] = 0
        untraversible = remove_small_objects(untraversible, min_size=64)
        untraversible = _binary_closing_compat(untraversible, disk(3))
        traversible = np.logical_not(untraversible)

        # 计算地板区域
        free_mask = 1 - np.logical_or(obstacles, objects)
        free_mask = np.logical_or(free_mask, navigable)
        floor = explored_area * free_mask
        floor = remove_small_objects(floor, min_size=400).astype(bool)
        floor = _binary_closing_compat(floor, footprint)
        traversible = np.logical_or(floor, traversible)
        
        # 计算边界（探索边缘）
        explored_area = _binary_closing_compat(explored_area, footprint)
        contours, _ = cv2.findContours(explored_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.zeros(full_map.shape[-2:], dtype=np.uint8)
        image = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=3)
        frontiers = np.logical_and(floor, image)
        frontiers = remove_small_objects(frontiers.astype(bool), min_size=64)

        return traversible, floor, frontiers.astype(np.uint8)
    
    def _save_floor_semantic_map(self, step: int, episode_id: int, full_map: np.ndarray):
        """保存包含floor语义层的分割地图可视化
        
        Args:
            step: 当前步数
            episode_id: episode ID
            full_map: 完整语义地图 (N+1, 480, 480)
        """
        # 提取各通道
        obstacles = full_map[0, ...].astype(bool)     # 障碍物（纯几何，高度>智能体）
        explored = full_map[1, ...].astype(bool)      # 已探索
        current_loc = full_map[2, ...].astype(bool)   # 当前位置
        
        # 创建彩色可视化 (480, 480, 3)
        h, w = obstacles.shape
        vis_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 颜色方案：
        # - 白色 (255,255,255): 未探索区域
        # - 浅灰色 (200,200,200): 已探索的空地（无障碍无物体）
        # - 黑色 (0,0,0): 障碍物（高度判断，墙体等）
        # - 浅绿色 (144,238,144): floor语义层（可行走地板）
        # - 红/蓝/黄/紫等: 各种语义物体（table, chair, kitchen等）
        
        # 1. 未探索区域 = 白色（默认背景）
        vis_image[:] = [255, 255, 255]
        
        # 2. 已探索区域 = 浅灰色
        vis_image[explored] = [200, 200, 200]
        
        # 3. 先绘制所有语义物体（彩色）
        if full_map.shape[0] > 4:  # 有语义通道
            semantic_channels = full_map[4:, ...]  # 所有语义通道
            
            # 为每个检测类别分配独特颜色
            color_palette = [
                [255, 0, 0],      # 0: 红色 (如 table)
                [0, 0, 255],      # 1: 蓝色 (如 chair)
                [255, 255, 0],    # 2: 黄色 (如 bed)
                [255, 0, 255],    # 3: 品红 (如 sofa)
                [0, 255, 255],    # 4: 青色 (如 cabinet)
                [255, 128, 0],    # 5: 橙色 (如 counter)
                [128, 0, 255],    # 6: 紫色 (如 sink)
                [0, 128, 255],    # 7: 天蓝 (如 refrigerator)
                [255, 128, 128],  # 8: 粉红
                [128, 255, 128],  # 9: 浅绿（注意和floor区分）
                [128, 128, 255],  # 10: 浅蓝
                [255, 255, 128],  # 11: 浅黄
            ]
            
            # 绘制每个检测到的语义类别
            for i, class_name in enumerate(self.detected_classes):
                if i >= semantic_channels.shape[0]:
                    break
                class_mask = semantic_channels[i] > 0.5  # 置信度阈值
                if np.any(class_mask):
                    color = color_palette[i % len(color_palette)]
                    vis_image[class_mask] = color
        
        # 4. Floor语义层 = 浅绿色 (144,238,144) Light Green
        floor_overlay = self.floor.astype(bool)
        vis_image[floor_overlay] = [144, 238, 144]
        
        # 5. 障碍物（纯几何高度判断）= 黑色（最高优先级）
        vis_image[obstacles] = [0, 0, 0]
        
        # 翻转图像（与其他地图可视化保持一致）
        vis_image = np.flipud(vis_image)
        
        # 6. 绘制当前位置和朝向箭头
        # 找到当前位置的中心点
        current_loc_flipped = np.flipud(current_loc)
        if np.any(current_loc_flipped):
            # 获取当前位置的质心
            y_coords, x_coords = np.where(current_loc_flipped)
            if len(y_coords) > 0:
                center_y = int(np.mean(y_coords))
                center_x = int(np.mean(x_coords))
                
                # 从full_pose获取朝向角度
                # 注意：这里需要从self.mapping_module获取当前位姿
                if hasattr(self, 'mapping_module') and hasattr(self.mapping_module, 'full_pose'):
                    heading = self.mapping_module.full_pose[0, -1]  # 弧度
                    
                    # 计算箭头终点（箭头长度为20像素）
                    arrow_length = 20
                    # Habitat坐标系：heading=0朝向+X轴（地图右侧）
                    # OpenCV坐标系：需要转换，y轴向下
                    end_x = int(center_x + arrow_length * np.cos(heading))
                    end_y = int(center_y - arrow_length * np.sin(heading))  # y轴反向
                    
                    # 绘制箭头（红色，粗线）
                    cv2.arrowedLine(
                        vis_image,
                        (center_x, center_y),  # 起点
                        (end_x, end_y),        # 终点
                        (0, 0, 255),           # 红色箭头
                        thickness=3,           # 线宽
                        tipLength=0.3          # 箭头尖端长度比例
                    )
                    
                    # 绘制中心点（黄色圆点，更醒目）
                    cv2.circle(vis_image, (center_x, center_y), 5, (0, 255, 255), -1)
        
        # 保存图像
        save_dir = os.path.join(self.config.RESULTS_DIR, "floor_semantic_map/eps_%d" % episode_id)
        os.makedirs(save_dir, exist_ok=True)
        fn = "{}/step-{}.png".format(save_dir, step)
        cv2.imwrite(fn, vis_image)
    
    def _maps_initialization(self):
        """初始化地图：重置环境 + 解析指令 + 初始化语义地图"""
        obs = self.envs.reset()  # 重置环境，获取初始观察
        self._process_llm_reply(obs[0])  # 解析 LLM 指令
        self.current_episode_id = self.envs.current_episodes()[0].episode_id
        print("current episode id: ", self.current_episode_id)
        
        # 初始化语义地图
        self.mapping_module.init_map_and_pose(num_detected_classes=len(self.detected_classes))
        batch_obs = self._batch_obs(obs)
        poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
        self.mapping_module(batch_obs, poses)
        full_map, full_pose, _ = self.mapping_module.update_map(0, self.detected_classes, self.current_episode_id)
        
        # 清空单步地图
        self.mapping_module.one_step_full_map.fill_(0.)
        self.mapping_module.one_step_local_map.fill_(0.)
    
    def _look_around(self):
        """环视 360 度（12 步 × 30° = 360°），建立初始地图
        
        核心流程：
        1. 循环 12 次，每次左转 30° (12 × 30° = 360°)
        2. 每次转向后：
           - 获取 RGB-D 观察
           - 语义分割（GroundedSAM）
           - 点云生成 + 坐标变换
           - 3D 体素投影 + 高度压缩
           - 多帧融合（取最大值）
           - 更新全局地图
        3. 环视结束后规划初始动作
        
        Returns:
            full_pose: (3,) [x, y, heading] 当前位姿（米）
            obs: dict 最后一帧观察
            dones: bool episode 是否结束
            infos: dict 附加信息
        """
        print("\n========== LOOK AROUND ==========\n")
        # 初始化返回变量
        full_pose, obs, dones, infos = None, None, None, None
        
        # ========== 循环 12 次，每次左转 30° ==========
        for step in range(0, 12):
            # ===== 步骤 1: 执行左转动作 (30°) =====
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                # HabitatSimActions.TURN_LEFT = 左转 30°
                actions.append({"action": HabitatSimActions.TURN_LEFT})
            
            # 在仿真环境中执行动作
            outputs = self.envs.step(actions)
            
            # 解包结果: obs=观察, _=奖励(不使用), dones=是否结束, infos=附加信息
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            # Save RGB frames if print_images is enabled
            if self.config.MAP.PRINT_IMAGES:
                rgb_frame = obs[0]['rgb'].astype(np.uint8)  # Get RGB from observation
                save_dir = os.path.join(self.config.RESULTS_DIR, "rgb_frames/eps_%d"%self.current_episode_id)
                os.makedirs(save_dir, exist_ok=True)
                fn = "{}/step-{}.png".format(save_dir, step)
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fn, bgr_frame)
            
            # ===== 步骤 2: 检查 episode 是否提前结束 =====
            if dones[0]:
                # 如果已结束（例如超时、碰撞等），直接返回
                return full_pose, obs, dones, infos
            
            # ===== 步骤 3: 更新语义地图（核心建图流程）=====
            
            # 3.1 预处理观察 → 语义分割
            # Line 517: 批处理观察 → 语义分割
            batch_obs = self._batch_obs(obs)
            # ↓ 展开调用链：
            # _batch_obs() [Line 346]
            #   → _preprocess_obs() [Line 349]
            #     → _concat_obs() [Line 203] 合并 RGB + Depth
            #     → _preprocess_state() [Line 210]
            #       → _get_sem_pred() [Line 220] 🔥 GroundedSAM 语义分割
            #         → segment_module.segment() [GroundedSAM]
            #           返回: masks (N, 480, 640)
            #       → _preprocess_depth() [Line 313]
            #       → 下采样 4x
            #   返回: batch_obs (1, 4+N, 160, 160)
            
            # 3.2 获取智能体位姿变化 (相对于上一步的位移)
            # sensor_pose: [Δx, Δy, Δθ] 单位: 米, 米, 弧度
            poses = torch.from_numpy(np.array([item['sensor_pose'] for item in obs])).float().to(self.device)
            
            # 3.3 调用 mapping_module 前向传播（核心建图）
            # mapping_module.forward() 执行:
            #   ① 点云生成: Depth → (120, 160, 3) 3D 点
            #   ② 坐标变换: 相机坐标系 → 智能体坐标系 → 世界坐标系
            #   ③ 体素投影: 点云 + 语义特征 → (N+1, 100, 100, 80) 3D 体素
            #   ④ 高度压缩: 沿 z 轴求和 → (N+1, 100, 100) 2D 地图
            #   ⑤ 位姿变换: agent_view → 旋转 + 平移 → local_map
            #   ⑥ 多帧融合: max(历史地图, 当前帧) → 更新 local_map
            self.mapping_module(batch_obs, poses)
            
            # 3.4 更新全局地图并获取当前状态
            # update_map() 执行:
            #   ① 更新当前位置标记 (3×3 区域)
            #   ② local_map → full_map (写回到全局地图)
            #   ③ 更新全局位姿
            # 返回:
            #   full_map: (1, N+4, 480, 480) 完整语义地图
            #     通道 0: 障碍物地图
            #     通道 1: 已探索区域
            #     通道 2: 当前位置
            #     通道 3: 已访问区域
            #     通道 4~: 各类别语义掩码 (如 floor, wall, kitchen 等)
            #   full_pose: (1, 3) [x, y, heading] 当前全局位姿（米）
            #   one_step_full_map: (1, N+4, 480, 480) 仅包含当前帧的地图
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
            
            # 3.5 清空单步地图（准备下一次循环）
            # one_step_*_map 只记录当前帧，每步都清空
            # 用途: 区分"新探索区域" vs "历史累积区域"
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            
            # ===== 步骤 4: 处理导航地图（提取可导航信息）=====
            
            # 4.1 从语义地图中提取导航相关信息
            # _process_map() 执行:
            #   ① 区分可导航类别 (floor, carpet) 和不可导航类别 (wall, table)
            #   ② 形态学处理 (闭运算填充小孔)
            #   ③ 计算可穿越区域 traversible
            #   ④ 计算地板区域 floor
            #   ⑤ 计算探索边界 frontiers (已探索区域的轮廓)
            # 返回:
            #   traversible: (480, 480) bool 可穿越区域
            #   floor: (480, 480) bool 地板区域
            #   frontiers: (480, 480) uint8 探索边界
            self.traversible, self.floor, self.frontiers = self._process_map(step, full_map[0])
            
            # Save floor map visualization if print_images is enabled
            if self.config.MAP.PRINT_IMAGES:
                self._save_floor_semantic_map(step, self.current_episode_id, full_map[0])
            
            # 4.2 处理当前步新探索的地板
            # 只处理 one_step_full_map，用于价值图计算
            self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])
                        
            # ===== 步骤 5: 计算价值图（目标导向的价值分布）=====
            
            # 5.1 使用 BLIP 计算当前视野与目标的语义相似度
            # BLIP (Bootstrapped Language-Image Pre-training):
            #   输入: RGB 图像 (480, 640, 3) + 目标文本 (如 "kitchen")
            #   输出: blip_value (160, 160) 每个像素与目标的相似度 [0, 1]
            # 原理: 视觉-语言对比学习，计算图像区域与文本描述的匹配度
            blip_value = self.value_map_module.get_blip_value(
                Image.fromarray(obs[0]['rgb']),  # 当前 RGB 观察
                self.destination                 # 目标描述 (如 "kitchen", "living room")
            )
            blip_value = blip_value.detach().cpu().numpy()
            
            # 5.2 融合多种信息生成价值图
            # value_map_module() 执行:
            #   ① 将 blip_value 投影到地图坐标系 (160×160 → 480×480)
            #   ② 结合语义通道 (目标类别的掩码，如 "kitchen" 通道)
            #   ③ 应用距离衰减 (离目标越远价值越低)
            #   ④ 排除碰撞区域 (价值置零)
            #   ⑤ 融合新探索区域奖励 (鼓励探索)
            # 返回: value_map (2, 480, 480)
            #   [0]: 原始价值图
            #   [1]: 处理后价值图 (会在后续乘以 history_map 和 direction_map)
            value_map = self.value_map_module(
                step,                    # 当前步数
                full_map[0],            # 完整语义地图 (N+4, 480, 480)
                self.floor,             # 地板区域 (480, 480)
                self.one_step_floor,    # 当前步新探索地板 (480, 480)
                self.collision_map,     # 碰撞地图 (480, 480)
                blip_value,             # BLIP 相似度 (160, 160)
                full_pose[0],           # 当前位姿 [x, y, heading]
                self.detected_classes,  # 已检测类别列表
                self.current_episode_id # 当前 episode ID (用于可视化)
            )
        
        # ========== 环视结束：规划初始动作 ==========
        
        # 使用 FMM (Fast Marching Method) 路径规划算法
        # policy() 执行:
        #   ① 将价值图作为目标场 (高价值区域 = 目标)
        #   ② FMM 扩散算法计算距离场 (每个位置到高价值区域的距离)
        #   ③ 梯度下降找到最优路径
        #   ④ 根据路径方向生成动作
        # 返回: {"action": 0/1/2/3} 
        #   0=STOP, 1=FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT
        self._action = self.policy(
            self.value_map_module.value_map[1],  # 价值图 (480, 480)
            self.collision_map,                  # 碰撞地图 (480, 480)
            full_map[0],                        # 完整语义地图 (N+4, 480, 480)
            self.floor,                         # 地板区域 (480, 480)
            self.traversible,                   # 可穿越区域 (480, 480)
            full_pose[0],                       # 当前位姿 [x, y, heading]
            self.frontiers,                     # 探索边界 (480, 480)
            self.detected_classes,              # 已检测类别列表
            self.destination_class,             # 目标类别列表 (如 ["kitchen"])
            self.classes,                       # 当前要检测的类别列表
            False,                              # search_destination: 是否搜索最终目标
            one_step_full_map[0],              # 当前帧地图 (N+4, 480, 480)
            self.current_detections,            # 当前检测结果 (用于目标验证)
            self.current_episode_id,            # episode ID (用于可视化)
            False,                              # replan: 是否强制重新规划
            step                                # 当前步数
        )
        
        # 返回最终状态
        return full_pose, obs, dones, infos
    
    def _use_keyboard_control(self):
        """手动键盘控制（用于调试）"""
        a = input("action:")
        if a == 'w':
           return {"action": 1}  # 前进
        elif a == 'a':
            return {"action": 2}  # 左转
        elif a == 'd':
            return {"action": 3}  # 右转
        else:
            return {"action": 0}  # 停止
    
    def reset(self) -> None:
        """重置所有状态，准备下一个 episode"""
        self.classes = []
        self.current_detections = None
        self.detected_classes = OrderedSet()
        self.floor = np.zeros(self.map_shape)
        self.one_step_floor = np.zeros(self.map_shape)
        self.frontiers = np.zeros(self.map_shape)
        self.traversible = np.zeros(self.map_shape)
        self.collision_map = np.zeros(self.map_shape)
        self.visited = np.zeros(self.map_shape)
        self.base_classes = copy.deepcopy(base_classes)
        
        # 重置所有模块
        self.policy.reset()
        self.mapping_module.reset()
        self.value_map_module.reset()
        self.history_module.reset()
    
    def rollout(self):
        """执行一个完整的 episode（包含多个子任务）
        
        这是 VLN 任务的核心执行函数，处理从初始化到完成的整个导航过程。
        支持复杂的自然语言指令，通过子任务分解、约束监控、价值图融合等机制
        实现鲁棒的室内导航。
        
        主要流程：
        ┌─────────────────────────────────────────┐
        │ 1. 初始化地图 (_maps_initialization)   │
        │    - 重置环境                           │
        │    - 解析 LLM 指令                      │
        │    - 初始化语义地图                     │
        └─────────────────────────────────────────┘
                        ↓
        ┌─────────────────────────────────────────┐
        │ 2. 环视建图 (_look_around)             │
        │    - 旋转 360° (12 步 × 30°)           │
        │    - 语义分割 + 地图构建                │
        │    - 规划初始动作                       │
        └─────────────────────────────────────────┘
                        ↓
        ┌─────────────────────────────────────────┐
        │ 3. 主导航循环 (步数 12-500)            │
        │    ├─ 更新轨迹点                        │
        │    ├─ 计算历史/方向约束地图             │
        │    ├─ 检查子任务约束                    │
        │    ├─ 切换子任务（如需要）              │
        │    ├─ 执行动作                          │
        │    ├─ 更新语义地图                      │
        │    ├─ 碰撞检测与异常恢复                │
        │    ├─ 计算价值图（BLIP）                │
        │    └─ FMM 路径规划                      │
        └─────────────────────────────────────────┘
        
        约束管理状态机：
        EXECUTING → (约束满足) → WAITING → (最小步数) → SWITCH_TASK
        
        异常恢复机制：
        - 连续碰撞 30 步 → 重新规划 (replan=True)
        - 价值图空 5 次 → 重新环视 360°
        - 超过最大约束步数 → 强制切换下一子任务
        
        Returns:
            None (结果通过 self._calculate_metric 记录)
        """
        # ═══════════════════════════════════════════════════════════════════
        # 阶段 1: 初始化与环视建图
        # ═══════════════════════════════════════════════════════════════════
        
        # 初始化语义地图，解析 LLM 指令分解结果
        # 调用: envs.reset() → _process_llm_reply() → mapping_module.init_map_and_pose()
        self._maps_initialization()
        
        # 环视 360° 建立初始地图 (12 步 × 30° = 360°)
        # 返回: 当前位姿、观察、结束标志、附加信息
        full_pose, obs, dones, infos = self._look_around()
        print("\n ========== START TO NAVIGATE ==========\n")
        
        # ═══════════════════════════════════════════════════════════════════
        # 阶段 2: 初始化导航状态变量
        # ═══════════════════════════════════════════════════════════════════
        
        # --- 轨迹追踪 ---
        trajectory_points = []  # 存储最近 2 个位置点 [(y1,x1), (y2,x2)]
                                # 用途: HistoryMap 模块，避免原地徘徊
                                # 在价值图上降低已访问区域的价值
        
        direction_points = []   # 存储最近 5 个位置点 [array([x1,y1]), ...]
                                # 用途: DirectionMap 模块，处理方向约束
                                # 例如指令"turn left"时，检查移动方向向量
        
        # --- 约束管理 ---
        constraint_steps = 0    # 当前子任务已执行的步数计数器
                                # 用途: 判断是否达到切换子任务的条件
                                # 范围: [MIN_CONSTRAINT_STEPS, MAX_CONSTRAINT_STEPS]
        
        start_to_wait = False   # 约束满足后的等待标志
                                # True: 约束已满足，等待最小步数后切换
                                # False: 正在执行约束
        
        search_destination = False  # 是否到达最后一个子任务标志
                                    # True: 开始搜索最终目标位置
                                    # False: 还在执行中间子任务
        
        # --- 异常恢复 ---
        collided = 0            # 连续碰撞/卡住的步数计数器
                                # ≥30: 触发重新规划 (replan=True)
                                # <0.2m/步 判定为卡住
        
        empty_value_map = 0     # 价值图为空的连续次数
                                # ≥5: 触发重新环视 360°
                                # ≤24×24 像素判定为空
        
        replan = False          # 是否需要重新规划路径标志
                                # True: 传递给 policy，强制重新计算路径
        
        # --- 方向约束 ---
        direction_map = np.ones(self.map_shape)  # (480, 480) 方向约束掩码
                                                 # 全1: 无方向限制
                                                 # 部分0: 屏蔽不符合方向的区域
        
        direction_map_exist = False  # 方向地图是否已计算标志
                                     # 避免重复计算相同的方向约束
        
        # --- 位姿追踪 ---
        last_action = None      # 上一步执行的动作 {"action": 0/1/2/3}
        current_action = None   # 当前步执行的动作
                                # 用途: 只在 FORWARD 动作时更新碰撞地图
        
        last_pose = None        # 上一步的位姿 [x, y, θ]
        current_pose = full_pose[0]  # 当前位姿 [x, y, θ] (米, 米, 弧度)
                                     # 用途: 计算位移，检测是否卡住
        
        start_check_pose = None # 开始检查方向约束时的位姿
                                # 用途: 计算从起始位置转过的角度
        
        self._action2 = None    # 键盘手动控制动作 (调试用)
        
        # ═══════════════════════════════════════════════════════════════════
        # 阶段 3: 获取第一个子任务信息
        # ═══════════════════════════════════════════════════════════════════
        
        # 找到第一个未完成的子任务索引 (初始全为 False，返回 0)
        current_idx = self.constraints_check.index(False)
        
        # 获取该子任务的 landmark 决策
        # 例如: [("kitchen", "move towards"), ("table", "move away")]
        landmarks = self.decisions[str(current_idx)]['landmarks']
        
        # 提取目标类别名称 (去掉决策动作)
        # 例如: ["kitchen", "table"]
        self.destination_class = [item[0] for item in landmarks]
        
        # 更新需要检测的类别列表 (基础类别 + 目标类别)
        # base_classes: ["floor", "wall", "door", ...] (常见室内物体)
        # 目标类别会被移到列表末尾，提高检测优先级
        self.classes = self._process_classes(self.base_classes, self.destination_class)
        
        # 获取当前子任务的约束条件
        # 例如: [("direction constraint", "turn left"), ("landmark constraint", "kitchen")]
        current_constraint = self.sub_constraints[str(current_idx)]
        
        # 提取约束类型列表
        # 例如: ["direction constraint", "landmark constraint"]
        all_constraint_types = [item[0] for item in current_constraint]
        
        # ═══════════════════════════════════════════════════════════════════
        # 阶段 4: 主导航循环 (步数 12-500)
        # ═══════════════════════════════════════════════════════════════════
        for step in range(12, self.max_step):
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.1 打印当前状态 (调试信息)                                 │
            # └─────────────────────────────────────────────────────────────┘
            print(f"\nepisode:{self.current_episode_id}, step:{step}")
            print(f"instr: {self.instruction}")  # 完整指令
            print(f"sub_instr_{current_idx}: {self.sub_instructions[current_idx]}")  # 当前子指令
            
            # 约束步数计数器递增
            constraint_steps += 1
            
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.2 更新位置和轨迹记录                                       │
            # └─────────────────────────────────────────────────────────────┘
            
            # 将位姿从米转换为像素坐标
            # full_pose[0][:2] = [x_m, y_m] → position = [x_px, y_px]
            position = full_pose[0][:2] * 100 / self.resolution  # 米 → 厘米 → 像素
            heading = full_pose[0][-1]  # 朝向角度 (弧度)
            print("full pose: ", full_pose[0])  # [x, y, θ] (米, 米, 弧度)
            
            # 转换坐标并限制在地图范围内
            # 注意: position[0]=x, position[1]=y，但地图坐标是 (y, x)
            y = min(int(position[0]), self.map_shape[0] - 1)  # 限制 0~479
            x = min(int(position[1]), self.map_shape[1] - 1)  # 限制 0~479
            
            # 标记当前位置为已访问
            self.visited[x, y] = 1
            
            # 更新轨迹点列表 (用于历史地图)
            trajectory_points.append((y, x))
            if len(trajectory_points) > 2:
                del trajectory_points[0]  # 保持最多 2 个点: [前一步, 当前步]
            
            # 更新方向点列表 (用于方向约束)
            direction_points.append(np.array([x, y]))
            if len(direction_points) > 5:
                del direction_points[0]  # 保持最多 5 个点，计算移动趋势
            
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.3 计算历史地图 (避免原地徘徊)                              │
            # └─────────────────────────────────────────────────────────────┘
            # HistoryMap: 在最近访问的区域绘制惩罚值
            # 原理: 连接 trajectory_points 的两个点，画一条直线
            #       在直线周围区域的价值图上乘以衰减系数 (如 0.5)
            # 效果: 智能体倾向于探索新区域，而非原路返回
            history_map = self.history_module(trajectory_points, step, self.current_episode_id)

            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.4 方向约束处理                                             │
            # └─────────────────────────────────────────────────────────────┘
            # 如果有方向约束 (如 "turn left")，记录起始位姿用于角度计算
            if "direction constraint" in all_constraint_types and start_check_pose is None:
                start_check_pose = full_pose[0]  # 记录开始检查时的位姿
            
            # 检查是否到达最后一个子任务
            if int(current_idx) >= len(self.sub_instructions) - 1:
                search_destination = True  # 开始搜索最终目标
                print("start to search destination")
                

            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.5 约束检查和子任务切换 (状态机核心逻辑)                    │
            # └─────────────────────────────────────────────────────────────┘
            # 只在还有未完成的子任务时执行
            if sum(self.constraints_check) < len(self.sub_instructions):
                
                # ┈┈┈┈┈ 4.5.1 计算方向约束地图 ┈┈┈┈┈
                # 如果当前约束包含方向约束且尚未计算
                if (len(current_constraint) > 0 
                    and current_constraint[0][0] == "direction constraint" 
                    and not direction_map_exist):
                    
                    # 提取方向类型 (如 "turn left", "turn right", "go straight")
                    direction = current_constraint[0][1]
                    
                    # 获取当前位置和 5 步前的位置 (用于计算移动向量)
                    if len(direction_points) < 5:
                        # 步数不足，使用当前位置
                        current_position = direction_points[-1]
                        last_five_position = direction_points[-1]
                    else:
                        # 使用最近 5 步的首尾位置
                        current_position = direction_points[-1]      # 当前
                        last_five_position = direction_points[0]     # 5步前
                    
                    # 调用 DirectionMap 模块计算方向掩码
                    # 原理: 根据移动向量和朝向角，判断是否满足方向要求
                    #   - "turn left": 屏蔽右侧和正前方区域
                    #   - "turn right": 屏蔽左侧和正前方区域
                    #   - "go straight": 屏蔽左右两侧区域
                    direction_map = self.direction_module(
                        current_position, last_five_position, heading,
                        direction, step, self.current_episode_id
                    )
                    direction_map_exist = True  # 标记已计算
                else:
                    # 无方向约束，地图全为 1 (不限制)
                    direction_map = np.ones(self.map_shape)
                
                # ┈┈┈┈┈ 4.5.2 检查约束是否满足 ┈┈┈┈┈
                # ConstraintsMonitor: 检查每个约束条件
                # 返回: [True, False, True, ...] 布尔列表
                # 约束类型:
                #   - "direction constraint": 检查转向角度
                #   - "landmark constraint": 检查是否看到目标物体
                #   - "distance constraint": 检查与landmark的距离
                check = self.constraints_monitor(
                    current_constraint,       # 当前约束列表
                    obs[0],                   # 当前观察
                    self.current_detections,  # 当前检测到的物体
                    self.classes,             # 检测类别列表
                    current_pose,             # 当前位姿
                    start_check_pose          # 开始检查时的位姿
                )
                print(current_constraint, check)  # 调试: 打印约束和检查结果
                
                # ┈┈┈┈┈ 4.5.3 处理方向约束满足 ┈┈┈┈┈
                # 如果方向约束已满足，重置方向地图 (解除限制)
                if (len(current_constraint) > 0 
                    and current_constraint[0][0] == "direction constraint" 
                    and check[0] == True):
                    direction_map = np.ones(self.map_shape)
                    direction_map_exist = False  # 允许下次重新计算
                
                # ┈┈┈┈┈ 4.5.4 更新未满足的约束 ┈┈┈┈┈
                if len(check) == 0:
                    # 空约束列表
                    print("empty constraint")
                elif sum(check) < len(check):
                    # 部分约束未满足，只保留未满足的
                    # 例如: constraints = [C1, C2, C3], check = [True, False, True]
                    #       → constraints = [C2] (只保留 C2)
                    current_constraint = [
                        current_constraint[i] 
                        for i in range(len(current_constraint)) 
                        if not check[i]  # 保留 check[i] == False 的约束
                    ]
                    all_constraint_types = [item[0] for item in current_constraint]
                
                # ┈┈┈┈┈ 4.5.5 判断是否进入等待状态 ┈┈┈┈┈
                # 满足条件:
                #   1. 所有约束都满足 (sum(check) == len(check))
                #   2. 或超过最大约束步数 (constraint_steps >= max)
                if (sum(check) == len(check) or 
                    constraint_steps >= self.max_constraint_steps):
                    if not start_to_wait:
                        start_to_wait = True  # 进入等待状态
                        self.constraints_check[current_idx] = True  # 标记子任务完成
                
                # ┈┈┈┈┈ 4.5.6 切换到下一个子任务 ┈┈┈┈┈
                # 满足条件:
                #   1. 已进入等待状态 (start_to_wait == True)
                #   2. 达到最小约束步数 (constraint_steps >= min)
                # 原因: 避免子任务切换过快，确保每个子任务执行一定时间
                if start_to_wait and (constraint_steps >= self.min_constraint_steps):
                    if False in self.constraints_check:
                        # 还有未完成的子任务，切换到下一个
                        current_idx = self.constraints_check.index(False)
                        print(f"sub_instr_{current_idx}: {self.sub_instructions[current_idx]}")
                        
                        # 更新新子任务的目标类别
                        landmarks = self.decisions[str(current_idx)]['landmarks']
                        if len(landmarks) > 0:
                            self.destination_class = [item[0] for item in landmarks]
                            self.classes = self._process_classes(
                                self.base_classes, self.destination_class
                            )
                        
                        # 更新新子任务的约束
                        current_constraint = self.sub_constraints[str(current_idx)]
                        all_constraint_types = [item[0] for item in current_constraint]
                        
                        # 重置位姿检查点
                        current_pose, start_check_pose = None, None
                    else:
                        # 所有子任务都完成
                        current_constraint, all_constraint_types = [], []
                        print("all constraints are done")
                    
                    # 重置约束步数和等待标志
                    constraint_steps = 0
                    start_to_wait = False
                    
            # 打印当前状态 (调试)
            print("current constraint: ", current_constraint)
            print("constraint_steps: ", constraint_steps)
                
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.6 更新导航目标                                             │
            # └─────────────────────────────────────────────────────────────┘
            # 根据当前约束更新 self.destination (用于 BLIP 查询)
            
            # 如果有非方向约束，使用约束中的 landmark 作为目标
            if len(current_constraint) > 0 and current_constraint[0][0] != "direction constraint":
                new_destination = current_constraint[0][1]  # 例如: "kitchen"
                
                # 如果是最后一个子任务，使用最终目标
                if current_idx >= len(self.sub_instructions) - 1:
                    self.destination = self.llm_reply['destination']  # LLM解析的最终目标
                else:
                    self.destination = new_destination  # 子任务的中间目标
            
            # 如果所有约束都完成且是最后一个子任务，使用最终目标
            if len(current_constraint) == 0 and current_idx >= len(self.sub_constraints) - 1:
                self.destination = self.llm_reply['destination']
                
            # 目标变化时，衰减价值图 (避免旧目标的影响)
            # 乘以 0.5: 保留部分历史信息，但降低其权重
            if self.destination != self.last_destination:
                self.value_map_module.value_map[...] *= 0.5
                self.last_destination = self.destination
                
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.7 异常恢复: 价值图为空检测                                 │
            # └─────────────────────────────────────────────────────────────┘
            # 价值图为空表示找不到导航目标，可能原因:
            #   1. 目标不在当前视野内
            #   2. 目标被遮挡或未被检测到
            #   3. 语义分割失败
            
            # 统计价值图中非零像素数量
            # 阈值: 24×24 = 576 像素 (约 1.2m × 1.2m 的区域)
            if np.sum(self.value_map_module.value_map[1].astype(bool)) <= 24**2:
                empty_value_map += 1  # 连续为空的次数
                constraint_steps = 0  # 重置约束步数 (不计入无效步数)
            else:
                empty_value_map = 0   # 重置计数器
            
            # 连续 5 次为空，触发重新环视
            if empty_value_map >= 5:
                print(f"[WARNING] Value map empty for {empty_value_map} steps, re-looking around...")
                full_pose, obs, dones, infos = self._look_around()  # 重新环视 360°
                
                # 检查环视后 episode 是否结束
                if dones[0]:
                    self._calculate_metric(infos)
                    break
                
                # 重置计数器
                empty_value_map = 0
                constraint_steps = 0
            
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.8 执行动作                                                 │
            # └─────────────────────────────────────────────────────────────┘
            actions = []
            for _ in range(self.config.NUM_ENVIRONMENTS):
                if self.keyboard_control:
                    # 手动控制模式 (调试用)
                    self._action2 = self._use_keyboard_control() 
                    actions.append(self._action2)
                else:
                    # 使用策略规划的动作
                    # self._action 在上一轮的最后或 _look_around() 中计算
                    actions.append(self._action)
            
            # 在仿真环境中执行动作
            outputs = self.envs.step(actions)
            obs, _, dones, infos = [list(x) for x in zip(*outputs)]
            
            # Save RGB frames if print_images is enabled
            if self.config.MAP.PRINT_IMAGES:
                rgb_frame = obs[0]['rgb'].astype(np.uint8)  # Get RGB from observation
                save_dir = os.path.join(self.config.RESULTS_DIR, "rgb_frames/eps_%d"%self.current_episode_id)
                os.makedirs(save_dir, exist_ok=True)
                fn = "{}/step-{}.png".format(save_dir, step)
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fn, bgr_frame)
            
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.9 检查 episode 是否结束                                   │
            # └─────────────────────────────────────────────────────────────┘
            if dones[0]:
                # Episode 结束原因可能是:
                #   1. 到达目标 (Success)
                #   2. 超时 (达到最大步数 500)
                #   3. 调用了 STOP 动作
                self._calculate_metric(infos)  # 计算评估指标
                break  # 退出导航循环
            
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.10 更新语义地图                                            │
            # └─────────────────────────────────────────────────────────────┘
            # 处理新观察: RGB-D → 语义分割 → 点云 → 地图投影
            batch_obs = self._batch_obs(obs)  # 预处理观察 (包含 GroundedSAM 分割)
            
            # 获取相对位姿变化
            poses = torch.from_numpy(
                np.array([item['sensor_pose'] for item in obs])
            ).float().to(self.device)  # [Δx, Δy, Δθ]
            
            # 调用 Mapping 模块前向传播
            self.mapping_module(batch_obs, poses)
            
            # 更新全局地图并获取当前状态
            full_map, full_pose, one_step_full_map = \
                self.mapping_module.update_map(step, self.detected_classes, self.current_episode_id)
            
            # 清空单步地图 (准备下一步)
            self.mapping_module.one_step_full_map.fill_(0.)
            self.mapping_module.one_step_local_map.fill_(0.)
            
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.11 处理导航地图 (提取可导航信息)                           │
            # └─────────────────────────────────────────────────────────────┘
            # 从语义地图中提取:
            #   - traversible: 可穿越区域 (无障碍物和不可导航物体)
            #   - floor: 地板区域 (可行走的平面)
            #   - frontiers: 探索边界 (已探索区域的轮廓)
            self.traversible, self.floor, self.frontiers = self._process_map(step, full_map[0])
            
            # Save floor map visualization if print_images is enabled
            if self.config.MAP.PRINT_IMAGES:
                self._save_floor_semantic_map(step, self.current_episode_id, full_map[0])
            
            # 处理当前步新探索的地板 (用于价值图的探索奖励)
            self.one_step_floor = self._process_one_step_floor(one_step_full_map[0])
            
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.12 异常恢复: 碰撞检测与卡住处理                            │
            # └─────────────────────────────────────────────────────────────┘
            # 检测智能体是否被卡住 (连续多步位移很小)
            
            # 保存上一步位姿
            last_pose = current_pose
            current_pose = full_pose[0]  # 更新当前位姿
            
            if last_pose is not None and current_pose is not None:
                # 计算两步之间的位移 (欧氏距离，单位: 像素)
                displacement = calculate_displacement(last_pose, current_pose, self.resolution)
                
                # 阈值: 0.2m = 20cm = 4 像素 (5cm/像素)
                # 如果位移 < 0.2m，认为是卡住或碰撞
                if displacement < 0.2 * 100 / self.resolution:  # 0.2m → 4 pixels
                    collided += 1  # 累计卡住步数
                else:
                    # 移动正常，重置计数器
                    collided = 0
                    replan = False
                
                # 连续卡住 30 步，触发重新规划
                if collided >= 30:
                    replan = True  # 告诉 policy 强制重新规划路径
                    print(f"[WARNING] {self.current_episode_id}: Stuck for {collided} steps\n")
                    
                    # 记录日志 (调试用)
                    fname = os.path.join(
                        self.config.EVAL_CKPT_PATH_DIR, 
                        f"r{self.local_rank}_w{self.world_size}_collision_stuck.txt"
                    )
                    with open(fname, "a") as f:
                        f.writelines(
                            f"id: {str(self.current_episode_id)}; "
                            f"step: {str(step)}; "
                            f"collided: {str(collided)}\n"
                        )
                
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.13 更新碰撞地图                                            │
            # └─────────────────────────────────────────────────────────────┘
            # 只在执行 FORWARD 动作时更新碰撞地图
            # 原因: TURN_LEFT/TURN_RIGHT 不会产生碰撞
            
            last_action = current_action
            current_action = self._action
            
            if last_pose is not None and current_action["action"] == 1:  # 1 = MOVE_FORWARD
                # 使用 FMM 算法检测从 last_pose 到 current_pose 的路径上是否有碰撞
                # 原理: 如果规划的路径和实际位移不符，说明发生了碰撞
                collision_map = collision_check_fmm(
                    last_pose, 
                    current_pose, 
                    self.resolution, 
                    self.mapping_module.map_shape
                )
                
                # 累积碰撞地图 (逻辑或运算)
                # 一旦某个位置被标记为碰撞，后续会持续避开
                self.collision_map = np.logical_or(self.collision_map, collision_map)
            
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.14 计算价值图 (BLIP 视觉-语言匹配)                         │
            # └─────────────────────────────────────────────────────────────┘
            # BLIP (Bootstrapped Language-Image Pre-training):
            #   - 输入: RGB 图像 + 目标文本 (如 "kitchen")
            #   - 输出: 每个像素与目标的语义相似度 [0, 1]
            #   - 原理: 视觉-语言对比学习
            
            blip_value = self.value_map_module.get_blip_value(
                Image.fromarray(obs[0]['rgb']),  # 当前 RGB 观察
                self.destination                 # 目标描述
            )
            blip_value = blip_value.detach().cpu().numpy()  # (160, 160)
            
            # 融合多种信息生成价值图
            # 输入:
            #   - blip_value: BLIP 语义相似度
            #   - full_map: 语义地图 (目标类别的掩码)
            #   - floor: 地板区域
            #   - one_step_floor: 新探索区域
            #   - collision_map: 碰撞区域
            # 输出:
            #   - value_map[0]: 原始价值图
            #   - value_map[1]: 处理后价值图 (会在下一步乘以 history_map 和 direction_map)
            value_map = self.value_map_module(
                step, 
                full_map[0], 
                self.floor, 
                self.one_step_floor, 
                self.collision_map, 
                blip_value, 
                full_pose[0], 
                self.detected_classes, 
                self.current_episode_id
            )
            
            # ┌─────────────────────────────────────────────────────────────┐
            # │ 4.15 路径规划 (FMM 算法)                                     │
            # └─────────────────────────────────────────────────────────────┘
            # Fast Marching Method (FMM):
            #   ① 将价值图作为目标场 (高价值区域 = 目标)
            #   ② FMM 扩散算法计算距离场 (每个位置到高价值区域的距离)
            #   ③ 梯度下降找到最优路径
            #   ④ 根据路径方向生成动作
            
            # 融合约束地图
            # value_map[1] * history_map * direction_map
            #   - history_map: 降低已访问区域的价值
            #   - direction_map: 屏蔽不符合方向约束的区域 (如果有)
            final_value_map = self.value_map_module.value_map[1] * history_map
            
            # 注意: direction_map 在当前实现中未直接乘入，而是在 policy 中处理
            # 如果需要应用方向约束，可以取消下面这行的注释:
            # final_value_map = final_value_map * direction_map
            
            self._action = self.policy(
                final_value_map,              # 融合后的价值图 (480, 480)
                self.collision_map,           # 碰撞地图 (480, 480)
                full_map[0],                  # 完整语义地图 (N+4, 480, 480)
                self.floor,                   # 地板区域 (480, 480)
                self.traversible,             # 可穿越区域 (480, 480)
                full_pose[0],                 # 当前位姿 [x, y, heading]
                self.frontiers,               # 探索边界 (480, 480)
                self.detected_classes,        # 已检测类别列表
                self.destination_class,       # 目标类别列表
                self.classes,                 # 当前要检测的类别列表
                search_destination,           # 是否搜索最终目标
                one_step_full_map[0],        # 当前帧地图 (N+4, 480, 480)
                self.current_detections,      # 当前检测结果
                self.current_episode_id,      # episode ID (用于可视化)
                replan,                       # 是否强制重新规划
                step                          # 当前步数
            )
            # self._action = {"action": 0/1/2/3}
            #   0 = STOP (到达目标或放弃)
            #   1 = MOVE_FORWARD (前进)
            #   2 = TURN_LEFT (左转 30°)
            #   3 = TURN_RIGHT (右转 30°)
    
    def eval(self):
        """评估主函数
        
        流程：
        1. 设置评估配置
        2. 初始化环境
        3. 加载真实轨迹
        4. 初始化策略模块
        5. 循环评估每个 episode
        6. 保存结果到 JSON
        """
        # ===== 初始化 =====
        self._set_eval_config()  # 设置配置
        self._init_envs()  # 初始化 Habitat 环境
        self._collect_val_traj()  # 加载真实轨迹（用于计算 NDTW）
        self._initialize_policy()  # 初始化所有策略模块
        
        # ===== 确定要评估的 episode 数量 =====
        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)  # 评估所有分配的 episodes
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
            
        # ===== 循环评估每个 episode =====
        self.state_eps = {}  # 存储每个 episode 的评估结果
        t1 = time.time()
        for i in tqdm(range(eps_to_eval)):
            self.rollout()  # 执行一个完整 episode
            self.reset()  # 重置状态
                    
        self.envs.close()
        
        # ===== 保存结果 =====
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(self.config.EVAL_CKPT_PATH_DIR, 
                             f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json"
                             )
        with open(fname, "w") as f:
            json.dump(self.state_eps, f, indent=2)
        t2 = time.time()
        logger.info(f"time: {t2 - t1}s")
        print("test time: ", t2 - t1)