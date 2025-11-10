import argparse
import random
import os
import json
from copy import deepcopy
import glob
from pprint import pprint

import numpy as np
import torch
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import Pool

from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.config.default import get_config
from vlnce_baselines.common.utils import seed_everything
    

def run_exp(exp_name, exp_config, run_type, nprocesses, opts):
    # ① 加载基础配置
    config = get_config(exp_config, opts)
    
    # ② 修改配置，添加实验名称后缀
    config.defrost()  # 解冻配置（允许修改）
    config.TENSORBOARD_DIR += exp_name      # data/tensorboard_dirs/exp_1
    config.CHECKPOINT_FOLDER += exp_name    # data/checkpoints/exp_1
    config.EVAL_CKPT_PATH_DIR += exp_name   # data/checkpoints/exp_1
    config.RESULTS_DIR += exp_name          # data/logs/eval_results/exp_1
    config.VIDEO_DIR += exp_name            # data/logs/video/exp_1
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE  # exp_1_log.txt
    config.freeze()  # 冻结配置（禁止修改）
    
    # ③ 创建必要目录
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.EVAL_CKPT_PATH_DIR, exist_ok=True)
    os.system("mkdir -p data/logs/running_log")
    
    # ④ 设置日志
    logger.add_filehandler('data/logs/running_log/' + config.LOG_FILE)
    logger.info(f"hyper parameters:\n{config.EVAL}")
    logger.info(f"llm reply file: {config.TASK_CONFIG.DATASET.LLM_REPLYS_PATH}")
    # dataset split, start multi-processes

    # ① 获取可用 GPU 数量
    num_devices = torch.cuda.device_count()  # 8 个 GPU
    print(f'num devices: {num_devices}, num processes: {nprocesses}')  # 8, 16
    
    # ② 加载 LLM 回复数据集
    with open(config.TASK_CONFIG.DATASET.LLM_REPLYS_PATH, 'r') as f:
        llm_reply_dataset = json.load(f)
    
    # ③ 提取所有 episode ID
    episode_ids = list(llm_reply_dataset.keys())  
    # 例如: ['1001', '1002', '1003', ..., '2000']  (假设 1000 个 episodes)
    
    # ④ 轮询分片 (Round-Robin)
    split_episode_ids = [episode_ids[i::nprocesses] for i in range(nprocesses)]
    # 进程 0: [1001, 1017, 1033, ...]  (每隔 16 个取一个)
    # 进程 1: [1002, 1018, 1034, ...]
    # ...
    # 进程 15: [1016, 1032, 1048, ...]

    configs = []
    for i, ep_ids in enumerate(split_episode_ids):
        # ① 深拷贝配置（避免进程间干扰）
        shared_config = deepcopy(config)
        shared_config.defrost()
        
        # ② 分配 GPU（循环分配）
        device_num = i % num_devices  # 16 个进程 → 8 个 GPU
        # 进程 0 → GPU 0
        # 进程 1 → GPU 1
        # ...
        # 进程 8 → GPU 0 (重新循环)
        
        # ③ 设置进程特定参数
        shared_config.local_rank = i              # 进程编号 (0-15)
        shared_config.world_size = nprocesses     # 总进程数 (16)
        shared_config.TORCH_GPU_ID = device_num   # PyTorch 使用的 GPU
        shared_config.TORCH_GPU_IDS = [device_num]
        shared_config.SIMULATOR_GPU_IDS = [device_num]  # Habitat 模拟器 GPU
        
        # ④ 分配该进程要处理的 episodes
        shared_config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = ep_ids
        
        shared_config.freeze()
        configs.append(shared_config)
    
    # 检测是否为键盘控制模式（单进程直接运行）
    if config.KEYBOARD_CONTROL:
        print("=" * 60)
        print("KEYBOARD CONTROL MODE - Running in main process")
        print("=" * 60)
        # 直接在主进程运行第一个配置（避免 multiprocessing input() 问题）
        worker(configs[0])
        print("\n" + "=" * 60)
        print("KEYBOARD CONTROL SESSION ENDED")
        print("=" * 60)
        return  # 键盘控制模式不需要聚合结果
    else:
        # ① 创建进程池 (16 个进程)
        pool = Pool(processes=nprocesses)
        
        # ② 并行执行 worker 函数
        pool.map(worker, configs)
        # 等价于:
        # for config in configs:
        #     worker(config)  # 但是是并行的
        
        # ③ 关闭进程池
        pool.close()  # 不再接受新任务
        pool.join()   # 等待所有任务完成

    # ① 找到所有进程输出的结果文件
    fns = glob.glob(config.CHECKPOINT_FOLDER + '/stats_ep_ckpt_*.json')
    # 匹配文件:
    # - stats_ep_ckpt_val_unseen_r0_w16.json   (进程 0 的结果)
    # - stats_ep_ckpt_val_unseen_r1_w16.json   (进程 1 的结果)
    # - ...
    # - stats_ep_ckpt_val_unseen_r15_w16.json  (进程 15 的结果)
    
    # ② 合并所有 episode 的结果
    summary = {}
    for fn in fns:
        with open(fn, 'r') as f:
            summary.update(json.load(f))
    # summary = {
    #   '1001': {'success': 1.0, 'spl': 0.85, ...},
    #   '1002': {'success': 0.0, 'spl': 0.00, ...},
    #   ...
    # }
    
    # ③ 初始化指标累积列表
    summary_metrics = {
        "steps_taken": [],
        "distance_to_goal": [],
        "success": [],
        "oracle_success": [],
        "path_length": [],
        "spl": [],
        "ndtw": [],
        "sdtw": [],
    }
    
    # ④ 收集所有 episode 的指标
    for epid, metric in summary.items():
        for k, v in metric.items():
            summary_metrics[k].append(v)
    
    # ⑤ 计算平均指标
    for k, v in summary_metrics.items():
        summary_metrics[k] = np.mean(v)
    
    # ⑥ 打印并保存最终结果
    pprint(summary_metrics)
    # {
    #   'success': 0.45,
    #   'spl': 0.38,
    #   'ndtw': 0.52,
    #   ...
    # }
    
    with open(config.CHECKPOINT_FOLDER + '/stats_ckpt_val_unseen.json', 'w') as f:
        json.dump(summary_metrics, f, indent=2)
        
def worker(config):
    seed_everything(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    TRAINER = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert TRAINER is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = TRAINER(config)
    trainer.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["eval"],
        required=True,
        help="run type of the experiment(train, eval, inference), only eval for zero-shot vln",
    )
    parser.add_argument(
        "--nprocesses",
        type=int,
        default=1,
        help="number of processes",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    print(args)
    
    mp.set_start_method('spawn', force=True)
    run_exp(**vars(args))