#!/bin/bash
# 手动键盘控制测试脚本（单GPU）
# 用法: bash run_r2r/manual_test.sh [episode_id]
# 例如: bash run_r2r/manual_test.sh 701

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# 默认 episode 为 701，可通过参数修改
EPISODE_ID=${1:-701}

echo "=========================================="
echo "手动键盘控制测试"
echo "Episode ID: $EPISODE_ID"
echo "按键: w=前进, a=左转, d=右转, 其他=停止"
echo "=========================================="

flag=" --exp_name manual_test
      --run-type eval
      --exp-config vlnce_baselines/config/exp1.yaml
      --nprocesses 1
      NUM_ENVIRONMENTS 1
      TRAINER_NAME ZS-Evaluator-mp
      TORCH_GPU_IDS [0]
      SIMULATOR_GPU_IDS [0]
      KEYBOARD_CONTROL 1
      TASK_CONFIG.DATASET.EPISODES_ALLOWED [$EPISODE_ID]
      "

CUDA_VISIBLE_DEVICES=0 python run_mp.py $flag
