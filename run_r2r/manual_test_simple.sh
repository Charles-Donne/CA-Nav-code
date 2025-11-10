#!/bin/bash
# 手动键盘控制测试（最简版）
# 单GPU，Episode 701

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

echo "=========================================="
echo "🎮 手动键盘控制模式"
echo "📍 Episode: 701"
echo ""
echo "按键说明:"
echo "  w - 前进 (MOVE_FORWARD)"
echo "  a - 左转 30° (TURN_LEFT)"
echo "  d - 右转 30° (TURN_RIGHT)"
echo "  其他键 - 停止 (STOP)"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=0 python run_mp.py \
    --exp_name manual_test_701 \
    --run-type eval \
    --exp-config vlnce_baselines/config/exp1.yaml \
    --nprocesses 1 \
    NUM_ENVIRONMENTS 1 \
    TRAINER_NAME ZS-Evaluator-mp \
    TORCH_GPU_IDS [0] \
    SIMULATOR_GPU_IDS [0] \
    KEYBOARD_CONTROL 1 \
    TASK_CONFIG.DATASET.EPISODES_ALLOWED [701] \
    MAP.VISUALIZE True \
    MAP.PRINT_IMAGES False
