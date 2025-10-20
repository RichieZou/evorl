#!/bin/bash

# ==============================================================================
#           🚀 使用 Hydra Multirun 的极限测试脚本 🚀
# ==============================================================================
# 功能：
#   使用 Hydra 的 multirun 功能同时扫描 num_envs 和 rollout_length
#   自动计算 minibatch_size 并过滤无效组合
# ==============================================================================

# --- ⚙️ 1. 在这里配置您的实验 ---

# 定义要扫描的参数
ENVS="brax/walker2d,brax/ant"
SEEDS="42"

# 要测试的 rollout_length 值（添加了遍历）
ROLLOUT_LENGTHS="256,512,1024"

# 要测试的不同 num_envs 值
NUM_ENVS_VALUES="1024,2048,4096,8192,16384,32768,65536"

# 定义固定的参数
PYTHON_SCRIPT="scripts/train_20_iteration.py"
AGENT_CONFIG="ppo"
PROJECT_NAME="EVORL_PPO_extreme_grid"
GPU_ID=1

# 默认的 minibatch_size（可以在 Hydra 中动态计算）
DEFAULT_MINIBATCH_SIZE=256

# --- 🚀 2. 执行逻辑 ---

echo "=============================================================================="
echo "Starting EXTREME TEST with Hydra Multirun..."
echo "Environments: ${ENVS}"
echo "Seeds: ${SEEDS}"
echo "Testing rollout_lengths: ${ROLLOUT_LENGTHS}"
echo "Testing num_envs: ${NUM_ENVS_VALUES}"
echo "=============================================================================="
echo ""

# 使用 Hydra 的 multirun 模式进行参数扫描
# Hydra 会自动为每个参数组合创建一个独立的运行
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PYTHON_SCRIPT} \
    --multirun \
    agent=${AGENT_CONFIG} \
    project=${PROJECT_NAME} \
    env=${ENVS} \
    seed=${SEEDS} \
    rollout_length=${ROLLOUT_LENGTHS} \
    num_envs=${NUM_ENVS_VALUES} \
    minibatch_size=${DEFAULT_MINIBATCH_SIZE}

echo ""
echo "=============================================================================="
echo "✅ 所有 Hydra multirun 实验已完成！"
echo "=============================================================================="

