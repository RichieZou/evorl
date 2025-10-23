#!/bin/bash

# ==============================================================================
#           🚀 循环调度的极限测试脚本 🚀
# ==============================================================================
# 功能：
#   使用循环调度，每个参数组合独立运行，避免 GPU 内存累积问题
# ==============================================================================

# --- ⚙️ 1. 在这里配置您的实验 ---

# 定义要扫描的参数（使用空格分隔的数组）
ENVS=("brax/walker2d")
SEEDS=(42)

# 要测试的 rollout_length 值
ROLLOUT_LENGTHS=(1024 512 256 128 64)

# 要测试的不同 num_envs 值
NUM_ENVS_VALUES=(64 128 256 512 1024)

# 定义固定的参数
PYTHON_SCRIPT="scripts/train.py"
AGENT_CONFIG="ppo_grid_search"
PROJECT_NAME="test_gird_1024"
GPU_ID=3

# 默认的 minibatch_size
DEFAULT_MINIBATCH_SIZE=128

# --- 🚀 2. 执行逻辑 ---

echo "=============================================================================="
echo "Starting EXTREME TEST with Loop Scheduling..."
echo "Environments: ${ENVS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Testing rollout_lengths: ${ROLLOUT_LENGTHS[@]}"
echo "Testing num_envs: ${NUM_ENVS_VALUES[@]}"
echo "=============================================================================="
echo ""

# 计算总任务数
TOTAL_TASKS=$((${#ENVS[@]} * ${#SEEDS[@]} * ${#ROLLOUT_LENGTHS[@]} * ${#NUM_ENVS_VALUES[@]}))
CURRENT_TASK=0

# 将数组转换为 Hydra 接受的逗号分隔字符串
ENVS_STR=$(IFS=,; echo "${ENVS[*]}")
SEEDS_STR=$(IFS=,; echo "${SEEDS[*]}")

# 遍历每个参数组合
for ne in "${NUM_ENVS_VALUES[@]}"; do
    for rl in "${ROLLOUT_LENGTHS[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        
        echo "=============================================================================="
        echo "🚀 [Task ${CURRENT_TASK}/${TOTAL_TASKS}] Running: num_envs=${ne}, rollout_length=${rl}"
        echo "=============================================================================="
        
        # 为每个参数组合启动独立的 Python 进程
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PYTHON_SCRIPT} \
            --multirun \
            agent=${AGENT_CONFIG} \
            project=${PROJECT_NAME} \
            env=${ENVS_STR} \
            seed=${SEEDS_STR} \
            rollout_length=${rl} \
            num_envs=${ne} \
            minibatch_size=${DEFAULT_MINIBATCH_SIZE} \
            env.max_episode_steps=1024
        
        # 检查退出状态
        if [ $? -eq 0 ]; then
            echo "✅ Task ${CURRENT_TASK} completed successfully"
        else
            echo "❌ Task ${CURRENT_TASK} failed"
        fi
        
        echo ""
    done
done

echo "=============================================================================="
echo "✅ 所有实验已完成！总共完成 ${TOTAL_TASKS} 个任务"
echo "=============================================================================="

