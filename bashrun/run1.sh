#!/bin/bash

# ==============================================================================
#           🚀 通用 Hydra 实验自动化脚本 🚀
# ==============================================================================
# 功能：
#   1. 自动遍历所有 num_envs 和 rollout_length 的组合。
#   2. 根据预设逻辑动态计算正确的 minibatch_size。
#   3. 自动跳过无效的参数组合。
#   4. 为每个有效的组合启动一次 Hydra multirun，由 Hydra 负责 env 和 seed 的扫描。
# ==============================================================================

# --- ⚙️ 1. 在这里配置您的实验 ---


# 定义要扫描的参数 (使用空格分隔)
ENVS=("brax/walker2d" "brax/ant")
SEEDS=(42 43 44)
ROLLOUT_LENGTHS=(1 4 16 64 256 1024)
NUM_ENVS=(2 8 32 128 512 2048)

# 定义固定的参数
PYTHON_SCRIPT="scripts/train_100_iteration.py"
AGENT_CONFIG="ppo_record"
PROJECT_NAME="EVORL_PPO_record"
GPU_ID=1

# 逻辑计算用的常量
DEFAULT_MINIBATCH_SIZE=256

# --- 🚀 2. 执行逻辑 (通常无需修改) ---

# 将Bash数组转换为Hydra接受的逗号分隔字符串
ENVS_STR=$(IFS=,; echo "${ENVS[*]}")
SEEDS_STR=$(IFS=,; echo "${SEEDS[*]}")

echo "Starting automated grid search..."
echo "Environments: ${ENVS_STR}"
echo "Seeds: ${SEEDS_STR}"
echo "=============================================================================="

# 遍历需要计算依赖关系的参数
for ne in "${NUM_ENVS[@]}"; do
    for rl in "${ROLLOUT_LENGTHS[@]}"; do

        total_rollout=$((ne * rl))
        minibatch_size_to_use=0

        # 应用核心逻辑来决定minibatch_size或跳过
        if [[ ${total_rollout} -lt ${DEFAULT_MINIBATCH_SIZE} ]]; then
            minibatch_size_to_use=${total_rollout}
        elif [[ $((total_rollout % DEFAULT_MINIBATCH_SIZE)) -eq 0 ]]; then
            minibatch_size_to_use=${DEFAULT_MINIBATCH_SIZE}
        else
            # 如果组合无效，打印信息并跳到下一个循环
            echo "[SKIP] Invalid combination: num_envs=${ne}, rollout_length=${rl}. Total steps (${total_rollout}) not divisible by ${DEFAULT_MINIBATCH_SIZE}."
            echo "------------------------------------------------------------------------------"
            continue
        fi

        # 对于每个有效组合，启动一次 multirun
        echo "[RUN] Launching multirun for: num_envs=${ne}, rollout_length=${rl}, minibatch_size=${minibatch_size_to_use}"
        
        CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PYTHON_SCRIPT} \
            --multirun \
            agent=${AGENT_CONFIG} \
            project=${PROJECT_NAME} \
            num_envs=${ne} \
            rollout_length=${rl} \
            minibatch_size=${minibatch_size_to_use} \
            env=${ENVS_STR} \
            seed=${SEEDS_STR}

        echo "[DONE] Multirun finished for this group."
        echo "=============================================================================="
    done
done

echo "✅ All valid experiment groups have been launched successfully."