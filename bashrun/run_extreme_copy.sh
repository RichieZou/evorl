#!/bin/bash

# ==============================================================================
#           🚀 极限测试 Hydra 实验自动化脚本 🚀
# ==============================================================================
# 功能：
#   固定 rollout_length=1024，测试不同规模的 num_envs：
#   测试 num_envs: 1024, 4096, 8192, 16384
# ==============================================================================

# --- ⚙️ 1. 在这里配置您的实验 ---

# 定义要扫描的参数
ENVS=("brax/hopper" "brax/humanoidstandup")
SEEDS=(42)

# 固定的 rollout_length
FIXED_ROLLOUT_LENGTH=1024

# 要测试的不同 num_envs 值
NUM_ENVS_TO_TEST=(2048 4096 8192 16384)

# 定义固定的参数
PYTHON_SCRIPT="scripts/train_100_iteration.py"
AGENT_CONFIG="ppo"
PROJECT_NAME="EVORL_PPO_extreme_test"
GPU_ID=0

# 逻辑计算用的常量
DEFAULT_MINIBATCH_SIZE=256

# --- 🚀 2. 执行逻辑 ---

# 将Bash数组转换为Hydra接受的逗号分隔字符串
ENVS_STR=$(IFS=,; echo "${ENVS[*]}")
SEEDS_STR=$(IFS=,; echo "${SEEDS[*]}")

echo "=============================================================================="
echo "Starting EXTREME NUM_ENVS TEST (Fixed rollout_length=${FIXED_ROLLOUT_LENGTH})..."
echo "Environments: ${ENVS_STR}"
echo "Seeds: ${SEEDS_STR}"
echo "Testing num_envs: ${NUM_ENVS_TO_TEST[*]}"
echo "=============================================================================="
echo ""

# 遍历所有要测试的 num_envs
for i in "${!NUM_ENVS_TO_TEST[@]}"; do
    ne="${NUM_ENVS_TO_TEST[$i]}"
    rl="${FIXED_ROLLOUT_LENGTH}"
    
    echo "📊 测试 $((i+1))/${#NUM_ENVS_TO_TEST[@]}: num_envs=${ne}, rollout_length=${rl}"
    
    total_rollout=$((ne * rl))
    minibatch_size_to_use=0
    
    # 应用核心逻辑来决定minibatch_size或跳过
    if [[ ${total_rollout} -lt ${DEFAULT_MINIBATCH_SIZE} ]]; then
        minibatch_size_to_use=${total_rollout}
    elif [[ $((total_rollout % DEFAULT_MINIBATCH_SIZE)) -eq 0 ]]; then
        minibatch_size_to_use=${DEFAULT_MINIBATCH_SIZE}
    else
        # 如果组合无效，打印信息并跳到下一个循环
        echo "   [SKIP] Invalid combination: Total steps (${total_rollout}) not divisible by ${DEFAULT_MINIBATCH_SIZE}."
        echo "------------------------------------------------------------------------------"
        echo ""
        continue
    fi
    
    echo "   Total rollout steps: ${total_rollout} (${ne} × ${rl})"
    echo "   Minibatch size: ${minibatch_size_to_use}"
    echo "   [RUN] 正在启动训练..."
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${PYTHON_SCRIPT} \
        --multirun \
        agent=${AGENT_CONFIG} \
        project=${PROJECT_NAME} \
        num_envs=${ne} \
        rollout_length=${rl} \
        minibatch_size=${minibatch_size_to_use} \
        env=${ENVS_STR} \
        seed=${SEEDS_STR}
    
    echo "   [DONE] 此num_envs配置测试完成。"
    echo "=============================================================================="
    echo ""
done

echo "✅ 所有num_envs极限测试已成功完成！"
echo "测试了 ${#NUM_ENVS_TO_TEST[@]} 种不同的num_envs配置（rollout_length固定为${FIXED_ROLLOUT_LENGTH}）"
