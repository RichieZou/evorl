#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析EVORL_PPO_swanlab.csv文件，为每个环境生成rollout_length × num_envs矩阵
并用颜色标注成功率
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
csv_file = '/home/yingjie/evorl/multirun/train_100_iteration/EVORL_PPO_swanlab.csv'
df = pd.read_csv(csv_file)

# 将eval/episode_returns和train_episode_return转换为数值类型，无效值设为NaN
df['eval/episode_returns'] = pd.to_numeric(df['eval/episode_returns'], errors='coerce')
df['train_episode_return'] = pd.to_numeric(df['train_episode_return'], errors='coerce')

# 打印基本信息
print("数据形状:", df.shape)
print("\n环境名称:", df['env.env_name'].unique())
print("种子数量:", sorted(df['seed'].unique()))
print("rollout_length取值:", sorted(df['rollout_length'].unique()))
print("num_envs取值:", sorted(df['num_envs'].unique()))

# 成功定义：同时具有eval/episode_returns和train_episode_return的有效取值

def calculate_success_rate(group):
    """计算成功率：同时具有eval和train两个指标的有效值才算成功"""
    if len(group) == 0:
        return np.nan
    
    # 检查同时具有两个有效值的记录
    has_eval = group['eval/episode_returns'].notna()
    has_train = group['train_episode_return'].notna()
    success_count = (has_eval & has_train).sum()
    total_count = len(group)
    
    return success_count / total_count

def create_matrix_heatmap(env_name, df_env, output_dir='./analysis_results_swanlab'):
    """为特定环境创建矩阵热力图"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有唯一的rollout_length和num_envs
    rollout_lengths = sorted(df_env['rollout_length'].unique())
    num_envs_list = sorted(df_env['num_envs'].unique())
    
    # 创建矩阵：行为rollout_length，列为num_envs
    matrix = np.zeros((len(rollout_lengths), len(num_envs_list)))
    matrix[:] = np.nan
    
    # 创建计数矩阵和成功计数矩阵，用于显示分数
    count_matrix = np.zeros((len(rollout_lengths), len(num_envs_list)), dtype=int)
    success_matrix = np.zeros((len(rollout_lengths), len(num_envs_list)), dtype=int)
    
    # 填充矩阵
    for i, rl in enumerate(rollout_lengths):
        for j, ne in enumerate(num_envs_list):
            # 获取该组合的所有实验结果
            mask = (df_env['rollout_length'] == rl) & (df_env['num_envs'] == ne)
            group = df_env[mask]
            
            if len(group) > 0:
                # 计算成功率：同时具有eval和train两个有效值
                has_eval = group['eval/episode_returns'].notna()
                has_train = group['train_episode_return'].notna()
                success_count = (has_eval & has_train).sum()
                total_count = len(group)
                
                matrix[i, j] = success_count / total_count
                count_matrix[i, j] = total_count
                success_matrix[i, j] = success_count
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 创建自定义颜色映射：红色(0) -> 黄色(0.5) -> 绿色(1.0)
    colors = ['#d73027', '#fee08b', '#1a9850']  # 红 -> 黄 -> 绿
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('success_rate', colors, N=n_bins)
    
    # 绘制热力图
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(num_envs_list)))
    ax.set_yticks(np.arange(len(rollout_lengths)))
    ax.set_xticklabels([f'{int(ne)}' for ne in num_envs_list], rotation=45, ha='right')
    ax.set_yticklabels([f'{int(rl)}' for rl in rollout_lengths])
    
    ax.set_xlabel('Num Envs', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rollout Length', fontsize=12, fontweight='bold')
    ax.set_title(f'Success Rate Matrix for {env_name}\n(Success = Both eval & train returns available)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 在每个格子中添加文本标注（成功数/总数）
    for i in range(len(rollout_lengths)):
        for j in range(len(num_envs_list)):
            if count_matrix[i, j] > 0:
                text = f'{success_matrix[i, j]}/{count_matrix[i, j]}'
                # 根据背景颜色选择文字颜色
                text_color = 'white' if matrix[i, j] < 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Success Rate', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    output_file = os.path.join(output_dir, f'{env_name}_success_matrix.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"保存图形到: {output_file}")
    
    # 同时保存PDF格式（矢量图，适合论文）
    output_file_pdf = os.path.join(output_dir, f'{env_name}_success_matrix.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"保存PDF到: {output_file_pdf}")
    
    plt.close()
    
    # 返回矩阵数据
    return matrix, rollout_lengths, num_envs_list

def create_summary_statistics(df_env, env_name, output_dir='./analysis_results_swanlab'):
    """创建详细统计表格"""
    
    # 获取所有唯一的rollout_length和num_envs
    rollout_lengths = sorted(df_env['rollout_length'].unique())
    num_envs_list = sorted(df_env['num_envs'].unique())
    
    # 创建详细的统计数据
    stats_data = []
    
    for rl in rollout_lengths:
        for ne in num_envs_list:
            mask = (df_env['rollout_length'] == rl) & (df_env['num_envs'] == ne)
            group = df_env[mask]
            
            if len(group) > 0:
                # 计算成功率：同时具有eval和train两个有效值
                has_eval = group['eval/episode_returns'].notna()
                has_train = group['train_episode_return'].notna()
                both_valid = has_eval & has_train
                success_count = both_valid.sum()
                total_count = len(group)
                
                # 获取有效的eval returns用于统计
                valid_eval_returns = group.loc[both_valid, 'eval/episode_returns']
                
                stats_data.append({
                    'rollout_length': rl,
                    'num_envs': ne,
                    'num_seeds': total_count,
                    'success_count': success_count,
                    'success_rate': success_count / total_count,
                    'mean_eval_return': valid_eval_returns.mean() if len(valid_eval_returns) > 0 else np.nan,
                    'std_eval_return': valid_eval_returns.std() if len(valid_eval_returns) > 0 else np.nan,
                    'min_eval_return': valid_eval_returns.min() if len(valid_eval_returns) > 0 else np.nan,
                    'max_eval_return': valid_eval_returns.max() if len(valid_eval_returns) > 0 else np.nan
                })
    
    # 创建DataFrame
    stats_df = pd.DataFrame(stats_data)
    
    # 保存到CSV
    output_file = os.path.join(output_dir, f'{env_name}_statistics.csv')
    stats_df.to_csv(output_file, index=False)
    print(f"保存统计数据到: {output_file}")
    
    return stats_df

# 主程序
if __name__ == '__main__':
    print("=" * 80)
    print("开始分析SwanLab训练结果")
    print("=" * 80)
    
    # 获取所有环境名称
    env_names = df['env.env_name'].unique()
    
    # 为每个环境创建矩阵和统计
    for env_name in env_names:
        print(f"\n处理环境: {env_name}")
        print("-" * 80)
        
        # 筛选该环境的数据
        df_env = df[df['env.env_name'] == env_name].copy()
        
        print(f"  数据点数量: {len(df_env)}")
        print(f"  种子数量: {df_env['seed'].nunique()}")
        print(f"  rollout_length取值数: {df_env['rollout_length'].nunique()}")
        print(f"  num_envs取值数: {df_env['num_envs'].nunique()}")
        
        # 创建矩阵热力图
        matrix, rollout_lengths, num_envs_list = create_matrix_heatmap(env_name, df_env)
        
        # 创建统计表格
        stats_df = create_summary_statistics(df_env, env_name)
        
        print(f"  完成 {env_name} 的分析")
    
    print("\n" + "=" * 80)
    print("分析完成！所有结果保存在 ./analysis_results_swanlab/ 目录中")
    print("=" * 80)

