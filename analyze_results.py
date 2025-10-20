#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练结果CSV文件，为每个环境生成rollout_length × total_timesteps矩阵
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
csv_file = '/home/yingjie/evorl/multirun/train_100_iteration/train_100_sumup.csv'
df = pd.read_csv(csv_file)

# 打印基本信息
print("数据形状:", df.shape)
print("\n环境名称:", df['env.env_name'].unique())
print("种子数量:", df['seed'].unique())
print("rollout_length取值:", sorted(df['rollout_length'].unique()))
print("num_envs取值:", sorted(df['num_envs'].unique()))

# 定义成功标准（可以根据需要调整）
# 这里以eval/episode_returns是否大于某个阈值来判断
# 对于hopper，通常认为>100为成功；对于walker2d，通常认为>500为成功
SUCCESS_THRESHOLDS = {
    'hopper': 100,
    'walker2d': 500
}

def calculate_success_rate(group, env_name):
    """计算成功率"""
    if len(group) == 0:
        return np.nan
    
    threshold = SUCCESS_THRESHOLDS.get(env_name, 0)
    # 过滤掉空值
    valid_returns = group['eval/episode_returns'].dropna()
    
    if len(valid_returns) == 0:
        return np.nan
    
    # 计算成功率
    success_count = (valid_returns > threshold).sum()
    total_count = len(valid_returns)
    
    return success_count / total_count

def create_matrix_heatmap(env_name, df_env, output_dir='./analysis_results'):
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
                # 计算成功率
                threshold = SUCCESS_THRESHOLDS.get(env_name, 0)
                valid_returns = group['eval/episode_returns'].dropna()
                
                if len(valid_returns) > 0:
                    success_count = (valid_returns > threshold).sum()
                    total_count = len(valid_returns)
                    
                    matrix[i, j] = success_count / total_count
                    count_matrix[i, j] = total_count
                    success_matrix[i, j] = success_count
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 12))
    
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
    ax.set_title(f'Success Rate Matrix for {env_name}\n(Threshold: {SUCCESS_THRESHOLDS.get(env_name, 0)})', 
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

def create_summary_statistics(df_env, env_name, output_dir='./analysis_results'):
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
                valid_returns = group['eval/episode_returns'].dropna()
                
                if len(valid_returns) > 0:
                    threshold = SUCCESS_THRESHOLDS.get(env_name, 0)
                    success_count = (valid_returns > threshold).sum()
                    
                    stats_data.append({
                        'rollout_length': rl,
                        'num_envs': ne,
                        'num_seeds': len(valid_returns),
                        'success_count': success_count,
                        'success_rate': success_count / len(valid_returns),
                        'mean_return': valid_returns.mean(),
                        'std_return': valid_returns.std(),
                        'min_return': valid_returns.min(),
                        'max_return': valid_returns.max()
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
    print("开始分析训练结果")
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
    print("分析完成！所有结果保存在 ./analysis_results/ 目录中")
    print("=" * 80)

