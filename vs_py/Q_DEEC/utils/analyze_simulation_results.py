# analyze_simulation_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np # 保留numpy
import logging
import shutil

# --- 全局配置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid") # 使用seaborn的样式，让图更好看

PROJECT_ROOT_ANALYSIS = Path(__file__).resolve().parent.parent
REPORTS_BASE_DIR = PROJECT_ROOT_ANALYSIS / "reports"
OUTPUT_ANALYSIS_DIR = REPORTS_BASE_DIR / "analysis_comparison_plots"

# --- 日志配置 ---
logger = logging.getLogger("AnalysisScript")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def load_all_performance_data(base_dir):
    """遍历所有实验子目录，加载并合并performance_log.csv。"""
    all_perf_df = []
    # 寻找所有以 'reports_' 开头的子目录
    for report_dir in base_dir.glob("reports_*"):
        if report_dir.is_dir():
            log_file = report_dir / "performance_log.csv"
            algorithm_name = report_dir.name.replace("reports_", "").upper() # 从目录名提取算法名
            
            if log_file.exists():
                try:
                    df = pd.read_csv(log_file)
                    df['algorithm'] = algorithm_name # 添加一列用于区分算法
                    all_perf_df.append(df)
                    logger.info(f"成功加载 {algorithm_name} 的性能日志 (共 {len(df)} 条记录)。")
                except pd.errors.EmptyDataError:
                    logger.warning(f"性能日志文件 {log_file} 为空，已跳过。")
                except Exception as e:
                    logger.error(f"读取文件 {log_file} 时出错: {e}")
            else:
                logger.warning(f"在目录 {report_dir} 中未找到 performance_log.csv。")
    
    if not all_perf_df:
        return pd.DataFrame()
    
    return pd.concat(all_perf_df, ignore_index=True)

def plot_comparison_charts(df_all):
    """在一张图上绘制所有算法的性能对比。"""
    if df_all.empty:
        logger.warning("没有加载到任何性能数据，无法生成对比图。")
        return

    # --- 1. 网络生命周期对比 (存活节点数) ---
    # (此部分无需修改)
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_all, x='round', y='alive_nodes', hue='algorithm', lw=2)
    plt.title('Network life cycle comparison: number of surviving nodes', fontsize=16)
    plt.xlabel('Simulation rounds (Round)', fontsize=12)
    plt.ylabel('Number of surviving nodes', fontsize=12)
    plt.legend(title='algorithm')
    plt.savefig(OUTPUT_ANALYSIS_DIR / "comparison_lifetime.png", dpi=150)
    plt.close()
    logger.info("已绘制：网络生命周期对比图")

    # --- 2. 平均能量消耗对比 ---
    # (此部分无需修改)
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_all, x='round', y='avg_energy', hue='algorithm', lw=2)
    plt.title('Comparison of average node energy consumption', fontsize=16)
    plt.xlabel('Simulation rounds (Round)', fontsize=12)
    plt.ylabel('Average Energy (J)', fontsize=12)
    plt.legend(title='algorithm')
    plt.savefig(OUTPUT_ANALYSIS_DIR / "comparison_avg_energy.png", dpi=150)
    plt.close()
    logger.info("已绘制：平均能量消耗对比图")

    # --- [核心修改] 3. 累计数据包投递率 (Cumulative PDR) 对比 ---
    
    # 首先，我们需要计算每个算法随时间变化的累计生成包数和累计接收包数
    # 使用 groupby 和 cumsum() 函数可以轻松实现
    df_all['packets_generated_cumulative'] = df_all.groupby('algorithm')['packets_generated'].cumsum()
    df_all['packets_to_bs_cumulative'] = df_all.groupby('algorithm')['packets_to_bs'].cumsum()

    # 然后，计算累计PDR
    # 使用 .replace(0, np.nan) 来避免除以零的错误，pandas在计算时会自动忽略NaN值
    df_all['pdr_cumulative'] = df_all['packets_to_bs_cumulative'] / df_all['packets_generated_cumulative'].replace(0, np.nan)

    plt.figure(figsize=(12, 7))
    # 绘图时，y轴直接使用我们新计算的 'pdr_cumulative' 列
    sns.lineplot(data=df_all, x='round', y='pdr_cumulative', hue='algorithm', lw=2)
    # 更新图表标题和Y轴标签，以反映新的指标
    plt.title('Cumulative Packet Delivery Rate (PDR) Comparison', fontsize=16)
    plt.xlabel('Simulation rounds (Round)', fontsize=12)
    plt.ylabel('Cumulative PDR', fontsize=12)
    # Y轴的范围现在可以严格限制在[0, 1]，因为累计PDR不可能超过1
    plt.ylim(0, 1.0)
    plt.legend(title='algorithm')
    plt.savefig(OUTPUT_ANALYSIS_DIR / "comparison_cumulative_pdr.png", dpi=150) # 保存为新文件名
    plt.close()
    logger.info("已绘制：[新] 累计数据包投递率对比图")


    # --- [核心修改] 4. 簇头数量动态对比 (2x2 小倍图) ---
    logger.info("开始处理和绘制最终版本的稳定化簇头数量图 (小倍图版本)...")

    # 步骤 1 & 2: 数据处理 (与之前相同)
    epoch_length = 20
    df_all['epoch'] = (df_all['round'] // epoch_length) * epoch_length
    
    try:
        df_ch_stats = df_all.groupby(['algorithm', 'epoch'])['num_ch'].agg(['mean', 'std']).reset_index()
        df_ch_stats['std'] = df_ch_stats['std'].fillna(0)
    except Exception as e:
        logger.error(f"在计算统计数据时出错: {e}")
        return

    # 步骤 3: 使用 Seaborn 的 `FacetGrid` 创建小倍图网格
    # FacetGrid 是专门用于创建小倍图的强大工具
    g = sns.FacetGrid(
        df_ch_stats, 
        col="algorithm",  # 按 'algorithm' 列来创建不同的子图列
        col_wrap=2,       # 每行最多显示2个子图，形成 2x2 网格
        height=4,         # 每个子图的高度
        aspect=1.5,       # 每个子图的宽高比
        sharey=True,      # 【关键】所有子图共享Y轴，确保对比的公平性
        legend_out=True
    )

    # 步骤 4: 在每个子图上绘制均值曲线
    # g.map() 函数会将指定的绘图函数应用到每个子图上
    g.map(sns.lineplot, 'epoch', 'mean', lw=2.5, errorbar=None) # errorbar=None 告诉lineplot不要画默认的置信区间

    # 步骤 5: 在每个子图上绘制标准差误差带
    # 我们需要遍历每个子图的Axes对象来添加fill_between
    for ax, (_, algo_data) in zip(g.axes.flat, g.facet_data()):
        ax.fill_between(
            algo_data['epoch'],
            algo_data['mean'] - algo_data['std'],
            algo_data['mean'] + algo_data['std'],
            alpha=0.2
        )
        # 为每个子图设置网格线
        ax.grid(True)

    # 步骤 6: 设置总标题和坐标轴标签
    # g.fig 是整个图的对象
    g.fig.suptitle('Cluster Head Dynamics (Mean and Std. Dev. per Epoch)', y=1.03, fontsize=16) 
    # g.set_axis_labels() 为所有共享的轴设置标签
    g.set_axis_labels('Simulation rounds (Round)', 'Number of Cluster Heads')
    # g.set_titles() 可以为每个子图设置标题，这里用算法名
    g.set_titles("Algorithm: {col_name}")
    
    # 调整布局，防止标题重叠
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # 保存图像
    plt.savefig(OUTPUT_ANALYSIS_DIR / "comparison_ch_count_grid.png", dpi=300)
    plt.close()
    logger.info("已绘制并保存最终版本的小倍图簇头数量图。")

def main_analysis():
    """主分析函数"""
    # --- 清理旧的分析图片 ---
    if OUTPUT_ANALYSIS_DIR.exists():
        try:
            shutil.rmtree(OUTPUT_ANALYSIS_DIR)
            logger.info(f"已删除旧的分析图片目录: {OUTPUT_ANALYSIS_DIR}")
        except OSError as e:
            logger.error(f"删除目录 {OUTPUT_ANALYSIS_DIR} 失败: {e}")
    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"已创建分析图片目录: {OUTPUT_ANALYSIS_DIR}")

    # --- 加载所有实验数据并绘图 ---
    all_perf_data = load_all_performance_data(REPORTS_BASE_DIR)
    
    if not all_perf_data.empty:
        # 为了安全，给 performance_log.csv 的列重命名，确保它们是可预测的
        # 这是个好习惯，防止 env.py 中日志头变化导致脚本出错
        expected_columns = [
            'round', 'alive_nodes', 'total_energy', 'avg_energy', 
            'num_ch', 'avg_ch_energy', 'avg_members', 'ch_load_variance', 
            'packets_generated', 'packets_to_bs', 'avg_delay'
        ]
        # 假设你的日志文件没有列头，或者你想强制使用这些列名
        # all_perf_data.columns = expected_columns 
        # 如果你的日志文件有列头，但你想确保它们没有前后空格
        all_perf_data.columns = [col.strip() for col in all_perf_data.columns]


        plot_comparison_charts(all_perf_data)
    else:
        logger.error("未能加载任何实验数据，分析结束。")
        logger.error("请确保 'reports' 目录下存在 'reports_xxx' 格式的子目录，并且其中包含 'performance_log.csv'。")

if __name__ == '__main__':
    main_analysis()