# analyze_simulation_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np # <--- [核心修正] 添加这一行
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
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_all, x='round', y='avg_energy', hue='algorithm', lw=2)
    plt.title('Comparison of average node energy consumption', fontsize=16)
    plt.xlabel('Simulation rounds (Round)', fontsize=12)
    plt.ylabel('Average Energy (J)', fontsize=12)
    plt.legend(title='algorithm')
    plt.savefig(OUTPUT_ANALYSIS_DIR / "comparison_avg_energy.png", dpi=150)
    plt.close()
    logger.info("已绘制：平均能量消耗对比图")

    # --- 3. 数据包投递率 (PDR) 对比 ---
    # 计算每个算法的滑动平均PDR，使曲线更平滑
    df_all['pdr_this_round'] = df_all['packets_to_bs'] / df_all['packets_generated'].replace(0, np.nan)
    df_all['pdr_ma'] = df_all.groupby('algorithm')['pdr_this_round'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
    
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_all, x='round', y='pdr_ma', hue='algorithm', lw=2)
    plt.title('Packet delivery rate comparison (50 rounds of sliding average)', fontsize=16)
    plt.xlabel('Simulation rounds (Round)', fontsize=12)
    plt.ylabel('PDR (sliding average)', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='algorithm')
    plt.savefig(OUTPUT_ANALYSIS_DIR / "comparison_pdr.png", dpi=150)
    plt.close()
    logger.info("已绘制：数据包投递率对比图")

    # --- 4. 簇头数量对比 ---
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df_all, x='round', y='num_ch', hue='algorithm', lw=2, alpha=0.8)
    plt.title('Comparison of cluster head quantity changes', fontsize=16)
    plt.xlabel('Simulation rounds (Round)', fontsize=12)
    plt.ylabel('Number of cluster heads', fontsize=12)
    plt.legend(title='algorithm')
    plt.savefig(OUTPUT_ANALYSIS_DIR / "comparison_ch_count.png", dpi=150)
    plt.close()
    logger.info("已绘制：簇头数量对比图")

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
        plot_comparison_charts(all_perf_data)
    else:
        logger.error("未能加载任何实验数据，分析结束。")
        logger.error("请确保 'reports' 目录下存在 'reports_xxx' 格式的子目录，并且其中包含 'performance_log.csv'。")

if __name__ == '__main__':
    main_analysis()