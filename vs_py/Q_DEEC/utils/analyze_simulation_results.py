# analyze_simulation_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import logging # 导入logging
import shutil # 用于删除文件夹

logger = logging.getLogger("AnalysisScript")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # 或 DEBUG

PROJECT_ROOT_ANALYSIS = Path(__file__).resolve().parent.parent # 假设脚本在utils下，或与reports同级
PERFORMANCE_LOG = PROJECT_ROOT_ANALYSIS / "reports" / "performance_log.csv"
CH_BEHAVIOR_LOG = PROJECT_ROOT_ANALYSIS / "reports" / "ch_behavior_log.csv"
OUTPUT_ANALYSIS_DIR = PROJECT_ROOT_ANALYSIS / "reports" / "simulation_analysis_plots"

def plot_performance_metrics(df_perf):
    if df_perf.empty:
        print("性能日志为空，无法绘图。")
        return

    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # 网络生命周期 - 存活节点数
    plt.figure(figsize=(10, 6))
    plt.plot(df_perf['round'], df_perf['alive_nodes'], label='Alive Nodes')
    plt.xlabel('Round')
    plt.ylabel('Number of Alive Nodes')
    plt.title('Network Lifetime: Alive Nodes over Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_ANALYSIS_DIR / "network_lifetime_alive_nodes.png")
    plt.close()
    print("已绘制：存活节点数变化图")

    # 平均能量
    plt.figure(figsize=(10, 6))
    plt.plot(df_perf['round'], df_perf['avg_energy'], label='Average Node Energy')
    plt.xlabel('Round')
    plt.ylabel('Average Energy (J)')
    plt.title('Average Node Energy over Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_ANALYSIS_DIR / "average_node_energy.png")
    plt.close()
    print("已绘制：平均能量变化图")

    # CH数量变化
    plt.figure(figsize=(10, 6))
    plt.plot(df_perf['round'], df_perf['num_ch'], label='Number of Cluster Heads', color='orange')
    plt.xlabel('Round')
    plt.ylabel('Number of CHs')
    plt.title('Number of Cluster Heads over Rounds')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_ANALYSIS_DIR / "ch_count_over_rounds.png")
    plt.close()
    print("已绘制：CH数量变化图")

    # 数据包投递率 (如果数据有效)
    if 'packets_generated' in df_perf.columns and 'packets_to_bs' in df_perf.columns:
        # 计算累积或滑动窗口的PDR可能更有意义
        df_perf['pdr_this_round'] = df_perf['packets_to_bs'] / df_perf['packets_generated'].replace(0, np.nan) # 避免除以0
        df_perf_pdr_valid = df_perf.dropna(subset=['pdr_this_round'])
        if not df_perf_pdr_valid.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(df_perf_pdr_valid['round'], df_perf_pdr_valid['pdr_this_round'], label='Packet Delivery Ratio (per round)', color='green', alpha=0.7)
            # 可以增加一个滑动平均来平滑曲线
            if len(df_perf_pdr_valid) >= 10:
                 plt.plot(df_perf_pdr_valid['round'], df_perf_pdr_valid['pdr_this_round'].rolling(window=10).mean(), label='PDR (10-round MA)', color='darkgreen')
            plt.xlabel('Round')
            plt.ylabel('Packet Delivery Ratio')
            plt.title('Packet Delivery Ratio over Rounds')
            plt.ylim(0, 1.1) # PDR在0到1之间
            plt.legend()
            plt.grid(True)
            plt.savefig(OUTPUT_ANALYSIS_DIR / "packet_delivery_ratio.png")
            plt.close()
            print("已绘制：数据包投递率图")
        else:
            print("无有效PDR数据进行绘制。")
    else:
        print("缺少 'packets_generated' 或 'packets_to_bs' 列，无法计算PDR。")


def plot_ch_behavior_metrics(df_ch_beh):
    if df_ch_beh.empty:
        print("CH行为日志为空，无法绘图。")
        return
    
    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # CH当选时的能量分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df_ch_beh['energy_at_election'], kde=True, bins=20)
    plt.xlabel('Energy at Election (J)')
    plt.ylabel('Frequency / Density')
    plt.title('Distribution of CH Energy at Time of Election')
    plt.grid(True)
    plt.savefig(OUTPUT_ANALYSIS_DIR / "ch_energy_at_election_dist.png")
    plt.close()
    print("已绘制：CH当选时能量分布图")

    # CH的最终成员数分布
    plt.figure(figsize=(10, 6))
    sns.histplot(df_ch_beh['final_members'], kde=True, discrete=True) # discrete=True for integer counts
    plt.xlabel('Number of Members per CH per Epoch')
    plt.ylabel('Frequency / Density')
    plt.title('Distribution of Members per CH')
    plt.grid(True)
    plt.savefig(OUTPUT_ANALYSIS_DIR / "ch_member_count_dist.png")
    plt.close()
    print("已绘制：CH成员数分布图")
    
    # CH能量与其成员数的关系 (散点图)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_ch_beh, x='energy_at_election', y='final_members', alpha=0.6)
    plt.xlabel('CH Energy at Election (J)')
    plt.ylabel('Final Number of Members')
    plt.title('CH Energy vs. Number of Members')
    plt.grid(True)
    plt.savefig(OUTPUT_ANALYSIS_DIR / "ch_energy_vs_members.png")
    plt.close()
    print("已绘制：CH能量与成员数关系图")

    # CH到BS距离与其成员数的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_ch_beh, x='dist_to_bs', y='final_members', alpha=0.6)
    plt.xlabel('CH Distance to BS (m)')
    plt.ylabel('Final Number of Members')
    plt.title('CH Distance to BS vs. Number of Members')
    plt.grid(True)
    plt.savefig(OUTPUT_ANALYSIS_DIR / "ch_dist_vs_members.png")
    plt.close()
    print("已绘制：CH到BS距离与成员数关系图")


def main_analysis():
    # --- 清理旧的分析图片 ---
    if OUTPUT_ANALYSIS_DIR.exists():
        try:
            shutil.rmtree(OUTPUT_ANALYSIS_DIR)
            print(f"已删除旧的分析图片目录: {OUTPUT_ANALYSIS_DIR}")
        except OSError as e:
            print(f"删除目录 {OUTPUT_ANALYSIS_DIR} 失败: {e}")
    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"已创建分析图片目录: {OUTPUT_ANALYSIS_DIR}")

    try:
        df_perf = pd.read_csv(PERFORMANCE_LOG)
        print(f"成功加载性能日志: {PERFORMANCE_LOG}")
        plot_performance_metrics(df_perf)
    except FileNotFoundError:
        print(f"性能日志文件 {PERFORMANCE_LOG} 未找到。")
    except pd.errors.EmptyDataError:
        print(f"性能日志文件 {PERFORMANCE_LOG} 为空。")
    except Exception as e:
        logger.error(f"读取或绘制性能日志时出错: {e}", exc_info=True)

    try:
        df_ch_beh = pd.read_csv(CH_BEHAVIOR_LOG)
        print(f"成功加载CH行为日志: {CH_BEHAVIOR_LOG}")
        plot_ch_behavior_metrics(df_ch_beh)
    except FileNotFoundError:
        print(f"CH行为日志文件 {CH_BEHAVIOR_LOG} 未找到。")
    except pd.errors.EmptyDataError:
        print(f"CH行为日志文件 {CH_BEHAVIOR_LOG} 为空。")
    except Exception as e:
        logger.error(f"读取或绘制CH行为日志时出错: {e}", exc_info=True)

if __name__ == '__main__':
    main_analysis()