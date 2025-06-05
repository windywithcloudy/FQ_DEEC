# analyze_discretization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path
import numpy as np # 引入numpy
import shutil # 用于删除文件夹

# 加载配置文件以获取当前的离散化边界
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
CONFIG_FILE = PROJECT_ROOT/ "config" / "config.yml" # 假设脚本和config在同一父目录下
RAW_STATE_LOG_FILE = PROJECT_ROOT.parent.parent / "raw_state_log.csv" # 日志文件路径
IMAGE_OUTPUT_DIR = PROJECT_ROOT / "reports" / "discretization_analysis"

def load_config(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config.get('discretization_params', {})
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到。")
        return {}
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        return {}

def plot_distribution_with_boundaries(data_series, title,filename_base, boundaries=None, num_bins_eq_width=None, is_normalized=False):
    plt.figure(figsize=(10, 6))
    sns.histplot(data_series, kde=True, stat="density", bins=50 if data_series.nunique() > 50 else 'auto') # kde=True 添加核密度估计

    current_boundaries_for_plot = []
    if boundaries: # 自定义边界
        # np.digitize(x, bins) 对于 bins=[b1,b2] 会产生0, 1, 2三个状态
        # 为了可视化，我们需要实际的分割线
        current_boundaries_for_plot = list(boundaries)
        # 为了显示所有区间，可以在两端加上数据的最小值和最大值（如果边界没有覆盖）
        # plot_min = min(data_series.min(), boundaries[0] if boundaries else data_series.min())
        # plot_max = max(data_series.max(), boundaries[-1] if boundaries else data_series.max())
        # all_lines = sorted(list(set([plot_min] + boundaries + [plot_max])))
        # for i in range(len(all_lines) -1):
        #     plt.axvline(all_lines[i], color='r', linestyle='--', alpha=0.7)
        #     plt.axvline(all_lines[i+1], color='r', linestyle='--', alpha=0.7)
        # 简单画出边界线
        for bound in boundaries:
            plt.axvline(bound, color='r', linestyle='--', linewidth=2, label=f'Boundary: {bound}')
        
    elif num_bins_eq_width: # 等宽分箱
        min_val = 0 if is_normalized else data_series.min()
        max_val = 1 if is_normalized else data_series.max() # 对于非归一化数据，这个max_val可能需要更稳健的估计
        
        # 如果数据范围本身就很小，num_bins_eq_width可能不适用
        if max_val > min_val :
            bin_width = (max_val - min_val) / num_bins_eq_width
            for i in range(1, num_bins_eq_width):
                boundary = min_val + i * bin_width
                current_boundaries_for_plot.append(boundary)
                plt.axvline(boundary, color='g', linestyle=':', linewidth=2, label=f'EqWidth Boundary: {boundary:.2f}')
        else:
            print(f"警告: {title} 的数据范围过小 ({min_val} - {max_val})，无法进行等宽分箱。")


    plt.title(f'Distribution of {title}\n(Red lines: Custom Boundaries, Green lines: Equal Width Boundaries)')
    plt.xlabel(title)
    plt.ylabel('Density / Frequency')
    
    # 移除重复的legend项
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label: # 只有当有label时才显示legend
        plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    output_filename = IMAGE_OUTPUT_DIR / f"{filename_base}_distribution.png"
    try:
        plt.savefig(output_filename, dpi=150)
        print(f"图片已保存到: {output_filename}")
    except Exception as e:
        print(f"保存图片 {output_filename} 失败: {e}")
    plt.close() # 关闭图像，释放内存

def main():

    # --- 管理图片输出文件夹 ---
    if IMAGE_OUTPUT_DIR.exists():
        try:
            shutil.rmtree(IMAGE_OUTPUT_DIR) # 删除旧的分析图片文件夹
            print(f"已删除旧的分析图片目录: {IMAGE_OUTPUT_DIR}")
        except OSError as e:
            print(f"删除目录 {IMAGE_OUTPUT_DIR} 失败: {e}. 请手动删除或检查权限。")
            # 可以选择退出或继续，但保存图片可能会失败
    try:
        IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=False) # 创建新的，exist_ok=False确保是新建
        print(f"已创建新的分析图片目录: {IMAGE_OUTPUT_DIR}")
    except FileExistsError: # 如果由于并发或其他原因文件夹已存在
        print(f"分析图片目录 {IMAGE_OUTPUT_DIR} 已存在。")
    except Exception as e:
        print(f"创建目录 {IMAGE_OUTPUT_DIR} 失败: {e}")
        return # 如果无法创建目录，则不继续

    disc_params = load_config(CONFIG_FILE)
    if not disc_params:
        print("无法加载离散化参数，退出分析。")
        return

    try:
        df_raw_states = pd.read_csv(RAW_STATE_LOG_FILE)
    except FileNotFoundError:
        print(f"原始状态日志文件 {RAW_STATE_LOG_FILE} 未找到。请先运行仿真并生成日志。")
        return
    except pd.errors.EmptyDataError:
        print(f"原始状态日志文件 {RAW_STATE_LOG_FILE} 为空。")
        return
    except Exception as e:
        print(f"读取原始状态日志文件出错: {e}")
        return

    print(f"已加载 {len(df_raw_states)} 条原始状态记录。")
    print("数据样本:")
    print(df_raw_states.head())
    print("\n数据描述:")
    print(df_raw_states.describe())


    # 1. 分析归一化能量 (e_self_raw)
    if 'e_self_raw' in df_raw_states.columns:
        energy_params = disc_params.get('energy', {})
        plot_distribution_with_boundaries(
            df_raw_states['e_self_raw'], 
            'Normalized Self Energy (e_self_raw)',
            "self_energy", # filename_base
            boundaries=energy_params.get('custom_boundaries'),
            num_bins_eq_width=energy_params.get('num_bins'),
            is_normalized=True
        )

    # 2. 分析上次当CH的时间 (t_last_ch_raw)
    if 't_last_ch_raw' in df_raw_states.columns:
        time_params = disc_params.get('time_since_last_ch', {})
        plot_distribution_with_boundaries(
            df_raw_states['t_last_ch_raw'],
            'Rounds Since Last CH (t_last_ch_raw)',
            "time_since_last_ch", # filename_base
            boundaries=time_params.get('boundaries')
        )

    # 3. 分析邻居数量 (n_neighbor_raw)
    if 'n_neighbor_raw' in df_raw_states.columns:
        neighbor_params = disc_params.get('neighbor_count', {})
        plot_distribution_with_boundaries(
            df_raw_states['n_neighbor_raw'],
            'Neighbor Count (n_neighbor_raw)',
            "neighbor_count",
            boundaries=neighbor_params.get('boundaries'),
            num_bins_eq_width=neighbor_params.get('num_bins')
        )
    
    # 4. 分析附近CH数量 (n_ch_nearby_raw)
    if 'n_ch_nearby_raw' in df_raw_states.columns:
        ch_count_params = disc_params.get('ch_count_nearby', {})
        plot_distribution_with_boundaries(
            df_raw_states['n_ch_nearby_raw'],
            'Nearby CH Count (n_ch_nearby_raw)',
            "ch_count_nearby",
            boundaries=ch_count_params.get('boundaries')
        )

    # 5. 分析归一化到BS的距离 (d_bs_normalized_raw)
    if 'd_bs_normalized_raw' in df_raw_states.columns:
        dist_bs_params = disc_params.get('distance_to_bs_normalized', {})
        plot_distribution_with_boundaries(
            df_raw_states['d_bs_normalized_raw'],
            'Normalized Distance to BS (d_bs_normalized_raw)',
            "dist_bs_normalized",
            boundaries=dist_bs_params.get('boundaries'),
            is_normalized=True
        )
        
    # 6. 分析平均邻居能量 (e_neighbor_avg_raw) - 假设你也记录了这个
    if 'e_neighbor_avg_raw' in df_raw_states.columns:
        avg_neighbor_energy_params = disc_params.get('avg_neighbor_energy', disc_params.get('energy', {}))
        plot_distribution_with_boundaries(
            df_raw_states['e_neighbor_avg_raw'],
            'Average Neighbor Energy (e_neighbor_avg_raw)',
            "avg_neighbor_energy",
            boundaries=avg_neighbor_energy_params.get('custom_boundaries'),
            num_bins_eq_width=avg_neighbor_energy_params.get('num_bins'),
            is_normalized=True
        )


if __name__ == '__main__':
    # 为了演示，可以尝试加载（如果文件已存在）
    if RAW_STATE_LOG_FILE.exists() and RAW_STATE_LOG_FILE.stat().st_size > 0 :
         main()
    else:
        print(f"{RAW_STATE_LOG_FILE} 不存在或为空。")