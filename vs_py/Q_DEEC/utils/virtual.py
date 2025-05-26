# utils/virtual.py
import matplotlib.pyplot as plt
from pathlib import Path
from .log import logger # 从同级目录的log.py导入logger

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统
plt.rcParams['axes.unicode_minus'] = False
DEFAULT_OUTPUT_IMAGE_FILE = "network_topology.png"

def visualize_network(nodes_data, base_station_pos, area_dims, filename=DEFAULT_OUTPUT_IMAGE_FILE, current_round=None, cluster_heads_ids=None, config=None):
    """
    可视化网络拓扑并保存到文件。

    参数:
        nodes_data (list): 节点信息列表，每个元素是一个包含 "position"键的字典。
                           例如: [{"id": 0, "position": [x, y], ...}, ...]
        base_station_pos (list): 基站坐标 [x, y]。
        area_dims (list): 网络区域尺寸 [width, height]。
        filename (str): 保存图像的文件名。
    """
    if not nodes_data:
        logger.info("没有节点可供可视化。")
        return
    logger.info(f"可视化包含 {len(nodes_data)} 个节点的网络拓扑 (轮次: {current_round if current_round is not None else 'N/A'})...")
    
    area_w, area_h = area_dims
    base_x, base_y = base_station_pos

    # 从配置中获取颜色 (如果提供了config)
    viz_colors = config.get('visualization', {}).get('colors', {}) if config else {}
    ch_color = viz_colors.get('cluster_head', '#FF4500')
    normal_color = viz_colors.get('normal_node', '#1E90FF')
    bs_color = viz_colors.get('base_station', '#32CD32')
    dead_color = viz_colors.get('dead_node', '#A9A9A9')
    
    plt.figure(figsize=(10,10))

    # 绘制节点
    normal_nodes_x, normal_nodes_y = [], []
    ch_nodes_x, ch_nodes_y = [], []
    dead_nodes_x, dead_nodes_y = [], []
    for node in nodes_data:
        if node["status"] == "dead":
            dead_nodes_x.append(node["position"][0])
            dead_nodes_y.append(node["position"][1])
        elif node["role"] == "cluster_head":
            ch_nodes_x.append(node["position"][0])
            ch_nodes_y.append(node["position"][1])
        else: # normal node
            normal_nodes_x.append(node["position"][0])
            normal_nodes_y.append(node["position"][1])
    if normal_nodes_x:
        plt.scatter(normal_nodes_x, normal_nodes_y, s=50, c=normal_color, alpha=0.8, label=f'普通节点 ({len(normal_nodes_x)})')
    if ch_nodes_x:
        plt.scatter(ch_nodes_x, ch_nodes_y, s=100, c=ch_color, marker='P', # P for Plus (filled) or X
                    edgecolors='black', linewidth=0.5, label=f'簇头 ({len(ch_nodes_x)})', zorder=3)
    if dead_nodes_x:
        plt.scatter(dead_nodes_x, dead_nodes_y, s=30, c=dead_color, marker='x', alpha=0.6, label=f'死亡节点 ({len(dead_nodes_x)})')
    
    # 绘制基站
    plt.scatter(base_x, base_y, s=180, c=bs_color, marker='^', edgecolors='black', linewidth=1, label='基站', zorder=4)
    # (可选) 绘制普通节点到其簇头的连线
    if cluster_heads_ids is not None: # 确保 cluster_heads_ids 传入的是ID列表
        for node in nodes_data:
            if node["status"] == "active" and node["role"] == "normal" and node["cluster_id"] != -1:
                # 找到簇头节点的数据
                ch_node = next((n for n in nodes_data if n["id"] == node["cluster_id"]), None)
                if ch_node and ch_node["status"] == "active": # 确保簇头也存活
                    plt.plot([node["position"][0], ch_node["position"][0]],
                             [node["position"][1], ch_node["position"][1]],
                             linestyle=':', linewidth=0.7, color='gray', alpha=0.5, zorder=1)
    
    title = '网络拓扑图'
    if current_round is not None:
        title += f' (第 {current_round} 轮)'
    plt.title(title, fontsize=16)
    plt.xlabel('X坐标 (米)', fontsize=12)
    plt.ylabel('Y坐标 (米)', fontsize=12)
    plt.xlim(-10, area_w + 10) # 留出一些边距
    plt.ylim(-10, area_h + 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
        
    # 确保输出目录存在
    output_dir = Path(filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plt.savefig(filename, dpi=150, bbox_inches='tight') # 降低dpi以加快保存速度
        logger.info(f"网络拓扑可视化已保存到 {filename}")
    except Exception as e:
        logger.error(f"保存可视化图像时出错: {e}")
    
    # 在脚本模式下，plt.show() 会阻塞，直到图形关闭
    # 如果是在循环中生成多张图，可能不希望阻塞
    # plt.show(block=False) # 非阻塞，但可能一闪而过
    # plt.pause(0.1) # 短暂暂停，让GUI有机会刷新
    plt.close() # 保存后关闭图形，避免打开过多窗口