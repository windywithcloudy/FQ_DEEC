# utils/virtual.py
import matplotlib.pyplot as plt
from .log import logger # 从同级目录的log.py导入logger

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统
DEFAULT_OUTPUT_IMAGE_FILE = "network_topology.png"

def visualize_network(nodes_data, base_station_pos, area_dims, filename=DEFAULT_OUTPUT_IMAGE_FILE):
    """
    可视化网络拓扑并保存到文件。

    参数:
        nodes_data (list): 节点信息列表，每个元素是一个包含 "position"键的字典。
                           例如: [{"id": 0, "position": [x, y], ...}, ...]
        base_station_pos (list): 基站坐标 [x, y]。
        area_dims (list): 网络区域尺寸 [width, height]。
        filename (str): 保存图像的文件名。
    """
    logger.info(f"可视化包含 {len(nodes_data)} 个节点的网络拓扑...")
    
    area_w, area_h = area_dims
    base_x, base_y = base_station_pos
    
    plt.figure(figsize=(9, 9))
    
    # 绘制节点
    if nodes_data:
        node_x_coords = [node["position"][0] for node in nodes_data]
        node_y_coords = [node["position"][1] for node in nodes_data]
        plt.scatter(node_x_coords, node_y_coords, s=50, c='royalblue', alpha=0.8, label=f'节点 ({len(nodes_data)})')
    
    # 绘制基站
    plt.scatter(base_x, base_y, s=180, c='crimson', marker='^', edgecolors='black', linewidth=1, label='基站')
    
    plt.title('网络拓扑图 (泊松圆盘采样)', fontsize=16)
    plt.xlabel('X坐标 (米)', fontsize=12)
    plt.ylabel('Y坐标 (米)', fontsize=12)
    plt.xlim(0, area_w)
    plt.ylim(0, area_h)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"网络拓扑可视化已保存到 {filename}")
    except Exception as e:
        logger.error(f"保存可视化图像时出错: {e}")
    
    try:
        plt.show()
    except Exception as e:
        logger.warning(f"显示图像时出错 (可能在无GUI环境运行): {e}")