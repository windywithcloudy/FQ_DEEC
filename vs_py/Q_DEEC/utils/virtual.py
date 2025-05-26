# utils/virtual.py
import matplotlib.pyplot as plt
from pathlib import Path
from .log import logger # 从同级目录的log.py导入logger

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统
plt.rcParams['axes.unicode_minus'] = False
DEFAULT_OUTPUT_IMAGE_FILE = "network_topology.png"

def visualize_network(nodes_data, base_station_pos, area_dims, filename=DEFAULT_OUTPUT_IMAGE_FILE, current_round=None, candidate_ch_ids=None, config=None):
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
    direct_bs_color = viz_colors.get('direct_bs_node', '#FFD700') # 获取新颜色
    
    plt.figure(figsize=(10,10))

    # 绘制节点
    normal_nodes_x, normal_nodes_y = [], []
    candidate_ch_nodes_x, candidate_ch_nodes_y = [], [] # 重命名
    dead_nodes_x, dead_nodes_y = [], []
    direct_bs_nodes_x, direct_bs_nodes_y = [], [] # 新增

    # 区分不同角色的节点
    final_active_ch_ids_for_viz = set() # 用于绘制连线的最终活跃CH
    if candidate_ch_ids: # 首先确定哪些是本轮的CH角色（在分阶段模型中，这可能在Q学习分配后才最终确定）
        # 简单处理：如果一个节点是候选CH，并且在Q学习分配后有成员或自己就是活跃的，才标为CH颜色
        # 在当前阶段，我们只知道候选CH，所以先按候选CH画
        # 后续Q学习阶段完成后，可以传入最终的活跃CH列表
        # 这里我们先假设 candidate_ch_ids 就是我们要标记为CH颜色的节点
        pass # 逻辑会放在下面的循环

    for node in nodes_data:
        if node["status"] == "dead":
            dead_nodes_x.append(node["position"][0])
            dead_nodes_y.append(node["position"][1])
        elif node.get("role_override") == "direct_to_bs" or node.get("can_connect_bs_directly"): # 检查直连BS状态
            direct_bs_nodes_x.append(node["position"][0])
            direct_bs_nodes_y.append(node["position"][1])
        elif candidate_ch_ids and node["id"] in candidate_ch_ids: # 如果是候选CH
             candidate_ch_nodes_x.append(node["position"][0])
             candidate_ch_nodes_y.append(node["position"][1])
             # 假设所有候选CH都是活跃的CH用于可视化连线 (在分阶段模型中，这可能需要调整)
             final_active_ch_ids_for_viz.add(node["id"])
        elif node["role"] == "cluster_head": # 如果Q学习分配后，节点的role被设为CH
            candidate_ch_nodes_x.append(node["position"][0]) # 也用CH颜色画
            candidate_ch_nodes_y.append(node["position"][1])
            final_active_ch_ids_for_viz.add(node["id"])
        else: # 普通节点
            normal_nodes_x.append(node["position"][0])
            normal_nodes_y.append(node["position"][1])

    if normal_nodes_x:
        plt.scatter(normal_nodes_x, normal_nodes_y, s=50, c=normal_color, alpha=0.8, label=f'普通节点 ({len(normal_nodes_x)})')
    if candidate_ch_nodes_x: # 使用候选CH列表绘制
        plt.scatter(candidate_ch_nodes_x, candidate_ch_nodes_y, s=100, c=ch_color, marker='P', 
                    edgecolors='black', linewidth=0.5, label=f'候选/活跃CH ({len(candidate_ch_nodes_x)})', zorder=3)
    if dead_nodes_x:
        plt.scatter(dead_nodes_x, dead_nodes_y, s=30, c=dead_color, marker='x', alpha=0.6, label=f'死亡节点 ({len(dead_nodes_x)})')
    if direct_bs_nodes_x: # 绘制直连BS节点
        plt.scatter(direct_bs_nodes_x, direct_bs_nodes_y, s=70, c=direct_bs_color, marker='D', # D for Diamond
                    edgecolors='black', linewidth=0.5, label=f'直连BS节点 ({len(direct_bs_nodes_x)})', zorder=3)

    plt.scatter(base_x, base_y, s=180, c=bs_color, marker='^', edgecolors='black', linewidth=1, label='基站', zorder=4)
    
    # 绘制连线
    if final_active_ch_ids_for_viz: 
        for node in nodes_data:
            # 普通节点连接到其CH
            if node["status"] == "active" and node.get("role_override") != "direct_to_bs" and \
               node["role"] == "normal" and node["cluster_id"] in final_active_ch_ids_for_viz:
                ch_node = next((n for n in nodes_data if n["id"] == node["cluster_id"]), None)
                if ch_node and ch_node["status"] == "active":
                    plt.plot([node["position"][0], ch_node["position"][0]],
                             [node["position"][1], ch_node["position"][1]],
                             linestyle=':', linewidth=0.7, color='gray', alpha=0.5, zorder=1)
            # 直连BS的节点连接到BS
            elif node["status"] == "active" and node.get("role_override") == "direct_to_bs":
                 plt.plot([node["position"][0], base_x],
                          [node["position"][1], base_y],
                          linestyle='--', linewidth=0.8, color=direct_bs_color, alpha=0.7, zorder=1)

    # ... (title, labels, grid, savefig, close - 与之前类似) ...
    title = '网络拓扑图'
    if current_round is not None:
        title += f' (第 {current_round} 轮)'
    plt.title(title, fontsize=16)
    plt.xlabel('X坐标 (米)', fontsize=12)
    plt.ylabel('Y坐标 (米)', fontsize=12)
    plt.xlim(-10, area_dims[0] + 10) 
    plt.ylim(-10, area_dims[1] + 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_dir = Path(filename).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        logger.info(f"网络拓扑可视化已保存到 {filename}")
    except Exception as e:
        logger.error(f"保存可视化图像时出错: {e}")
    plt.close()