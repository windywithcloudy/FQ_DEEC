# utils/virtual.py
import matplotlib.pyplot as plt
from pathlib import Path
from .log import logger 

plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
DEFAULT_OUTPUT_IMAGE_FILE = "network_topology.png"
DIRECT_BS_NODE_TYPE_STR_VIZ = "DIRECT_BS_NODE" # 与env.py中一致或映射

def visualize_network(nodes_data, base_station_pos, area_dims, filename=DEFAULT_OUTPUT_IMAGE_FILE, 
                      current_round=None, 
                      # candidate_ch_ids, # 这个参数现在可能更多的是 confirmed_cluster_heads_for_epoch
                      confirmed_chs_in_epoch=None, # 改为接收本epoch确定的CH列表
                      config=None):
    if not nodes_data:
        logger.info("没有节点可供可视化。")
        return
    
    if confirmed_chs_in_epoch is None:
        confirmed_chs_in_epoch = [] 
    logger.info(f"可视化包含 {len(nodes_data)} 个节点的网络拓扑 (轮次: {current_round if current_round is not None else 'N/A'})...")
    
    area_w, area_h = area_dims
    base_x, base_y = base_station_pos
    bs_id_for_routing = -1 # 与env.py中定义一致

    viz_colors = config.get('visualization', {}).get('colors', {}) if config else {}
    ch_color = viz_colors.get('cluster_head', '#FF4500')
    normal_color = viz_colors.get('normal_node', '#1E90FF')
    bs_color = viz_colors.get('base_station', '#32CD32')
    dead_color = viz_colors.get('dead_node', '#A9A9A9')
    direct_bs_color = viz_colors.get('direct_bs_node', '#FFD700')
    ch_route_color = viz_colors.get('ch_route_line', '#8A2BE2') # 新增：CH间路由颜色 (例如紫色)
    ch_to_direct_node_route_color = viz_colors.get('ch_to_direct_node_route_line', '#FFA500') # 例如橙色


    plt.figure(figsize=(10,10))

    normal_nodes_x, normal_nodes_y = [], []
    active_ch_nodes_x, active_ch_nodes_y = [], [] # 只画活跃的CH
    dead_nodes_x, dead_nodes_y = [], []
    direct_bs_nodes_x, direct_bs_nodes_y = [], []

    active_ch_ids_map = {ch_id: nodes_data[ch_id] for ch_id in confirmed_chs_in_epoch if nodes_data[ch_id]["status"] == "active"}


    for node in nodes_data:
        if node["status"] == "dead":
            dead_nodes_x.append(node["position"][0])
            dead_nodes_y.append(node["position"][1])
        elif node.get("role_override") == "direct_to_bs":
            direct_bs_nodes_x.append(node["position"][0])
            direct_bs_nodes_y.append(node["position"][1])
        elif node["id"] in active_ch_ids_map: # 如果是本epoch确认的活跃CH
             active_ch_nodes_x.append(node["position"][0])
             active_ch_nodes_y.append(node["position"][1])
        elif node["role"] == "normal": # 其他活跃的普通节点
            normal_nodes_x.append(node["position"][0])
            normal_nodes_y.append(node["position"][1])
        # 其他情况（例如，是候选但未最终成为活跃CH的）可以不画或用不同标记

    if normal_nodes_x:
        plt.scatter(normal_nodes_x, normal_nodes_y, s=50, c=normal_color, alpha=0.8, label=f'普通节点 ({len(normal_nodes_x)})')
    if active_ch_nodes_x: 
        plt.scatter(active_ch_nodes_x, active_ch_nodes_y, s=100, c=ch_color, marker='P',  # type: ignore
                    edgecolors='black', linewidth=0.5, label=f'活跃CH ({len(active_ch_nodes_x)})', zorder=3)
    if dead_nodes_x:
        plt.scatter(dead_nodes_x, dead_nodes_y, s=30, c=dead_color, marker='x', alpha=0.6, label=f'死亡节点 ({len(dead_nodes_x)})') # type: ignore
    if direct_bs_nodes_x: 
        plt.scatter(direct_bs_nodes_x, direct_bs_nodes_y, s=70, c=direct_bs_color, marker='D',  # type: ignore
                    edgecolors='black', linewidth=0.5, label=f'直连BS节点 ({len(direct_bs_nodes_x)})', zorder=3)

    plt.scatter(base_x, base_y, s=180, c=bs_color, marker='^', edgecolors='black', linewidth=1, label='基站', zorder=4) # type: ignore
    
    # --- 绘制连线 ---
    for node in nodes_data:
        if node["status"] != "active":
            continue

        # 1. 普通节点连接到其CH
        if node.get("role_override") != "direct_to_bs" and \
           node["role"] == "normal" and \
           node["cluster_id"] in active_ch_ids_map: # 确保连接到的是本epoch的活跃CH
            ch_node = active_ch_ids_map.get(node["cluster_id"]) # 从map中获取，保证是活跃的
            if ch_node: # ch_node 肯定存在且活跃
                plt.plot([node["position"][0], ch_node["position"][0]],
                         [node["position"][1], ch_node["position"][1]],
                         linestyle=':', linewidth=0.7, color='gray', alpha=0.5, zorder=1)
        
        # 2. 直连BS的节点连接到BS
        elif node.get("role_override") == "direct_to_bs":
             plt.plot([node["position"][0], base_x],
                      [node["position"][1], base_y],
                      linestyle='--', linewidth=0.8, color=direct_bs_color, alpha=0.7, zorder=1)
        
        # 3. *** 新增：活跃CH连接到其选择的下一跳 ***
        elif node["id"] in active_ch_ids_map: # 如果当前节点是活跃CH
            chosen_next_hop_id = node.get("chosen_next_hop_id")
            if chosen_next_hop_id is not None: # 并且它已经选择了一个下一跳
                next_hop_pos = None
                line_style = '-.' # CH到CH的路由用点划线
                line_color = ch_route_color
                line_width = 1.0

                if chosen_next_hop_id == bs_id_for_routing: # 下一跳是基站
                    next_hop_pos = [base_x, base_y]
                    line_style = '-' # CH到BS的路由用实线
                elif 0 <= chosen_next_hop_id < len(nodes_data) and \
                     nodes_data[chosen_next_hop_id]["status"] == "active":
                    
                    next_hop_node_data_viz = nodes_data[chosen_next_hop_id]
                    if next_hop_node_data_viz["id"] in active_ch_ids_map: # 下一跳是另一个活跃CH
                        next_hop_pos = next_hop_node_data_viz["position"]
                        # current_line_color 和 style 保持 ch_route_color 和 '-.'
                    elif next_hop_node_data_viz.get("can_connect_bs_directly", False): # !!! 新增：下一跳是直连BS的节点 !!!
                        next_hop_pos = next_hop_node_data_viz["position"]
                        current_line_color = ch_to_direct_node_route_color # 使用新颜色
                # else: 下一跳无效或不是活跃CH，不画

                if next_hop_pos:
                    plt.plot([node["position"][0], next_hop_pos[0]],
                            [node["position"][1], next_hop_pos[1]],
                            linestyle=line_style, linewidth=line_width, color=line_color, # 使用 current_ 前缀的变量
                            alpha=0.8, zorder=2) # zorder=2 使其在普通节点连线之上
                    # (可选) 绘制箭头指示方向
                    # plt.arrow(node["position"][0], node["position"][1],
                    #           next_hop_pos[0] - node["position"][0],
                    #           next_hop_pos[1] - node["position"][1],
                    #           head_width=5, head_length=7, fc=line_color, ec=line_color, length_includes_head=True, alpha=0.8, zorder=2)


    # ... (title, labels, grid, savefig, close - 与之前类似) ...
    title = '网络拓扑图'
    if current_round is not None: title += f' (第 {current_round} 轮)'
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