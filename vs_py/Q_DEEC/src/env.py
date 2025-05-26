import yaml
from pathlib import Path
import numpy as np
import math
import random
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.log import logger # 从 utils 包导入 logger
from utils.fuzzy import NormalNodeCHSelectionFuzzySystem 


def _poisson_disk_sampling(width, height, min_dist, num_nodes_target, k_samples):
    """
    使用Bridson的泊松圆盘采样算法生成节点位置。
    这是一个辅助函数，供 WSNEnv._init_nodes 调用。
    """
    if num_nodes_target == 0:
        return []
    if min_dist <= 0:
        logger.warning("泊松圆盘采样：min_dist 必须为正。如果 num_nodes_target > 0，将退化为随机放置。")
        return [[random.uniform(0, width), random.uniform(0, height)] for _ in range(num_nodes_target)]

    cell_size = min_dist / math.sqrt(2) # 2D
    grid_cols = math.ceil(width / cell_size)
    grid_rows = math.ceil(height / cell_size)
    grid = [[None for _ in range(grid_rows)] for _ in range(grid_cols)]
    
    points = []
    active_list = []

    def add_point_to_sampler(p_coord):
        points.append(p_coord)
        active_list.append(p_coord)
        px, py = p_coord
        grid_x = math.floor(px / cell_size)
        grid_y = math.floor(py / cell_size)
        grid[grid_x][grid_y] = p_coord

    if num_nodes_target > 0:
        first_point = [random.uniform(0, width), random.uniform(0, height)]
        add_point_to_sampler(first_point)
        #logger.debug(f"泊松圆盘采样：初始点位于 [{first_point[0]:.2f}, {first_point[1]:.2f}]")

    # 上限是为了防止当 min_dist 过小时生成过多的点
    # 通常泊松盘采样会填满空间，然后我们再从中选取num_nodes_target个
    # 这里设置一个比num_nodes_target稍大的上限，比如 num_nodes_target * 2 或 1.5
    # 或者，可以允许它生成更多，然后在最后采样。
    # 为了更符合“生成num_nodes_target个”的意图，我们先生成足够多的点，然后采样。
    # 一个更鲁棒的方法是持续生成直到活动列表为空，然后看点数。
    # 此处采用原逻辑：生成点数上限为 num_nodes_target * 2
    max_points_to_generate = num_nodes_target * 2 if num_nodes_target > 0 else 0
    #if num_nodes_target == 0: max_points_to_generate = 0


    while active_list and len(points) < max_points_to_generate:
        active_idx = random.randrange(len(active_list))
        current_point = active_list[active_idx]
        cx, cy = current_point
        found_candidate_for_current_point = False

        for _ in range(k_samples):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(min_dist, 2 * min_dist) 
            nx = cx + radius * math.cos(angle)
            ny = cy + radius * math.sin(angle)

            if 0 <= nx < width and 0 <= ny < height:
                candidate_p = [nx, ny]
                candidate_gx = math.floor(nx / cell_size)
                candidate_gy = math.floor(ny / cell_size)
                is_valid_candidate = True
                
                # 检查邻域 (候选点单元格周围的5x5单元格，因为一个点可能影响到半径为2*cell_size的区域)
                # 对于Bridson算法，检查半径r内的点，所以检查点所在单元格及其周围8个单元格就够了（如果cell_size=r/sqrt(d)）。
                # 但为了安全，检查更广一点的区域（例如5x5个单元格）是常见的，确保覆盖所有可能的冲突。
                # 这里的检查范围是基于候选点，检查其周围2个单元格的距离。
                for gx_offset in range(-2, 3): 
                    for gy_offset in range(-2, 3):
                        check_gx = candidate_gx + gx_offset
                        check_gy = candidate_gy + gy_offset
                        if 0 <= check_gx < grid_cols and 0 <= check_gy < grid_rows:
                            neighbor_in_cell = grid[check_gx][check_gy]
                            if neighbor_in_cell:
                                dist_sq = (nx - neighbor_in_cell[0])**2 + (ny - neighbor_in_cell[1])**2
                                if dist_sq < min_dist**2:
                                    is_valid_candidate = False
                                    break 
                        if not is_valid_candidate: break
                    if not is_valid_candidate: break
                
                if is_valid_candidate:
                    add_point_to_sampler(candidate_p)
                    found_candidate_for_current_point = True
                    if len(points) >= max_points_to_generate: break # 提前退出
            if found_candidate_for_current_point and len(points) >= max_points_to_generate: break # 提前退出        
             
        
        if not found_candidate_for_current_point:
            active_list.pop(active_idx)

    logger.info(f"泊松圆盘采样（最小距离={min_dist:.2f}m, k={k_samples}）初步生成了 {len(points)} 个点。")

    if not points: # 如果一个点都没生成
        if num_nodes_target > 0:
            logger.warning("泊松圆盘采样未能生成任何点。将退化为随机放置。")
            return [[random.uniform(0, width), random.uniform(0, height)] for _ in range(num_nodes_target)]
        return []

    if len(points) < num_nodes_target:
        logger.warning(
            f"泊松圆盘采样仅能生成 {len(points)} 个节点，少于目标数量 {num_nodes_target}。"
            f" 这可能是因为对于给定的区域和节点数量，最小距离 ({min_dist:.2f}m) 设置过大。"
            " 将使用所有已生成的节点，并随机补充不足的节点（如果需要）。"
        )
        # 如果需要严格数量，可以补充随机点，但这会破坏泊松分布的特性
        # current_points = list(points) # 复制
        # while len(current_points) < num_nodes_target:
        #     # 简单的随机添加，可能不满足min_dist
        #     new_point = [random.uniform(0, width), random.uniform(0, height)]
        #     # (更复杂的逻辑：尝试添加满足min_dist的点，但这会很慢)
        #     current_points.append(new_point)
        # return current_points
        return random.sample(points, len(points)) # 返回所有能生成的点
    else:
        #logger.info(f"从 {len(points)} 个生成的点中随机选择 {num_nodes_target} 个节点。")
        return random.sample(points, num_nodes_target)

class WSNEnv:
    def __init__(self, config_path=None):
        # 自动计算配置文件绝对路径
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yml"
        else:
            config_path = Path(config_path)
        
        logger.info(f"从 {config_path} 加载配置。")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            logger.info("配置加载成功。")
        except FileNotFoundError:
            logger.error(f"配置文件 {config_path} 未找到。")
            raise
        except yaml.YAMLError as e:
            logger.error(f"解析YAML文件 {config_path} 出错: {e}")
            raise
        except Exception as e:
            logger.error(f"加载配置时发生未知错误: {e}")
            raise
        # 初始化节点列表（不构建拓扑）
        self.nodes = []
        self.current_round = 0
        self.cluster_heads = [] # 存储当前轮次的簇头ID

        deec_cfg = self.config.get('deec', {})
        self.p_opt_initial = deec_cfg.get('p_opt', 0.1) # 期望的簇头比例
        self.p_opt_current = self.p_opt_initial # 当前轮次使用的p_opt，可以动态调整
        self.max_communication_range_increase_factor = deec_cfg.get('max_comm_range_increase_factor', 1.5) # 允许通信范围增加的最大倍数
        self.min_ch_to_node_ratio_target = deec_cfg.get('min_ch_to_node_ratio', 0.05) # 目标最小簇头与节点比例，用于动态调整p_opt

        # E_total_initial 用于DEEC计算，每个节点的初始能量可能不同，DEEC原版假设相同
        # 我们这里假设所有节点的初始能量相同，取第一个节点的初始能量作为参考
        self.E0 = self.config.get('energy', {}).get('initial', 1.0) 

        self._init_nodes()
        logger.info(f"WSN 环境初始化完成，共 {len(self.nodes)} 个节点。")

    def   _init_nodes(self):
        logger.info("开始初始化节点位置和属性...")
        network_cfg = self.config.get('network', {})
        energy_cfg = self.config.get('energy', {})

        node_count_target = network_cfg.get('node_count', 0)
        area_width, area_height = network_cfg.get('area_size', [100, 100])
        
        if node_count_target == 0:
            logger.info("目标节点数量为0，不生成任何节点。")
            self.nodes = []
            return

        # 获取泊松圆盘采样参数 (如果config中没有，则使用默认值)
        DEFAULT_MIN_DIST_FACTOR = network_cfg.get('poisson_disk_min_dist_factor')
        DEFAULT_K_SAMPLES_BEFORE_REJECTION = network_cfg.get('poisson_disk_k_samples')
        min_dist_factor = network_cfg.get('poisson_disk_min_dist_factor', DEFAULT_MIN_DIST_FACTOR)
        k_samples = network_cfg.get('poisson_disk_k_samples', DEFAULT_K_SAMPLES_BEFORE_REJECTION)

        # 计算最小距离 r
        # r = sqrt((area_width * area_height * C) / node_count_target)
        # 避免 node_count_target 为0时除零错误 (虽然上面已经处理了 node_count_target == 0 的情况)
        if node_count_target > 0 :
            min_r_distance = math.sqrt((area_width * area_height * min_dist_factor) / node_count_target)
        else: # Should not happen if node_count_target is 0 due to earlier return
            min_r_distance = 0 # Or some other sensible default / error

        #logger.info(f"目标节点数量: {node_count_target}, 区域: {area_width}x{area_height}m")
        #logger.info(f"泊松圆盘采样参数: 最小距离因子 C={min_dist_factor}, k_samples={k_samples}")
        #logger.info(f"计算得到的泊松圆盘采样最小距离 (r): {min_r_distance:.2f}m")

        # 生成节点位置
        node_positions = _poisson_disk_sampling(
            width=area_width,
            height=area_height,
            min_dist=min_r_distance,
            num_nodes_target=node_count_target,
            k_samples=k_samples
        )
        
        initial_energy = energy_cfg.get('initial') # 默认初始能量
        #logger.info(f"为 {len(node_positions)} 个节点设置初始能量: {initial_energy} J")

        self.nodes = []
        for i, pos in enumerate(node_positions):
            node_data = {
                "id": i,
                "position": [pos[0], pos[1]], #确保是列表
                "energy": initial_energy,
                "initial_energy": initial_energy, # DEEC需要初始能量
                "tx_count": 0,
                "rx_count": 0,
                "status": "active", # 可以添加其他初始状态
                "neighbors":[],
                "role": "normal", # 新增：normal, cluster_head
                "cluster_id": -1, # 所属簇头的ID，-1表示未加入任何簇
                "is_CH_candidate_this_round": False, # DEEC中，节点是否在本轮有资格成为CH
                "CH_selection_counter": 0, # DEEC中，节点成为CH的次数（用于 Tn(i)）
                "base_communication_range": network_cfg.get('communication_range', 100), # 基础通信范围
                "current_communication_range": network_cfg.get('communication_range', 100) # 当前通信范围，可以动态调整
            }
            self.nodes.append(node_data)
            # logger.debug(f"已创建节点 {i}: 位置 [{pos[0]:.2f}, {pos[1]:.2f}], 能量 {initial_energy} J")
        
        if len(self.nodes) != node_count_target and len(node_positions) == node_count_target : # 修正判断条件
             logger.warning(f"最终生成的节点数量 ({len(self.nodes)}) 与目标数量 ({node_count_target}) 不符，但位置已生成。")
        elif len(self.nodes) != len(node_positions):
             logger.warning(f"最终生成的节点数量 ({len(self.nodes)}) 与位置数量 ({len(node_positions)}) 不符。")

    def _calculate_current_average_energy(self):
        """计算当前网络中所有存活节点的平均能量"""
        total_energy = 0
        alive_nodes_count = 0
        for node in self.nodes:
            if node["status"] == "active":
                total_energy += node["energy"]
                alive_nodes_count += 1
        return total_energy / alive_nodes_count if alive_nodes_count > 0 else 0
    
    def deec_cluster_head_election(self):
        """执行DEEC协议的簇头选举"""
        logger.info(f"第 {self.current_round} 轮：开始DEEC簇头选举...")
        self.cluster_heads = [] # 清空上一轮的簇头
        for node in self.nodes: # 重置角色
            if node["status"] == "active":
                node["role"] = "normal"
                node["cluster_id"] = -1
                node["current_communication_range"] = node["base_communication_range"] 
        
        num_alive_nodes = self.get_alive_nodes()
        if num_alive_nodes == 0:
            logger.info("没有存活节点，无法选举簇头。")
            return

        # E_avg_current ( E_bar(r) in DEEC paper)
        current_avg_energy = self._calculate_current_average_energy()
        if current_avg_energy <= 0: # 防止除零
            logger.warning("当前网络平均能量为0或负，无法计算DEEC概率。")
            return

        # 期望成为CH的节点数
        num_expected_chs_ideal  = self.p_opt_current * num_alive_nodes
        eligible_nodes_for_ch = [n for n in self.nodes if n["status"] == "active"]
        
        # 对每个节点计算其成为CH的概率并选举
        temp_ch_candidates = []
        for node in eligible_nodes_for_ch:
            prob_i = self.p_opt_current * (node["energy"] / current_avg_energy) if current_avg_energy > 0 else 0
            prob_i = min(1.0, prob_i) if prob_i > 0 else 0

            # 简化：每个合格节点独立以 prob_i 概率成为CH候选
            # 实际DEEC中，节点成为CH后在一个epoch内通常不再参与选举
            # 我们这里简化为每轮都可能重新选举，但可以通过 CH_selection_counter 来影响概率（如果需要更复杂的DEEC）
            if random.random() < prob_i:
                temp_ch_candidates.append(node["id"])
        
        # 如果候选CH数量远超或远少于期望，可以做一些调整
        # 这里我们先直接使用选出的候选，后续通过assign_nodes_to_clusters后的孤立节点情况来调整p_opt
        if temp_ch_candidates:
            # 如果选出的候选过多，可以按能量或其他标准筛选，或随机选取接近期望数量的
            if len(temp_ch_candidates) > num_expected_chs_ideal * 1.5 and num_expected_chs_ideal > 0:
                 logger.debug(f"初步选出 {len(temp_ch_candidates)} 个CH候选，多于期望 {num_expected_chs_ideal:.1f}*1.5，将进行筛选。")
                 # 按能量排序，选能量高的 (或者随机选)
                 temp_ch_candidates.sort(key=lambda id: self.nodes[id]["energy"], reverse=True)
                 self.cluster_heads = temp_ch_candidates[:max(1,int(num_expected_chs_ideal * 1.2))] # 保留下限为1，上限为期望的1.2倍
            else:
                 self.cluster_heads = temp_ch_candidates
            
            for ch_id in self.cluster_heads:
                self.nodes[ch_id]["role"] = "cluster_head"
                self.nodes[ch_id]["cluster_id"] = ch_id
            logger.info(f"DEEC选举完成，共选出 {len(self.cluster_heads)} 个簇头: {self.cluster_heads}")
        
        if not self.cluster_heads and num_alive_nodes > 0:
            logger.warning("本轮未能通过概率选举出任何簇头。")
            # 强制选择 (保持之前的逻辑)
            highest_energy_node = max((n for n in self.nodes if n["status"] == "active"), key=lambda x: x["energy"], default=None)
            if highest_energy_node:
                highest_energy_node["role"] = "cluster_head"
                highest_energy_node["cluster_id"] = highest_energy_node["id"]
                self.cluster_heads.append(highest_energy_node["id"])
                logger.info(f"强制选择能量最高的节点 {highest_energy_node['id']} 作为簇头。")

        logger.info(f"DEEC选举完成，共选出 {len(self.cluster_heads)} 个簇头: {self.cluster_heads}")
        if not self.cluster_heads and num_alive_nodes > 0:
            logger.warning("本轮未能选举出任何簇头。可能需要调整DEEC参数或网络状态不佳。")
            # 可以考虑强制选择一个能量最高的节点作为CH，以保证网络连通性（如果需要）
            if num_alive_nodes > 0:
                highest_energy_node = max((n for n in self.nodes if n["status"] == "active"), key=lambda x: x["energy"], default=None)
                if highest_energy_node:
                    highest_energy_node["role"] = "cluster_head"
                    highest_energy_node["cluster_id"] = highest_energy_node["id"]
                    self.cluster_heads.append(highest_energy_node["id"])
                    logger.info(f"强制选择能量最高的节点 {highest_energy_node['id']} 作为簇头。")

    def assign_nodes_to_clusters(self, attempt=1):
        if not self.cluster_heads:
            logger.info("没有簇头可选，节点无法加入簇。")
            for node in self.nodes:
                if node["status"] == "active" and node["role"] == "normal":
                    node["cluster_id"] = -1
            return len([n for n in self.nodes if n["status"] == "active" and n["role"] == "normal"]) # 返回孤立节点数

        logger.info(f"开始将普通节点分配给簇头 (尝试次数: {attempt})...")
        num_assigned = 0
        isolated_nodes_count = 0

        for node_data in self.nodes: # 使用不同的变量名以避免覆盖外部的node
            if node_data["status"] == "active" and node_data["role"] == "normal":
                node_data["cluster_id"] = -1 # 每轮尝试前先重置
                min_dist_to_ch = float('inf')
                assigned_ch_id = -1
                
                # 在当前尝试中，节点使用的通信范围
                # 第一次尝试使用基础通信范围，后续尝试可以增加
                current_comm_range = node_data["base_communication_range"]
                if attempt > 1:
                    # 简单增加通信范围的策略，可以更精细化
                    increase_factor = min(1 + (attempt - 1) * 0.25, self.max_communication_range_increase_factor)
                    current_comm_range = node_data["base_communication_range"] * increase_factor
                node_data["current_communication_range"] = current_comm_range


                for ch_id in self.cluster_heads:
                    if not (0 <= ch_id < len(self.nodes)) or self.nodes[ch_id]["status"] == "dead":
                        continue
                    
                    distance = self.calculate_distance(node_data["id"], ch_id)
                    # 节点必须在CH的通信范围内，并且CH也在节点的当前通信范围内
                    # (简化：假设对称信道，只检查一方是否在另一方的通信范围内即可)
                    if distance <= node_data["current_communication_range"] and distance <= self.nodes[ch_id]["current_communication_range"]: # CH也用current_communication_range
                        if distance < min_dist_to_ch:
                            min_dist_to_ch = distance
                            assigned_ch_id = ch_id
                
                if assigned_ch_id != -1:
                    node_data["cluster_id"] = assigned_ch_id
                    num_assigned +=1
                    # logger.debug(f"节点 {node_data['id']} 加入簇头 {assigned_ch_id} (距离: {min_dist_to_ch:.2f}m, 通信范围: {current_comm_range:.1f}m)。")
                else:
                    # logger.debug(f"节点 {node_data['id']} 未能找到可达的簇头加入 (通信范围: {current_comm_range:.1f}m)。")
                    isolated_nodes_count +=1
        
        logger.info(f"共 {num_assigned} 个普通节点完成了簇分配。剩余孤立节点数: {isolated_nodes_count} (尝试次数: {attempt})")
        return isolated_nodes_count
    
    def handle_isolated_nodes(self):
        """处理孤立节点，尝试增加通信范围或调整p_opt"""
        max_attempts = 3 # 最多尝试增加通信范围的次数
        for attempt in range(1, max_attempts + 1):
            isolated_count = self.assign_nodes_to_clusters(attempt=attempt)
            if isolated_count == 0:
                logger.info(f"所有节点均已分配到簇 (在第 {attempt} 次尝试后)。")
                return True # 所有节点都分配好了
            logger.info(f"第 {attempt} 次分配后，仍有 {isolated_count} 个孤立节点。")

        # 如果多次增加通信范围后仍有孤立节点，考虑下一轮增加p_opt
        num_alive_normal_nodes = len([n for n in self.nodes if n["status"] == "active" and n["role"] == "normal" and n["cluster_id"] == -1])
        if num_alive_normal_nodes > 0 : # 确认是普通节点孤立
            # 动态调整 p_opt_current
            # 简单策略：如果孤立节点比例过高，轻微增加 p_opt
            # 注意：p_opt不应无限增大，例如不超过初始值的2-3倍或一个上限（如0.3）
            current_ch_ratio = len(self.cluster_heads) / self.get_alive_nodes() if self.get_alive_nodes() > 0 else 0
            if current_ch_ratio < self.min_ch_to_node_ratio_target or isolated_count > self.get_alive_nodes() * 0.1: # 如果CH比例过低或孤立节点超过10%
                self.p_opt_current = min(self.p_opt_initial * 2, self.p_opt_current * 1.1, 0.3) 
                logger.info(f"由于存在较多孤立节点或CH比例低，下一轮 p_opt 调整为: {self.p_opt_current:.3f}")
            elif len(self.cluster_heads) > self.get_alive_nodes() * self.p_opt_initial * 1.5 : # 如果CH过多
                self.p_opt_current = max(self.p_opt_initial * 0.8, self.p_opt_current * 0.95) # 适当减少
                logger.info(f"由于CH数量过多，下一轮 p_opt 调整为: {self.p_opt_current:.3f}")
            else:
                 # 如果孤立节点不多，且CH数量合适，可以逐渐恢复p_opt
                if self.p_opt_current > self.p_opt_initial:
                    self.p_opt_current = max(self.p_opt_initial, self.p_opt_current * 0.98)


        # 对于本轮仍然孤立的节点，它们将不参与数据传输或直接尝试联系基站（如果策略允许）
        # 在我们的模型中，它们暂时就是孤立的。
        return isolated_count == 0
    
    def simulate_round_energy_consumption(self):
        """模拟一轮中由于基本操作（如感知、空闲监听）产生的能量消耗"""
        # 这是一个非常简化的模型，你可以根据需要调整
        idle_listening_cost_per_round = self.config.get('energy',{}).get('idle_listening_per_round', 1e-5) # J
        sensing_cost_per_round = self.config.get('energy',{}).get('sensing_per_round', 5e-6) # J
        
        nodes_to_kill_this_round = []
        for node in self.nodes:
            if node["status"] == "active":
                cost = idle_listening_cost_per_round + sensing_cost_per_round
                node["energy"] -= cost
                if node["energy"] < 0:
                    node["energy"] = 0
                    nodes_to_kill_this_round.append(node["id"]) # 先记录，后处理，避免在迭代中修改列表导致问题
        
        for node_id in nodes_to_kill_this_round:
            if self.nodes[node_id]["status"] == "active": # 再次确认，防止重复kill
                 self.kill_node(node_id)
        
    def calculate_distance(self, node1_idx, node2_idx):
        """计算两个节点之间的欧氏距离 (基于节点ID)"""
        # 修改为接受节点ID，或直接接受节点字典
        node1 = self.nodes[node1_idx]
        node2 = self.nodes[node2_idx]
        x1, y1 = node1["position"]
        x2, y2 = node2["position"]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def update_energy(self, node_id, distance, is_tx=True):
        """
        更新节点能量（基于一阶无线电模型）
        - is_tx=True: 发送数据
        - is_tx=False: 接收数据
        """
        energy_cfg = self.config.get('energy', {})
        E_elec = energy_cfg.get('rx_cost') # 电子能量消耗，假设每比特
        tx_amp_fs = energy_cfg.get('tx_amp_fs', 10e-12)    # 自由空间模型的放大器能量 (epsilon_fs)
        tx_amp_mp = energy_cfg.get('tx_amp_mp', 0.0013e-12) # 多径衰落模型的放大器能量 (epsilon_mp)
        threshold_d0 = energy_cfg.get('threshold_d0')
        rx_cost_per_bit = energy_cfg.get('rx_cost')
        packet_size_bits = self.config["simulation"]["packet_size"]
        if node_id < 0 or node_id >= len(self.nodes):
            logger.error(f"更新能量：无效的节点ID {node_id}")
            return

        node = self.nodes[node_id]
        if node["energy"] <= 0: # 节点能量耗尽
            # logger.warning(f"节点 {node_id} 能量已耗尽，无法执行操作。")
            return

        energy_cost = 0
        if is_tx:
            # 发送能量 E_TX(k, d) = E_elec*k + E_amp*k*d^alpha
            cost_elec = E_elec * packet_size_bits
            if distance < threshold_d0:
                cost_amp = tx_amp_fs * packet_size_bits * (distance ** 2)
            else:
                cost_amp = tx_amp_mp * packet_size_bits * (distance ** 4)
            energy_cost = cost_elec + cost_amp
            node["tx_count"] += 1
        else:
            # 接收能量 E_RX(k) = E_elec*k
            energy_cost = rx_cost_per_bit * packet_size_bits # 或者 E_elec * packet_size_bits
            node["rx_count"] += 1
        
        node["energy"] -= energy_cost
        if node["energy"] < 0:
            node["energy"] = 0 # 能量不能为负
            node["status"] = "dead" # 标记节点死亡
            # logger.info(f"节点 {node_id} 能量耗尽。"

    def _build_spatial_index(self):
        from scipy.spatial import KDTree
        active_node_positions = [n["position"] for n in self.nodes if n["status"] == "active"]
        self.active_node_ids_for_kdtree = [n["id"] for n in self.nodes if n["status"] == "active"]
        if not active_node_positions:
            self._position_tree = None
            return
        self._position_tree = KDTree(active_node_positions)
        
    def get_node_neighbors(self, node_id, max_distance=None):
        if not (0 <= node_id < len(self.nodes)):
             logger.error(f"get_node_neighbors: 无效的节点ID {node_id}")
             return []
        
        # 确保索引在使用前已构建或更新
        # 简单起见，每次调用都重建（对于动态网络可能需要更优方案）
        # 或者在每轮开始时更新一次
        self._build_spatial_index() 
        if self._position_tree is None:
            return []
            
        node = self.nodes[node_id]
        if node["status"] == "dead": return []

        max_dist = max_distance if max_distance is not None else node["communication_range"]
        
        # KDTree查询返回的是在 active_node_positions 列表中的索引
        indices_in_kdtree = self._position_tree.query_ball_point(node["position"], max_dist)
        
        # 将KDTree索引映射回原始节点ID
        neighbor_ids = []
        for kdtree_idx in indices_in_kdtree:
            original_node_id = self.active_node_ids_for_kdtree[kdtree_idx]
            if original_node_id != node_id: # 排除自身
                 neighbor_ids.append(original_node_id)
        return neighbor_ids

    def _handle_packet_transmission(self,node_id):
        pass

    def get_network_energy(self):
        logger.info("get_network_energy")
        total_energy = 0
        for node in self.nodes:
            if node["status"] != "dead":
                total_energy += node["energy"]
        return total_energy
    
    def get_alive_nodes(self):
        logger.info("get_alive_nodes")
        alive_nodes = 0
        for node in self.nodes:
            if node["status"] != "dead":
                alive_nodes += 1
        return alive_nodes
    
    def kill_node(self,node_id):
        # ... (保持不变) ...
        if not (0 <= node_id < len(self.nodes)):
            logger.error(f"kill_node:无效的节点ID {node_id}")
            return False
        cur_node = self.nodes[node_id]
        if cur_node["status"] == "dead": return True # 已经死了
        cur_node["status"] = "dead"
        cur_node["energy"] = 0 # 确保能量为0
        logger.info(f"节点 {node_id} 已被标记为死亡。")
        return True

    def step(self, current_round_num):
        """执行一个仿真轮次的逻辑"""
        self.current_round = current_round_num
        logger.info(f"--- 开始第 {self.current_round} 轮 ---")

        for node in self.nodes:
            if node["status"] == "active":
                 node["current_communication_range"] = node["base_communication_range"]
        # 1. DEEC簇头选举
        self.deec_cluster_head_election()

        # 2. 普通节点加入簇 (简化为选择最近的CH)
        if self.cluster_heads:
            self.handle_isolated_nodes() # 这个函数内部会调用 assign_nodes_to_clusters 多次
        else:
            logger.warning(f"第 {self.current_round} 轮：没有簇头，所有活跃节点均为孤立。")
            # 如果没有CH，可以考虑下一轮增加p_opt
            self.p_opt_current = min(self.p_opt_initial * 1.5, self.p_opt_current * 1.2, 0.3)
            logger.info(f"由于没有CH选出，下一轮 p_opt 调整为: {self.p_opt_current:.3f}")


        # --- 后续可以添加数据传输、Q学习决策等逻辑 ---
        # 例如，普通节点基于Q学习选择CH (如果DEEC只是提供了候选)
        # CH基于Q学习选择下一跳等

        # 模拟能量消耗 (非常简化，实际应基于具体操作)
        # for node_data in self.nodes:
        #     if node_data["status"] == "active":
        #         # 假设每个节点每轮都有一些基础消耗或操作消耗
        #         # self.update_energy(node_data["id"], distance=10, is_tx=True) # 示例性消耗
        #         pass
        
        self.simulate_round_energy_consumption()
        logger.info(f"--- 第 {self.current_round} 轮结束 ---")
        alive_nodes = self.get_alive_nodes()
        logger.info(f"轮次结束时存活节点数: {alive_nodes}")
        if alive_nodes == 0:
            logger.info("所有节点已死亡，仿真可以提前结束。")
            return False # 返回False表示仿真可以结束
        return True # 返回True表示仿真继续

    def _get_packet_loss_rate(self, distance):
        """基于距离的Log-normal阴影模型"""
        PL_d0 = 55  # 参考距离d0=1m时的路径损耗(dB)
        path_loss = PL_d0 + 10 * 3.0 * np.log10(distance) + np.random.normal(0, 4)
        snr = 10 - path_loss  # 假设发射功率10dBm
        return 1 / (1 + np.exp(snr - 5))  # Sigmoid模拟丢包率
