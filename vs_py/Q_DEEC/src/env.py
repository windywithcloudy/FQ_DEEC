import yaml
from pathlib import Path
import numpy as np
import math
import random
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.log import logger # 从 utils 包导入 logger
from utils.fuzzy import NormalNodeCHSelectionFuzzySystem, RewardWeightsFuzzySystemForCHCompetition


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
        self.cluster_heads = []
        self.candidate_cluster_heads = [] # 重命名: 存储本轮的候选簇头ID

        deec_cfg = self.config.get('deec', {})
        self.p_opt_initial = deec_cfg.get('p_opt', 0.1) # 期望的簇头比例
        self.p_opt_current = self.p_opt_initial # 当前轮次使用的p_opt，可以动态调整
        self.max_communication_range_increase_factor = deec_cfg.get('max_comm_range_increase_factor', 1.5) # 允许通信范围增加的最大倍数
        self.min_ch_to_node_ratio_target = deec_cfg.get('min_ch_to_node_ratio', 0.05) # 目标最小簇头与节点比例，用于动态调整p_opt
        self.location_factor_enabled = deec_cfg.get('location_factor_enabled', False)
        self.optimal_bs_dist_min = deec_cfg.get('optimal_bs_dist_min', 50)
        self.optimal_bs_dist_max = deec_cfg.get('optimal_bs_dist_max', 200)
        self.penalty_factor_too_close = deec_cfg.get('penalty_factor_too_close', 0.5)
        self.penalty_factor_too_far = deec_cfg.get('penalty_factor_too_far', 0.5)

        # Q-learning parameters from config (or defaults)
        q_learning_cfg = self.config.get('q_learning', {})
        self.alpha_compete = float(q_learning_cfg.get('alpha_compete_ch', 0.1))
        self.gamma_compete = float(q_learning_cfg.get('gamma_compete_ch', 0.9))
        self.epsilon_compete_initial = float(q_learning_cfg.get('epsilon_compete_ch_initial', 0.5))
        self.epsilon_compete_decay = float(q_learning_cfg.get('epsilon_compete_ch_decay', 0.995))
        self.epsilon_compete_min = float(q_learning_cfg.get('epsilon_compete_ch_min', 0.01))
        self.current_epsilon_compete = self.epsilon_compete_initial
        self.reward_weights_adjuster = RewardWeightsFuzzySystemForCHCompetition(self.config)

        # E_total_initial 用于DEEC计算，每个节点的初始能量可能不同，DEEC原版假设相同
        # 我们这里假设所有节点的初始能量相同，取第一个节点的初始能量作为参考
        self.E0 = self.config.get('energy', {}).get('initial', 1.0) 

        self._init_nodes()
        self.confirmed_cluster_heads_previous_round = []
        self.confirmed_cluster_heads_current_round = []
        logger.info(f"WSN 环境初始化完成，共 {len(self.nodes)} 个节点。")

    def   _init_nodes(self):
        logger.info("开始初始化节点位置和属性...")
        network_cfg = self.config.get('network', {})
        energy_cfg = self.config.get('energy', {})

        node_count_target = network_cfg.get('node_count', 0)
        area_width, area_height = network_cfg.get('area_size', [100, 100])
        self.network_diagonal = math.sqrt(area_width**2 + area_height**2) if area_width > 0 and area_height > 0 else 353.55
        
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
                "time_since_last_ch": random.randint(0, 20), # 初始随机化，避免同步
                "q_table_compete_ch": {}, 
                "q_table_select_ch": {}, # 用于普通节点选择CH的Q表
                # 使用从config加载的Q学习参数
                "alpha_compete": self.alpha_compete,
                "gamma_compete": self.gamma_compete,
                "epsilon_compete": self.epsilon_compete_initial, # 每个节点维护自己的epsilon
                # ... (其他已有属性如 base_communication_range) ...
                 "base_communication_range": network_cfg.get('communication_range', 100), 
                 "current_communication_range": network_cfg.get('communication_range', 100) 
            }
            self.nodes.append(node_data)
            # logger.debug(f"已创建节点 {i}: 位置 [{pos[0]:.2f}, {pos[1]:.2f}], 能量 {initial_energy} J")
        
        if len(self.nodes) != node_count_target and len(node_positions) == node_count_target : # 修正判断条件
             logger.warning(f"最终生成的节点数量 ({len(self.nodes)}) 与目标数量 ({node_count_target}) 不符，但位置已生成。")
        elif len(self.nodes) != len(node_positions):
             logger.warning(f"最终生成的节点数量 ({len(self.nodes)}) 与位置数量 ({len(node_positions)}) 不符。")
    
    def discretize_normalized_energy(self, normalized_energy, num_bins=5):
        # ... (如之前定义) ...
        if not (0 <= normalized_energy <= 1.0001):
            normalized_energy = np.clip(normalized_energy, 0, 1)
        return min(int(normalized_energy * num_bins), num_bins - 1)

    def discretize_time_since_last_ch(self, rounds_since_last_ch):
        # ... (如之前定义) ...
        if rounds_since_last_ch <= 20: return 0
        elif rounds_since_last_ch <= 60: return 1
        else: return 2

    def discretize_neighbor_count(self, count, max_neighbors_ref=15, num_bins=3): # max_neighbors_ref 作为参考
        # ... (如之前定义) ...
        count = np.clip(count, 0, max_neighbors_ref * 1.5) # 允许略微超出参考最大值
        if count <= max_neighbors_ref / 3: return 0      
        elif count <= max_neighbors_ref * 2 / 3: return 1 
        else: return 2  

    def discretize_ch_count_nearby(self, count):
        # ... (如之前定义) ...
        if count == 0: return 0 
        elif count <= 2: return 1 
        else: return 2  

    def discretize_normalized_distance_to_bs(self, normalized_distance):
        # ... (如之前定义) ...
        normalized_distance = np.clip(normalized_distance, 0, 1)
        if normalized_distance < 0.25: return 0    
        elif normalized_distance < 0.75: return 1  
        else: return 2   

    def get_discrete_state_tuple_for_competition(self, node_id):
        # ... (实现，调用上面的离散化函数，返回一个元组) ..
        raw_state = self.get_node_state_for_ch_competition(node_id)
        if raw_state is None: return None

        d_bs_normalized = raw_state["d_bs"] / self.network_diagonal if self.network_diagonal > 0 else 0
        
        # 估算一个参考最大邻居数，例如基于初始节点密度
        avg_node_density_ref = self.config.get('network',{}).get('node_count',100) / \
                               (self.config.get('network',{}).get('area_size',[1,1])[0] * \
                                self.config.get('network',{}).get('area_size',[1,1])[1])
        comm_r = self.nodes[node_id]['base_communication_range']
        expected_neighbors_ref = avg_node_density_ref * np.pi * (comm_r**2)
        max_n_ref = max(15, int(expected_neighbors_ref * 2)) # 取一个上限


        state_tuple = (
            self.discretize_normalized_energy(raw_state["e_self"]),
            self.discretize_time_since_last_ch(raw_state["t_last_ch"]),
            self.discretize_neighbor_count(raw_state["n_neighbor"], max_neighbors_ref=max_n_ref),
            self.discretize_normalized_energy(raw_state["e_neighbor_avg"]),
            self.discretize_ch_count_nearby(raw_state["n_ch_nearby"]),
            self.discretize_normalized_distance_to_bs(d_bs_normalized)
        )
        return state_tuple


    def get_node_state_for_ch_competition(self, node_id):
        # ... (实现，与之前讨论类似，确保返回字典) ...
        node = self.nodes[node_id]
        if node["status"] == "dead": return None

        e_self = node["energy"] / node["initial_energy"] if node["initial_energy"] > 0 else 0
        t_last_ch = node["time_since_last_ch"]
        
        # 重新获取邻居，因为网络可能变化
        # 注意：get_node_neighbors 可能需要 self._build_spatial_index()
        # 为避免在循环中重复构建，可以在每轮开始时构建一次全局索引
        # 或者 get_node_neighbors 内部处理
        # neighbors_ids = self.get_node_neighbors(node_id, node["current_communication_range"])
        # 简化：假设邻居列表在节点字典中已更新 (例如每轮开始时)
        # 你需要在每轮开始时调用一个方法来更新所有节点的邻居列表
        # self.update_all_node_neighbors() # <--- 你需要实现这个
        
        # 临时的邻居获取，效率不高，但能工作
        temp_neighbors_ids = []
        if not hasattr(self, '_position_tree') or self._position_tree is None: # 确保索引存在
            self._build_spatial_index()

        if self._position_tree:
            indices_in_kdtree = self._position_tree.query_ball_point(node["position"], node["current_communication_range"])
            for kdtree_idx in indices_in_kdtree:
                original_node_id = self.active_node_ids_for_kdtree[kdtree_idx]
                if original_node_id != node_id:
                     temp_neighbors_ids.append(original_node_id)
        
        n_neighbor = len(temp_neighbors_ids)
        e_neighbor_avg = 0
        if n_neighbor > 0:
            sum_neighbor_energy_normalized = sum(
                (self.nodes[nid]["energy"] / self.nodes[nid]["initial_energy"] if self.nodes[nid]["initial_energy"] > 0 else 0)
                for nid in temp_neighbors_ids if self.nodes[nid]["status"] == "active" # 确保邻居是活的
            )
            e_neighbor_avg = sum_neighbor_energy_normalized / n_neighbor if n_neighbor > 0 else 0 #再次检查n_neighbor
        
        n_ch_nearby = 0
        # 使用 confirmed_cluster_heads_previous_round
        for ch_id_prev in self.confirmed_cluster_heads_previous_round: 
            if ch_id_prev != node_id and self.nodes[ch_id_prev]["status"] == "active":
                if self.calculate_distance(node_id, ch_id_prev) <= node["current_communication_range"] * 1.5:
                    n_ch_nearby += 1
        
        d_bs = self.calculate_distance_to_bs(node_id)
        return {
            "e_self": e_self, "t_last_ch": t_last_ch, "n_neighbor": n_neighbor,
            "e_neighbor_avg": e_neighbor_avg, "n_ch_nearby": n_ch_nearby, "d_bs": d_bs
        }


    def get_q_value_compete_ch(self, node_id, state_tuple, action):
        # ... (如之前定义) ...
        node = self.nodes[node_id]
        q_table = node["q_table_compete_ch"]
        return q_table.get(state_tuple, {}).get(action, 0.0) 

    def update_q_value_compete_ch(self, node_id, state_tuple, action, reward, next_state_tuple): # next_state_tuple is now required
        # ... (实现 Q-learning 更新，使用 next_state_tuple 计算 next_max_q) ...
        node = self.nodes[node_id]
        q_table = node["q_table_compete_ch"]
        alpha = node["alpha_compete"] # 从节点特定参数获取
        gamma = node["gamma_compete"]

        old_q_value = self.get_q_value_compete_ch(node_id, state_tuple, action)
        
        next_max_q = 0.0
        if next_state_tuple is not None: # 如果不是终止状态 (通常在WSN中不会有明确的终止状态除非节点死亡)
            q_next_action0 = self.get_q_value_compete_ch(node_id, next_state_tuple, 0)
            q_next_action1 = self.get_q_value_compete_ch(node_id, next_state_tuple, 1)
            next_max_q = max(q_next_action0, q_next_action1)
        # else: 节点死亡，next_max_q = 0

        new_q_value = old_q_value + alpha * (reward + gamma * next_max_q - old_q_value)
        
        if state_tuple not in q_table:
            q_table[state_tuple] = {0: 0.0, 1: 0.0} # 初始化两个动作的Q值
        q_table[state_tuple][action] = new_q_value
        # logger.debug(f"Node {node_id} Q_compete update: S={state_tuple}, A={action}, R={reward:.2f}, OldQ={old_q_value:.3f}, NewQ={new_q_value:.3f}, NextMaxQ={next_max_q:.3f}")


    def calculate_reward_for_ch_competition(self, node_id, action_taken, 
                                            actual_members_joined=0, 
                                            is_uncovered_after_all_selections=False):
        """计算节点竞争CH后的奖励，使用模糊逻辑调整的权重。"""
        node = self.nodes[node_id]
        raw_state = self.get_node_state_for_ch_competition(node_id) # 获取原始状态值
        if raw_state is None: return 0.0

        # --- 获取用于模糊逻辑的输入 ---
        # 1. 网络整体平均能量 (归一化)
        current_total_energy = sum(n['energy'] for n in self.nodes if n['status'] == 'active')
        current_total_initial_energy = sum(n['initial_energy'] for n in self.nodes if n['status'] == 'active' and n['initial_energy'] > 0)
        net_energy_level_normalized = current_total_energy / current_total_initial_energy if current_total_initial_energy > 0 else 0
        
        # 2. 当前决策节点的能量 (归一化) - raw_state["e_self"] 已经是了
        node_self_energy_normalized = raw_state["e_self"]

        # 3. 网络中CH与活跃节点比例
        num_alive = self.get_alive_nodes()
        ch_density_global_val = len(self.confirmed_cluster_heads_current_round) / num_alive if num_alive > 0 else 0
        # 参考p_opt进行调整，使其与模糊集定义对应
        #p_opt_ref = self.config.get('fuzzy',{}).get('ch_compete_reward_weights',{}).get('p_opt_reference_for_density', 0.1)
        # ch_density_global_val = ch_density_global_val # 或者 ch_density_global_val / p_opt_ref 如果模糊集是基于倍数定义的

        # 4. 当前节点到BS的归一化距离
        ch_to_bs_dis_normalized = raw_state["d_bs"] / self.network_diagonal if self.network_diagonal > 0 else 0
        
        # --- 调用模糊系统获取动态权重 ---
        # 确保 self.reward_weights_adjuster 已经实例化
        if not hasattr(self, 'reward_weights_adjuster'): # 惰性初始化或在 __init__ 中确保
            from utils.fuzzy import RewardWeightsFuzzySystemForCHCompetition # 假设在 utils.fuzzy
            self.reward_weights_adjuster = RewardWeightsFuzzySystemForCHCompetition(self.config)

        fuzzy_reward_weights = self.reward_weights_adjuster.compute_reward_weights(
            current_net_energy_level=net_energy_level_normalized,
            current_node_self_energy=node_self_energy_normalized,
            current_ch_density_global=ch_density_global_val,
            current_ch_to_bs_dis_normalized=ch_to_bs_dis_normalized # 传递新参数
        )

        reward = 0.0

        base_penalty_distance_component = self.config.get('rewards',{}).get('ch_compete',{}).get('distance_penalty_unit', 2.0)
        if action_taken == 1: # 只有当节点尝试成为CH时，位置惩罚才重要
        # 方案1：w_dis 直接作为一个乘性惩罚因子（如果w_dis > 1 表示惩罚大）
        # reward -= (fuzzy_reward_weights['w_dis'] - 1.0) * base_penalty_distance_component 
        # 解释: 如果w_dis=1.5 (距离不好), 惩罚 = 0.5 * base; 如果w_dis=0.5 (距离好), 惩罚 = -0.5 * base (即奖励)

        # 方案2：更明确的惩罚项，其大小由w_dis调节
        # 先判断距离是否真的“不好”（例如，基于原始距离是否在模糊集的“Low”或“High”区域有高隶属度）
        # 这个判断逻辑可以放在奖励函数内部，或者让模糊系统直接输出一个“距离惩罚等级”
        # 假设 raw_state["d_bs"] 是原始距离
            distance_category_for_penalty = "Medium" # 默认
            if raw_state["d_bs"] < self.optimal_bs_dist_min * 0.8: # 举例：非常近
                distance_category_for_penalty = "Too_Close"
            elif raw_state["d_bs"] > self.optimal_bs_dist_max * 1.2: # 举例：非常远
                distance_category_for_penalty = "Too_Far"

            if distance_category_for_penalty != "Medium":
                # fuzzy_reward_weights['w_dis'] 应该反映了对这个距离的“不满意度”
                # 如果 w_dis 输出的是惩罚强度 [0,1] (0=不惩罚, 1=最大惩罚)
                # reward -= fuzzy_reward_weights['w_dis'] * base_penalty_distance_component
                # 如果 w_dis 输出的是调整因子 [0.5, 1.5]
                if fuzzy_reward_weights['w_dis'] > 1.0: # 表明距离不好
                    reward -= (fuzzy_reward_weights['w_dis'] - 1.0) * base_penalty_distance_component * 2 # 放大惩罚效果
                # 如果距离好 (w_dis < 1.0)，可以选择不给奖励，或者给少量奖励
                # else:
                #    reward += (1.0 - fuzzy_reward_weights['w_dis']) * base_penalty_distance_component * 0.5 # 示例：给一半的奖励
        
        # --- 定义基础奖励/惩罚组件的量级 (可以从config加载) ---
        base_reward_members_unit = self.config.get('rewards',{}).get('ch_compete',{}).get('member_join_unit', 2.0)
        base_reward_energy_self_unit = self.config.get('rewards',{}).get('ch_compete',{}).get('self_energy_unit', 5.0)
        base_penalty_cost_ch = self.config.get('rewards',{}).get('ch_compete',{}).get('cost_of_being_ch', 5.0)
        base_reward_rotation_unit = self.config.get('rewards',{}).get('ch_compete',{}).get('rotation_unit', 0.1)
        base_penalty_missed_opportunity = self.config.get('rewards',{}).get('ch_compete',{}).get('missed_opportunity', 10.0)
        base_penalty_uncovered = self.config.get('rewards',{}).get('ch_compete',{}).get('self_uncovered', 8.0)
        base_reward_conserve_energy_low_self = self.config.get('rewards',{}).get('ch_compete',{}).get('conserve_energy_low_self', 3.0)
        base_reward_passivity_ch_enough = self.config.get('rewards',{}).get('ch_compete',{}).get('passivity_ch_enough', 2.0)
        optimal_ch_nearby_threshold = self.config.get('deec',{}).get('optimal_ch_nearby_threshold', 2)


        if action_taken == 1: # 尝试成为CH
            # 成员收益 (核心)
            reward += fuzzy_reward_weights['w_members_factor'] * base_reward_members_unit * actual_members_joined
            
            # 自身能量贡献 (如果能量高)
            if raw_state["e_self"] > 0.6: # 示例阈值
                reward += fuzzy_reward_weights['w_energy_self_factor'] * base_reward_energy_self_unit * raw_state["e_self"]
            
            # 成为CH的成本
            reward -= fuzzy_reward_weights['w_cost_ch_factor'] * base_penalty_cost_ch

            # 轮换收益
            reward += fuzzy_reward_weights['w_rotation_factor'] * base_reward_rotation_unit * raw_state["t_last_ch"]

            # 扎堆惩罚 (如果附近CH过多)
            if raw_state["n_ch_nearby"] > optimal_ch_nearby_threshold:
                 # 可以让 w_cost_ch_factor 也影响这个惩罚的力度
                reward -= fuzzy_reward_weights['w_cost_ch_factor'] * 5.0 * (raw_state["n_ch_nearby"] - optimal_ch_nearby_threshold)
            
            base_distance_impact = self.config.get('rewards',{}).get('ch_compete',{}).get('distance_impact_unit', 5.0)
            reward_adjustment_from_distance = (1.0 - fuzzy_reward_weights['w_dis']) * base_distance_impact
            reward += reward_adjustment_from_distance

        else: # 选择不成为CH (action_taken == 0)
            # 节省能量 (如果自身能量低)
            if raw_state["e_self"] < 0.3: # 示例阈值
                reward += base_reward_conserve_energy_low_self
            
            # 明智地不当CH (如果附近CH已足够)
            if raw_state["n_ch_nearby"] >= optimal_ch_nearby_threshold:
                reward += base_reward_passivity_ch_enough
            
            # 错失良机惩罚
            if raw_state["n_ch_nearby"] < optimal_ch_nearby_threshold and \
               raw_state["e_self"] > 0.7 and \
               raw_state["e_neighbor_avg"] < raw_state["e_self"]: # 自己条件好，邻居差，且缺CH
                reward -= base_penalty_missed_opportunity # 应该由模糊权重调整，例如 w_missed_opp_factor

            # 自己未被覆盖的惩罚
            if is_uncovered_after_all_selections:
                reward -= base_penalty_uncovered
        
        # logger.debug(f"Node {node_id} CompeteCH Reward: Action={action_taken}, Members={actual_members_joined}, Uncovered={is_uncovered_after_all_selections}, Final_R={reward:.2f}")
        return reward


    def finalize_ch_roles(self, ch_declarations_this_round):
        # ... (与之前方案类似，但要更新 time_since_last_ch) ...
        if not ch_declarations_this_round:
            self.confirmed_cluster_heads_current_round = []
            # 所有非直连BS的活跃节点 time_since_last_ch++
            for node in self.nodes:
                if node["status"] == "active" and not node.get("can_connect_bs_directly", False):
                    node["time_since_last_ch"] += 1
            return

        num_alive_eligible = len([
            n for n in self.nodes 
            if n["status"] == "active" and not n.get("can_connect_bs_directly", False)
        ])
        if num_alive_eligible == 0: # 以防万一
             self.confirmed_cluster_heads_current_round = []
             logger.info("没有符合CH选举资格的活跃节点。")
             return
        p_opt_ref = self.config.get('deec',{}).get('p_opt', 0.1) 
        target_ch_count_ideal = max(1, int(num_alive_eligible * p_opt_ref))

        comm_range_avg = self.config.get('network', {}).get('communication_range', 100.0)
        too_close_factor = self.config.get('deec', {}).get('ch_finalize_too_close_factor', 0.8)
        medium_density_outer_factor = self.config.get('deec', {}).get('ch_finalize_medium_outer_factor', 1.0) # 通常等于通信范围
        
        threshold_too_close = comm_range_avg * too_close_factor
        # threshold_medium_outer = comm_range_avg * medium_density_outer_factor # 这个是中等密度环带的外边界

        max_chs_in_medium_ring = self.config.get('deec', {}).get('ch_finalize_max_in_medium_ring', 3)
        min_total_chs_after_filter = max(1, int(target_ch_count_ideal * 0.7)) # 筛选后至少保留的CH数量

        logger.debug(f"CH最终确定参数：理想CH数={target_ch_count_ideal}, 过近阈值={threshold_too_close:.1f}m, "
                     f"中等环带最大CH数={max_chs_in_medium_ring}, 筛选后最小总CH数={min_total_chs_after_filter}")
        
        # --- 步骤1: 初步能量筛选和数量控制 (与之前类似，但更宽松一些) ---
        # 按能量排序宣告者
        ch_declarations_this_round.sort(key=lambda id_val: self.nodes[id_val]["energy"], reverse=True)
        
        # 初步候选列表，数量可以比理想值略多，给后续空间筛选留余地
        preliminary_candidates = ch_declarations_this_round[:max(min_total_chs_after_filter, int(target_ch_count_ideal * 1.5))]
        logger.debug(f"初步能量筛选后，候选CH数量: {len(preliminary_candidates)}")

        if not preliminary_candidates:
            self.confirmed_cluster_heads_current_round = []
            logger.info("能量筛选后没有候选CH。")
            # 更新 time_since_last_ch
            for node in self.nodes:
                if node["status"] == "active" and not node.get("can_connect_bs_directly", False):
                    node["time_since_last_ch"] += 1
            return
        
        finalized_chs_step1 = []
        # 已经按能量排序，所以能量高的会先被加入finalized_chs_step1
        for cand_id in preliminary_candidates:
            node_cand = self.nodes[cand_id]
            if node_cand["status"] != "active": continue

            is_too_close_to_existing_final = False
            for final_id in finalized_chs_step1:
                # 确保 final_id 仍然是活跃的 (虽然不太可能在这里变)
                if self.nodes[final_id]["status"] != "active": continue 
                dist = self.calculate_distance(cand_id, final_id)
                if dist < threshold_too_close:
                    is_too_close_to_existing_final = True
                    # logger.debug(f"CH候选 {cand_id} 因与已确认CH {final_id} 距离过近 ({dist:.1f}m) 而被跳过。")
                    break
            if not is_too_close_to_existing_final:
                finalized_chs_step1.append(cand_id)
        
        logger.debug(f"移除过近CH后，候选CH数量: {len(finalized_chs_step1)}")

        finalized_chs_step2 = []
        target_upper_bound = max(min_total_chs_after_filter, int(target_ch_count_ideal * 1.1))

        if len(finalized_chs_step1) > target_upper_bound:
            logger.debug(f"CH数量 ({len(finalized_chs_step1)}) 仍多于目标上限 ({target_upper_bound})，按能量进一步削减。")
            # finalized_chs_step1 已经是按能量排序的，直接取前面部分
            finalized_chs_step2 = finalized_chs_step1[:target_upper_bound]
        else:
            finalized_chs_step2 = finalized_chs_step1
        

         # --- 步骤4: 确保至少有 min_total_chs_after_filter 个CH (如果可能) ---
        if len(finalized_chs_step2) < min_total_chs_after_filter and preliminary_candidates:
            logger.debug(f"筛选后CH数量 ({len(finalized_chs_step2)}) 少于最小目标 ({min_total_chs_after_filter})，尝试从初步候选者中补充。")
            # 尝试从 preliminary_candidates 中补充一些与 finalized_chs_step2 不冲突（不太近）的节点
            # 这个补充逻辑需要小心，避免重新引入扎堆
            # 简化：如果数量太少，就直接用 finalized_chs_step1（即只做了距离去重，没做数量削减）
            # 但要确保 finalized_chs_step1 本身也满足最小数量，否则可能需要从原始宣告者中选
            if len(finalized_chs_step1) >= min_total_chs_after_filter :
                 self.confirmed_cluster_heads_current_round = finalized_chs_step1[:max(min_total_chs_after_filter, len(finalized_chs_step1))] # 取较多的一方
            elif preliminary_candidates: # 如果step1也不够，从最初的按能量排序的里面取
                 self.confirmed_cluster_heads_current_round = preliminary_candidates[:min(len(preliminary_candidates), min_total_chs_after_filter*2)] # 允许略多，后续再精简
                 # 这里应该重新跑一遍距离去重，但为了简化，暂时这样
                 logger.warning("CH数量过少，补充逻辑可能不够完善，请关注CH分布。")
                 # 实际上，如果到这一步CH还很少，说明宣告成为CH的节点本身就很少或能量分布不均
                 # 此时 p_opt 可能需要调整
            else: # 实在没候选了
                 self.confirmed_cluster_heads_current_round = []

        else:
            self.confirmed_cluster_heads_current_round = finalized_chs_step2

        # --- 更新节点角色和计时器 ---
        for node_data_val in self.nodes:
            if node_data_val["status"] == "active":
                if node_data_val["id"] in self.confirmed_cluster_heads_current_round:
                    node_data_val["role"] = "cluster_head"
                    node_data_val["cluster_id"] = node_data_val["id"] 
                    node_data_val["time_since_last_ch"] = 0 
                elif not node_data_val.get("can_connect_bs_directly", False): 
                    node_data_val["role"] = "normal" # 确保其他节点是normal
                    node_data_val["time_since_last_ch"] += 1
                    node_data_val["cluster_id"] = -1 
        
        logger.info(f"最终确认本轮活跃CH ({len(self.confirmed_cluster_heads_current_round)}个): {self.confirmed_cluster_heads_current_round}")

    def step(self, current_round_num):
        self.current_round = current_round_num
        logger.info(f"--- 开始第 {self.current_round} 轮 (Epsilon Compete: {self.current_epsilon_compete:.3f}) ---")

        # 0. 更新和准备工作
        self.confirmed_cluster_heads_previous_round = list(self.confirmed_cluster_heads_current_round) # 保存上一轮的CH
        self.confirmed_cluster_heads_current_round = []
        self.identify_direct_bs_nodes()
        for node in self.nodes:
            if node["status"] == "active":
                 node["current_communication_range"] = node["base_communication_range"]
                 # 更新每个节点的epsilon (如果它们有独立的epsilon)
                 # node["epsilon_compete"] = max(self.epsilon_compete_min, node["epsilon_compete"] * self.epsilon_compete_decay)
        # 或者全局更新 epsilon
        self.current_epsilon_compete = max(self.epsilon_compete_min, self.current_epsilon_compete * self.epsilon_compete_decay)


        # --- 实例化或更新模糊逻辑系统 ---
        # 如果是全局共享的，并且其输入（如网络平均能量）会变，需要更新
        # 这里假设 reward_weight_adjuster 和 normal_node_ch_selector_fuzzy 在 __init__ 中创建
        # 如果 NormalNodeCHSelectionFuzzySystem 需要动态的 node_sum, cluster_sum，则在此时更新或重新创建
        # self.normal_node_ch_selector_fuzzy.update_network_stats(len(self.nodes), len(self.confirmed_cluster_heads_current_round)) # 假设有此方法
        
        # 获取用于模糊调整奖励权重的输入
        # net_energy_level_for_fuzzy = self._calculate_current_average_energy() / self.E0 if self.E0 > 0 else 0
        # ch_density_for_fuzzy = len(self.confirmed_cluster_heads_previous_round) / self.get_alive_nodes() if self.get_alive_nodes() > 0 else 0
        # ^^^ 这些应该在循环内，基于每个节点的视角或全局状态


        # === 阶段1: 节点通过Q学习竞争成为CH ===
        logger.info("开始阶段1：节点竞争成为CH...")
        ch_declarations_this_round = []
        competition_log = {} # node_id -> {"state_tuple": S, "action": A, "raw_state": raw_S}

        nodes_eligible_for_competition = [
            n for n in self.nodes
            if n["status"] == "active" and not n.get("can_connect_bs_directly", False)
        ]

        for node_data in nodes_eligible_for_competition:
            node_id = node_data["id"]
            state_tuple = self.get_discrete_state_tuple_for_competition(node_id)
            if state_tuple is None: continue

            action_to_take = 0
            if random.random() < self.current_epsilon_compete: # 使用全局衰减的epsilon
                action_to_take = random.choice([0, 1])
            else:
                q0 = self.get_q_value_compete_ch(node_id, state_tuple, 0)
                q1 = self.get_q_value_compete_ch(node_id, state_tuple, 1)
                action_to_take = 1 if q1 > q0 else (0 if q0 > q1 else random.choice([0,1])) # 如果相等则随机
            
            competition_log[node_id] = {"state_tuple": state_tuple, "action": action_to_take, 
                                        "raw_state": self.get_node_state_for_ch_competition(node_id)} # 保存原始状态用于奖励
            if action_to_take == 1:
                ch_declarations_this_round.append(node_id)
        
        logger.info(f"CH竞争阶段：{len(ch_declarations_this_round)} 个节点宣告想成为CH: {ch_declarations_this_round}")

        # === 阶段1.5: 从宣告者中最终确定本轮CH ===
        self.finalize_ch_roles(ch_declarations_this_round)

        # === 阶段2: 普通节点使用Q学习选择已确定的CH ===
        logger.info("开始阶段2：普通节点Q学习选择簇头...")
        # TODO: 在这里集成你的 NormalNodeCHSelectionFuzzySystem 和普通节点的Q学习选择逻辑
        # 普通节点会从 self.confirmed_cluster_heads_current_round 中选择
        # 你需要记录普通节点选择CH的决策，用于后续更新其 q_table_select_ch
        # 暂时使用之前的随机占位符，但要确保它从 confirmed_cluster_heads_current_round 中选
        num_nodes_assigned_by_q_learning_placeholder = 0
        # ... (之前的随机选择占位符逻辑，确保它从 self.confirmed_cluster_heads_current_round 中选) ...
        # ... 并且更新 node_data["cluster_id"] ...
        # ... (这个占位符需要你自己用完整的Q学习逻辑替换) ...
        # 创建 NormalNodeCHSelectionFuzzySystem 实例
        # 注意：node_sum 和 cluster_sum 应该是当前轮次的实际值
        # current_active_nodes = self.get_alive_nodes()
        # current_confirmed_chs_count = len(self.confirmed_cluster_heads_current_round)
        # normal_node_fuzzy_logic = NormalNodeCHSelectionFuzzySystem(node_sum=current_active_nodes, 
        #                                                          cluster_sum=max(1, current_confirmed_chs_count)) #避免除零
        
        # 示例：如果用随机选择，并标记哪些普通节点做了选择
        for node_data in self.nodes:
            if node_data["status"] == "active" and node_data["role"] == "normal" and \
               not node_data.get("can_connect_bs_directly", False) and \
               node_data["cluster_id"] == -1: # 确保只为未分配的普通节点选择
                
                eligible_chs_for_node = []
                if self.confirmed_cluster_heads_current_round:
                    for ch_id in self.confirmed_cluster_heads_current_round:
                        if self.nodes[ch_id]["status"] == "active":
                            distance = self.calculate_distance(node_data["id"], ch_id)
                            if distance <= node_data["current_communication_range"] and \
                               distance <= self.nodes[ch_id]["current_communication_range"]:
                                eligible_chs_for_node.append(ch_id)
                
                if eligible_chs_for_node:
                    # TODO: Replace random choice with Q-learning decision using NormalNodeCHSelectionFuzzySystem
                    chosen_ch_id = random.choice(eligible_chs_for_node) 
                    node_data["cluster_id"] = chosen_ch_id
                    num_nodes_assigned_by_q_learning_placeholder +=1


        # === 阶段2之后: 处理仍然孤立的普通节点 ===
        # ... (可以保留之前的 handle_isolated_nodes 或 assign_nodes_to_clusters 作为备用) ...
        # ... 但现在 assign_nodes_to_clusters 的候选CH应为 self.confirmed_cluster_heads_current_round ...
        
        # === 奖励计算与Q表更新 (在所有动作完成后) ===
        logger.info("计算奖励并更新Q表 (CH竞争)...")
        
        # 实例化奖励权重模糊调整器 (如果它是动态的，可能每轮或每次调用都需要)
        # reward_weights_adjuster = RewardWeightsFuzzySystemForCHCompetition(self.config) # 确保config传递正确
        # 或者在 __init__ 中创建 self.reward_weights_adjuster

        for node_id, log_info in competition_log.items():
            node = self.nodes[node_id]
            if node["status"] == "dead" and node["energy"] > 0: # 节点可能在本轮的其他操作中死亡
                 logger.warning(f"Node {node_id} status is dead but energy > 0. Forcing energy to 0.")
                 node["energy"] = 0 # 确保死亡节点能量为0
            if node["status"] == "dead": # 如果节点在本轮后续操作中死亡，S'的maxQ为0
                reward = self.calculate_reward_for_ch_competition(node_id, log_info["action"], 0, True, None,0) # 假设死亡=未覆盖
                self.update_q_value_compete_ch(node_id, log_info["state_tuple"], log_info["action"], reward, None) # None for next_state
                continue

            actual_members = 0
            if node["role"] == "cluster_head":
                actual_members = len([m_node for m_node in self.nodes if m_node.get("cluster_id") == node_id and m_node["status"]=='active'])
            
            is_uncovered = (node["role"] == "normal" and node["cluster_id"] == -1 and not node.get("can_connect_bs_directly", False))

            # 获取用于模糊调整奖励权重的输入
            # net_energy_level = self._calculate_current_average_energy() / self.E0 if self.E0 > 0 else 0
            # node_self_e_for_fuzzy = node["energy"] / node["initial_energy"] if node["initial_energy"] > 0 else 0
            # ch_density_for_fuzzy = len(self.confirmed_cluster_heads_current_round) / self.get_alive_nodes() if self.get_alive_nodes() > 0 else 0
            # current_fuzzy_reward_weights = self.reward_weight_adjuster.compute_reward_weights(
            #     net_energy_level, node_self_e_for_fuzzy, ch_density_for_fuzzy
            # )
            # 暂时不使用模糊调整奖励权重，以简化初始实现
            current_fuzzy_reward_weights = None 


            reward = self.calculate_reward_for_ch_competition(node_id, log_info["action"], actual_members, is_uncovered)
            
            # 获取下一状态 S' (本轮结束，下一轮开始时的状态)
            # 注意：这里的能量是本轮所有消耗计算完之前的能量，实际S'的能量会更低
            # 为了更准确，S'的能量应该是调用 simulate_round_energy_consumption 之后的
            # 这是一个简化：我们用当前状态信息（除了能量会变）来估计下一状态的非能量部分
            # 能量部分则需要在能耗计算后再确定。
            # 这是一个复杂点，简单处理是假设下一状态的非能量部分与当前相似，能量部分会降低。
            # 或者，如之前讨论，对 next_max_q 做简化。

            # 简化：假设下一状态的非能量部分与当前决策时的原始状态 raw_state 相似
            # 能量会降低，t_last_ch 会变化
            next_raw_state_estimate = dict(log_info["raw_state"]) # 复制一份
            # 预估能量消耗 (非常粗略)
            estimated_energy_consumption = 0.01 * node["initial_energy"] if node["initial_energy"] > 0 else 0.01
            next_raw_state_estimate["e_self"] = max(0, log_info["raw_state"]["e_self"] - (estimated_energy_consumption / (node["initial_energy"] if node["initial_energy"] > 0 else 1)))
            if log_info["action"] == 1 and node["role"] == "cluster_head": # 如果本轮当了CH
                next_raw_state_estimate["t_last_ch"] = 0
            else:
                next_raw_state_estimate["t_last_ch"] = log_info["raw_state"]["t_last_ch"] + 1
            # n_neighbor, e_neighbor_avg, n_ch_nearby 也会变，但精确预测S'很难
            # 这里我们用一个简化的S'，或者直接将next_max_q设为0（如果单步优化）
            
            # 为了让代码能跑起来，暂时将next_state_tuple设为None，即不考虑多步优化（gamma=0的效果）
            # 你需要根据你的Q学习理论设计来完善这里对S'和next_max_q的处理
            next_state_tuple_for_update = None # <--- TODO: 关键的简化，需要后续完善
            # 如果要计算 next_state_tuple_for_update:
            # next_s_dict = self.get_node_state_for_ch_competition(node_id) # 获取下一轮开始前的状态 (需要模拟能量消耗后)
            # if next_s_dict:
            #     next_state_tuple_for_update = tuple(self.discretize_value(v, k) for k,v in next_s_dict.items())
            # else: # 节点可能死了
            #     next_state_tuple_for_update = None


            self.update_q_value_compete_ch(node_id, log_info["state_tuple"], log_info["action"], reward, next_state_tuple_for_update)

        # TODO: 更新普通节点选择CH的Q表

        # === 调整下一轮的p_opt (如果仍然需要DEEC的p_opt作为某种参考或目标CH比例) ===
        # self.adjust_p_opt_for_next_round(...) 

        # === 模拟本轮的能量消耗 ===
        # 注意：这里的能耗应该是与Q学习决策和数据传输分离的背景能耗
        # 或者，如果你的Q学习奖励已经完全包含了所有能耗，这里就不需要再减了
        # 我之前的 simulate_round_energy_consumption 包含了通信，需要调整
        self.simulate_base_round_energy_consumption() # 假设有一个只处理背景能耗的函数
        
        logger.info(f"--- 第 {self.current_round} 轮结束 ---")
        # ... (日志和返回)
        return True

    def simulate_base_round_energy_consumption(self):
        """只模拟与Q学习动作无关的背景能耗，如空闲监听、基础感知"""
        # ... (只包含 idle_listening 和 sensing cost 的逻辑) ...
        energy_cfg = self.config.get('energy', {})
        try:
            idle_listening_cost_per_round = float(energy_cfg.get('idle_listening_per_round', 1e-6)) # 调小默认值
            sensing_cost_per_round = float(energy_cfg.get('sensing_per_round', 5e-7))
        except (ValueError, TypeError):
            idle_listening_cost_per_round = 1e-6
            sensing_cost_per_round = 5e-7
        
        nodes_to_kill_this_round = []
        for node in self.nodes:
            if node["status"] == "active":
                cost = idle_listening_cost_per_round + sensing_cost_per_round
                if self.consume_node_energy(node["id"], cost): # consume_node_energy 会处理死亡
                    pass # 节点仍然存活
                # else: 节点已在此处被标记为死亡 (如果能量耗尽)
        # kill_node 的调用现在在 consume_node_energy 内部

    # ... (其他辅助函数如 calculate_distance, calculate_distance_to_bs, get_alive_nodes, kill_node, 
    #      calculate_transmission_energy, consume_node_energy, _build_spatial_index, get_node_neighbors)
    # 你需要确保这些函数都存在且功能正确

    def _calculate_current_average_energy(self):
        """计算当前网络中所有存活节点的平均能量"""
        total_energy = 0
        alive_nodes_count = 0
        for node in self.nodes:
            if node["status"] == "active":
                total_energy += node["energy"]
                alive_nodes_count += 1
        return total_energy / alive_nodes_count if alive_nodes_count > 0 else 0
    
    

    def _calculate_current_average_energy(self):
        """
        计算当前网络的平均能量
        """
        alive_nodes = [node for node in self.nodes if node["status"] == "active"]
        if not alive_nodes:
            return 0
        
        total_energy = sum(node["energy"] for node in alive_nodes)
        return total_energy / len(alive_nodes)

    def get_alive_nodes(self):
        """
        获取存活节点数量
        """
        return len([node for node in self.nodes if node["status"] == "active"])

    def calculate_distance_to_base_station(self, node_id):
        """
        计算节点到基站的距离
        """
        node = self.nodes[node_id]
        base_pos = self.config['network']['base_position']
        
        dx = node['position'][0] - base_pos[0]
        dy = node['position'][1] - base_pos[1]
        
        return math.sqrt(dx*dx + dy*dy)

    

    def identify_direct_bs_nodes(self):
        """识别可以直接与基站通信的节点"""
        direct_comm_threshold = self.config.get('network', {}).get('direct_bs_comm_threshold', 50)
        # min_energy_for_direct_bs = self.config.get('energy', {}).get('min_energy_direct_bs', self.E0 * 0.1)
        # 使用浮动比例或绝对值，确保E0被正确初始化
        initial_e = self.E0 if hasattr(self, 'E0') and self.E0 > 0 else self.config.get('energy', {}).get('initial', 1.0)
        min_energy_for_direct_bs = self.config.get('energy', {}).get('min_energy_direct_bs_factor', 0.1) * initial_e


        count_direct = 0
        for node in self.nodes:
            if node["status"] == "active":
                node["can_connect_bs_directly"] = False # 每轮重置
                node["role_override"] = None         # 重置角色覆盖

                d_to_bs = self.calculate_distance_to_bs(node["id"])
                if d_to_bs <= direct_comm_threshold and node["energy"] > min_energy_for_direct_bs:
                    node["can_connect_bs_directly"] = True
                    node["role_override"] = "direct_to_bs" 
                    node["cluster_id"] = -2 # 特殊标记，表示直连BS
                    count_direct +=1
        if count_direct > 0:
            logger.info(f"识别出 {count_direct} 个节点将直接与BS通信。")
    
    
    def assign_nodes_to_clusters(self, attempt=1, nodes_to_assign=None, candidate_chs=None):
        """
        辅助函数：尝试将指定的普通节点分配给指定的候选簇头。
        主要用于Q学习选择失败后的备用策略或特定场景。
        """
        if nodes_to_assign is None: # 默认处理所有未分配的普通节点
            nodes_to_assign = [n for n in self.nodes if n["status"] == "active" and n["role"] == "normal" and n["cluster_id"] == -1]
        
        if candidate_chs is None: # 默认使用本轮的候选CH
            candidate_chs = self.candidate_cluster_heads

        if not candidate_chs:
            logger.info(f"Assign attempt {attempt}: 没有候选簇头，节点无法加入簇。")
            return len(nodes_to_assign) 

        logger.info(f"Assign attempt {attempt}: 开始将 {len(nodes_to_assign)} 个普通节点分配给 {len(candidate_chs)} 个候选簇头...")
        num_newly_assigned_this_attempt = 0
        
        for node_data in nodes_to_assign:
            if node_data["cluster_id"] != -1: continue # 如果已经被分配（可能通过Q学习），则跳过

            min_dist_to_ch = float('inf')
            assigned_ch_id = -1
            
            current_comm_range = node_data["base_communication_range"]
            if attempt > 1:
                increase_factor = min(1 + (attempt - 1) * 0.25, self.max_communication_range_increase_factor)
                current_comm_range = node_data["base_communication_range"] * increase_factor
            node_data["current_communication_range"] = current_comm_range

            for ch_id in candidate_chs:
                if not (0 <= ch_id < len(self.nodes)) or self.nodes[ch_id]["status"] == "dead":
                    continue
                # 候选CH也应该使用其当前通信范围（尽管在DEEC选举阶段可能还没特殊调整）
                if self.nodes[ch_id]["id"] not in self.candidate_cluster_heads : continue # 确保是候选CH

                distance = self.calculate_distance(node_data["id"], ch_id)
                if distance <= node_data["current_communication_range"] and \
                   distance <= self.nodes[ch_id]["current_communication_range"]:
                    if distance < min_dist_to_ch:
                        min_dist_to_ch = distance
                        assigned_ch_id = ch_id
            
            if assigned_ch_id != -1:
                node_data["cluster_id"] = assigned_ch_id
                # node_data["role"] = "member" # 可以有一个更细致的角色
                num_newly_assigned_this_attempt +=1
        
        remaining_isolated = len([n for n in nodes_to_assign if n["cluster_id"] == -1])
        logger.info(f"Assign attempt {attempt}: 新分配了 {num_newly_assigned_this_attempt} 个节点。仍有 {remaining_isolated} 个来自输入列表的节点未分配。")
        return remaining_isolated
    
    def adjust_p_opt_for_next_round(self, isolated_normal_nodes_after_q_learning):
        """根据本轮Q学习分配后的孤立普通节点情况和候选CH数量，调整下一轮的p_opt_current"""
        num_alive = self.get_alive_nodes()
        if num_alive == 0: return

        # 调整p_opt的逻辑，现在基于候选CH的数量和Q学习分配后的结果
        num_candidates = len(self.candidate_cluster_heads)
        expected_candidates = self.p_opt_initial * num_alive # 基于初始p_opt的期望

        if num_candidates < expected_candidates * 0.5 or isolated_normal_nodes_after_q_learning > num_alive * 0.1:
            # 如果候选CH太少，或者Q学习后孤立节点太多，增加p_opt
            self.p_opt_current = min(self.p_opt_initial * 2.0, self.p_opt_current * 1.15, 0.35)
            logger.info(f"由于候选CH不足或Q学习后孤立节点多，下一轮 p_opt 调整为: {self.p_opt_current:.3f}")
        elif num_candidates > expected_candidates * 1.8:
            # 如果候选CH过多
            self.p_opt_current = max(self.p_opt_initial * 0.7, self.p_opt_current * 0.90)
            logger.info(f"由于候选CH过多，下一轮 p_opt 调整为: {self.p_opt_current:.3f}")
        else:
            # 逐渐恢复到初始p_opt
            if self.p_opt_current > self.p_opt_initial:
                self.p_opt_current = max(self.p_opt_initial, self.p_opt_current * 0.98)
            elif self.p_opt_current < self.p_opt_initial:
                self.p_opt_current = min(self.p_opt_initial, self.p_opt_current * 1.02)
    
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

    def calculate_distance(self, node1_idx, node2_idx):
        """计算两个节点之间的欧氏距离 (基于节点ID)"""
        # 修改为接受节点ID，或直接接受节点字典
        node1 = self.nodes[node1_idx]
        node2 = self.nodes[node2_idx]
        x1, y1 = node1["position"]
        x2, y2 = node2["position"]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
       
    def simulate_round_energy_consumption(self):
        """模拟一轮中由于基本操作和数据传输产生的能量消耗"""
        energy_cfg = self.config.get('energy', {})
        try:
            idle_listening_cost_per_round = float(energy_cfg.get('idle_listening_per_round', 1e-5))
            sensing_cost_per_round = float(energy_cfg.get('sensing_per_round', 5e-6))
        except (ValueError, TypeError):
            logger.error("配置文件中的 idle_listening_per_round 或 sensing_per_round 不是有效的数字。请检查 config.yml。")
            idle_listening_cost_per_round = 1e-5
            sensing_cost_per_round = 5e-6
        
        packet_size_bits = self.config.get("simulation",{}).get("packet_size", 4000)

        for node_data in self.nodes: # 使用不同的变量名
            if node_data["status"] != "active":
                continue

            current_node_id = node_data["id"]
            total_cost_this_round = idle_listening_cost_per_round + sensing_cost_per_round

            if node_data["role"] == "normal" and node_data["cluster_id"] != -1 and node_data["cluster_id"] != -2 : # 已加入簇且非直连BS
                # 假设普通节点每轮都向其CH发送一个数据包
                ch_id = node_data["cluster_id"]
                if 0 <= ch_id < len(self.nodes) and self.nodes[ch_id]["status"] == "active":
                    distance_to_ch = self.calculate_distance(current_node_id, ch_id)
                    
                    # 普通节点发送成本
                    tx_c = self.calculate_transmission_energy(distance_to_ch, packet_size_bits, is_tx_operation=True)
                    total_cost_this_round += tx_c
                    node_data["tx_count"] += 1 # 手动增加计数器

                    # 对应CH的接收成本
                    rx_c_for_ch = self.calculate_transmission_energy(0, packet_size_bits, is_tx_operation=False)
                    if self.consume_node_energy(ch_id, rx_c_for_ch): # CH消耗接收能量
                         self.nodes[ch_id]["rx_count"] += 1
                else:
                    logger.warning(f"普通节点 {current_node_id} 的簇头 {ch_id} 无效或已死亡，本轮不发送。")
            
            elif node_data.get("role_override") == "direct_to_bs": # 直连BS的节点
                distance_to_bs = self.calculate_distance_to_bs(current_node_id)
                tx_c_direct = self.calculate_transmission_energy(distance_to_bs, packet_size_bits, is_tx_operation=True)
                total_cost_this_round += tx_c_direct
                node_data["tx_count"] += 1
                # BS不计算能耗

            elif node_data["role"] == "cluster_head":
                # CH 接收来自成员节点的数据 (这部分已在上面普通节点发送时计算并扣除CH能量)
                # CH 可能需要聚合数据 (假设有固定聚合成本)
                aggregation_cost = float(energy_cfg.get('aggregation_cost_per_packet', 5e-9)) * packet_size_bits # 示例
                # 假设CH聚合了收到的所有数据包（简化：这里只算一次聚合成本，实际应基于成员数）
                # 查找有多少成员连接到这个CH
                num_members = len([m_node for m_node in self.nodes if m_node.get("cluster_id") == current_node_id and m_node["status"] == "active"])
                total_cost_this_round += aggregation_cost * max(1, num_members) # 至少聚合一次（自身数据）

                # CH 将聚合后的数据发送给基站 (或下一跳CH，这里简化为直接到BS)
                distance_to_bs_for_ch = self.calculate_distance_to_bs(current_node_id)
                # 假设CH发送一个（可能更大的）聚合数据包
                aggregated_packet_size_factor = float(self.config.get("simulation", {}).get("ch_aggregated_packet_factor", 1.0)) # 聚合后数据包大小因子
                tx_c_ch = self.calculate_transmission_energy(distance_to_bs_for_ch, packet_size_bits * aggregated_packet_size_factor, is_tx_operation=True)
                total_cost_this_round += tx_c_ch
                node_data["tx_count"] += 1

            # 最终从节点扣除本轮总成本
            self.consume_node_energy(current_node_id, total_cost_this_round)
        

    
    def calculate_distance_to_bs(self, node1_idx):
        node1 = self.nodes[node1_idx]
        x1, y1 = node1["position"]
        return ((x1 - 250) ** 2 + (y1 - 250) ** 2) ** 0.5
    
    def calculate_transmission_energy(self, distance, packet_size_bits, is_tx_operation=True):
        """
        计算一次发送或接收操作的能量成本。
        Args:
            distance (float): 传输距离 (米). 对于接收，此参数可能不直接使用，但保留接口一致性。
            packet_size_bits (int): 数据包大小 (比特).
            is_tx_operation (bool): True表示发送操作，False表示接收操作。
        Returns:
            float: 计算得到的能量成本 (焦耳).
        """
        energy_cfg = self.config.get('energy', {})
        try:
            E_elec = float(energy_cfg.get('rx_cost', 50e-9)) # 也用作发送电路能耗
            tx_amp_default = float(energy_cfg.get('tx_amp', 0.0013e-12)) # 一个备用值
            tx_amp_fs = float(energy_cfg.get('tx_amp_fs', 10e-12))
            tx_amp_mp = float(energy_cfg.get('tx_amp_mp', tx_amp_default)) # 如果没有mp，用tx_amp或另一个默认
            threshold_d0 = float(energy_cfg.get('threshold_d0', 87.7))
            rx_cost_per_bit_config = float(energy_cfg.get('rx_cost', 50e-9))

        except (ValueError, TypeError) as e:
            logger.error(f"配置文件中能量参数无效: {e}. 使用默认值.")
            E_elec = 50e-9
            tx_amp_fs = 10e-12
            tx_amp_mp = 0.0013e-12
            threshold_d0 = 87.7
            rx_cost_per_bit_config = 50e-9

        energy_cost_val = 0.0
        if is_tx_operation:
            cost_elec = E_elec * packet_size_bits
            if distance < 0: # 防御性编程
                logger.warning(f"计算发送能耗时距离为负: {distance}，将使用0处理。")
                distance = 0
            if threshold_d0 <=0: # 防御性编程
                 logger.warning(f"距离阈值 d0 ({threshold_d0}) 无效，默认使用自由空间模型。")
                 cost_amp = tx_amp_fs * packet_size_bits * (distance ** 2)
            elif distance < threshold_d0:
                cost_amp = tx_amp_fs * packet_size_bits * (distance ** 2)
            else:
                cost_amp = tx_amp_mp * packet_size_bits * (distance ** 4)
            energy_cost_val = cost_elec + cost_amp
        else: # 接收操作
            energy_cost_val = rx_cost_per_bit_config * packet_size_bits
        
        return energy_cost_val
    
    def consume_node_energy(self, node_id, energy_to_consume):
        """
        从指定节点消耗能量，并更新其状态。
        Args:
            node_id (int): 节点ID.
            energy_to_consume (float): 要消耗的能量值.
        Returns:
            bool: 如果能量成功消耗且节点仍然存活，返回True。如果节点死亡或ID无效，返回False。
        """
        if not (0 <= node_id < len(self.nodes)):
            logger.error(f"消耗能量：无效的节点ID {node_id}")
            return False
        
        node = self.nodes[node_id]
        if node["status"] == "dead":
            return False # 已经死亡，不能再消耗

        if not isinstance(node["energy"], (int, float)):
            logger.error(f"节点 {node_id} 的能量值类型不正确: {node['energy']}. 无法消耗能量。")
            return False
        if not isinstance(energy_to_consume, (int, float)) or energy_to_consume < 0:
            logger.error(f"要消耗的能量值无效: {energy_to_consume}.")
            return False # 消耗的能量不能是负数或无效类型

        node["energy"] -= energy_to_consume
        if node["energy"] < 0:
            node["energy"] = 0
        
        if node["energy"] == 0: # 检查是否精确为0，或者非常接近0也可以认为是死亡
            self.kill_node(node_id) # kill_node 内部会设置 status="dead"
            return False # 节点死亡
        return True


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
        #logger.info("get_alive_nodes")
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

    

    def _get_packet_loss_rate(self, distance):
        """基于距离的Log-normal阴影模型"""
        PL_d0 = 55  # 参考距离d0=1m时的路径损耗(dB)
        path_loss = PL_d0 + 10 * 3.0 * np.log10(distance) + np.random.normal(0, 4)
        snr = 10 - path_loss  # 假设发射功率10dBm
        return 1 / (1 + np.exp(snr - 5))  # Sigmoid模拟丢包率
