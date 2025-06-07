import yaml
from pathlib import Path
import numpy as np
import networkx as nx
import math
import random
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.log import logger # 从 utils 包导入 logger
from utils.fuzzy import NormalNodeCHSelectionFuzzySystem, RewardWeightsFuzzySystemForCHCompetition,CHToBSPathSelectionFuzzySystem
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CH_BEHAVIOR_LOG_FILE = PROJECT_ROOT / "reports" / "ch_behavior_log.csv"

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
        self.BS_ID = -1
        self.DIRECT_BS_NODE_TYPE_STR = "DIRECT_BS_NODE"
        self.CH_TYPE_STR = "CH"
        self.BS_TYPE_STR = "BS"
        self.current_round = 0
        self.cluster_heads = []
        self.candidate_cluster_heads = [] # 重命名: 存储本轮的候选簇头ID
        self.confirmed_cluster_heads_for_epoch = []      # 在本Epoch内固定的CH ID列表
        self.confirmed_cluster_heads_previous_epoch = [] # 上一个Epoch的CH，用于CH竞争状态计算
        self.packets_in_transit = {}
        self.ch_forwarding_buffer_size  = self.config.get('q_learning', {}).get('ch_buffer_size', 5)

        deec_cfg = self.config.get('deec', {})
        self.p_opt_initial = deec_cfg.get('p_opt', 0.1) # 期望的簇头比例
        self.p_opt_current = self.p_opt_initial # 当前轮次使用的p_opt，可以动态调整
        self.max_communication_range_increase_factor = deec_cfg.get('max_comm_range_increase_factor', 1.5) # 允许通信范围增加的最大倍数
        self.min_ch_to_node_ratio_target = deec_cfg.get('min_ch_to_node_ratio', 0.05) # 目标最小簇头与节点比例，用于动态调整p_opt
        self.location_factor_enabled = deec_cfg.get('location_factor_enabled', True)
        self.optimal_bs_dist_min = deec_cfg.get('optimal_bs_dist_min', 50)
        self.optimal_bs_dist_max = deec_cfg.get('optimal_bs_dist_max', 200)
        self.penalty_factor_too_close = deec_cfg.get('penalty_factor_too_close', 0.5)
        self.penalty_factor_too_far = deec_cfg.get('penalty_factor_too_far', 0.5)
        self.epoch_length = int(deec_cfg.get('epoch_length', 20)) # 读取Epoch长度
        if self.epoch_length <= 0:
            logger.warning("Epoch length_must be positive. Setting to default 20.")
            self.epoch_length = 20

        # Q-learning parameters from config (or defaults)
        q_learning_cfg = self.config.get('q_learning', {})
        self.alpha_compete = float(q_learning_cfg.get('alpha_compete_ch', 0.1))
        self.gamma_compete = float(q_learning_cfg.get('gamma_compete_ch', 0.9))
        self.epsilon_compete_initial = float(q_learning_cfg.get('epsilon_compete_ch_initial', 0.5))
        self.epsilon_compete_decay = float(q_learning_cfg.get('epsilon_compete_ch_decay', 0.995))
        self.epsilon_compete_min = float(q_learning_cfg.get('epsilon_compete_ch_min', 0.01))
        self.ch_switching_hysteresis = float(q_learning_cfg.get('ch_switching_hysteresis', 5.0)) # 新增：切换滞后阈值
        ch_management_cfg = q_learning_cfg.get('ch_management', {}) # 或者 self.config.get('deec', {})
        self.enable_ch_capacity_limit = ch_management_cfg.get('enable_capacity_limit', True)
        self.ch_max_members_factor = float(ch_management_cfg.get('max_members_factor', 1.5))
        rewards_cfg = self.config.get('rewards', {}).get('select_ch', {})
        self.ch_rejection_penalty = float(rewards_cfg.get('rejection_penalty', -30.0))
        self.current_epsilon_compete = self.epsilon_compete_initial
        self.competition_log_for_current_epoch = {}

        self.disc_params = self.config.get('discretization_params', {})
        # 为每个状态获取其参数，并提供默认值以防配置缺失
        self.energy_disc_bins = self.disc_params.get('energy', {}).get('num_bins', 5)
        # self.energy_disc_boundaries = self.disc_params.get('energy', {}).get('custom_boundaries', [0.2, 0.4, 0.6, 0.8])
        self.time_disc_boundaries = self.disc_params.get('time_since_last_ch', {}).get('boundaries', [20, 60])     
        self.neighbor_count_disc_bins = self.disc_params.get('neighbor_count', {}).get('num_bins', 3)
        # self.neighbor_count_disc_boundaries = self.disc_params.get('neighbor_count', {}).get('boundaries', [5, 15])
        self.ch_count_disc_boundaries = self.disc_params.get('ch_count_nearby', {}).get('boundaries', [0, 2])    
        self.dist_bs_disc_boundaries = self.disc_params.get('distance_to_bs_normalized', {}).get('boundaries', [0.25, 0.75])

        self.sim_packets_generated_total = 0
        self.sim_packets_delivered_bs_total = 0
        self.sim_total_delay_for_delivered_packets = 0.0 # 如果计算精确时延
        self.sim_total_hops_for_delivered_packets_this_round = 0 # 如果用跳数作代理
        self.sim_num_packets_counted_for_hops_this_round = 0   # 如果用跳数作代理
        
        self.sim_packets_generated_this_round = 0
        self.sim_packets_delivered_bs_this_round = 0 # <<< --- 这个是报错的属性 ---
        self.sim_total_delay_this_round = 0.0 # 如果计算精确时延
        self.sim_num_packets_for_delay_this_round = 0 # 如果计算精确时延

        self.ch_path_fuzzy_logic = CHToBSPathSelectionFuzzySystem(self.config)
        if not hasattr(self, 'reward_weights_adjuster'):
            from utils.fuzzy import RewardWeightsFuzzySystemForCHCompetition
            self.reward_weights_adjuster = RewardWeightsFuzzySystemForCHCompetition(self.config)
        if not hasattr(self, 'normal_node_fuzzy_logic'):
            from utils.fuzzy import NormalNodeCHSelectionFuzzySystem # 确保导入路径正确
            self.normal_node_fuzzy_logic = NormalNodeCHSelectionFuzzySystem(self.config)

        # E_total_initial 用于DEEC计算，每个节点的初始能量可能不同，DEEC原版假设相同
        # 我们这里假设所有节点的初始能量相同，取第一个节点的初始能量作为参考
        self.E0 = self.config.get('energy', {}).get('initial', 1.0) 

        self._init_nodes()
        self.confirmed_cluster_heads_previous_round = []
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
                "q_table_select_next_hop": {},
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
        num_bins = self.energy_disc_bins
        if not (0 <= normalized_energy <= 1.0001):
            normalized_energy = np.clip(normalized_energy, 0, 1)
        return min(int(normalized_energy * num_bins), num_bins - 1)
        # 如果使用自定义边界:
        # boundaries = self.energy_disc_boundaries 
        # return np.digitize(normalized_energy, boundaries, right=False) # right=False: [b1, b2), right=True: (b1, b2]

    def discretize_time_since_last_ch(self, rounds_since_last_ch):
        # np.digitize 会返回一个索引，0表示小于第一个边界，1表示在第一和第二个边界之间，以此类推
        # boundaries = [20, 60] -> 3 states: val <= 20 (state 0), 20 < val <= 60 (state 1), val > 60 (state 2)
        return np.digitize(rounds_since_last_ch, self.time_disc_boundaries)

    def discretize_neighbor_count(self, count, max_neighbors_ref=None): # max_neighbors_ref 作为参考
        # 示例：如果config中定义了 neighbor_count.boundaries
        if 'boundaries' in self.disc_params.get('neighbor_count', {}):
            return np.digitize(count, self.disc_params['neighbor_count']['boundaries'])
        else: # 使用 num_bins 进行等宽分箱，需要一个大致的上限
            # 这个上限最好也是可配置的或动态估算的
            # 假设一个硬编码的上限，或者从config读取
            approx_max_n_count = self.config.get('network',{}).get('node_count', 100) // 2 # 粗略估计
            num_bins = self.neighbor_count_disc_bins
            clipped_count = np.clip(count, 0, approx_max_n_count)
            bin_width = approx_max_n_count / num_bins if num_bins > 0 else approx_max_n_count
            if bin_width == 0: return 0 # 避免除零
            return min(int(clipped_count / bin_width), num_bins - 1) 

    def discretize_ch_count_nearby(self, count):
        return np.digitize(count, self.ch_count_disc_boundaries) 

    def discretize_normalized_distance_to_bs(self, normalized_distance):
        normalized_distance = np.clip(normalized_distance, 0, 1) # 确保在[0,1]
        return np.digitize(normalized_distance, self.dist_bs_disc_boundaries) 

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
        with open("raw_state_log.csv", "a") as f:
            if f.tell() == 0: #写入表头
                f.write("round,node_id,e_self_raw,t_last_ch_raw,n_neighbor_raw,e_neighbor_avg_raw,n_ch_nearby_raw,d_bs_normalized_raw\n")
            f.write(f"{self.current_round},{node_id},"
                    f"{raw_state['e_self']},{raw_state['t_last_ch']},"
                    f"{raw_state['n_neighbor']},{raw_state['e_neighbor_avg']},"
                    f"{raw_state['n_ch_nearby']},"
                    f"{raw_state['d_bs'] / self.network_diagonal if self.network_diagonal > 0 else 0}\n")

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

        if self._position_tree: 
            scan_radius_for_members = node["current_communication_range"] 
            try:
                # 明确指定参数 r
                indices_in_kdtree = self._position_tree.query_ball_point(x=node["position"], r=scan_radius_for_members,p=2.0)
                for kdtree_idx in indices_in_kdtree:
                    original_node_id = self.active_node_ids_for_kdtree[kdtree_idx] 
                    if original_node_id != node_id:
                         temp_neighbors_ids.append(original_node_id)
            except Exception as e_kdtree: # 捕获可能的KDTree查询错误
                logger.error(f"KDTree query_ball_point 错误 for node {node_id} (pos {node['position']}, r {scan_radius_for_members}): {e_kdtree}", exc_info=True)
        
        n_neighbor = len(temp_neighbors_ids)
        e_neighbor_avg = 0
        if n_neighbor > 0:
            sum_neighbor_energy_normalized = sum(
                (self.nodes[nid]["energy"] / self.nodes[nid]["initial_energy"] if self.nodes[nid]["initial_energy"] > 0 else 0)
                for nid in temp_neighbors_ids 
            )
            e_neighbor_avg = sum_neighbor_energy_normalized / n_neighbor if n_neighbor > 0 else 0 #再次检查n_neighbor
        
        n_ch_nearby = 0
        # 使用 confirmed_cluster_heads_previous_round
        scan_radius_for_chs = node["current_communication_range"] * 1.5
        for ch_id_prev in self.confirmed_cluster_heads_previous_epoch: 
            if ch_id_prev != node_id and self.nodes[ch_id_prev]["status"] == "active": # 确保参考的旧CH还是活的
                if self.calculate_distance(node_id, ch_id_prev) <= scan_radius_for_chs:
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
                                            is_uncovered_after_all_selections=False
                                            ):
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
        ch_density_global_val = len(self.confirmed_cluster_heads_for_epoch) / num_alive if num_alive > 0 else 0
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
                base_crowding_penalty = 15.0
                reward -= fuzzy_reward_weights['w_cost_ch_factor'] * base_crowding_penalty * (raw_state["n_ch_nearby"] - optimal_ch_nearby_threshold)
            
            base_distance_impact = self.config.get('rewards',{}).get('ch_compete',{}).get('distance_impact_unit', 5.0)
            reward_adjustment_from_distance = (1.0 - fuzzy_reward_weights['w_dis']) * base_distance_impact
            reward += reward_adjustment_from_distance

        else: # 选择不成为CH (action_taken == 0)
            # 节省能量 (如果自身能量低)
            if raw_state["e_self"] < 0.3: # 示例阈值
                reward += base_reward_conserve_energy_low_self
            
            # 明智地不当CH (如果附近CH已足够)
            if raw_state["n_ch_nearby"] >= optimal_ch_nearby_threshold:
                reward += base_reward_passivity_ch_enough * raw_state["n_ch_nearby"]
            
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


    def finalize_ch_roles(self, ch_declarations_this_epoch):
        """
        [最终版本-带覆盖增益] 从宣告者中确定本Epoch的活跃CH。
        该函数使用贪心算法，在保证连通性的前提下，平衡覆盖范围、节点能量和中心度。
        """
        logger.info("CH最终确定阶段 (基于覆盖增益和图论)：从 {} 个宣告者中确定本Epoch的CH...".format(len(ch_declarations_this_epoch)))
        
        # 0. 准备工作
        nodes_eligible = [n for n in self.nodes if n["status"] == "active" and not n.get("can_connect_bs_directly", False)]
        num_alive_eligible = len(nodes_eligible)
        all_normal_nodes_set = {n["id"] for n in nodes_eligible if n.get("role") != "cluster_head"} # 所有需要被覆盖的节点

        if not ch_declarations_this_epoch or num_alive_eligible == 0:
            self.confirmed_cluster_heads_for_epoch = []
            return

        # 1. 构建连通性图并初筛 (与之前版本相同)
        G = nx.Graph()
        G.add_nodes_from(ch_declarations_this_epoch)
        G.add_node(self.BS_ID)
        for i in range(len(ch_declarations_this_epoch)):
            for j in range(i + 1, len(ch_declarations_this_epoch)):
                ch1_id, ch2_id = ch_declarations_this_epoch[i], ch_declarations_this_epoch[j]
                if self.calculate_distance(ch1_id, ch2_id) <= self.nodes[ch1_id]["base_communication_range"]:
                    G.add_edge(ch1_id, ch2_id)
        for ch_id in ch_declarations_this_epoch:
            if self.calculate_distance_to_bs(ch_id) <= self.nodes[ch_id]["base_communication_range"]:
                G.add_edge(ch_id, self.BS_ID)
        
        connected_to_bs_candidates = []
        if self.BS_ID in G:
            for component in nx.connected_components(G):
                if self.BS_ID in component:
                    component.remove(self.BS_ID)
                    connected_to_bs_candidates = list(component)
                    break
        
        if not connected_to_bs_candidates:
            logger.error("无任何宣告者能连通BS！选择能量最高的宣告者作为唯一CH。")
            ch_declarations_this_epoch.sort(key=lambda nid: self.nodes[nid]["energy"], reverse=True)
            self.confirmed_cluster_heads_for_epoch = ch_declarations_this_epoch[:1]
            # 后续的角色更新逻辑会处理这一个CH
            self._update_node_roles_and_timers()
            return

        # 2. 贪心迭代选择CH
        
        # a. 初始化
        potential_chs = set(connected_to_bs_candidates)
        final_ch_list = []
        covered_normal_nodes = set()
        
        # 获取配置参数
        ch_finalize_cfg = self.config.get('deec', {}).get('ch_finalize_rules', {})
        w_coverage = ch_finalize_cfg.get('coverage_weight', 0.5)
        w_energy = ch_finalize_cfg.get('energy_weight', 0.3)
        w_centrality = ch_finalize_cfg.get('centrality_weight', 0.2)
        
        area_w, area_h = self.config['network']['area_size']
        center_x, center_y = area_w / 2, area_h / 2
        max_radius = math.sqrt(center_x**2 + center_y**2)
        
        ideal_ch_count = max(1, int(num_alive_eligible * self.p_opt_initial))
        max_ch_count = int(ideal_ch_count * self.config.get('deec', {}).get('ch_finalize_rules', {}).get('max_count_factor', 1.2))

        # b. 迭代循环
        while len(final_ch_list) < max_ch_count and len(covered_normal_nodes) < len(all_normal_nodes_set):
            best_candidate_id = -1
            max_score = -float('inf')

            # 在每一轮迭代中，为所有剩余的候选者计算当前带来的边际效益
            for cand_id in potential_chs:
                node = self.nodes[cand_id]
                
                # i. 计算覆盖增益 (能覆盖多少个“新”节点)
                neighbors = self.get_node_neighbors(cand_id, node["base_communication_range"])
                uncovered_neighbors = {nid for nid in neighbors if nid in all_normal_nodes_set and nid not in covered_normal_nodes}
                coverage_gain = len(uncovered_neighbors)

                # ii. 计算中心度分数
                node_x, node_y = node["position"]
                dist_to_center = math.sqrt((node_x - center_x)**2 + (node_y - center_y)**2)
                centrality_score = 1.0 - (dist_to_center / max_radius) if max_radius > 0 else 0.5
                
                # iii. 能量分数
                energy_score = node["energy"] / self.E0 if self.E0 > 0 else 0

                # iv. 综合分数
                # 注意：coverage_gain的量级可能远大于其他两项，需要归一化或调整权重
                # 这里我们用一个简单的归一化，假设平均邻居数是10
                normalized_coverage_gain = coverage_gain / 10.0 
                
                score = (w_coverage * normalized_coverage_gain +
                         w_energy * energy_score +
                         w_centrality * centrality_score)
                
                if score > max_score:
                    max_score = score
                    best_candidate_id = cand_id

            if best_candidate_id == -1:
                break # 没有更多可选择的候选者了
            
            # c. 将本轮最优的候选者选为CH，并更新状态
            final_ch_list.append(best_candidate_id)
            potential_chs.remove(best_candidate_id) # 从候选池中移除
            
            # 更新已覆盖的普通节点集合
            neighbors = self.get_node_neighbors(best_candidate_id, self.nodes[best_candidate_id]["base_communication_range"])
            for neighbor_id in neighbors:
                if neighbor_id in all_normal_nodes_set:
                    covered_normal_nodes.add(neighbor_id)
        
        self.confirmed_cluster_heads_for_epoch = final_ch_list
        
        # 3. 更新所有节点的角色和状态
        self._update_node_roles_and_timers()

    def _update_node_roles_and_timers(self):
        """[新辅助函数] 根据最终的CH列表，更新所有节点的角色和计时器。"""
        confirmed_ch_set = set(self.confirmed_cluster_heads_for_epoch)
        for node_data in self.nodes:
            if node_data["status"] == "active":
                if node_data["id"] in confirmed_ch_set:
                    node_data["role"] = "cluster_head"
                    node_data["cluster_id"] = node_data["id"]
                    node_data["time_since_last_ch"] = 0
                    node_data['initial_energy_as_ch_this_epoch'] = node_data["energy"]
                elif not node_data.get("can_connect_bs_directly", False):
                    node_data["role"] = "normal"
                    node_data["cluster_id"] = -1
                    if "time_since_last_ch" not in node_data: node_data["time_since_last_ch"] = 0
                    node_data["time_since_last_ch"] += 1
        
        logger.info(f"最终确认本Epoch活跃CH ({len(self.confirmed_cluster_heads_for_epoch)}个): {self.confirmed_cluster_heads_for_epoch}")
    
    def _update_ch_competition_q_tables_at_epoch_end(self):
        # ... (基本与你之前的实现一致，确保使用正确的变量) ...
        # ... (S' 的获取需要基于当前轮次结束时的状态，而不是上个epoch开始时的competition_log中的raw_state)
        logger.info(f"Epoch { (self.current_round -1) // self.epoch_length } 结束: 更新CH竞争Q表...") # 应该是上一个epoch
        if not self.competition_log_for_current_epoch:
            return

        for node_id, log_info in self.competition_log_for_current_epoch.items():
            node = self.nodes[node_id]
            
            # 获取 S (做出决策时的状态) 和 A (做出的动作)
            state_tuple_s = log_info["state_tuple"]
            action_taken_a = log_info["action"]

            # 计算奖励 R (基于整个epoch的表现)
            actual_members_this_epoch = 0
            if self.nodes[node_id]["role"] == "cluster_head" and node_id in self.confirmed_cluster_heads_for_epoch: # 确认它真的是本epoch的CH
                actual_members_this_epoch = len([
                    m_node for m_node in self.nodes 
                    if m_node.get("cluster_id") == node_id and m_node["status"] == 'active'
                ])
            
            is_uncovered_at_epoch_end = (
                self.nodes[node_id]["role"] == "normal" and 
                self.nodes[node_id]["cluster_id"] == -1 and 
                not self.nodes[node_id].get("can_connect_bs_directly", False)
            )
            
            # ... (计算模糊权重 fuzzy_reward_weights) ...
            # ... (调用 self.calculate_reward_for_ch_competition) ...
            reward_compete = self.calculate_reward_for_ch_competition( # 这个函数内部会调用模糊系统
                node_id, action_taken_a, 
                actual_members_this_epoch, 
                is_uncovered_at_epoch_end
            )
            
            # 获取下一状态 S' (即当前epoch结束，下一epoch开始前的状态)
            next_state_tuple_for_update = None
            if node["status"] == "active": # 只有节点存活才有下一状态
                next_state_tuple_for_update = self.get_discrete_state_tuple_for_competition(node_id)
            
            self.update_q_value_compete_ch(
                node_id, 
                state_tuple_s, 
                action_taken_a,
                reward_compete,
                next_state_tuple_for_update 
            )
        if (self.current_round % self.epoch_length == 0 and self.current_round > 0) or \
   self.current_round == self.config.get('simulation',{}).get('total_rounds',10)-1 : # 确保是在epoch边    
            current_epoch_num_for_log = (self.current_round -1) // self.epoch_length # 上一个刚结束的epoch
            if self.current_round == self.config.get('simulation',{}).get('total_rounds',10)-1 and \
            self.current_round % self.epoch_length != 0 : # 如果是最后但不满一个epoch
                current_epoch_num_for_log = self.current_round // self.epoch_length


            with open(CH_BEHAVIOR_LOG_FILE, "a", encoding="utf-8") as f_ch_log:
                for ch_id_log in self.confirmed_cluster_heads_for_epoch: # 使用本epoch的CH列表
                    ch_node_log = self.nodes[ch_id_log]
                    if ch_node_log["status"] == "active" and ch_node_log["role"] == "cluster_head": # 确保是活跃CH
                        
                        # 获取选举时的状态 (需要从 competition_log_for_current_epoch 中获取)
                        # 注意：competition_log 中的 "raw_state_for_log" 存的是宣告时的状态
                        election_time_energy = ch_node_log["initial_energy"] # 简化：假设选举时是满能量，或需要回溯
                        time_since_last_ch_at_election = 0 # 简化：或从log获取
                        neighbors_at_election = 0 # 简化：或从log获取
                        dist_to_bs_val = self.calculate_distance_to_bs(ch_id_log)

                        # 获取最终成员数
                        final_members_count = len([
                            m_node for m_node in self.nodes
                            if m_node.get("cluster_id") == ch_id_log and m_node["status"] == 'active'
                        ])
                        
                        # 从 competition_log 获取宣告时的信息
                        # 这个log是在epoch开始时记录的，而CH角色是在finalize_ch_roles后确定的
                        # CH_BEHAVIOR_LOG 应该记录的是 *最终成为CH* 的节点在 *它们宣告并最终被选为CH的那个epoch开始时* 的一些状态
                        # 以及它们在那个epoch结束时的表现（如成员数）

                        # 正确的逻辑应该是，在 finalize_ch_roles 之后，对于每一个 confirmed_ch,
                        # 查其在 competition_log_for_current_epoch 中的记录（如果它是通过Q学习宣告的）
                        # 或者，更简单地，CH_BEHAVIOR_LOG 记录的是CH在 *被确认后的状态* 和 *epoch结束时的表现*
                        
                        # 让我们采用“CH被确认后的状态” + “epoch结束时的表现”
                        # 选举时的能量，可以近似认为是它在成为CH那个epoch开始时的能量。
                        # 这需要你在 finalize_ch_roles 后，或者在 _update_ch_competition_q_tables_at_epoch_end 中
                        # 对于每个最终的CH，记录其当时的能量等信息。

                        # 简化版：假设记录的是CH在一个epoch服务期间的一些平均或最终状态
                        # 为了更准确记录 "energy_at_election"，你需要在 finalize_ch_roles 中，
                        # 当一个节点被确认为CH时，立即记录它当时的能量。
                        # 或者，如果CH选举只在epoch开始，那么CH的初始能量就是它在该epoch开始时的能量。

                        # 假设我们记录的是epoch开始时它作为CH的初始状态和epoch结束时的成员数
                        # energy_at_election 应该是该CH在本epoch开始时的能量
                        # 你可以在 finalize_ch_roles 后，为每个 confirmed_ch 记录一个 "energy_at_start_of_epoch_as_ch"
                        
                        # 临时的简化记录：
                        # energy_at_election: 使用节点在当前（epoch结束时）的能量可能不准确，
                        # 最好是在它刚被选为CH时（epoch初）记录。
                        # 如果 competition_log_for_current_epoch 存了宣告时的能量，可以用那个。
                        # 我们假设 'initial_energy_as_ch_this_epoch' 存在节点属性中，
                        # 它在 finalize_ch_roles 中被设置。
                        energy_at_election_val = ch_node_log.get('initial_energy_as_ch_this_epoch', ch_node_log["energy"])


                        f_ch_log.write(f"{current_epoch_num_for_log},{ch_id_log},"
                                    f"{energy_at_election_val:.4f}," # 需要确保这个值的准确性
                                    f"{ch_node_log['time_since_last_ch']}," # 这是它当上CH后变为0，然后每轮增加，这里可能是0
                                    f"0," # 简化：邻居数在选举时
                                    f"{dist_to_bs_val:.2f},"
                                    f"{final_members_count}\n")
        self.competition_log_for_current_epoch = {} # 清空为下一个epoch准备

    def get_q_value_select_ch(self, node_id, ch_id):
        node = self.nodes[node_id]
        return node["q_table_select_ch"].get(ch_id, 0.0) # 默认为0

    def update_q_value_select_ch(self, node_id, ch_id, reward): # 简化为单步优化
        node = self.nodes[node_id]
        q_table = node["q_table_select_ch"]
        alpha = node.get("alpha_select_ch", 0.1) # 从节点获取或用默认
        
        old_q_value = q_table.get(ch_id, 0.0)
        # Simplified Q-update (like multi-armed bandit, or Q-learning with gamma=0 or next_max_q=0)
        new_q_value = old_q_value + alpha * (reward - old_q_value)
        q_table[ch_id] = new_q_value
        # logger.debug(f"Node {node_id} Q_select_CH update: CH={ch_id}, R={reward:.2f}, OldQ={old_q_value:.3f}, NewQ={new_q_value:.3f}")

    def calculate_reward_for_selecting_ch(self, normal_node_id, chosen_ch_id, 
                                          fuzzy_weights, 
                                          transmission_successful=True, 
                                          actual_energy_spent_tx=0):
        # --- 获取基础奖励/惩罚单位值 (从config) ---
        reward_cfg = self.config.get('rewards', {}).get('select_ch', {})
        R_success_val = float(reward_cfg.get('transmission_success', 50))
        R_fail_val = float(reward_cfg.get('transmission_fail', -50))
        
        energy_factor_scale = float(reward_cfg.get('energy_component_scale', 20)) # 用于缩放能量项贡献
        path_factor_scale = float(reward_cfg.get('path_component_scale', 30))     # 用于缩放路径项贡献
        load_factor_scale = float(reward_cfg.get('load_component_scale', 20))     # 用于缩放负载项贡献
        dist_bs_factor_scale = float(reward_cfg.get('dist_bs_component_scale', 10))# 用于缩放距离项贡献

        # --- 1. 结果奖励 ---
        reward = R_success_val if transmission_successful else R_fail_val

        node_n = self.nodes[normal_node_id]
        node_ch = self.nodes[chosen_ch_id]

        # --- 2. CH能量相关奖励 (越高越好) ---
        # e_cluster_normalized 是 CH 的归一化能量 [0,1]
        # 我们希望能量高时奖励为正，能量低时为负（相对于一个中点0.5）
        ch_energy_metric = (node_ch["energy"] / node_ch["initial_energy"] if node_ch["initial_energy"] > 0 else 0) - 0.5
        reward += fuzzy_weights['w_e_ch'] * ch_energy_metric * energy_factor_scale

        # --- 3. 路径相关奖励 (能耗低、成功率高越好) ---
        # actual_energy_spent_tx (需要归一化或与一个参考值比较)
        # 假设 avg_e_send_to_ch 是一个参考平均能耗
        # avg_e_send_to_ch_ref = self.config.get('energy',{}).get('avg_send_to_ch_ref', 0.001) # 从config获取参考值
        # path_cost_metric = -(actual_energy_spent_tx / avg_e_send_to_ch_ref) if avg_e_send_to_ch_ref > 0 else -10 # 示例，能耗越低越好
        # 简化：直接用能耗惩罚，能耗越低惩罚越小
        path_cost_metric = -actual_energy_spent_tx * 1000 # 放大一点，假设能耗是小数
        
        # r_success_with_ch (需要获取或估计普通节点与该CH的历史成功率)
        # 暂时假设 r_success_with_ch = 1.0 (理想) 或从节点属性获取
        r_success_metric = node_n.get("history_success_with_ch", {}).get(chosen_ch_id, 0.8) # 示例：默认0.8
        path_metric = path_cost_metric + r_success_metric * 10 # 示例组合

        reward += fuzzy_weights['w_path'] * path_metric * path_factor_scale # 注意：path_factor_scale 可能需要调整

        # --- 4. CH负载相关奖励 (负载低越好) ---
        # p_cluster_ratio_j 是 CH 的负载比率 (actual_load / avg_load)
        # avg_load_ref = node_n.get("avg_load_per_ch_for_fuzzy", 10) # 这个应该从 NormalNodeCHSelectionFuzzySystem 的实例获取或传入
        # 假设我们能直接获取到CH的成员数
        ch_member_count = len([m_node for m_node in self.nodes if m_node.get("cluster_id") == chosen_ch_id and m_node["status"]=='active'])
        # 负载惩罚：成员越多，惩罚越大 (相对于一个理想值)
        ideal_members_per_ch = self.config.get('deec',{}).get('ideal_members_per_ch_ref', 
                                  max(1, self.get_alive_nodes() / (len(self.confirmed_cluster_heads_for_epoch) if self.confirmed_cluster_heads_for_epoch else 1) )
                                 ) 
        load_metric = -(ch_member_count - ideal_members_per_ch) # 成员数超过理想值则为负
        reward += fuzzy_weights['w_load'] * load_metric * load_factor_scale
        
        # --- 5. CH到BS距离相关奖励 (距离近越好) ---
        # d_c_base_normalized_j 是 CH 到BS的归一化距离 [0,1]
        d_c_base_j = self.calculate_distance_to_bs(chosen_ch_id)
        d_c_base_normalized_j = d_c_base_j / self.network_diagonal if self.network_diagonal > 0 else 0
        dist_bs_metric = (0.5 - d_c_base_normalized_j) # 越近值越大 (0.5是中点)
        reward += fuzzy_weights['w_dist_bs'] * dist_bs_metric * dist_bs_factor_scale
        
        # logger.debug(f"Node {normal_node_id} reward for CH {chosen_ch_id}: R_total={reward:.2f} [Outcome_R={reward - (fuzzy_weights['w_e_ch'] * ...)}, FuzzyWeightedTerms=...]")
        return reward

    def get_q_value_select_next_hop(self, ch_node_id, next_hop_node_id):
        """
        获取簇头 ch_node_id 选择 next_hop_node_id 作为下一跳的Q值。

        Args:
            ch_node_id (int): 当前做决策的簇头的ID。
            next_hop_node_id (int): 候选下一跳的ID 
                                    (可以是另一个CH的ID，或者是代表基站的特殊ID，例如 -1)。

        Returns:
            float: 对应的Q值。如果Q表中没有记录，则返回一个默认值 (例如0.0)。
        """
        # 1. 参数验证 (可选，但推荐)
        if not (0 <= ch_node_id < len(self.nodes)):
            logger.error(f"get_q_value_select_next_hop: 无效的 ch_node_id: {ch_node_id}")
            return 0.0 # 或者抛出异常

        ch_node_data = self.nodes[ch_node_id]

        if ch_node_data["status"] == "dead" or ch_node_data["role"] != "cluster_head":
            logger.warning(f"get_q_value_select_next_hop: 节点 {ch_node_id} 不是一个活跃的簇头。")
            return 0.0 # 非CH或死亡节点不应有下一跳Q值

        # 2. 获取该CH的下一跳Q表
        # 确保 q_table_select_next_hop 键存在于节点数据中
        if "q_table_select_next_hop" not in ch_node_data:
            # logger.debug(f"Node {ch_node_id} 的 q_table_select_next_hop 不存在，初始化为空字典。")
            ch_node_data["q_table_select_next_hop"] = {} # 如果不存在则初始化

        q_table = ch_node_data["q_table_select_next_hop"]

        # 3. 从Q表中获取对应下一跳的Q值
        #    如果 next_hop_node_id 不在q_table中，get方法会返回第二个参数（默认值）
        default_q_value = 0.0 # 中性初始化，鼓励探索未知动作
        # 你也可以考虑一个小的负值作为默认，以略微惩罚未探索的动作，
        # 但0.0对于ε-greedy通常足够了，因为ε会处理探索。
        # default_q_value = -0.1 

        q_value = q_table.get(next_hop_node_id, default_q_value)
        
        # logger.debug(f"Node {ch_node_id} Q_select_next_hop: NextHop={next_hop_node_id}, Q_val={q_value:.3f}")
        return q_value

    def calculate_reward_for_selecting_next_hop(self, 
                                                ch_node_id, 
                                                chosen_next_hop_id, # 可以是其他CH的ID，或BS的特殊ID (如 -1)
                                                fuzzy_weights, # 从 CHPathSelectionFuzzySystem 计算得到的权重
                                                transmission_successful, # 本次传输到下一跳是否成功
                                                actual_energy_spent_tx, # 本次发送到下一跳的实际能耗
                                                data_advanced_amount, # 数据向BS进展的量 (例如，距离的减少量)
                                                is_next_hop_bs, # chosen_next_hop_id 是否是基站
                                                next_hop_energy_normalized=None, # 下一跳的归一化能量 (如果是CH)
                                                next_hop_load_ratio=None, # 下一跳CH的负载比率 (如果是CH)
                                                is_next_hop_direct_bs_node=False
                                            ):
        """
        计算CH选择特定下一跳后的即时奖励。

        Args:
            ch_node_id (int): 做决策的CH的ID。
            chosen_next_hop_id (int): 被选中的下一跳的ID。
            fuzzy_weights (dict): 从CHPathSelectionFuzzySystem得到的动态权重因子。
                e.g., {'w_nh_progress': val, 'w_nh_energy_cost': val, ...}
            transmission_successful (bool): 本次传输到下一跳是否成功。
            actual_energy_spent_tx (float): 本次发送的实际能耗。
            data_advanced_amount (float): 数据向BS进展的量。
                                        正值表示更接近BS，0或负值表示没有进展或更远。
            is_next_hop_bs (bool): chosen_next_hop_id 是否是基站。
            next_hop_energy_normalized (float, optional): 下一跳CH的归一化能量。
            next_hop_load_ratio (float, optional): 下一跳CH的负载比率。

        Returns:
            float: 计算得到的综合奖励值。
        """
        # --- 获取基础奖励/惩罚单位值 (从config) ---
        # 你需要在 config.yml 中为这些参数定义合理的值
        # rewards: ch_select_next_hop:
        #   reach_bs_bonus: 100
        #   transmission_fail_penalty: -100
        #   data_progress_unit: 1.0  # 每单位进展量(如米)的奖励系数
        #   energy_cost_penalty_unit: 1000 # 每单位能耗的惩罚系数 (乘以能耗值)
        #   next_hop_low_energy_penalty_unit: -20
        #   next_hop_high_load_penalty_unit: -15
        #   reliability_bonus_unit: 10 # (如果使用链路质量/成功率)
        
        reward_cfg = self.config.get('rewards', {}).get('ch_select_next_hop', {})
        R_REACH_BS = float(reward_cfg.get('reach_bs_bonus', 100))
        R_FAIL_TRANSMISSION = float(reward_cfg.get('transmission_fail_penalty', -100))
        
        K_PROGRESS = float(reward_cfg.get('data_progress_unit', 1.0))
        K_ENERGY_COST = float(reward_cfg.get('energy_cost_penalty_unit', 2000)) # 调整这个系数以平衡能耗
        K_NH_LOW_ENERGY = float(reward_cfg.get('next_hop_low_energy_penalty_unit', -20)) # 惩罚选择低能量下一跳
        K_NH_HIGH_LOAD = float(reward_cfg.get('next_hop_high_load_penalty_unit', -15))  # 惩罚选择高负载下一跳
        # K_RELIABILITY = float(reward_cfg.get('reliability_bonus_unit', 10)) # 如果有更明确的可靠性指标

        # --- 初始化总奖励 ---
        total_reward = 0.0

        # --- 1. 传输结果奖励/惩罚 ---
        if not transmission_successful:
            total_reward += R_FAIL_TRANSMISSION
            # 如果传输失败，后续很多奖励可能不适用或应打折扣
            # logger.debug(f"CH {ch_node_id} to NH {chosen_next_hop_id}: Transmission FAILED. Penalty: {R_FAIL_TRANSMISSION}")
            return total_reward # 传输失败，直接返回大惩罚

        # 如果传输成功:
        # logger.debug(f"CH {ch_node_id} to NH {chosen_next_hop_id}: Transmission SUCCESSFUL.")

        # --- 2. 数据成功到达基站的巨大奖励 ---
        if is_next_hop_bs:
            total_reward += R_REACH_BS
            # logger.debug(f"  Reached BS! Bonus: {R_REACH_BS}")
        
        # --- 3. 路径进展奖励 (核心驱动力) ---
        # data_advanced_amount: 正值表示更接近BS
        # fuzzy_weights['w_nh_progress'] 应该是一个 [0,1] 或 [0.5, 1.5] 的因子
        # 假设 w_nh_progress 是 [0,1] 直接作为权重
        progress_reward = fuzzy_weights.get('w_nh_progress', 0.5) * data_advanced_amount * K_PROGRESS
        total_reward += progress_reward
        # logger.debug(f"  Path Progress: amount={data_advanced_amount:.2f}, weight={fuzzy_weights.get('w_nh_progress', 0.5):.2f}, reward_comp={progress_reward:.2f}")

        # --- 4. 能耗惩罚 ---
        # actual_energy_spent_tx 是正值
        # fuzzy_weights['w_nh_energy_cost'] 应该是一个 [0,1] 或 [0.5, 1.5] 的因子
        # 假设 w_nh_energy_cost 是 [0,1]，值越大表示越关注能耗（即惩罚越大）
        energy_penalty = - (fuzzy_weights.get('w_nh_energy_cost', 0.5) * actual_energy_spent_tx * K_ENERGY_COST)
        total_reward += energy_penalty
        # logger.debug(f"  Energy Cost: cost={actual_energy_spent_tx:.2e}, weight={fuzzy_weights.get('w_nh_energy_cost', 0.5):.2f}, penalty_comp={energy_penalty:.2f}")

        # --- 5. 下一跳CH的属性考量 (如果下一跳不是BS) ---
        if not is_next_hop_bs and not is_next_hop_direct_bs_node:
            # a. 下一跳CH的能量 (可持续性)
            # next_hop_energy_normalized is [0,1]
            # fuzzy_weights['w_nh_sustainability'] (或 w_fur)
            # 我们希望能量高时有正贡献，能量低时有负贡献
            if next_hop_energy_normalized is not None:
                # 将能量映射到 [-0.5, 0.5] (0.5是中点)
                energy_metric_nh = next_hop_energy_normalized - 0.5 
                sustainability_reward = fuzzy_weights.get('w_fur', 0.5) * energy_metric_nh * abs(K_NH_LOW_ENERGY) * 2 # 乘以2使其与惩罚幅度匹配
                total_reward += sustainability_reward
                # logger.debug(f"  NextHop Energy: norm_E={next_hop_energy_normalized:.2f}, weight={fuzzy_weights.get('w_fur', 0.5):.2f}, reward_comp={sustainability_reward:.2f}")
            
            # b. 下一跳CH的负载
            # next_hop_load_ratio is [0, ~3], 1是平均负载
            # fuzzy_weights['w_nh_load_neighbor']
            # 我们希望负载低时有正贡献（或无惩罚），负载高时有负贡献
            if next_hop_load_ratio is not None:
                # 将负载比率映射，例如超过1时开始有显著惩罚
                # load_metric_nh = 1.0 - next_hop_load_ratio # 如果 ratio=0.5, metric=0.5; ratio=1, metric=0; ratio=2, metric=-1
                load_metric_nh = 0
                if next_hop_load_ratio > 1.2: # 超过平均负载20%开始惩罚
                    load_metric_nh = -(next_hop_load_ratio - 1.2) 
                elif next_hop_load_ratio < 0.8: # 低于平均负载20%给点小奖励
                    load_metric_nh = (0.8 - next_hop_load_ratio) * 0.5 # 奖励幅度小一些

                load_penalty = fuzzy_weights.get('w_load_neighbor', 0.5) * load_metric_nh * abs(K_NH_HIGH_LOAD) 
                total_reward += load_penalty
                # logger.debug(f"  NextHop Load: ratio={next_hop_load_ratio:.2f}, weight={fuzzy_weights.get('w_load_neighbor', 0.5):.2f}, penalty_comp={load_penalty:.2f}")

        # --- 6. (可选) 基于与下一跳的直接链路质量/历史成功率的奖励 ---
        # 你在模糊逻辑输入中有 R_c_success (与下一跳的历史通信成功率)
        # 可以在这里再对其进行一次奖励，或者认为它已经通过影响 transmission_successful 和模糊权重体现了
        # 例如:
        # if transmission_successful: # 只有传输成功才考虑这个
        #    r_success_val_for_reward = self.nodes[ch_node_id].get("history_success_with_nh", {}).get(chosen_next_hop_id, 0.8)
        #    reliability_bonus = fuzzy_weights.get('w_nh_reliability', 0.5) * (r_success_val_for_reward - 0.5) * K_RELIABILITY # 假设w_nh_reliability存在
        #    total_reward += reliability_bonus
        #    logger.debug(f"  Reliability to NH: rate={r_success_val_for_reward:.2f}, weight={fuzzy_weights.get('w_nh_reliability',0.5):.2f}, bonus_comp={reliability_bonus:.2f}")


        # logger.info(f"CH {ch_node_id} -> NH {chosen_next_hop_id}: Final Reward = {total_reward:.3f}")
        return total_reward

    def update_q_value_select_next_hop(self, 
                                    ch_node_id, 
                                    chosen_next_hop_id, # 实际选择的下一跳的ID
                                    reward,             # 选择该下一跳后获得的即时奖励 R
                                    # 可选参数，用于更完整的Q学习 (Sarsa 或 Q-learning)
                                    next_state_available_next_hops=None, # 下一状态S'时，该CH可选择的下一跳列表 [(nh_id1, dist1), ...]
                                    is_terminal_next_hop=False # chosen_next_hop_id 是否是最终目标 (BS)
                                    ):
        """
        更新簇头 ch_node_id 选择 next_hop_node_id 作为下一跳的Q值。

        Args:
            ch_node_id (int): 当前做决策的簇头的ID。
            chosen_next_hop_id (int): 被选中的下一跳的ID。
            reward (float): 选择该下一跳后获得的即时奖励 R.
            next_state_available_next_hops (list, optional): 
                在下一状态S'时，该CH可选择的下一跳的列表。
                每个元素可以是 next_hop_id，或者 (next_hop_id, other_info) 用于更复杂的S'。
                如果为None或空，则max_q_next_state将为0 (单步优化或回合结束)。
            is_terminal_next_hop (bool): 指示 chosen_next_hop_id 是否是最终目标（如BS）。
                                        如果为True，则下一状态的价值通常为0。
        """
        if ch_node_id == -1 or not (0 <= ch_node_id < len(self.nodes)):
            # 记录一个错误，说明有逻辑试图更新BS的Q表，这是不应该发生的
            logger.error(f"update_q_value_select_next_hop: 尝试为一个无效的ID({ch_node_id})更新Q表。")
            return
        
        # 1. 参数验证和获取节点数据
        if not (0 <= ch_node_id < len(self.nodes)):
            logger.error(f"update_q_value_select_next_hop: 无效的 ch_node_id: {ch_node_id}")
            return

        ch_node_data = self.nodes[ch_node_id]

        if ch_node_data["status"] == "dead" or ch_node_data["role"] != "cluster_head":
            # logger.warning(f"update_q_value_select_next_hop: 节点 {ch_node_id} 不是一个活跃的簇头。")
            return 

        # 2. 获取Q学习参数和Q表
        # 确保 q_table_select_next_hop 键存在
        if "q_table_select_next_hop" not in ch_node_data:
            ch_node_data["q_table_select_next_hop"] = {}
        
        q_table = ch_node_data["q_table_select_next_hop"]
        
        # 从节点特定配置或全局配置获取alpha和gamma
        q_cfg = self.config.get('q_learning', {})
        alpha = ch_node_data.get("alpha_select_next_hop", float(q_cfg.get('alpha_ch_hop', 0.1)))
        gamma = ch_node_data.get("gamma_select_next_hop", float(q_cfg.get('gamma_ch_hop', 0.9)))

        # 3. 获取旧的Q值
        old_q_value = q_table.get(chosen_next_hop_id, 0.0) # 如果是第一次选这个下一跳，Q值为0

        # 4. 计算 max_A' Q(S', A') (下一状态的最大期望Q值)
        max_q_next_state = 0.0
        if not is_terminal_next_hop and next_state_available_next_hops:
            # 如果选择的下一跳不是BS，并且我们有关于下一状态S'的信息
            # S' 是CH在选择了当前下一跳并发送数据、消耗能量后，在下一轮（或下一个决策点）的状态
            # next_state_available_next_hops 是在S'时，该CH可以接触到的所有下一跳
            
            # 我们需要为S'时的每个可用下一跳获取其Q值
            # 注意：这里的S'对于表格Q学习来说比较难定义，因为它会变化。
            # 简化处理：
            #   - 假设下一状态S'与当前状态S（用于选择chosen_next_hop_id时的状态）相似，
            #     或者我们不直接使用S'来索引Q表，而是直接查找下一跳的Q值。
            #   - 如果Q表是 Q_ch[next_hop_id]，那么我们直接从这个Q表中找下一状态的最大Q值。
            
            q_values_for_next_state_actions = []
            for nh_info in next_state_available_next_hops:
                nh_id_in_next_state = nh_info[0] # 假设nh_info是 (id, type, dist)
                # 获取在下一状态S'时，选择nh_id_in_next_state的Q值
                # 这里的核心是如何定义和获取Q(S', a')
                # 简单版本：直接用当前的Q表来估计下一状态的Q值
                q_values_for_next_state_actions.append(self.get_q_value_select_next_hop(ch_node_id, nh_id_in_next_state))
                
            if q_values_for_next_state_actions:
                max_q_next_state = max(q_values_for_next_state_actions)
        # 如果 is_terminal_next_hop 为 True (即到达BS)，则 max_q_next_state 保持为 0.0

        # 5. 应用Q学习更新规则 (Bellman方程)
        # Q(S, A) ← Q(S, A) + α * [R + γ * max_A' Q(S', A') - Q(S, A)]
        # 在我们的简化Q表中，S是隐含的（当前CH的状态），A是chosen_next_hop_id
        new_q_value = old_q_value + alpha * (reward + gamma * max_q_next_state - old_q_value)
        
        q_table[chosen_next_hop_id] = new_q_value
        # node["q_table_select_next_hop"] = q_table # 不需要，因为q_table是字典的引用

        # logger.debug(f"CH {ch_node_id} Q_select_next_hop updated: NextHop={chosen_next_hop_id}, R={reward:.2f}, OldQ={old_q_value:.3f}, NewQ={new_q_value:.3f}, NextMaxQ={max_q_next_state:.3f}")

    def step(self, current_round_num):
        """
        执行一轮完整的网络仿真。
        该函数是高层协调器，调用各个阶段的辅助函数。
        """
        self.current_round = current_round_num
        logger.info(f"--- 开始第 {self.current_round} 轮 ---")

        # 阶段 0: 每轮开始前的准备工作
        self._prepare_for_new_round()

        # 阶段 1: Epoch 开始时的特殊处理 (CH 竞争与确认)
        if self.current_round % self.epoch_length == 0:
            self._run_epoch_start_phase()

        # 阶段 2: 普通节点选择簇头
        self._run_normal_node_selection_phase()
        
        # 阶段 2.5: CH 容量限制与协调
        if self.enable_ch_capacity_limit:
            self._run_ch_capacity_check_phase()
        self._remedy_isolated_nodes() # 新增调用
        # 阶段 3: CH 选择下一跳进行路由
        self._run_ch_routing_phase()
        
        # 阶段 4: 执行本轮所有暂存的能量消耗
        self._execute_energy_consumption_for_round()

        # 阶段 5: 更新并记录本轮的性能指标
        self._update_and_log_performance_metrics()
        
        logger.info(f"--- 第 {self.current_round} 轮结束 ---")
        
        # 检查仿真是否应结束 (例如, 所有节点都死亡)
        if self.get_alive_nodes() == 0:
            logger.info("网络中所有节点均已死亡，仿真结束。")
            return False
            
        return True

    def _prepare_for_new_round(self):
            """阶段 0: 为新一轮仿真做准备。"""
            self._build_spatial_index()
            self.identify_direct_bs_nodes()
            
            # 重置每轮的统计数据
            self.sim_packets_generated_this_round = 0
            self.sim_packets_delivered_bs_this_round = 0
            self.sim_total_delay_this_round = 0.0
            self.sim_num_packets_for_delay_this_round = 0

            for node in self.nodes:
                if node["status"] == "active":
                    # 重置通信范围和路由选择
                    node["current_communication_range"] = node["base_communication_range"]
                    node["chosen_next_hop_id"] = None
                    
                    # 清空上一轮的待消耗能量
                    node["pending_tx_energy"] = 0.0
                    node["pending_rx_energy"] = 0.0
                    node["pending_aggregation_energy"] = 0.0
                    # 假设每个活跃节点每轮都产生一个包
                    self.sim_packets_generated_this_round += 1
                    self.sim_packets_generated_total += 1
                    
                    # 我们可以给节点一个标志，表示它有数据待处理
                    node["has_data_to_send"] = True
                    # Epsilon 衰减 (普通节点选CH)
                    if node["role"] == "normal" and not node.get("can_connect_bs_directly", False):
                        min_eps = self.config.get('q_learning',{}).get('epsilon_select_ch_min', 0.01)
                        decay = self.config.get('q_learning',{}).get('epsilon_select_ch_decay_per_round', 0.998)
                        current_eps = node.get("epsilon_select_ch", self.config.get('q_learning',{}).get('epsilon_select_ch_initial', 0.3))
                        node["epsilon_select_ch"] = max(min_eps, current_eps * decay)
    
    def _run_epoch_start_phase(self):
        """阶段 1: 处理新 Epoch 开始时的逻辑，包括 CH 竞争和确认。"""
        logger.info(f"--- ***** 新 Epoch 开始 (轮次 {self.current_round}) ***** ---")
        
        # 更新上一个 epoch 的 CH 竞争 Q 表
        if self.current_round > 0:
            self._update_ch_competition_q_tables_at_epoch_end()
        
        # 为 CH 竞争状态计算保存上一个 epoch 的 CH 列表
        self.confirmed_cluster_heads_previous_epoch = list(self.confirmed_cluster_heads_for_epoch)
        
        # CH 竞争 Q 学习的 Epsilon 衰减
        self.current_epsilon_compete = max(
            self.epsilon_compete_min,
            self.current_epsilon_compete * (self.epsilon_compete_decay ** self.epoch_length)
        )
        logger.info(f"Epoch 开始，CH 竞争 Epsilon 更新为: {self.current_epsilon_compete:.4f}")

        # 节点通过 Q 学习决定是否宣告成为 CH
        ch_declarations = self._run_ch_competition_decision()
        
        # 从宣告者中最终确定本 Epoch 的活跃 CH
        self.finalize_ch_roles(ch_declarations)
    
    def _run_ch_competition_decision(self):
        """节点通过 Q 学习决定是否宣告成为 CH，返回宣告列表。"""
        logger.info("Epoch 开始：节点通过 Q 学习竞争成为 CH...")
        ch_declarations_this_epoch = []
        self.competition_log_for_current_epoch = {}  # 清空并开始记录

        nodes_eligible = [
            n for n in self.nodes
            if n["status"] == "active" and not n.get("can_connect_bs_directly", False)
        ]

        for node in nodes_eligible:
            node_id = node["id"]
            state_tuple = self.get_discrete_state_tuple_for_competition(node_id)
            if state_tuple is None:
                continue

            action = 0
            if random.random() < self.current_epsilon_compete:
                action = random.choice([0, 1])
            else:
                q0 = self.get_q_value_compete_ch(node_id, state_tuple, 0)
                q1 = self.get_q_value_compete_ch(node_id, state_tuple, 1)
                action = 1 if q1 > q0 else (0 if q0 > q1 else random.choice([0, 1]))
            
            # 记录决策以备 epoch 结束时更新 Q 表
            self.competition_log_for_current_epoch[node_id] = {
                "state_tuple": state_tuple, "action": action
            }

            if action == 1:
                ch_declarations_this_epoch.append(node_id)
        
        logger.info(f"Epoch CH 竞争：{len(ch_declarations_this_epoch)} 个节点宣告想成为 CH。")
        return ch_declarations_this_epoch
    
    def _run_normal_node_selection_phase(self):
        """
        [最终修正版] 阶段 2: 普通节点使用 Q 学习选择 CH。
        此函数只负责决策和Q-table更新，不处理任何物理过程。
        """        
        logger.info("开始阶段2：普通节点Q学习选择簇头...")
        num_nodes_switched_ch_this_round = 0
        num_nodes_initially_assigned_this_round = 0
        
        # 获取用于模糊逻辑的参考值
        avg_load_per_confirmed_ch_ref = self.get_alive_nodes() / len(self.confirmed_cluster_heads_for_epoch) if self.confirmed_cluster_heads_for_epoch else 10
        AVG_E_SEND_REF_NORMAL_NODE = 0.001 # 用于模糊逻辑归一化的参考值

        for node_data in self.nodes:
            if not (node_data["status"] == "active" and 
                    node_data["role"] == "normal" and 
                    not node_data.get("can_connect_bs_directly", False)):
                continue

            node_id = node_data["id"]
            current_assigned_ch_id = node_data["cluster_id"]
            
            # 1. 寻找可达的CH
            reachable_chs_info = []
            if self.confirmed_cluster_heads_for_epoch:
                for ch_id_cand in self.confirmed_cluster_heads_for_epoch:
                    if self.nodes[ch_id_cand]["status"] == "active":
                        distance = self.calculate_distance(node_id, ch_id_cand)
                        if distance <= node_data["base_communication_range"]:
                            reachable_chs_info.append((ch_id_cand, distance))
            
            if not reachable_chs_info:
                node_data["cluster_id"] = -1
                continue

            # 2. Q学习决策 (利用/探索)
            q_values_for_reachable_chs = {ch_id: self.get_q_value_select_ch(node_id, ch_id) for ch_id, _ in reachable_chs_info}
            
            potential_new_ch_id = max(q_values_for_reachable_chs, key=lambda k:q_values_for_reachable_chs[k])
            best_q_for_new_ch = q_values_for_reachable_chs[potential_new_ch_id]

            chosen_for_this_round_ch_id = -1
            current_epsilon_select = node_data.get("epsilon_select_ch", 0.3)

            if random.random() < current_epsilon_select:
                chosen_for_this_round_ch_id = random.choice([ch_info[0] for ch_info in reachable_chs_info])
            else:
                q_current_ch = q_values_for_reachable_chs.get(current_assigned_ch_id, -float('inf'))
                if current_assigned_ch_id == -1 or current_assigned_ch_id not in q_values_for_reachable_chs:
                    chosen_for_this_round_ch_id = potential_new_ch_id
                elif best_q_for_new_ch > q_current_ch + self.ch_switching_hysteresis:
                    chosen_for_this_round_ch_id = potential_new_ch_id
                else:
                    chosen_for_this_round_ch_id = current_assigned_ch_id
            
            if chosen_for_this_round_ch_id == -1: continue # 以防万一

            # 3. 为被选中的CH计算奖励并更新Q表
            # 无论最终是否切换，我们都为“被选中的那个CH”更新一次Q表
            ch_node_chosen = self.nodes[chosen_for_this_round_ch_id]
            dist_to_chosen_ch = self.calculate_distance(node_id, chosen_for_this_round_ch_id)
            
            # a. 模拟一次“理想”传输，以计算奖励
            r_success_with_chosen_ch_norm = node_data.get("history_success_with_ch", {}).get(chosen_for_this_round_ch_id, 0.9)
            transmission_success_for_reward = (random.random() < r_success_with_chosen_ch_norm)
            
            # b. 计算预期的发送能耗，用于奖励计算
            packet_size = self.config.get("simulation",{}).get("packet_size",4000)
            expected_e_send = self.calculate_transmission_energy(dist_to_chosen_ch, packet_size)
            
            # c. 获取模糊逻辑的输入
            dc_base_chosen_ch = self.calculate_distance_to_bs(chosen_for_this_round_ch_id)
            e_cluster_chosen_ch_norm = ch_node_chosen["energy"] / ch_node_chosen["initial_energy"] if ch_node_chosen["initial_energy"] > 0 else 0
            load_actual_chosen_ch = len([m for m in self.nodes if m.get("cluster_id") == chosen_for_this_round_ch_id])
            p_cluster_ratio_for_fuzzy = load_actual_chosen_ch / avg_load_per_confirmed_ch_ref if avg_load_per_confirmed_ch_ref > 0 else 0
            e_send_total_ratio_for_fuzzy = expected_e_send / AVG_E_SEND_REF_NORMAL_NODE if AVG_E_SEND_REF_NORMAL_NODE > 0 else 0
            
            fuzzy_weights = self.normal_node_fuzzy_logic.compute_weights(
                current_dc_base=dc_base_chosen_ch,
                current_e_cluster_normalized=e_cluster_chosen_ch_norm,
                current_p_cluster_ratio_val=p_cluster_ratio_for_fuzzy, 
                current_r_success_normalized=r_success_with_chosen_ch_norm,
                current_e_send_total_ratio_val=e_send_total_ratio_for_fuzzy 
            )

            # d. 计算奖励并更新Q表
            reward = self.calculate_reward_for_selecting_ch(
                node_id, chosen_for_this_round_ch_id, fuzzy_weights,
                transmission_successful=transmission_success_for_reward,
                actual_energy_spent_tx=expected_e_send
            )
            self.update_q_value_select_ch(node_id, chosen_for_this_round_ch_id, reward)
            
            # 4. 最终确定本轮的cluster_id
            if current_assigned_ch_id != chosen_for_this_round_ch_id:
                if current_assigned_ch_id == -1:
                    num_nodes_initially_assigned_this_round += 1
                else:
                    num_nodes_switched_ch_this_round += 1
            node_data["cluster_id"] = chosen_for_this_round_ch_id

        logger.info(f"Q学习选择CH阶段：{num_nodes_initially_assigned_this_round} 个节点初次分配，{num_nodes_switched_ch_this_round} 个节点切换了CH。")
        
    def _run_ch_capacity_check_phase(self):
        """阶段 2.5: 检查并处理 CH 容量超限的问题。"""
        logger.info("开始阶段 2.5：CH 容量限制检查...")
        if self.enable_ch_capacity_limit and self.confirmed_cluster_heads_for_epoch:
            num_rejections_this_round = 0

            # 计算网络平均每个CH应服务的成员数 (参考值)
            num_alive_non_direct_bs_nodes = len([
                n for n in self.nodes
                if n["status"] == "active" and not n.get("can_connect_bs_directly", False) and n["role"] == "normal"
            ])
            num_active_chs_for_capacity = len(self.confirmed_cluster_heads_for_epoch)
            
            if num_active_chs_for_capacity > 0:
                avg_members_per_ch_ideal = num_alive_non_direct_bs_nodes / num_active_chs_for_capacity
                # 使用 factor 计算最大成员数
                max_members_for_ch_calculated = max(1, int(avg_members_per_ch_ideal * self.ch_max_members_factor))
                # 如果使用绝对值，则是:
                # max_members_for_ch_calculated = self.ch_max_absolute_members
                logger.debug(f"  CH容量限制：理想平均成员数={avg_members_per_ch_ideal:.2f}, 计算得到的最大成员数上限={max_members_for_ch_calculated}")
            else:
                # 没有活跃CH，无法进行容量限制，或设置一个默认上限（如果普通节点依然选了某些ID）
                max_members_for_ch_calculated = 10 # 任意默认值，理论上不应发生

            for ch_id in self.confirmed_cluster_heads_for_epoch:
                ch_node_data_cap = self.nodes[ch_id] # 使用不同变量名
                if ch_node_data_cap["status"] != "active" or ch_node_data_cap["role"] != "cluster_head":
                    continue

                # 找出所有选择了这个CH的普通节点
                current_members_of_ch = [
                    node_member for node_member in self.nodes
                    if node_member["status"] == "active" and \
                       node_member["role"] == "normal" and \
                       not node_member.get("can_connect_bs_directly", False) and \
                       node_member.get("cluster_id") == ch_id
                ]
                
                num_current_members = len(current_members_of_ch)
                # logger.debug(f"  CH {ch_id} 当前有 {num_current_members} 个成员，容量上限 {max_members_for_ch_calculated}.")

                if num_current_members > max_members_for_ch_calculated:
                    num_to_reject = num_current_members - max_members_for_ch_calculated
                    logger.info(f"  CH {ch_id} 过载 (成员数 {num_current_members} > 上限 {max_members_for_ch_calculated})。需要拒绝 {num_to_reject} 个成员。")

                    # 拒绝策略：例如，拒绝距离最远的成员
                    # 可以先按距离排序，然后拒绝多余的
                    current_members_of_ch.sort(key=lambda m_node: self.calculate_distance(m_node["id"], ch_id), reverse=True)
                    
                    for i_reject in range(num_to_reject): # 使用不同循环变量
                        if i_reject < len(current_members_of_ch): # 确保列表还有元素
                            rejected_node = current_members_of_ch[i_reject]
                            rejected_node_id = rejected_node["id"]
                            
                            logger.info(f"    CH {ch_id} 拒绝节点 {rejected_node_id} (距离: {self.calculate_distance(rejected_node_id, ch_id):.2f}m)。")
                            
                            # 1. 更新被拒绝节点的 cluster_id
                            rejected_node["cluster_id"] = -1 # 标记为未分配
                            
                            # 2. 更新被拒绝节点对该CH的Q值 (施加惩罚)
                            #   需要获取被拒绝节点在选择该CH时使用的模糊权重和能耗等信息，
                            #   或者更简单地，直接更新Q值，惩罚该(CH_ID)动作。
                            #   这里采用简化的直接Q值惩罚，不重新计算完整奖励。
                            #   注意：这可能不是最精确的Q学习更新，但作为惩罚机制是可行的。
                            
                            # 简化的Q值更新：直接在旧Q值基础上减去惩罚或乘以一个衰减因子
                            # 或者，如果能拿到当时的fuzzy_weights和消耗，可以重新调用calculate_reward_for_selecting_ch，但结果设为极低
                            # 最简单：
                            self.update_q_value_select_ch(rejected_node_id, ch_id, self.ch_rejection_penalty) # 使用配置的惩罚值
                            if "collected_raw_packets" in ch_node_data_cap and ch_node_data_cap["collected_raw_packets"] > 0:
                                ch_node_data_cap["collected_raw_packets"] -= 1
                            
                            # 将被拒绝节点的数据重新标记为“待发送”
                            rejected_node["has_data_to_send"] = True
                            num_rejections_this_round += 1
                        else:
                            break # 如果排序后的成员列表已空

            if num_rejections_this_round > 0:
                logger.info(f"CH容量限制阶段：共拒绝了 {num_rejections_this_round} 个节点的加入请求。")


    def _run_ch_routing_phase(self):
        """
        [最终版-V3] 阶段 3: 活跃 CH 通过 Q 学习选择下一跳。
        集成了遗留包处理、主动数据融合、缓存管理和全局拓扑引导。
        """
        logger.info("开始阶段 3：活跃 CH 通过 Q 学习选择下一跳...")

        # 1. "交接班"：处理上一轮落选CH的遗留数据包
        current_ch_set = set(self.confirmed_cluster_heads_for_epoch)
        packets_to_reassign = []
        for holder_id, buffers in list(self.packets_in_transit.items()):
            if holder_id not in current_ch_set:
                logger.warning(f"节点 {holder_id} 在本轮落选CH，其持有的数据包需要处理。")
                packets_to_reassign.extend(buffers.get("forwarding_buffer", []))
                packets_to_reassign.extend(buffers.get("generation_buffer", []))
                del self.packets_in_transit[holder_id]

        for packet in packets_to_reassign:
            last_valid_hop = packet["path"][-1]
            if last_valid_hop in self.nodes:
                best_new_holder, min_dist = None, float('inf')
                for ch_id in current_ch_set:
                    if ch_id not in self.packets_in_transit: self.packets_in_transit[ch_id] = {"forwarding_buffer": [], "generation_buffer": []}
                    if len(self.packets_in_transit[ch_id]["forwarding_buffer"]) < self.ch_forwarding_buffer_size:
                        dist = self.calculate_distance(last_valid_hop, ch_id)
                        if dist < min_dist:
                            min_dist, best_new_holder = dist, ch_id
                if best_new_holder:
                    logger.info(f"数据包 (源: {packet['source_ch']}) 从落选CH处重新分配到新CH {best_new_holder} 的转发缓冲区。")
                    self.packets_in_transit[best_new_holder]["forwarding_buffer"].append(packet)
                else:
                    logger.warning(f"数据包 (源: {packet['source_ch']}) 重新分配失败，找不到有空闲转发缓存的新持有者。")

        # 2. CH主动融合本轮新数据到“生成缓冲区”
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            if self.nodes[ch_id]["status"] == "active":
                if ch_id not in self.packets_in_transit: self.packets_in_transit[ch_id] = {"forwarding_buffer": [], "generation_buffer": []}
                
                raw_packets_sources = [node_id for node_id, node in enumerate(self.nodes) if (node.get("cluster_id") == ch_id or node["id"] == ch_id) and node.get("has_data_to_send")]
                
                if raw_packets_sources:
                    if not self.packets_in_transit[ch_id]["generation_buffer"]:
                        uid = f"{self.current_round}-{ch_id}"
                        new_packet = {"source_ch": ch_id, "gen_round": self.current_round, "path": [ch_id], "uid": uid, "num_raw_packets": len(raw_packets_sources), "original_sources": raw_packets_sources}
                        self.packets_in_transit[ch_id]["generation_buffer"].append(new_packet)
                        agg_cost_per_bit = self.config.get('energy', {}).get('aggregation_cost_per_bit', 5e-9)
                        base_packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
                        total_raw_size = base_packet_size * len(raw_packets_sources)
                        agg_energy = agg_cost_per_bit * total_raw_size
                        self.nodes[ch_id]["pending_aggregation_energy"] += agg_energy
                        for node_id in raw_packets_sources: self.nodes[node_id]["has_data_to_send"] = False
                        logger.debug(f"CH {ch_id} 融合了 {len(raw_packets_sources)} 个原始包到生成缓冲区。")
                    else:
                        logger.warning(f"CH {ch_id} 的生成缓冲区已被占用，本轮 {len(raw_packets_sources)} 个新生成的原始包丢失。")

        # 3. 构建当前轮次的CH路由图并计算最短路径
        G_ch_routing = nx.Graph()
        active_ch_list = [ch_id for ch_id in self.confirmed_cluster_heads_for_epoch if self.nodes[ch_id]["status"] == "active"]
        G_ch_routing.add_nodes_from(active_ch_list)
        G_ch_routing.add_node(self.BS_ID)
        for i in range(len(active_ch_list)):
            for j in range(i + 1, len(active_ch_list)):
                ch1_id, ch2_id = active_ch_list[i], active_ch_list[j]
                if self.calculate_distance(ch1_id, ch2_id) <= self.nodes[ch1_id]["base_communication_range"]:
                    G_ch_routing.add_edge(ch1_id, ch2_id)
        for ch_id in active_ch_list:
            if self.calculate_distance_to_bs(ch_id) <= self.nodes[ch_id]["base_communication_range"]:
                G_ch_routing.add_edge(ch_id, self.BS_ID)
        
        shortest_path_lengths = {}
        if self.BS_ID in G_ch_routing:
            try:
                path_lengths_from_nx = nx.single_source_shortest_path_length(G_ch_routing, self.BS_ID)
                shortest_path_lengths = dict(path_lengths_from_nx)
            except nx.NetworkXNoPath:
                logger.warning("在当前CH拓扑中，BS节点可能孤立。")
        for ch_id in active_ch_list:
            if ch_id not in shortest_path_lengths:
                shortest_path_lengths[ch_id] = float('inf')

        # 4. 决策阶段
        transfer_intentions = []
        # 遍历所有当前持有数据包的CH ID
        for ch_id in list(self.packets_in_transit.keys()):
            # 在循环开始时，重新、安全地获取该CH的缓存
            buffers = self.packets_in_transit.get(ch_id)
            
            # 防御性检查：如果该CH的条目在处理过程中被移除了，则跳过
            if not buffers:
                continue
                
            if self.nodes[ch_id]["status"] != "active": continue

            packet_to_send, source_buffer_type = None, None
            
            # 现在可以安全地使用 .get()
            if buffers.get("forwarding_buffer"):
                packet_to_send = buffers["forwarding_buffer"][0]
                source_buffer_type = "forwarding_buffer"
            elif buffers.get("generation_buffer"):
                packet_to_send = buffers["generation_buffer"][0]
                source_buffer_type = "generation_buffer"
            
            if not packet_to_send: continue
            
            if len(packet_to_send["path"]) > self.config.get("simulation", {}).get("max_packet_hops", 15):
                logger.warning(f"数据包 (UID: {packet_to_send.get('uid')}) 在CH {ch_id} 处超最大跳数，被丢弃。")
                self.packets_in_transit[ch_id].pop(0)
                continue
            
            final_candidates_info = self._find_candidate_next_hops(ch_id)
            chosen_next_hop_id = self._decide_next_hop_with_q_learning(ch_id, final_candidates_info, packet_to_send["path"], shortest_path_lengths)

            if chosen_next_hop_id != -100:
                chosen_nh_info = next((info for info in final_candidates_info if info[0] == chosen_next_hop_id), (chosen_next_hop_id, self.BS_TYPE_STR, self.calculate_distance_to_bs(ch_id)) if chosen_next_hop_id == self.BS_ID else None)
                if not chosen_nh_info: continue
                chosen_nh_type, dist_to_nh = chosen_nh_info[1], chosen_nh_info[2]
                
                success_rate = self._get_transmission_success_rate(ch_id, chosen_next_hop_id, chosen_nh_type, dist_to_nh)
                is_transmission_successful = (random.random() < success_rate)

                reward, next_hops, is_terminal = self._calculate_routing_reward_and_next_state(ch_id, chosen_next_hop_id, chosen_nh_type, dist_to_nh, is_transmission_successful, final_candidates_info)
                self.update_q_value_select_next_hop(ch_id, chosen_next_hop_id, reward, next_hops, is_terminal)

                if is_transmission_successful:
                    transfer_intentions.append((ch_id, chosen_next_hop_id, packet_to_send, chosen_nh_type, source_buffer_type))
                    self.nodes[ch_id]["chosen_next_hop_id"] = chosen_next_hop_id
                else:
                    self._penalize_failed_routing_action(ch_id, chosen_next_hop_id)

        # 5. 执行阶段
        delivery_attempts = {}
        for sender_id, receiver_id, packet, receiver_type, source_buffer_type in transfer_intentions:
            is_bs_like = (receiver_type == self.BS_TYPE_STR or receiver_type == self.DIRECT_BS_NODE_TYPE_STR)
            if is_bs_like:
                self._execute_successful_transfer(sender_id, receiver_id, packet, receiver_type, source_buffer_type)
            else:
                if receiver_id not in delivery_attempts: delivery_attempts[receiver_id] = []
                delivery_attempts[receiver_id].append((sender_id, packet, receiver_type, source_buffer_type))

        for receiver_id, attempts in delivery_attempts.items():
            if receiver_id not in self.packets_in_transit: self.packets_in_transit[receiver_id] = {"forwarding_buffer": [], "generation_buffer": []}
            available_slots = self.ch_forwarding_buffer_size - len(self.packets_in_transit[receiver_id]["forwarding_buffer"])

            if available_slots <= 0:
                logger.warning(f"CH {receiver_id} 的转发缓存已满，拒绝所有 {len(attempts)} 个传入请求。")
                for sender_id, _, _, _ in attempts:
                    self._penalize_failed_routing_action(sender_id, receiver_id)
                continue

            sorted_attempts = sorted(attempts, key=lambda item: self.nodes[item[0]]["energy"], reverse=True)
            for i, (sender_id, packet, receiver_type, source_buffer_type) in enumerate(sorted_attempts):
                if i < available_slots:
                    self._execute_successful_transfer(sender_id, receiver_id, packet, receiver_type, source_buffer_type)
                else:
                    logger.warning(f"CH {sender_id} 到 {receiver_id} 的传输因接收方转发缓存满而失败。")
                    self._penalize_failed_routing_action(sender_id, receiver_id)

        # 6. 清理和更新
        for holder_id, buffer in list(self.packets_in_transit.items()):
            if self.nodes[holder_id]["status"] == "dead":
                logger.warning(f"数据包持有者 {holder_id} 已死亡，其缓存的 {len(buffer)} 个数据包丢失。")
                del self.packets_in_transit[holder_id]

        for ch_id in self.confirmed_cluster_heads_for_epoch:
            if self.nodes[ch_id]["status"] == "active":
                min_eps = self.config.get('q_learning',{}).get('epsilon_ch_hop_min',0.01)
                decay = self.config.get('q_learning',{}).get('epsilon_ch_hop_decay_per_round',0.998)
                current_eps = self.nodes[ch_id].get("epsilon_select_next_hop", self.config.get('q_learning',{}).get('epsilon_ch_hop_initial',0.2))
                self.nodes[ch_id]["epsilon_select_next_hop"] = max(min_eps, current_eps * decay)

    def _decide_next_hop_with_q_learning(self, ch_id, candidates_info, path_history, shortest_paths):
        """
        [新版本] 使用Q学习、地理贪心和全局最短路径，从候选者中决定下一跳。
        """
        ch_node_data = self.nodes[ch_id]
        current_epsilon_ch_hop = ch_node_data.get("epsilon_select_next_hop", 0.2)
        
        # 从config获取各个奖励的权重因子
        q_cfg = self.config.get('q_learning', {})
        q_factor = q_cfg.get('q_value_factor', 1.0)
        geo_factor = q_cfg.get('geography_reward_factor', 0.05)
        path_factor = q_cfg.get('shortest_path_reward_factor', 10.0)

        valid_next_hops_q_values = {}
        
        for nh_id, nh_type, dist in candidates_info:
            if nh_id in path_history:
                continue

            # a. 历史经验Q值
            q_original = self.get_q_value_select_next_hop(ch_id, nh_id) if nh_id != self.BS_ID else 0.0

            # b. 地理进展奖励 (基于物理距离)
            dist_ch_to_bs = self.calculate_distance_to_bs(ch_id)
            dist_nh_to_bs = self.calculate_distance_to_bs(nh_id) if nh_id != self.BS_ID else 0.0
            geo_reward = (dist_ch_to_bs - dist_nh_to_bs) * geo_factor

            # c. 全局路径奖励 (基于网络拓扑跳数)
            current_path_len = shortest_paths.get(ch_id, float('inf'))
            next_hop_path_len = shortest_paths.get(nh_id, float('inf'))
            path_reward = 0.0
            if current_path_len != float('inf') and next_hop_path_len < current_path_len:
                # 奖励的幅度与跳数的减少量成正比
                path_reward = (current_path_len - next_hop_path_len) * path_factor

            # d. 综合决策Q值
            q_for_decision = (q_factor * q_original) + geo_reward + path_reward
            valid_next_hops_q_values[nh_id] = q_for_decision
            logger.debug(f"  - Cand: {nh_id}, OrigQ: {q_original:.2f}, GeoR: {geo_reward:.2f}, PathR: {path_reward:.2f}, FinalQ: {q_for_decision:.2f}")

        if not valid_next_hops_q_values:
            # 找不到非环路下一跳，检查是否能直连BS作为最后手段 (逃生舱)
            if self.calculate_distance_to_bs(ch_id) <= ch_node_data["base_communication_range"]:
                logger.warning(f"CH {ch_id} 找不到常规下一跳，启动逃生模式：直连BS。")
                return self.BS_ID
            else:
                logger.warning(f"CH {ch_id} 没有有效的下一跳，且无法直连BS。")
                return -100

        if random.random() < current_epsilon_ch_hop:
            return random.choice(list(valid_next_hops_q_values.keys()))
        else:
            return max(valid_next_hops_q_values, key=lambda k: valid_next_hops_q_values[k])


    def _execute_energy_consumption_for_round(self):
        """
        [最终修正版] 在每轮结束时，统一计算并消耗所有节点的能量。
        """
        energy_cfg = self.config.get('energy', {})
        idle_listening_cost = float(energy_cfg.get('idle_listening_per_round', 1e-6))
        sensing_cost = float(energy_cfg.get('sensing_per_round', 5e-7))
        packet_size = self.config.get("simulation", {}).get("packet_size", 4000)

        # 1. 处理普通节点到CH的数据传递和能耗
        for node in self.nodes:
            if node["status"] == "active" and node["role"] == "normal" and node.get("cluster_id", -1) >= 0:
                ch_id = node["cluster_id"]
                # 确保CH是存在的且活跃的
                if 0 <= ch_id < len(self.nodes) and self.nodes[ch_id]["status"] == "active":
                    
                    # 假设普通节点到CH的传输总是尝试的，并模拟其成功率
                    # 注意：这里的成功率只影响数据是否被收集，不影响能耗（因为能量总是要花的）
                    success_rate = node.get("history_success_with_ch", {}).get(ch_id, 0.9) * 0.98
                    is_successful = (random.random() < success_rate)

                    # a. 计算并暂存发送能耗
                    dist = self.calculate_distance(node["id"], ch_id)
                    tx_energy = self.calculate_transmission_energy(dist, packet_size)
                    node["pending_tx_energy"] += tx_energy
                    node["tx_count"] += 1
                    
                    if is_successful:
                        # b. 只有传输成功，CH才消耗接收能量，并标记数据被收集
                        rx_energy = self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)
                        self.nodes[ch_id]["pending_rx_energy"] += rx_energy
                        self.nodes[ch_id]["rx_count"] += 1
                        
                        # 标记数据已被CH收集
                        node["has_data_to_send"] = False

        # 2. 处理直连BS节点的发送行为
        for node in self.nodes:
            if node["status"] == "active" and node.get("can_connect_bs_directly", False):
                # a. 计算并暂存发送能耗
                dist_to_bs = self.calculate_distance_to_bs(node["id"])
                tx_energy = self.calculate_transmission_energy(dist_to_bs, packet_size)
                node["pending_tx_energy"] += tx_energy
                node["tx_count"] += 1
                
                # b. 标记数据已被处理
                node["has_data_to_send"] = False
                
                # c. 统计数据包送达 (假设直连总是成功)
                self.sim_packets_delivered_bs_this_round += 1
                self.sim_packets_delivered_bs_total += 1
                self.sim_total_delay_this_round += 1
                self.sim_num_packets_for_delay_this_round += 1

        # 3. 汇总并扣除所有节点的总能耗
        for node in self.nodes:
            if node["status"] == "active":
                total_cost_this_round = (
                    node.get("pending_tx_energy", 0.0) +
                    node.get("pending_rx_energy", 0.0) +
                    node.get("pending_aggregation_energy", 0.0) +
                    idle_listening_cost +
                    sensing_cost
                )
                if total_cost_this_round > 0:
                    self.consume_node_energy(node["id"], total_cost_this_round)

    def _update_and_log_performance_metrics(self):
        """阶段 5: 更新并记录本轮的性能指标。"""
        # (这个函数主要用于整理日志记录代码，使其不散落在 step 函数各处)
        
        # 统计当前轮次的最终状态
        alive_nodes_list = [n for n in self.nodes if n["status"] == "active"]
        num_alive = len(alive_nodes_list)
        total_energy_now = sum(n["energy"] for n in alive_nodes_list)
        avg_energy_now = total_energy_now / num_alive if num_alive > 0 else 0
        
        active_chs_list = [self.nodes[ch_id] for ch_id in self.confirmed_cluster_heads_for_epoch if self.nodes[ch_id]["status"] == "active"]
        num_ch_now = len(active_chs_list)
        avg_ch_energy_now = sum(n["energy"] for n in active_chs_list) / num_ch_now if num_ch_now > 0 else 0
        
        members_counts = [len([m for m in alive_nodes_list if m.get("cluster_id") == ch["id"]]) for ch in active_chs_list]
        avg_members_now = sum(members_counts) / num_ch_now if num_ch_now > 0 else 0
        ch_load_variance_now = np.var(members_counts) if members_counts else 0

        avg_delay_this_round = (self.sim_total_delay_this_round / self.sim_num_packets_for_delay_this_round) \
                               if self.sim_num_packets_for_delay_this_round > 0 else 0

        # 写入性能日志
        with open(PROJECT_ROOT / "reports" / "performance_log.csv", "a", encoding="utf-8") as f_perf:
            f_perf.write(f"{self.current_round},{num_alive},{total_energy_now:.4f},{avg_energy_now:.4f},"
                         f"{num_ch_now},{avg_ch_energy_now:.4f},{avg_members_now:.2f},{ch_load_variance_now:.2f},"
                         f"{self.sim_packets_generated_this_round},{self.sim_packets_delivered_bs_this_round},{avg_delay_this_round:.4f}\n")
        
        # === 调整下一轮的p_opt (如果仍然需要DEEC的p_opt作为某种参考或目标CH比例) ===
        if (self.current_round + 1) % self.epoch_length == 0 or \
           self.current_round == self.config.get('simulation',{}).get('total_rounds',10)-1:
            
            # 统计本epoch最终的孤立节点情况
            isolated_normal_nodes_at_epoch_end = 0
            for node_data_val in self.nodes:
                if node_data_val["status"] == "active" and \
                   node_data_val["role"] == "normal" and \
                   node_data_val["cluster_id"] == -1 and \
                   not node_data_val.get("can_connect_bs_directly", False):
                    isolated_normal_nodes_at_epoch_end += 1
            
            logger.info(f"Epoch {self.current_round // self.epoch_length} 结束: "
                        f"孤立普通节点数 = {isolated_normal_nodes_at_epoch_end}, "
                        f"CH数 = {len(self.confirmed_cluster_heads_for_epoch)}")
            
            self.adjust_p_opt_for_next_round(isolated_normal_nodes_at_epoch_end)

    def _remedy_isolated_nodes(self):
        logger.info("开始对孤立普通节点进行补救...")
        isolated_nodes = [n for n in self.nodes if n["status"] == "active" and n["role"] == "normal" and n["cluster_id"] == -1]
        if not isolated_nodes: return
        num_alive_non_direct_bs_nodes = len([
                n for n in self.nodes
                if n["status"] == "active" and not n.get("can_connect_bs_directly", False) and n["role"] == "normal"
            ])
        num_active_chs_for_capacity = len(self.confirmed_cluster_heads_for_epoch)
        
        if num_active_chs_for_capacity > 0:
            avg_members_per_ch_ideal = num_alive_non_direct_bs_nodes / num_active_chs_for_capacity
            # 使用 factor 计算最大成员数
            max_members_for_ch_calculated = max(1, int(avg_members_per_ch_ideal * self.ch_max_members_factor))
            # 如果使用绝对值，则是:
            # max_members_for_ch_calculated = self.ch_max_absolute_members
            logger.debug(f"  CH容量限制：理想平均成员数={avg_members_per_ch_ideal:.2f}, 计算得到的最大成员数上限={max_members_for_ch_calculated}")
        else:
            # 没有活跃CH，无法进行容量限制，或设置一个默认上限（如果普通节点依然选了某些ID）
            max_members_for_ch_calculated = 10 # 任意默认值，理论上不应发生
        # 找到所有未满员的活跃CH
        available_chs = []
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            # 计算当前成员数
            current_members = len([m for m in self.nodes if m.get("cluster_id") == ch_id])
            if current_members < max_members_for_ch_calculated:
                available_chs.append(ch_id)

        if not available_chs:
            logger.warning("没有未满员的CH可供孤立节点加入。")
            return

        # 让孤立节点尝试加入最近的、未满员的CH
        num_remedied = 0
        for node in isolated_nodes:
            min_dist = float('inf')
            best_ch = -1
            for ch_id in available_chs:
                dist = self.calculate_distance(node["id"], ch_id)
                if dist < node["base_communication_range"] and dist < min_dist:
                    min_dist = dist
                    best_ch = ch_id
            
            if best_ch != -1:
                node["cluster_id"] = best_ch
                num_remedied += 1
        
        logger.info(f"孤立节点补救完成，{num_remedied} / {len(isolated_nodes)} 个节点成功加入新簇。")

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

    def _penalize_failed_routing_action(self, ch_id, failed_action_id):
            """对失败的路由动作施加Q值惩罚。"""
            if ch_id == -1 or not (0 <= ch_id < len(self.nodes)):
                logger.error(f"_penalize_failed_routing_action: 尝试为一个无效的ID({ch_id})施加惩罚。")
                return
            # (这是你原来“惩罚悬停”的逻辑)
            penalty = self.config.get('rewards', {}).get('ch_select_next_hop', {}).get('routing_failure_penalty', -50.0)
            q_table = self.nodes[ch_id].get("q_table_select_next_hop", {})
            old_q = q_table.get(failed_action_id, 0.0)
            alpha = self.nodes[ch_id].get("alpha_select_next_hop", 0.1)
            new_q = old_q + alpha * (penalty - old_q)
            q_table[failed_action_id] = new_q
            self.nodes[ch_id]["q_table_select_next_hop"] = q_table
            logger.warning(f"CH {ch_id}: 因路由失败，对选择下一跳 {failed_action_id} 的动作施加了惩罚。Q值从 {old_q:.2f} 更新为 {new_q:.2f}。")
    

    def _stage_routing_energy_costs(self, ch_id, next_hop_id, distance, packet):
        """
        [最终修正版] 暂存一次路由传输的TX/RX能耗，考虑可变包大小。
        聚合能耗已在包生成时计算。
        """
        if not packet:
            logger.warning(f"在暂存能耗时，CH {ch_id} 传入的packet为空，跳过。")
            return
            
        # 1. 根据包内含的原始数据量，计算有效的传输包大小
        base_packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        num_raw_packets = packet.get("num_raw_packets", 1)
        fusion_efficiency_factor = self.config.get('energy', {}).get('fusion_efficiency_factor', 0.1)
        
        if num_raw_packets > 1:
            effective_packet_size = base_packet_size * (1 + (num_raw_packets - 1) * fusion_efficiency_factor)
        else:
            effective_packet_size = base_packet_size
            
        logger.debug(f"CH {ch_id} 发送融合包，包含 {num_raw_packets} 个原始包，有效大小: {effective_packet_size:.0f} bits")

        # 2. 计算并暂存发送方 (CH) 的TX能耗
        tx_energy = self.calculate_transmission_energy(distance, effective_packet_size, is_tx_operation=True)
        self.nodes[ch_id]["pending_tx_energy"] += tx_energy
        self.nodes[ch_id]["tx_count"] += 1

        # 3. 计算并暂存接收方的RX能耗 (如果接收方不是BS)
        if next_hop_id != self.BS_ID and 0 <= next_hop_id < len(self.nodes) and self.nodes[next_hop_id]["status"] == "active":
            rx_energy = self.calculate_transmission_energy(0, effective_packet_size, is_tx_operation=False)
            self.nodes[next_hop_id]["pending_rx_energy"] += rx_energy
            self.nodes[next_hop_id]["rx_count"] += 1

    def _get_transmission_success_rate(self, ch_id, next_hop_id, next_hop_type, distance):
        """获取到下一跳的传输成功率。"""
        # (这是你之前潜在问题2的解决方案)
        if next_hop_type == "BS" or next_hop_type == "DIRECT_BS_NODE":
            # 对BS或直连节点使用更可靠的成功率模型
            if distance < 100: return 0.99
            if distance < 200: return 0.95
            return 0.90
        else:
            # 对其他CH使用历史成功率
            return self.nodes[ch_id].get("history_success_with_nh", {}).get(next_hop_id, 0.9)        
    
    def _find_candidate_next_hops(self, ch_id):
        """为指定的CH寻找所有可能的下一跳候选。"""
        ch_node_data = self.nodes[ch_id]
        bs_id_for_routing = -1
        DIRECT_BS_NODE_TYPE_STR = "DIRECT_BS_NODE"
        
        # 动态通信范围调整
        ch_base_comm_range = ch_node_data["base_communication_range"]
        ch_max_comm_range = self.config.get('deec', {}).get('ch_communication_range_max', ch_base_comm_range * 1.5)
        num_range_steps = self.config.get('deec', {}).get('ch_communication_range_steps', 3)
        
        range_values = [ch_base_comm_range]
        if num_range_steps > 1 and ch_max_comm_range > ch_base_comm_range:
            increment = (ch_max_comm_range - ch_base_comm_range) / (num_range_steps - 1)
            for i_step in range(1, num_range_steps):
                range_values.append(min(ch_base_comm_range + i_step * increment, ch_max_comm_range))
        range_values = sorted(list(set(range_values)))

        candidate_next_hops_info_all_ranges = []
        for comm_range_attempt in range_values:
            # a. 其他活跃CH
            for other_ch_id in self.confirmed_cluster_heads_for_epoch:
                if other_ch_id != ch_id and self.nodes[other_ch_id]["status"] == "active":
                    dist = self.calculate_distance(ch_id, other_ch_id)
                    if dist <= comm_range_attempt and dist <= self.nodes[other_ch_id]["base_communication_range"]:
                        candidate_next_hops_info_all_ranges.append((other_ch_id, "CH", dist))
            # b. 基站BS
            dist_to_bs = self.calculate_distance_to_bs(ch_id)
            if dist_to_bs <= comm_range_attempt:
                candidate_next_hops_info_all_ranges.append((bs_id_for_routing, "BS", dist_to_bs))
            # c. 直连BS节点
            for other_node in self.nodes:
                if other_node["status"] == "active" and other_node.get("can_connect_bs_directly", False) and other_node["id"] != ch_id:
                    dist = self.calculate_distance(ch_id, other_node["id"])
                    if dist <= comm_range_attempt:
                        # 使用 self.DIRECT_BS_NODE_TYPE_STR 作为类型标记
                        candidate_next_hops_info_all_ranges.append((other_node["id"], self.DIRECT_BS_NODE_TYPE_STR, dist))
        
        # 去重，保留距离最近的
        final_candidates = {}
        for nh_id, nh_type, dist in candidate_next_hops_info_all_ranges:
            if nh_id not in final_candidates or dist < final_candidates[nh_id][2]:
                final_candidates[nh_id] = (nh_id, nh_type, dist)
        
        return list(final_candidates.values())
    
    def _calculate_routing_reward_and_next_state(self, ch_id, chosen_nh_id, chosen_nh_type, 
                                                 dist_to_nh, success_flag, all_candidates_info):
        """
        计算路由奖励和Q-learning所需的下一状态信息。
        这是一个辅助函数，封装了所有与奖励计算相关的复杂逻辑。

        Args:
            ch_id (int): 做决策的CH的ID。
            chosen_nh_id (int): 被选中的下一跳的ID。
            chosen_nh_type (str): 被选中的下一跳的类型 ("CH", "BS", "DIRECT_BS_NODE")。
            dist_to_nh (float): 到下一跳的距离。
            success_flag (bool): 本次传输是否成功。
            all_candidates_info (list): 当前CH本轮所有可达的候选下一跳信息。

        Returns:
            tuple: (reward, next_state_hops_info, is_terminal)
        """
        ch_node_data = self.nodes[ch_id]
        bs_id_for_routing = -1
        DIRECT_BS_NODE_TYPE_STR = "DIRECT_BS_NODE"

        # --- 1. 收集模糊逻辑的输入 ---
        
        # a. 获取下一跳的属性
        next_hop_node_obj = None
        is_next_hop_bs_like = (chosen_nh_type == "BS" or chosen_nh_type == DIRECT_BS_NODE_TYPE_STR)
        
        dc_bs_of_nh = 0.0
        e_of_nh_norm = 1.0  # 对BS假设能量无限
        load_actual_of_nh = 0
        
        if not is_next_hop_bs_like:
            if 0 <= chosen_nh_id < len(self.nodes):
                next_hop_node_obj = self.nodes[chosen_nh_id]
                dc_bs_of_nh = self.calculate_distance_to_bs(chosen_nh_id)
                e_of_nh_norm = (next_hop_node_obj["energy"] / next_hop_node_obj["initial_energy"]) if next_hop_node_obj["initial_energy"] > 0 else 0
                load_actual_of_nh = len([m for m in self.nodes if m.get("cluster_id") == chosen_nh_id and m.get("status") == "active"])
        else: # 如果是直连BS节点
            if chosen_nh_type == DIRECT_BS_NODE_TYPE_STR and 0 <= chosen_nh_id < len(self.nodes):
                 next_hop_node_obj = self.nodes[chosen_nh_id]
                 dc_bs_of_nh = self.calculate_distance_to_bs(chosen_nh_id)
                 e_of_nh_norm = (next_hop_node_obj["energy"] / next_hop_node_obj["initial_energy"]) if next_hop_node_obj["initial_energy"] > 0 else 0
                 load_actual_of_nh = 1 # 负载可以看作是1（它自己）

        # b. 获取与本次传输相关的属性
        r_success_with_nh_norm = self._get_transmission_success_rate(ch_id, chosen_nh_id, chosen_nh_type, dist_to_nh)
        packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        actual_e_tx_to_nh = self.calculate_transmission_energy(dist_to_nh, packet_size, is_tx_operation=True)
        max_send_e_ref = self.calculate_transmission_energy(ch_node_data["base_communication_range"], packet_size, True)
        e_ctx_cost_norm = actual_e_tx_to_nh / max_send_e_ref if max_send_e_ref > 0 else 0
        e_ctx_cost_norm = np.clip(e_ctx_cost_norm, 0, 1)
        
        # c. 获取网络平均负载作为参考
        avg_load_for_nh_ref = self.get_alive_nodes() / len(self.confirmed_cluster_heads_for_epoch) if self.confirmed_cluster_heads_for_epoch else 10

        # --- 2. 调用模糊逻辑系统计算权重 ---
        fuzzy_weights = self.ch_path_fuzzy_logic.compute_weights(
            current_dc_bs_neighbor=dc_bs_of_nh,
            current_e_c_neighbor=e_of_nh_norm,
            current_load_c_actual=load_actual_of_nh,
            current_r_c_success=r_success_with_nh_norm,
            current_e_ctx_cost_normalized=e_ctx_cost_norm,
            avg_load_for_neighbor_ch=avg_load_for_nh_ref
        )

        # --- 3. 计算奖励 (Reward) ---
        # a. 计算路径进展量
        dist_ch_to_bs = self.calculate_distance_to_bs(ch_id)
        data_advanced_amount = dist_ch_to_bs - dc_bs_of_nh
        
        # b. 计算下一跳的负载比率
        next_hop_load_ratio = load_actual_of_nh / avg_load_for_nh_ref if avg_load_for_nh_ref > 0 else 0

        # c. 调用奖励计算函数
        reward = self.calculate_reward_for_selecting_next_hop(
            ch_node_id=ch_id,
            chosen_next_hop_id=chosen_nh_id,
            fuzzy_weights=fuzzy_weights,
            transmission_successful=success_flag,
            actual_energy_spent_tx=actual_e_tx_to_nh,
            data_advanced_amount=data_advanced_amount,
            is_next_hop_bs=(chosen_nh_type == "BS"),
            next_hop_energy_normalized=e_of_nh_norm if not is_next_hop_bs_like else None,
            next_hop_load_ratio=next_hop_load_ratio if chosen_nh_type == "CH" else None,
            is_next_hop_direct_bs_node=(chosen_nh_type == DIRECT_BS_NODE_TYPE_STR)
        )
        
        # d. (可选) 为成功送达逻辑终点增加额外奖励
        if success_flag and is_next_hop_bs_like:
            reach_bonus = self.config.get('rewards', {}).get('ch_select_next_hop', {}).get('reach_bs_via_direct_node_bonus', 90)
            reward += reach_bonus

        # --- 4. 确定下一状态信息 (Next State Info) ---
        is_terminal = success_flag and is_next_hop_bs_like
        
        next_state_hops_info = []
        if not is_terminal and self.nodes[ch_id]["status"] == "active":
            # 假设下一状态的可选动作与当前状态相同
            next_state_hops_info = all_candidates_info

        return reward, next_state_hops_info, is_terminal
    # ... (其他辅助函数如 calculate_distance, calculate_distance_to_bs, get_alive_nodes, kill_node, 
    #      calculate_transmission_energy, consume_node_energy, _build_spatial_index, get_node_neighbors)
    # 你需要确保这些函数都存在且功能正确

    # in env.py, class WSNEnv

    def _execute_successful_transfer(self, sender_id, receiver_id, packet, receiver_type, source_buffer_type):
        """
        [双缓冲区版] 执行一次成功的数据包传递，更新所有相关状态。
        """
        logger.debug(f"执行传输: {sender_id} -> {receiver_id} (包源: {packet['source_ch']})")
        
        # 1. 暂存能量消耗
        dist = self.calculate_distance(sender_id, receiver_id) if receiver_id != -1 else self.calculate_distance_to_bs(sender_id)
        self._stage_routing_energy_costs(sender_id, receiver_id, dist, packet) # 传递packet以计算可变大小

        # 2. 从发送方的对应缓存队列中移除该数据包
        packet_uid = packet.get("uid")
        if sender_id in self.packets_in_transit:
            if source_buffer_type in self.packets_in_transit[sender_id]:
                self.packets_in_transit[sender_id][source_buffer_type] = [
                    p for p in self.packets_in_transit[sender_id][source_buffer_type] if p.get("uid") != packet_uid
                ]

        # 3. 处理接收方
        is_logically_at_bs = (receiver_type == self.BS_TYPE_STR or receiver_type == self.DIRECT_BS_NODE_TYPE_STR)
        
        if is_logically_at_bs:
            # 数据包送达，统计信息
            num_delivered = packet.get("num_raw_packets", 1)
            self.sim_packets_delivered_bs_this_round += num_delivered
            self.sim_packets_delivered_bs_total += num_delivered
            delay = self.current_round - packet["gen_round"] + 1
            self.sim_total_delay_this_round += delay * num_delivered
            self.sim_num_packets_for_delay_this_round += num_delivered
            #logger.info(f"数据包 (UID: {packet_uid}) 成功送达逻辑终点 {receiver_id}，包含 {num_delivered} 个原始包。")
        else:  # 接收方是普通CH
            # 数据包总是进入接收方的“转发缓冲区”
            packet["path"].append(receiver_id)
            if receiver_id not in self.packets_in_transit:
                self.packets_in_transit[receiver_id] = {"forwarding_buffer": [], "generation_buffer": []}
            self.packets_in_transit[receiver_id]["forwarding_buffer"].append(packet)

    def _calculate_current_average_energy(self):
        """计算当前网络中所有存活节点的平均能量"""
        total_energy = 0
        alive_nodes_count = 0
        for node in self.nodes:
            if node["status"] == "active":
                total_energy += node["energy"]
                alive_nodes_count += 1
        return total_energy / alive_nodes_count if alive_nodes_count > 0 else 0
    
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
        isolated_count = 0
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
        try:
            self._position_tree = KDTree(active_node_positions)
        # logger.debug(f"空间索引构建/更新完毕，包含 {len(active_node_positions)} 个活跃节点。")
        except Exception as e:
            logger.error(f"构建KDTree时发生错误: {e}", exc_info=True)
            self._position_tree = None
        
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

        max_dist_query = max_distance if max_distance is not None else node["current_communication_range"] # 使用 current_communication_range
        
        indices_in_kdtree = [] # 初始化
        try:
            # 明确指定参数 r
            indices_in_kdtree = self._position_tree.query_ball_point(x=node["position"], r=max_dist_query,p=2.0)
        except Exception as e_kdtree_neighbors:
            logger.error(f"KDTree query_ball_point 错误 in get_node_neighbors for node {node_id} (pos {node['position']}, r {max_dist_query}): {e_kdtree_neighbors}", exc_info=True)
        
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
