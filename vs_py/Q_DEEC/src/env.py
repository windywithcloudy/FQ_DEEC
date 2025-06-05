import yaml
from pathlib import Path
import numpy as np
import math
import random
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.log import logger # 从 utils 包导入 logger
from utils.fuzzy import NormalNodeCHSelectionFuzzySystem, RewardWeightsFuzzySystemForCHCompetition,CHToBSPathSelectionFuzzySystem
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
        self.confirmed_cluster_heads_for_epoch = []      # 在本Epoch内固定的CH ID列表
        self.confirmed_cluster_heads_previous_epoch = [] # 上一个Epoch的CH，用于CH竞争状态计算

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
                "current_data_path":[],
                "source_ch_of_current_packet" : -1,
                "gen_round_of_current_packet" :-1,
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

    def finalize_ch_roles(self, ch_declarations_this_epoch):
        logger.info(f"CH最终确定阶段：从 {len(ch_declarations_this_epoch)} 个宣告者中确定本Epoch的CH...")
        if not ch_declarations_this_epoch:
            self.confirmed_cluster_heads_for_epoch = []
            logger.info("没有节点宣告成为CH，本Epoch无CH。")
            for node in self.nodes: # 更新所有非直连BS节点的time_since_last_ch
                if node["status"] == "active" and not node.get("can_connect_bs_directly", False):
                    node["time_since_last_ch"] += 1
            return

        num_alive_eligible = len([n for n in self.nodes if n["status"] == "active" and not n.get("can_connect_bs_directly", False)])
        target_ch_count_ideal = max(1, int(num_alive_eligible * self.p_opt_current)) # 使用当前的p_opt

        # --- (与之前 deec_candidate_election 中类似的筛选逻辑) ---
        # 1. 初步能量筛选和数量控制
        ch_declarations_this_epoch.sort(key=lambda id_val: self.nodes[id_val]["energy"], reverse=True)
        preliminary_candidates = ch_declarations_this_epoch[:max(target_ch_count_ideal, int(target_ch_count_ideal * 1.5))]

        # 2. 基于距离的冗余CH移除
        final_refined_candidates = []
        if preliminary_candidates:
            comm_range_avg = self.config.get('network', {}).get('communication_range', 100.0)
            min_ch_dist_factor = self.config.get('deec', {}).get('ch_finalize_too_close_factor', 0.7) # 使用 finalize 的因子
            d_min_ch_dist = min_ch_dist_factor * comm_range_avg
            
            for cand_id in preliminary_candidates: # 已经是能量排序的
                node_cand = self.nodes[cand_id]
                if node_cand["status"] != "active": continue
                is_too_close = False
                for final_id in final_refined_candidates:
                    if self.nodes[final_id]["status"] != "active": continue
                    if self.calculate_distance(cand_id, final_id) < d_min_ch_dist:
                        is_too_close = True; break
                if not is_too_close: final_refined_candidates.append(cand_id)
        
        # 3. 数量再平衡 (确保不太少也不太多)
        min_total_chs_target = max(1, int(target_ch_count_ideal * 0.8))
        max_total_chs_target = max(min_total_chs_target, int(target_ch_count_ideal * 1.2))

        if len(final_refined_candidates) < min_total_chs_target and preliminary_candidates:
            # 如果筛选后太少，从能量较高的初步候选中补充（但要小心再次引入扎堆）
            # 简单处理：如果太少，就用初步筛选后、距离去重前的列表，再限制数量
            logger.warning(f"距离筛选后CH ({len(final_refined_candidates)}) 过少，尝试从能量筛选结果补充。")
            temp_recheck = preliminary_candidates[:max_total_chs_target] # 取能量高的
            comm_range_avg = self.config.get('network', {}).get('communication_range', 100.0)
            min_ch_dist_factor = self.config.get('deec', {}).get('ch_finalize_too_close_factor', 0.7) # 使用 finalize 的因子
            d_min_ch_dist = min_ch_dist_factor * comm_range_avg
            # 再次进行简化距离筛选
            final_refined_candidates = []
            for cand_id in temp_recheck:
                is_too_close = False
                for final_id in final_refined_candidates:
                    if self.calculate_distance(cand_id, final_id) < d_min_ch_dist * 0.9 : # 用略小的阈值尝试填充
                        is_too_close = True; break
                if not is_too_close: final_refined_candidates.append(cand_id)
            self.confirmed_cluster_heads_for_epoch = final_refined_candidates[:max_total_chs_target]

        elif len(final_refined_candidates) > max_total_chs_target:
            self.confirmed_cluster_heads_for_epoch = final_refined_candidates[:max_total_chs_target]
        else:
            self.confirmed_cluster_heads_for_epoch = final_refined_candidates
        
        # 如果最终还是没有CH，且有合格节点，强制选一个
        if not self.confirmed_cluster_heads_for_epoch and num_alive_eligible > 0:
            eligible_nodes = [n for n in self.nodes if n["status"] == "active" and not n.get("can_connect_bs_directly", False)]
            if eligible_nodes:
                highest_e_node = max(eligible_nodes, key=lambda x:x["energy"])
                self.confirmed_cluster_heads_for_epoch.append(highest_e_node["id"])
                logger.info(f"最终无CH，强制选择能量最高合格节点 {highest_e_node['id']} 为CH。")


        # --- 更新所有节点的角色和 time_since_last_ch ---
        for node_data in self.nodes:
            if node_data["status"] == "active":
                if node_data["id"] in self.confirmed_cluster_heads_for_epoch:
                    node_data["role"] = "cluster_head"
                    node_data["cluster_id"] = node_data["id"] 
                    node_data["time_since_last_ch"] = 0 
                elif not node_data.get("can_connect_bs_directly", False): 
                    node_data["role"] = "normal"
                    # time_since_last_ch 只在它不是CH时才增加
                    if node_data.get("time_since_last_ch") is None: node_data["time_since_last_ch"] = 0 # 初始化
                    node_data["time_since_last_ch"] += 1 
                    node_data["cluster_id"] = -1 # 普通节点初始未分配，等待Q学习选择
        
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
                                  max(1, self.get_alive_nodes() / (len(self.confirmed_cluster_heads_current_round) if self.confirmed_cluster_heads_current_round else 1) )
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
        self.current_round = current_round_num
        logger.info(f"--- 开始第 {self.current_round} 轮  ---")

        # 0. 更新和准备工作
        self._build_spatial_index()
        self.confirmed_cluster_heads_current_round = []
        self.identify_direct_bs_nodes()
        for node_data_loop in self.nodes: # 使用不同的变量名
            if node_data_loop["status"] == "active":
                 node_data_loop["current_communication_range"] = node_data_loop["base_communication_range"]
                 node_data_loop["chosen_next_hop_id"] = None
                 node_data_loop["current_data_path"] = [node_data_loop["id"]]
                 if node_data_loop["role"] == "normal" and not node_data_loop.get("can_connect_bs_directly", False):
                    min_eps_select = self.config.get('q_learning',{}).get('epsilon_select_ch_min',0.01)
                    decay_select = self.config.get('q_learning',{}).get('epsilon_select_ch_decay_per_round',0.998) # 按轮衰减
                    current_eps_select = node_data_loop.get("epsilon_select_ch", self.config.get('q_learning',{}).get('epsilon_select_ch_initial',0.3))
                    node_data_loop["epsilon_select_ch"] = max(min_eps_select, current_eps_select * decay_select)


        # --- 实例化或更新模糊逻辑系统 ---
        # 如果是全局共享的，并且其输入（如网络平均能量）会变，需要更新
        # 这里假设 reward_weight_adjuster 和 normal_node_ch_selector_fuzzy 在 __init__ 中创建
        # 如果 NormalNodeCHSelectionFuzzySystem 需要动态的 node_sum, cluster_sum，则在此时更新或重新创建
        # self.normal_node_ch_selector_fuzzy.update_network_stats(len(self.nodes), len(self.confirmed_cluster_heads_current_round)) # 假设有此方法
        
        # 获取用于模糊调整奖励权重的输入
        # net_energy_level_for_fuzzy = self._calculate_current_average_energy() / self.E0 if self.E0 > 0 else 0
        # ch_density_for_fuzzy = len(self.confirmed_cluster_heads_previous_round) / self.get_alive_nodes() if self.get_alive_nodes() > 0 else 0
        # ^^^ 这些应该在循环内，基于每个节点的视角或全局状态


        # === Epoch 开始时的特殊处理 ===
        if self.current_round % self.epoch_length == 0:
            logger.info(f"--- ***** 新 Epoch 开始 (轮次 {self.current_round}) ***** ---")
            if self.current_round > 0: # 第0轮没有上一个epoch
                self._update_ch_competition_q_tables_at_epoch_end() # 更新上一个epoch的CH竞争Q表
            # 保存上一epoch的CH（如果需要用于CH竞争状态）
            self.confirmed_cluster_heads_previous_epoch = list(self.confirmed_cluster_heads_for_epoch)
            
            # CH竞争Q学习的epsilon可以在epoch开始时衰减
            self.current_epsilon_compete = max(
                self.epsilon_compete_min, 
                self.current_epsilon_compete * (self.epsilon_compete_decay ** self.epoch_length) # 或者一个固定的epoch衰减
            ) 
            logger.info(f"Epoch开始，CH竞争Epsilon更新为: {self.current_epsilon_compete:.4f}")

            # --- 阶段1: 节点通过Q学习决定是否宣告成为CH (仅在Epoch开始时) ---
            logger.info("Epoch开始：节点通过Q学习竞争成为CH...")
            ch_declarations_this_epoch = [] 
            self.competition_log_for_current_epoch = {} # 清空并开始记录本epoch的

            nodes_eligible_for_competition = [
                n for n in self.nodes
                if n["status"] == "active" and not n.get("can_connect_bs_directly", False)
            ]

            for node_data_compete in nodes_eligible_for_competition: # 使用不同变量名
                node_id_compete = node_data_compete["id"]
                # 状态S是基于上一个epoch结束时的CH分布 (self.confirmed_cluster_heads_previous_epoch)
                state_tuple_compete = self.get_discrete_state_tuple_for_competition(node_id_compete)
                if state_tuple_compete is None: continue

                action_to_take = 0 
                if random.random() < self.current_epsilon_compete:
                    action_to_take = random.choice([0, 1])
                else:
                    q0 = self.get_q_value_compete_ch(node_id_compete, state_tuple_compete, 0)
                    q1 = self.get_q_value_compete_ch(node_id_compete, state_tuple_compete, 1)
                    action_to_take = 1 if q1 > q0 else (0 if q0 > q1 else random.choice([0,1]))
                
                # 记录的是做出决策时的状态 S 和动作 A
                self.competition_log_for_current_epoch[node_id_compete] = {
                    "state_tuple": state_tuple_compete, "action": action_to_take
                }
                if action_to_take == 1:
                    ch_declarations_this_epoch.append(node_id_compete)
                raw_state_values_for_log = self.get_node_state_for_ch_competition(node_id_compete) # 获取原始状态
                if raw_state_values_for_log: # 确保获取成功
                    # 将原始状态（绝对值）也存起来用于日志
                    self.competition_log_for_current_epoch[node_id_compete] = {
                        "state_tuple": state_tuple_compete, 
                        "action": action_to_take,
                        "raw_state_for_log": { # 存储日志需要的原始值
                            "e_self_abs": self.nodes[node_id_compete]["energy"], # 记录当前绝对能量
                            "t_last_ch": raw_state_values_for_log["t_last_ch"],
                            "n_neighbor": raw_state_values_for_log["n_neighbor"],
                            "d_bs_abs": raw_state_values_for_log["d_bs"] # 记录原始距离
                            # 可以根据需要添加其他原始值
                        }
                    }
                else: # raw_state_values_for_log is None (节点可能状态不对)
                    self.competition_log_for_current_epoch[node_id_compete] = {
                        "state_tuple": state_tuple_compete, 
                        "action": action_to_take,
                        "raw_state_for_log": {} # 存一个空字典避免后续get失败
                    }
            
            logger.info(f"Epoch CH竞争：{len(ch_declarations_this_epoch)} 个节点宣告想成为CH: {ch_declarations_this_epoch}")

            # --- 阶段1.5: 从宣告者中最终确定本Epoch的活跃CH ---
            # finalize_ch_roles 会填充 self.confirmed_cluster_heads_for_epoch 并更新节点角色和time_since_last_ch
            self.finalize_ch_roles(ch_declarations_this_epoch) 
            
            # --- 为本Epoch的CH竞争决策计算奖励并更新Q表 ---
            # 这个奖励是基于整个Epoch的表现，还是基于宣告时的预期？
            # 简单起见，我们可以在Epoch结束时，或者在CH角色确定后，根据一些即时指标（如CH是否真的被选上，位置是否好）更新。
            # 更准确的是在Epoch结束时，根据CH的实际表现（服务了多少节点，数据传输成功率等）来更新。
            # 为了简化，我们先在CH角色确定后，根据一些简单指标（如是否被选上，位置）给一个即时奖励。
            # actual_members_joined 在这个时间点还不知道，因为普通节点还没选。
            logger.info("Epoch开始：更新CH竞争Q表 (基于宣告和最终确认)...")
            for node_id, log_info in self.competition_log_for_current_epoch.items(): # 使用正确的变量名
                node = self.nodes[node_id]
                if node["status"] == "dead": 
                    reward_compete = self.calculate_reward_for_ch_competition(
                        node_id, log_info["action"], 0, True
                    )
                    self.update_q_value_compete_ch(node_id, log_info["state_tuple"], log_info["action"], reward_compete, None)
                    continue

                actual_members = 0
                if node["role"] == "cluster_head":
                    actual_members = len([m_node for m_node in self.nodes if m_node.get("cluster_id") == node_id and m_node["status"]=='active'])
                is_uncovered = (node["role"] == "normal" and node["cluster_id"] == -1 and not node.get("can_connect_bs_directly", False))
                
                # ... (获取模糊权重 current_fuzzy_reward_weights) ...
                # (确保 self.reward_weights_adjuster 已实例化)
                if not hasattr(self, 'reward_weights_adjuster'):
                    from utils.fuzzy import RewardWeightsFuzzySystemForCHCompetition
                    self.reward_weights_adjuster = RewardWeightsFuzzySystemForCHCompetition(self.config)
                # ... (计算模糊输入) ...
                net_energy_level_norm = self._calculate_current_average_energy() / self.E0 if self.E0 > 0 else 0
                node_self_energy_norm = node["energy"] / node["initial_energy"] if node["initial_energy"] > 0 else 0
                current_ch_density_global_val = len(self.confirmed_cluster_heads_for_epoch) / self.get_alive_nodes() if self.get_alive_nodes() > 0 else 0 # 使用 for_epoch
                current_ch_to_bs_dis_norm = (self.calculate_distance_to_bs(node_id) / self.network_diagonal) if self.network_diagonal > 0 else 0

                current_fuzzy_reward_weights = self.reward_weights_adjuster.compute_reward_weights(
                    net_energy_level_norm, node_self_energy_norm, 
                    current_ch_density_global_val, current_ch_to_bs_dis_norm
                )


                reward_compete = self.calculate_reward_for_ch_competition(
                    node_id, log_info["action"], actual_members, is_uncovered, 
                    # fuzzy_reward_weights=current_fuzzy_reward_weights # 如果你的函数签名需要它
                ) # 注意：我之前的 calculate_reward_for_ch_competition 已经改为内部计算模糊权重

                # ... (获取 next_state_tuple_for_update) ...
                # ... (如之前讨论的简化或更复杂的 S' 估计) ...
                next_state_tuple_for_update = None # 临时简化

                self.update_q_value_compete_ch(node_id, log_info["state_tuple"], log_info["action"], reward_compete, next_state_tuple_for_update)


        # === 阶段2: 普通节点使用Q学习选择已确定的CH ===
        logger.info("开始阶段2：普通节点Q学习选择簇头...")
        num_nodes_switched_ch_this_round = 0
        num_nodes_initially_assigned_this_round = 0
        num_nodes_assigned_this_round = 0
        
        
        # 获取用于模糊逻辑的平均负载参考 (只计算一次，如果CH列表不变)
        # 或者，如果 NormalNodeCHSelectionFuzzySystem 内部不处理这个，就在这里计算
        avg_load_per_confirmed_ch_ref = self.get_alive_nodes() / len(self.confirmed_cluster_heads_current_round) \
                                        if self.confirmed_cluster_heads_current_round else 10 # 默认10个成员/CH

        # 硬编码 avg_e_send_total_for_normal_node_ref (方案A)
        AVG_E_SEND_REF_NORMAL_NODE = 0.001 # 你可以调整这个值，或者从一个不常用的config位置读取
        logger.debug(f"Using hardcoded AVG_E_SEND_REF_NORMAL_NODE: {AVG_E_SEND_REF_NORMAL_NODE}")


        for node_data in self.nodes:
            if node_data["status"] == "active" and \
               node_data["role"] == "normal" and \
               not node_data.get("can_connect_bs_directly", False): 

                node_id = node_data["id"]
                current_assigned_ch_id = node_data["cluster_id"] # 获取当前连接的CH (可能是-1)
                reachable_chs_info = [] # 存储 (ch_id, distance_to_ch)
                if self.confirmed_cluster_heads_for_epoch: # 使用本epoch固定的CH列表
                    for ch_id_cand in self.confirmed_cluster_heads_for_epoch:
                        if self.nodes[ch_id_cand]["status"] == "active": # 确保CH是活的
                            distance = self.calculate_distance(node_id, ch_id_cand)
                            if distance <= node_data["current_communication_range"] and \
                               distance <= self.nodes[ch_id_cand]["current_communication_range"]:
                                reachable_chs_info.append((ch_id_cand, distance))
                
                if not reachable_chs_info:
                    # logger.debug(f"Node {node_id} 没有可达的活跃CH。如果之前有连接，则断开。")
                    node_data["cluster_id"] = -1 # 如果没有可达的，则变为未分配
                    continue

                best_q_for_new_ch = -float('inf')
                potential_new_ch_id = -1
                
                # 评估所有可达的CH
                q_values_for_reachable_chs = {}
                for ch_id_cand, _ in reachable_chs_info:
                    q_values_for_reachable_chs[ch_id_cand] = self.get_q_value_select_ch(node_id, ch_id_cand)
                    if q_values_for_reachable_chs[ch_id_cand] > best_q_for_new_ch:
                        best_q_for_new_ch = q_values_for_reachable_chs[ch_id_cand]
                        potential_new_ch_id = ch_id_cand
                    elif q_values_for_reachable_chs[ch_id_cand] == best_q_for_new_ch and potential_new_ch_id != -1:
                        if random.random() < 0.5: potential_new_ch_id = ch_id_cand
                    elif potential_new_ch_id == -1 : # 第一个候选
                        potential_new_ch_id = ch_id_cand

                chosen_for_this_round_ch_id = -1
                # epsilon_select_ch 应该在 node_data 中维护和衰减
                current_epsilon_select = node_data.get("epsilon_select_ch", 
                                         self.config.get('q_learning',{}).get('epsilon_select_ch_initial',0.3))

                if random.random() < current_epsilon_select: # 探索
                    chosen_for_this_round_ch_id = random.choice([ch_info[0] for ch_info in reachable_chs_info])
                    # logger.debug(f"Node {node_id} (Select_CH) 探索选择了 CH {chosen_for_this_round_ch_id}")
                else: # 利用
                    if current_assigned_ch_id == -1 or current_assigned_ch_id not in q_values_for_reachable_chs: 
                        # 情况1：当前未连接CH，或当前CH已不可达/死亡 -> 选择Q值最高的potential_new_ch_id
                        chosen_for_this_round_ch_id = potential_new_ch_id
                        # logger.debug(f"Node {node_id} (Select_CH) 未连接或当前CH无效，利用选择了新CH {chosen_for_this_round_ch_id} (Q={best_q_for_new_ch:.2f})")
                    else:
                        # 情况2：当前已连接CH，比较新CH是否显著更优
                        q_current_ch = q_values_for_reachable_chs.get(current_assigned_ch_id, -float('inf'))
                        if potential_new_ch_id != -1 and best_q_for_new_ch > q_current_ch + self.ch_switching_hysteresis:
                            chosen_for_this_round_ch_id = potential_new_ch_id # 切换到更优的CH
                            # logger.debug(f"Node {node_id} (Select_CH) 切换到更优CH {chosen_for_this_round_ch_id} (Q_new={best_q_for_new_ch:.2f} > Q_curr={q_current_ch:.2f} + Hys={self.ch_switching_hysteresis})")
                        else:
                            chosen_for_this_round_ch_id = current_assigned_ch_id # 保持当前CH
                            # logger.debug(f"Node {node_id} (Select_CH) 保持当前CH {chosen_for_this_round_ch_id} (Q_new={best_q_for_new_ch:.2f} 未显著优于 Q_curr={q_current_ch:.2f})")
                    if chosen_for_this_round_ch_id == -1 and reachable_chs_info: # 如果上面逻辑都没选出来（不太可能），随机选一个
                        chosen_for_this_round_ch_id = random.choice([ch_info[0] for ch_info in reachable_chs_info])
                
                if chosen_for_this_round_ch_id != -1:
                    if current_assigned_ch_id == -1 and chosen_for_this_round_ch_id != -1:
                        num_nodes_initially_assigned_this_round +=1
                    elif current_assigned_ch_id != -1 and current_assigned_ch_id != chosen_for_this_round_ch_id:
                        num_nodes_switched_ch_this_round +=1

                    node_data["cluster_id"] = chosen_for_this_round_ch_id
                    num_nodes_assigned_this_round += 1
                    
                    ch_node_chosen = self.nodes[chosen_for_this_round_ch_id]
                    dc_base_chosen_ch = self.calculate_distance_to_bs(chosen_for_this_round_ch_id)
                    e_cluster_chosen_ch_norm = ch_node_chosen["energy"] / ch_node_chosen["initial_energy"] if ch_node_chosen["initial_energy"] > 0 else 0
                    
                    # 负载比率：CH当前成员数 / 平均每个CH应服务成员数
                    # 注意：这里的成员数应该是CH被当前node_id选择 *之前* 的。
                    # 为了简化，我们用它当前的成员数（可能包含了本轮其他已选它的节点）
                    # 或者，更简单的是，在普通节点选择CH时，假设CH的初始负载为0或很小，
                    # 因为这是CH刚被选出来，还没有成员。
                    # 我们用一个简化的“负载潜力”概念，或者直接用一个较小的值。
                    # 实际负载会在所有普通节点选择完毕后才能准确统计。
                    # 为了让模糊逻辑的 p_cluster_ratio 有意义，我们需要一个对CH负载的估计。
                    # 方案：先让所有普通节点“意向选择”，然后统计每个CH的意向成员数作为负载。
                    # 简化：暂时使用一个固定的低负载比率或基于CH自身属性的估计。
                    # 这里，我们用一个简单的负载比率，假设avg_load_per_confirmed_ch_ref是理想的每个CH的成员数
                    # 而ch_node_chosen当前的成员数是 load_actual_chosen_ch
                    # 实际负载可能是动态变化的，这里用一个估计值
                    load_actual_chosen_ch = len([m for m in self.nodes if m.get("cluster_id") == chosen_for_this_round_ch_id and m.get("status") == 'active']) 
                    p_cluster_ratio_for_fuzzy = load_actual_chosen_ch / avg_load_per_confirmed_ch_ref if avg_load_per_confirmed_ch_ref > 0 else 0
                    
                    r_success_with_chosen_ch_norm = node_data.get("history_success_with_ch", {}).get(chosen_for_this_round_ch_id, 0.9) 
                    
                    dist_to_chosen_ch = self.calculate_distance(node_id, chosen_for_this_round_ch_id) # 重新获取，因为reachable_chs中存了
                    packet_size = self.config.get("simulation",{}).get("packet_size",4000)
                    actual_e_send_to_chosen_ch = self.calculate_transmission_energy(dist_to_chosen_ch, packet_size, is_tx_operation=True)
                    
                    e_send_total_ratio_for_fuzzy = actual_e_send_to_chosen_ch / AVG_E_SEND_REF_NORMAL_NODE if AVG_E_SEND_REF_NORMAL_NODE > 0 else 0
                    
                    fuzzy_weights_for_chosen_ch = self.normal_node_fuzzy_logic.compute_weights(
                        current_dc_base=dc_base_chosen_ch,
                        current_e_cluster_normalized=e_cluster_chosen_ch_norm,
                        current_p_cluster_ratio_val=p_cluster_ratio_for_fuzzy, 
                        current_r_success_normalized=r_success_with_chosen_ch_norm,
                        current_e_send_total_ratio_val=e_send_total_ratio_for_fuzzy 
                    )

                    # 模拟传输并获取奖励
                    transmission_success = True if random.random() < r_success_with_chosen_ch_norm * 0.98 else False # 略微降低一点实际成功率

                    reward_for_selection = self.calculate_reward_for_selecting_ch(
                        node_id, chosen_for_this_round_ch_id, fuzzy_weights_for_chosen_ch,
                        transmission_successful=transmission_success,
                        actual_energy_spent_tx=actual_e_send_to_chosen_ch
                    )
                    
                    self.update_q_value_select_ch(node_id, chosen_for_this_round_ch_id, reward_for_selection)

                    # --- 实际能量消耗 ---
                    # 普通节点发送
                    if transmission_success : # 只有成功才认为CH收到了
                        if self.consume_node_energy(node_id, actual_e_send_to_chosen_ch):
                            node_data["tx_count"] += 1
                        # CH接收
                        energy_rx_ch = self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)
                        if self.consume_node_energy(chosen_for_this_round_ch_id, energy_rx_ch):
                            self.nodes[chosen_for_this_round_ch_id]["rx_count"] +=1
            
                # 更新节点的epsilon_select_ch
                min_eps_select = self.config.get('q_learning',{}).get('epsilon_select_ch_min',0.01)
                decay_select = self.config.get('q_learning',{}).get('epsilon_select_ch_decay',0.995)
                current_epsilon_select = max(min_eps_select, current_epsilon_select * decay_select)
                node_data["epsilon_select_ch"] = current_epsilon_select # 保存更新后的epsilon

        logger.info(f"Q学习选择CH阶段：{num_nodes_initially_assigned_this_round} 个节点初次分配，{num_nodes_switched_ch_this_round} 个节点切换了CH。")

        # === 阶段2之后: 处理仍然孤立的普通节点 ===
        # ... (可以保留之前的 handle_isolated_nodes 或 assign_nodes_to_clusters 作为备用) ...
        # ... 但现在 assign_nodes_to_clusters 的候选CH应为 self.confirmed_cluster_heads_current_round ...


        # === 阶段3: 活跃CH通过Q学习选择下一跳 (每轮执行) ===
        logger.info("开始阶段3：活跃CH通过Q学习选择下一跳...")
        bs_id_for_routing = -1 # 特殊ID表示基站
        DIRECT_BS_NODE_TYPE_STR = "DIRECT_BS_NODE" # 定义类型字符串

        if self.confirmed_cluster_heads_for_epoch:
            for ch_id in self.confirmed_cluster_heads_for_epoch:
                ch_node_data = self.nodes[ch_id]
                if ch_node_data["status"] != "active" or ch_node_data["role"] != "cluster_head":
                    continue 
                
                # 假设CH至少有自己的数据要发送，或者根据成员数量判断是否有数据
                # has_data = True # 简化：假设总有数据
                # if not has_data: continue
                if not ch_node_data.get("current_data_path") or ch_node_data.get("source_ch_of_current_packet", -1) == -1: # 或者检查 source_ch_of_current_packet == -1
                    ch_node_data["current_data_path"] = [ch_id]
                    ch_node_data["source_ch_of_current_packet"] = ch_id
                    ch_node_data["gen_round_of_current_packet"] = self.current_round
                    logger.debug(f"CH {ch_id} 作为源头/开始新数据流，产生融合包于轮次 {self.current_round}. Path: {ch_node_data['current_data_path']}")

                current_path_history = list(ch_node_data["current_data_path"]) # 使用副本进行决策
                # 确保当前CH的ID在路径的末尾（如果它是被传递过来的，上游应该已经加了）
                if not current_path_history or current_path_history[-1] != ch_id:
                    # 这种情况理论上不应该发生，如果路径传递正确的话
                    # 但作为防御，如果路径为空或末尾不是自己，则重置为源头
                    logger.warning(f"CH {ch_id}: current_data_path ({current_path_history}) 状态异常，重置为源头路径。")
                    ch_node_data["current_data_path"] = [ch_id]
                    ch_node_data["source_ch_of_current_packet"] = ch_id # 源信息也应重置
                    ch_node_data["gen_round_of_current_packet"] = self.current_round
                    current_path_history = list(ch_node_data["current_data_path"])

                # --- 调试日志：打印当前CH信息 ---
                #logger.info(f"[CH_DECISION_TRACE] CH {ch_id} (Energy: {ch_node_data['energy']:.4f}) 开始决策...")

                #logger.debug(f"CH {ch_id} 开始决策，当前数据路径历史: {current_path_history}")
                # --- 动态通信范围调整循环 (来自你之前的方案) ---
                ch_base_comm_range = ch_node_data["base_communication_range"]
                ch_max_comm_range = self.config.get('deec', {}).get('ch_communication_range_max', 200.0)
                num_range_steps = self.config.get('deec', {}).get('ch_communication_range_steps', 3)
                
                range_values = [ch_base_comm_range]
                if num_range_steps > 1 and ch_max_comm_range > ch_base_comm_range:
                    increment = (ch_max_comm_range - ch_base_comm_range) / (num_range_steps - 1)
                    for i_step in range(1, num_range_steps): # 使用不同的循环变量名
                        range_values.append(min(ch_base_comm_range + i_step * increment, ch_max_comm_range))
                range_values = sorted(list(set(range_values)))

                chosen_next_hop_id_for_ch = -100 
                action_taken_ch_hop = -100
                best_q_for_this_ch_overall = -float('inf') # 用于在所有范围档位中比较Q值
                
                candidate_next_hops_info_all_ranges = [] # 存储所有范围找到的候选

                for comm_range_attempt in range_values:
                    ch_node_data["current_communication_range"] = comm_range_attempt
                    # logger.debug(f"  CH {ch_id} 尝试范围: {comm_range_attempt:.1f}m")

                    current_attempt_candidates_info = [] # (next_hop_id, type_str, distance_float)
                    # a. 其他活跃CH
                    for other_ch_id_cand in self.confirmed_cluster_heads_for_epoch:
                        if other_ch_id_cand == ch_id or self.nodes[other_ch_id_cand]["status"] != "active":
                            continue
                        dist_to_other_ch = self.calculate_distance(ch_id, other_ch_id_cand)
                        if dist_to_other_ch <= ch_node_data["current_communication_range"] and \
                           dist_to_other_ch <= self.nodes[other_ch_id_cand]["current_communication_range"]:
                            current_attempt_candidates_info.append((other_ch_id_cand, "CH", dist_to_other_ch))
                    # b. 基站BS
                    dist_to_bs_val = self.calculate_distance_to_bs(ch_id)
                    if dist_to_bs_val <= ch_node_data["current_communication_range"]:
                        current_attempt_candidates_info.append((bs_id_for_routing, "BS", dist_to_bs_val))
                    
                    for other_node_idx, other_node_data_cand in enumerate(self.nodes):
                        if other_node_data_cand["status"] == "active" and \
                           other_node_data_cand.get("can_connect_bs_directly", False) and \
                           other_node_data_cand["id"] != ch_id:
                            
                            dist_to_direct_node = self.calculate_distance(ch_id, other_node_data_cand["id"])
                            if dist_to_direct_node <= ch_node_data["current_communication_range"] and \
                               dist_to_direct_node <= other_node_data_cand["current_communication_range"]:
                                current_attempt_candidates_info.append((other_node_data_cand["id"], DIRECT_BS_NODE_TYPE_STR, dist_to_direct_node))
                    
                    if current_attempt_candidates_info:
                        candidate_next_hops_info_all_ranges.extend(current_attempt_candidates_info)
                        # 策略1: 找到就用当前范围的候选者 (如果你想这样，这里就 break)
                        # break 
                max_hops_config = self.config.get("simulation", {}).get("max_packet_hops", 15)
                if len(current_path_history) > max_hops_config:
                    logger.warning(f"CH {ch_id}: 数据包路径 {current_path_history} (源CH: {ch_node_data.get('source_ch_of_current_packet')}) 已超最大跳数 {max_hops_config}。丢弃。")
                    ch_node_data["chosen_next_hop_id"] = None 
                    ch_node_data["current_data_path"] = [] # 清空路径，表示此包处理结束（丢弃）
                    ch_node_data["source_ch_of_current_packet"] = -1 
                    ch_node_data["gen_round_of_current_packet"] = -1
                    continue # 处理下一个CH
                
                # --- Q学习决策 (基于所有尝试范围找到的候选者) ---
                final_candidates_to_evaluate_info = []
                if candidate_next_hops_info_all_ranges:
                    temp_dict_nh = {} # key: nh_id, value: (nh_id, type, distance)
                    for nh_id_eval, nh_type_eval, nh_dist_eval in candidate_next_hops_info_all_ranges:
                        if nh_id_eval not in temp_dict_nh or nh_dist_eval < temp_dict_nh[nh_id_eval][2]: # 比较元组中第3个元素（距离）
                            temp_dict_nh[nh_id_eval] = (nh_id_eval, nh_type_eval, nh_dist_eval)
                    final_candidates_to_evaluate_info = list(temp_dict_nh.values())

                if final_candidates_to_evaluate_info:
                    candidate_ids_types = [(info[0], info[1]) for info in final_candidates_to_evaluate_info]
                    # logger.info(f"[CH_DECISION_TRACE] CH {ch_id} 原始候选: {candidate_ids_types}")
                if not final_candidates_to_evaluate_info:
                    logger.warning(f"CH {ch_id} 在所有尝试的通信范围内均未找到可达的下一跳。")
                    ch_node_data["chosen_next_hop_id"] = None
                    continue

                # --- Q学习决策选择下一跳 ---
                current_epsilon_ch_hop = ch_node_data.get("epsilon_select_next_hop", 
                                           self.config.get('q_learning',{}).get('epsilon_ch_hop_initial',0.2))

                chosen_next_hop_id_for_ch = -100 # 初始化为无效值
        
                # 2. 过滤候选下一跳，排除已在路径历史中的节点
                valid_next_hops_q_values = {}
                #logger.info(f"[CH_DECISION_TRACE] CH {ch_id} 评估有效候选 (Path: {current_path_history}):")
                for nh_id_cand, nh_type_cand, dist_cand in final_candidates_to_evaluate_info:
                    if nh_id_cand == bs_id_for_routing: # BS总是有效目标
                        valid_next_hops_q_values[nh_id_cand] = self.get_q_value_select_next_hop(ch_id, nh_id_cand)
                        #is_valid_due_to_path = True
                        # logger.info(f"[CH_DECISION_TRACE]   - Cand: {nh_id_cand} (Type: {nh_type_cand}), Q_val: {q_val_for_cand:.3f} (BS, always valid path-wise)")
                    elif nh_id_cand not in current_path_history: # 如果不在路径中，则是有效候选
                            valid_next_hops_q_values[nh_id_cand] = self.get_q_value_select_next_hop(ch_id, nh_id_cand)
                            #is_valid_due_to_path = True
                            # logger.info(f"[CH_DECISION_TRACE]   - Cand: {nh_id_cand} (Type: {nh_type_cand}), Q_val: {q_val_for_cand:.3f} (Not in path, valid)")
                    else:
                        logger.debug(f"CH {ch_id} 排除候选下一跳 {nh_id_cand}，因为它已在路径 {current_path_history} 中。")
                        # 可以选择给一个极低的Q值，或者干脆不加入valid_next_hops_q_values
                    
                    #logger.info(f"[CH_DECISION_TRACE] CH {ch_id} 最终有效候选及Q值: {valid_next_hops_q_values}")
                
                if valid_next_hops_q_values: # 先确保非空                    
                    if random.random() < current_epsilon_ch_hop:
                        chosen_next_hop_id_for_ch = random.choice(list(valid_next_hops_q_values.keys()))
                    else:
                        max_q = -float('inf')
                        # 找到最大Q值
                        for q_val_check in valid_next_hops_q_values.values():
                            if q_val_check > max_q:
                                max_q = q_val_check
                        if max_q == -float('inf'): # 所有选项都极差
                            logger.warning(f"CH {ch_id}: 所有有效下一跳的Q值都是-inf或无效: {valid_next_hops_q_values}")
                            # 随机选一个，或者标记为失败
                            # chosen_next_hop_id_for_ch = random.choice(list(valid_next_hops_q_values.keys())) # 即使是-inf也选一个
                            chosen_next_hop_id_for_ch = -100 # 或者标记为无法选择
                        else:
                        # 找到所有Q值等于最大Q值的候选
                            tied_best_hops = [nh_id for nh_id, q_v in valid_next_hops_q_values.items() if abs(q_v - max_q) < 1e-9] # 用精度比较浮点数

                            if tied_best_hops:
                                chosen_next_hop_id_for_ch = random.choice(tied_best_hops)
                                best_q_val_nh_log = valid_next_hops_q_values[chosen_next_hop_id_for_ch] # 用于日志
                                logger.debug(f"CH {ch_id} (Select NH) 利用选择了有效下一跳 {chosen_next_hop_id_for_ch} (Q={best_q_val_nh_log:.2f}) 从 {len(tied_best_hops)} 个并列最优中选择。")
                            else:
                                # 这个分支理论上不应该被执行，如果 valid_next_hops_q_values 非空
                                logger.error(f"CH {ch_id} 在利用分支中，valid_next_hops_q_values非空但tied_best_hops为空! Q_values: {valid_next_hops_q_values}")
                                chosen_next_hop_id_for_ch = -100 
                else: # valid_next_hops_q_values 为空
                    logger.warning(f"CH {ch_id} 没有有效的（非环路）下一跳可选...")
                    chosen_next_hop_id_for_ch = -100 # 或者 None
                    if chosen_next_hop_id_for_ch != -100:
                        logger.info(f"[CH_DECISION_TRACE] CH {ch_id} 利用选择了: {chosen_next_hop_id_for_ch} (MaxQ: {max_q:.3f})")
                    else:
                        logger.error(f"[CH_DECISION_TRACE] CH {ch_id} 利用时未能选择 (Tied_best_hops empty, should not happen if valid_next_hops_q_values not empty)")

                
                
                # --- 动作执行与Q表更新 ---
                #logger.info(f"[FINAL_CHOICE_DEBUG] CH {ch_id}: Epsilon-Greedy后，最终 chosen_next_hop_id_for_ch = {chosen_next_hop_id_for_ch}, 类型: {type(chosen_next_hop_id_for_ch)}")
                if chosen_next_hop_id_for_ch != -100 and chosen_next_hop_id_for_ch is not None:
                    #logger.info(f"[FINAL_CHOICE_DEBUG] CH {ch_id}: 进入了成功选择的IF块，chosen_next_hop_id_for_ch = {chosen_next_hop_id_for_ch}")
                    try: # <<< --- 整体 try 块开始 ---
                        action_taken_ch_hop = chosen_next_hop_id_for_ch
                        ch_node_data["chosen_next_hop_id"] = chosen_next_hop_id_for_ch # 记录选择
                        #logger.debug(f"[SUCCESS_BLOCK_TRACE] CH {ch_id}: 动作已记录 chosen_next_hop_id = {ch_node_data['chosen_next_hop_id']}")

                        chosen_nh_info = next((info for info in final_candidates_to_evaluate_info if info[0] == chosen_next_hop_id_for_ch), None)
                        if not chosen_nh_info:
                            logger.error(f"[CRITICAL_ERROR] CH {ch_id}: 选择了下一跳 {chosen_next_hop_id_for_ch} 但在 final_candidates_to_evaluate_info 中找不到它！原始候选: {[info[0] for info in final_candidates_to_evaluate_info]}")
                            # 这种严重错误通常不应该发生，如果发生了，后续逻辑会基于错误的 chosen_nh_type
                            # 为了安全，可以强制让它走失败路径
                            raise ValueError(f"Chosen hop {chosen_next_hop_id_for_ch} not in final_candidates_to_evaluate_info")

                        chosen_nh_type = chosen_nh_info[1]
                        logger.debug(f"[SUCCESS_BLOCK_TRACE] CH {ch_id}: 下一跳 {chosen_next_hop_id_for_ch} 类型为 {chosen_nh_type}, 距离 {chosen_nh_info[2]:.2f}")

                        # next_hop_is_bs_flag 的判断应该基于 chosen_next_hop_id_for_ch 是否等于 bs_id_for_routing
                        # chosen_nh_type 只是辅助确认
                        next_hop_is_bs_flag = (chosen_next_hop_id_for_ch == bs_id_for_routing)
                        if next_hop_is_bs_flag and chosen_nh_type != "BS":
                            logger.warning(f"[TYPE_MISMATCH_WARN] CH {ch_id}: chosen_next_hop_id {chosen_next_hop_id_for_ch} 是 BS_ID, 但类型是 {chosen_nh_type}")
                        chosen_next_hop_is_direct_bs_node_flag = (chosen_nh_type == DIRECT_BS_NODE_TYPE_STR)
                        logger.debug(f"[SUCCESS_BLOCK_TRACE] CH {ch_id}: next_hop_is_bs_flag={next_hop_is_bs_flag}, chosen_next_hop_is_direct_bs_node_flag={chosen_next_hop_is_direct_bs_node_flag}")


                        # [BS_CHOICE_DEBUG] 日志块，现在可以更准确地基于 next_hop_is_bs_flag
                        bs_as_candidate_info_debug = next((info for info in final_candidates_to_evaluate_info if info[0] == bs_id_for_routing), None) # 确保用ID查找
                        if bs_as_candidate_info_debug:
                            bs_cand_id_debug = bs_as_candidate_info_debug[0]
                            is_bs_valid_pathwise_debug = bs_cand_id_debug in valid_next_hops_q_values
                            q_val_for_bs_debug = valid_next_hops_q_values.get(bs_cand_id_debug, "N/A")
                            
                            #logger.info(f"[BS_CHOICE_DEBUG] CH {ch_id}: BS ({bs_cand_id_debug}) 是候选。路径有效性: {is_bs_valid_pathwise_debug}。Q(CH,BS): {q_val_for_bs_debug}")
                            #if chosen_next_hop_id_for_ch == bs_cand_id_debug: # 即 next_hop_is_bs_flag 为 True
                            #    logger.info(f"[BS_CHOICE_DEBUG] CH {ch_id}: *** 选择了BS ({bs_cand_id_debug}) ***")
                            #elif is_bs_valid_pathwise_debug : 
                            #    logger.info(f"[BS_CHOICE_DEBUG] CH {ch_id}: BS ({bs_cand_id_debug}) 是有效候选但未被选择。最终选择: {chosen_next_hop_id_for_ch} (Type: {chosen_nh_type})")


                        next_hop_node_obj = None 
                        if not next_hop_is_bs_flag and chosen_next_hop_id_for_ch >= 0: 
                            next_hop_node_obj = self.nodes[chosen_next_hop_id_for_ch]
                            if not next_hop_node_obj: # 以防万一
                                logger.error(f"[CRITICAL_ERROR] CH {ch_id}: next_hop_node_obj 为 None，但下一跳ID为 {chosen_next_hop_id_for_ch} 且不是BS。")
                                raise ValueError("Failed to get next_hop_node_obj")
                        #logger.debug(f"[SUCCESS_BLOCK_TRACE] CH {ch_id}: next_hop_node_obj is {'None' if not next_hop_node_obj else next_hop_node_obj['id']}")


                        # 1. 收集模糊输入
                        dc_bs_of_nh = 0.0 
                        e_of_nh_norm = 1.0 
                        load_actual_of_nh = 0 
                        avg_load_for_nh_ref_val = self.get_alive_nodes() / len(self.confirmed_cluster_heads_for_epoch) \
                                                    if self.confirmed_cluster_heads_for_epoch else 10
                        
                        if chosen_next_hop_is_direct_bs_node_flag and next_hop_node_obj:
                            dc_bs_of_nh = self.calculate_distance_to_bs(chosen_next_hop_id_for_ch)
                            e_of_nh_norm = (next_hop_node_obj["energy"] / next_hop_node_obj["initial_energy"] 
                                            if next_hop_node_obj["initial_energy"] > 0 else 0)
                            load_actual_of_nh = 1 
                        elif not next_hop_is_bs_flag and next_hop_node_obj: # 是另一个CH
                            dc_bs_of_nh = self.calculate_distance_to_bs(chosen_next_hop_id_for_ch)
                            e_of_nh_norm = (next_hop_node_obj["energy"] / next_hop_node_obj["initial_energy"] 
                                            if next_hop_node_obj["initial_energy"] > 0 else 0)
                            load_actual_of_nh = len([m for m in self.nodes if m.get("cluster_id") == chosen_next_hop_id_for_ch and m.get("status") =="active"])
                        # 如果是BS, dc_bs_of_nh, e_of_nh_norm, load_actual_of_nh 保持默认值 (0, 1, 0)

                        r_success_with_nh_norm = ch_node_data.get("history_success_with_nh", {}).get(chosen_next_hop_id_for_ch, 0.9)
                        if next_hop_is_bs_flag: r_success_with_nh_norm = 0.99 

                        dist_to_chosen_nh_val = chosen_nh_info[2] # 从 chosen_nh_info 获取距离，这里不应再是 None
                        
                        packet_size_for_ch_val = self.config.get("simulation",{}).get("packet_size",4000)
                        actual_e_tx_to_nh_val = self.calculate_transmission_energy(dist_to_chosen_nh_val, packet_size_for_ch_val, is_tx_operation=True)
                        
                        max_send_e_ref_ch_curr = self.calculate_transmission_energy(ch_node_data["current_communication_range"], packet_size_for_ch_val, True)
                        e_ctx_cost_to_nh_norm_val = actual_e_tx_to_nh_val / max_send_e_ref_ch_curr if max_send_e_ref_ch_curr > 0 else 0
                        e_ctx_cost_to_nh_norm_val = np.clip(e_ctx_cost_to_nh_norm_val, 0, 1)
                        
                        logger.debug(f"[FUZZY_INPUT_TRACE] CH {ch_id} -> NH {chosen_next_hop_id_for_ch} (Type {chosen_nh_type}): "
                                    f"dc_bs_nh={dc_bs_of_nh:.2f}, e_nh_norm={e_of_nh_norm:.2f}, load_nh={load_actual_of_nh}, "
                                    f"r_succ_nh={r_success_with_nh_norm:.2f}, e_cost_norm={e_ctx_cost_to_nh_norm_val:.2f}, avg_load_ref={avg_load_for_nh_ref_val:.2f}")

                        fuzzy_weights_for_ch_path = self.ch_path_fuzzy_logic.compute_weights(
                            current_dc_bs_neighbor=dc_bs_of_nh,
                            current_e_c_neighbor=e_of_nh_norm,
                            current_load_c_actual=load_actual_of_nh,
                            current_r_c_success=r_success_with_nh_norm,
                            current_e_ctx_cost_normalized=e_ctx_cost_to_nh_norm_val,
                            avg_load_for_neighbor_ch=avg_load_for_nh_ref_val
                        )
                        logger.debug(f"[FUZZY_OUTPUT_TRACE] CH {ch_id} -> NH {chosen_next_hop_id_for_ch}: Fuzzy Weights = {fuzzy_weights_for_ch_path}")

                        transmission_success_ch_hop_flag = True if random.random() < r_success_with_nh_norm else False
                        logger.debug(f"[SUCCESS_BLOCK_TRACE] CH {ch_id}: Transmission to {chosen_next_hop_id_for_ch} success_flag = {transmission_success_ch_hop_flag}")
                        
                        # 更新历史成功率 (确保 ch_node_data["history_success_with_nh"] 已初始化为字典)
                        if "history_success_with_nh" not in ch_node_data: ch_node_data["history_success_with_nh"] = {} # 防御性初始化
                        ch_node_data["history_success_with_nh"][chosen_next_hop_id_for_ch] = \
                            (r_success_with_nh_norm * 0.9) + (1.0 * 0.1 if transmission_success_ch_hop_flag else 0.0 * 0.1)
                        
                        data_advanced_metric_val = 0
                        dist_ch_to_bs_val = self.calculate_distance_to_bs(ch_id)
                        R_REACH_BS_VIA_DIRECT_NODE = float(self.config.get('rewards', {}).get('ch_select_next_hop', {}).get('reach_bs_via_direct_node_bonus', 90))

                        temp_reward_for_reaching_bs_like_target = 0
                        if next_hop_is_bs_flag and transmission_success_ch_hop_flag:
                            data_advanced_metric_val = dist_ch_to_bs_val 
                        elif chosen_next_hop_is_direct_bs_node_flag and transmission_success_ch_hop_flag and next_hop_node_obj:
                            dist_direct_node_to_bs = self.calculate_distance_to_bs(chosen_next_hop_id_for_ch)
                            data_advanced_metric_val = dist_ch_to_bs_val - dist_direct_node_to_bs 
                            temp_reward_for_reaching_bs_like_target = R_REACH_BS_VIA_DIRECT_NODE 
                        elif not next_hop_is_bs_flag and next_hop_node_obj and transmission_success_ch_hop_flag: 
                            dist_nh_to_bs_val = self.calculate_distance_to_bs(chosen_next_hop_id_for_ch)
                            data_advanced_metric_val = dist_ch_to_bs_val - dist_nh_to_bs_val
                        logger.debug(f"[REWARD_INPUT_TRACE] CH {ch_id} -> NH {chosen_next_hop_id_for_ch}: data_adv={data_advanced_metric_val:.2f}, temp_reach_bonus={temp_reward_for_reaching_bs_like_target}")
                        
                        reward_ch_hop = self.calculate_reward_for_selecting_next_hop(
                            ch_node_id=ch_id, 
                            chosen_next_hop_id=chosen_next_hop_id_for_ch, 
                            fuzzy_weights=fuzzy_weights_for_ch_path,
                            transmission_successful=transmission_success_ch_hop_flag,
                            actual_energy_spent_tx=actual_e_tx_to_nh_val,
                            data_advanced_amount=data_advanced_metric_val,
                            is_next_hop_bs=next_hop_is_bs_flag, 
                            next_hop_energy_normalized=e_of_nh_norm if not next_hop_is_bs_flag else None,
                            next_hop_load_ratio=(load_actual_of_nh / avg_load_for_nh_ref_val if avg_load_for_nh_ref_val > 0 else 0) if not next_hop_is_bs_flag else None,
                            is_next_hop_direct_bs_node=chosen_next_hop_is_direct_bs_node_flag
                        )
                        if chosen_next_hop_is_direct_bs_node_flag and transmission_success_ch_hop_flag:
                            reward_ch_hop += temp_reward_for_reaching_bs_like_target
                        logger.debug(f"[REWARD_OUTPUT_TRACE] CH {ch_id} -> NH {chosen_next_hop_id_for_ch}: Calculated Reward = {reward_ch_hop:.3f}")

                        next_available_hops_for_s_prime_info = []
                        is_terminal_for_q_update = (next_hop_is_bs_flag and transmission_success_ch_hop_flag) or \
                                                (chosen_next_hop_is_direct_bs_node_flag and transmission_success_ch_hop_flag) 

                        if ch_node_data["status"] == "active" and not is_terminal_for_q_update:
                            next_available_hops_for_s_prime_info = final_candidates_to_evaluate_info 
                        logger.debug(f"[Q_UPDATE_INPUT_TRACE] CH {ch_id} -> NH {action_taken_ch_hop}: Reward={reward_ch_hop:.3f}, Terminal={is_terminal_for_q_update}, Next_Hops_Count={len(next_available_hops_for_s_prime_info)}")
                        
                        self.update_q_value_select_next_hop(
                            ch_id, action_taken_ch_hop, reward_ch_hop,
                            next_state_available_next_hops=next_available_hops_for_s_prime_info,
                            is_terminal_next_hop=is_terminal_for_q_update
                        )
                        logger.debug(f"[SUCCESS_BLOCK_TRACE] CH {ch_id}: Q-table updated for action to {action_taken_ch_hop}")

                        if transmission_success_ch_hop_flag:
                            # 8. 能量消耗
                            logger.debug(f"[ENERGY_TRACE] CH {ch_id}: Attempting to consume TX energy {actual_e_tx_to_nh_val:.4e}")
                            if self.consume_node_energy(ch_id, actual_e_tx_to_nh_val):
                                ch_node_data["tx_count"] += 1
                                logger.debug(f"[ENERGY_TRACE] CH {ch_id}: TX energy consumed. Remaining: {ch_node_data['energy']:.4f}")
                            else:
                                logger.warning(f"[ENERGY_TRACE] CH {ch_id}: Died after TX energy consumption.")
                                # 如果节点死亡，可能不应再进行路径传递等操作，但Q值已更新

                            if transmission_success_ch_hop_flag and not next_hop_is_bs_flag and next_hop_node_obj and next_hop_node_obj["status"] == "active": # 确保下一跳也活着
                                energy_rx_next_hop_val = self.calculate_transmission_energy(0, packet_size_for_ch_val, is_tx_operation=False) # Renamed variable
                                logger.debug(f"[ENERGY_TRACE] NH {chosen_next_hop_id_for_ch}: Attempting to consume RX energy {energy_rx_next_hop_val:.4e}")
                                if self.consume_node_energy(chosen_next_hop_id_for_ch, energy_rx_next_hop_val):
                                    next_hop_node_obj["rx_count"] +=1 
                                    logger.debug(f"[ENERGY_TRACE] NH {chosen_next_hop_id_for_ch}: RX energy consumed. Remaining: {next_hop_node_obj['energy']:.4f}")
                                else:
                                    logger.warning(f"[ENERGY_TRACE] NH {chosen_next_hop_id_for_ch}: Died after RX energy consumption.")
                        
                        # 路径传递逻辑 (只为下一跳是CH的情况更新其current_data_path)
                       # b. 数据包送达BS的统计 和 时延计算 (只在这里做一次)
                            is_logically_at_bs = (next_hop_is_bs_flag or chosen_next_hop_is_direct_bs_node_flag)
                            if is_logically_at_bs:
                                self.sim_packets_delivered_bs_this_round += 1
                                self.sim_packets_delivered_bs_total += 1
                                logger.debug(f"CH {ch_id} (Path: {current_path_history}) 将融合包送达BS逻辑入口 (NH: {chosen_next_hop_id_for_ch})")

                                original_gen_round = ch_node_data.get("gen_round_of_current_packet", -1)
                                original_source_ch = ch_node_data.get("source_ch_of_current_packet", -1)
                                if original_gen_round != -1:
                                    delay = self.current_round - original_gen_round + 1
                                    self.sim_total_delay_this_round += delay
                                    self.sim_num_packets_for_delay_this_round += 1
                                    logger.debug(f"  Fusion packet (SourceCH: {original_source_ch}, GenRound: {original_gen_round}) delay: {delay} rounds")
                                else:
                                    logger.warning(f"  CH {ch_id} delivered fusion packet to BS-like target, but missing original_gen_round info!")
                                
                                # 清理当前CH的这个数据流信息
                                ch_node_data["current_data_path"] = [] 
                                ch_node_data["source_ch_of_current_packet"] = -1 
                                ch_node_data["gen_round_of_current_packet"] = -1
                            
                            # c. 路径传递给下一跳CH (如果下一跳是CH且未到逻辑BS)
                            elif chosen_nh_type == "CH" and next_hop_node_obj and next_hop_node_obj["status"] == "active": # 确保不是逻辑BS
                                path_passed_on = list(current_path_history) 
                                if next_hop_node_obj["id"] not in path_passed_on:
                                    path_passed_on.append(next_hop_node_obj["id"])
                                else: # Should not happen due to earlier filtering
                                    logger.error(f"[PATH_ERROR_CRITICAL] CH {ch_id}: Next hop CH {next_hop_node_obj['id']} is ALREADY in path {path_passed_on}!")
                                
                                max_hops_config = self.config.get("simulation", {}).get("max_packet_hops", 15)
                                if len(path_passed_on) > max_hops_config:
                                    logger.warning(f"[PATH_LIMIT_EXCEEDED] CH {ch_id}: Path {path_passed_on} for NH {next_hop_node_obj['id']} exceeds max hops {max_hops_config}. Path not passed, data effectively dropped here.")
                                    # 可以在这里给一个额外的负奖励信号，或者依赖传输失败的奖励
                                    # 如果TTL超限，逻辑上这个包不应该被下一跳接收并继续处理其路径
                                    ch_node_data["current_data_path"] = [] 
                                    ch_node_data["source_ch_of_current_packet"] = -1 
                                    ch_node_data["gen_round_of_current_packet"] = -1
                                else:
                                    next_hop_node_obj["current_data_path"] = path_passed_on
                                    next_hop_node_obj["source_ch_of_current_packet"] = ch_node_data["source_ch_of_current_packet"]
                                    next_hop_node_obj["gen_round_of_current_packet"] = ch_node_data["gen_round_of_current_packet"]
                                    logger.debug(f"[PATH_PASS_TRACE] CH {ch_id} -> CH {next_hop_node_obj['id']}. Path for next hop: {path_passed_on}, SourceInfo: SrcCH={ch_node_data['source_ch_of_current_packet']}, GenRnd={ch_node_data['gen_round_of_current_packet']}")
                                    
                                    # 成功传递给下一个CH后，清空当前CH的这个数据流状态
                                    ch_node_data["current_data_path"] = [] 
                                    ch_node_data["source_ch_of_current_packet"] = -1 
                                    ch_node_data["gen_round_of_current_packet"] = -1
                        
                        else: # transmission_success_ch_hop_flag is False
                            logger.warning(f"[TRANSMISSION_FAIL] CH {ch_id} -> NH {chosen_next_hop_id_for_ch} (Type: {chosen_nh_type}). Data not passed.")
                            # 数据未成功发送，路径和源信息不传递，当前CH的current_data_path等信息保留，可能会在下一轮重试（如果它还是CH）
                            
                        #logger.info(f"[SUCCESS_BLOCK_TRACE] CH {ch_id}: Successfully processed choice for NH {chosen_next_hop_id_for_ch}. END of try block.")

                    except Exception as e_inner:
                        logger.error(f"[CRITICAL_ERROR_IN_IF_BLOCK] CH {ch_id} 在处理选定下一跳 {chosen_next_hop_id_for_ch}  时发生内部错误: {e_inner}", exc_info=True)
                        ch_node_data["chosen_next_hop_id"] = None # 发生错误，标记为未选择
                        # chosen_next_hop_id_for_ch = -100 # 确保后续的else会捕获这个状态 (如果需要)

                # 这个 if/else 结构现在是基于 try-except 执行后的 chosen_next_hop_id_for_ch 状态
                # 或者更直接地，依赖 ch_node_data["chosen_next_hop_id"] 是否为 None
                if ch_node_data["chosen_next_hop_id"] is None: # 如果在try块中出错并设为None，或初始就没选对
                    # logger.info(f"[FINAL_CHOICE_DEBUG] CH {ch_id}: 未进入成功选择的IF块（或IF块内出现问题导致选择被重置），chosen_next_hop_id_for_ch 的最终状态用于判断是 {chosen_next_hop_id_for_ch if 'chosen_next_hop_id_for_ch' in locals() else 'NOT_SET_IN_EPS_GREEDY'}")
                    # ch_node_data["chosen_next_hop_id"] 已经在上面设为 None 了
                    if ch_node_data["current_data_path"]: # 路径未清空，说明数据未成功送达逻辑终点
                            logger.warning(f"CH {ch_id} 未能为其数据（路径: {ch_node_data['current_data_path']}）选择或成功发送给下一跳。数据可能被丢弃或缓存。")
                                
                # 更新CH的epsilon_select_next_hop
                min_eps_ch_hop = self.config.get('q_learning',{}).get('epsilon_ch_hop_min',0.01)
                decay_ch_hop = self.config.get('q_learning',{}).get('epsilon_ch_hop_decay_per_round',0.998)
                ch_node_data["epsilon_select_next_hop"] = max(min_eps_ch_hop, current_epsilon_ch_hop * decay_ch_hop)

                
        

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

        # === 模拟本轮的能量消耗 ===
        # 注意：这里的能耗应该是与Q学习决策和数据传输分离的背景能耗
        # 或者，如果你的Q学习奖励已经完全包含了所有能耗，这里就不需要再减了
        # 我之前的 simulate_round_energy_consumption 包含了通信，需要调整
        self.simulate_base_round_energy_consumption() # 
                
        alive_nodes_list = [n for n in self.nodes if n["status"] == "active"]
        num_alive = len(alive_nodes_list)
        total_energy_now = sum(n["energy"] for n in alive_nodes_list)
        avg_energy_now = total_energy_now / num_alive if num_alive > 0 else 0
        
        active_chs_list = [self.nodes[ch_id_rec] for ch_id_rec in self.confirmed_cluster_heads_for_epoch if self.nodes[ch_id_rec]["status"] == "active"] # 使用本epoch确认的CH
        num_ch_now = len(active_chs_list)
        avg_ch_energy_now = sum(n["energy"] for n in active_chs_list) / num_ch_now if num_ch_now > 0 else 0
        
        members_counts = []
        if num_ch_now > 0:
            for ch_node_rec in active_chs_list:
                members_counts.append(len([m_node for m_node in alive_nodes_list if m_node.get("cluster_id") == ch_node_rec["id"]]))
        avg_members_now = sum(members_counts) / num_ch_now if num_ch_now > 0 else 0
        ch_load_variance_now = np.var(members_counts) if members_counts else 0

        self.sim_total_delay_this_round = 0.0
        self.sim_num_packets_for_delay_this_round = 0
        # 假设你有一些变量记录数据包信息 (需要你在仿真中实现这些统计)
        packets_generated_this_round = getattr(self, 'sim_packets_generated', 0) # 从env属性获取
        packets_to_bs_this_round = getattr(self, 'sim_packets_delivered_bs', 0)
        avg_delay_this_round = getattr(self, 'sim_avg_delay', 0)

        with open(PROJECT_ROOT / "reports" / "performance_log.csv", "a", encoding="utf-8") as f_perf:
            f_perf.write(f"{self.current_round},{num_alive},{total_energy_now:.4f},{avg_energy_now:.4f},"
                        f"{num_ch_now},{avg_ch_energy_now:.4f},{avg_members_now:.2f},{ch_load_variance_now:.2f},"
                        f"{packets_generated_this_round},{packets_to_bs_this_round},{avg_delay_this_round:.4f}\n")
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
