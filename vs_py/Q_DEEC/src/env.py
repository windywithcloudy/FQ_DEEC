import yaml
from pathlib import Path
import numpy as np
import networkx as nx
import math
import random
import sys
sys.path.append(str(Path(__file__).parent.parent))
import logging # 导入标准的logging库
logger = logging.getLogger("WSN_Simulation")
from utils.fuzzy import NormalNodeCHSelectionFuzzySystem, RewardWeightsFuzzySystemForCHCompetition,CHToBSPathSelectionFuzzySystem,CHSelectionStrategyFuzzySystem, CHDeclarationFuzzySystem


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
    def __init__(self, config_path=None,performance_log_path=None, ch_behavior_log_path=None):
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

        self.performance_log_path = performance_log_path
        self.ch_behavior_log_path = ch_behavior_log_path
        # 初始化节点列表（不构建拓扑）
        self.nodes = []
        self.BS_ID = -1
        self.NO_PATH_ID = -100 # [新增] 定义一个独特的“无路可走”ID
        self.DIRECT_BS_NODE_TYPE_STR = "DIRECT_BS_NODE"
        self.CH_TYPE_STR = "CH"
        self.BS_TYPE_STR = "BS"
        self.current_round = 0
        self.cluster_heads = []
        self.candidate_cluster_heads = [] # 重命名: 存储本轮的候选簇头ID
        self.confirmed_cluster_heads_for_epoch = []      # 在本Epoch内固定的CH ID列表
        self.confirmed_cluster_heads_previous_epoch = [] # 上一个Epoch的CH，用于CH竞争状态计算
        self.gateway_chs_for_epoch = []
        self.regular_chs_for_epoch = []
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


        self.strategy_fuzzy_logic = CHSelectionStrategyFuzzySystem(self.config)
        # 用于计算PDR移动平均值
        self.pdr_history = [] 
        self.pdr_ma_window = 50 # 可配置
        self.coverage_gini_last_epoch = 0.5 # 基尼系数初始设为中等不均衡，鼓励探索
        self.declaration_fuzzy_logic = CHDeclarationFuzzySystem(self.config)
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
                "last_epoch_choice": {'ch_id': -1, 'is_successful': False}, # [新增] 记录上一个epoch的选择和结果
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

    # in env.py
    def get_discrete_state_tuple_for_competition(self, node_id):
        raw_state = self.get_node_state_for_ch_competition(node_id)
        if raw_state is None: return None

        # [*** 核心修改：引入更精细的拓扑状态 ***]
        
        # 1. 能量与邻居状态 (不变)
        d_energy_self = self.discretize_normalized_energy(raw_state["e_self"])
        d_energy_neighbor_avg = self.discretize_normalized_energy(raw_state["e_neighbor_avg"])
        d_neighbor_count = self.discretize_neighbor_count(raw_state["n_neighbor"])
        d_ch_count_nearby = self.discretize_ch_count_nearby(raw_state["n_ch_nearby"])

        # 2. 到BS的归一化距离
        dist_to_bs_normalized = raw_state["d_bs"] / self.network_diagonal if self.network_diagonal > 0 else 0
        d_dist_to_bs = self.discretize_normalized_distance_to_bs(dist_to_bs_normalized)

        # 3. 到网络中心的归一化距离
        center_pos = self.config['network']['base_position'] # 假设中心和BS重合
        dist_to_center = self.calculate_distance_to_point(node_id, center_pos)
        dist_to_center_normalized = dist_to_center / (self.network_diagonal / 2) if self.network_diagonal > 0 else 0
        d_dist_to_center = self.discretize_normalized_distance_to_bs(dist_to_center_normalized) # 复用BS距离的离散化

        # 新的状态元组，包含了明确的拓扑位置信息
        state_tuple = (
            d_energy_self,
            d_energy_neighbor_avg,
            d_neighbor_count,
            d_dist_to_bs,
            d_dist_to_center,
            d_ch_count_nearby
            # 我们移除了 time_since_last_ch，因为它和能量/贡献有很强的相关性，可能引入噪声
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
        
        d_bs = self.calculate_distance_to_base_station(node_id)
        return {
            "e_self": e_self, "t_last_ch": t_last_ch, "n_neighbor": n_neighbor,
            "e_neighbor_avg": e_neighbor_avg, "n_ch_nearby": n_ch_nearby, "d_bs": d_bs
        }

    def get_q_value_compete_ch(self, node_id, state_tuple, action):
        # ... (如之前定义) ...
        node = self.nodes[node_id]
        q_table = node["q_table_compete_ch"]
        return q_table.get(state_tuple, {}).get(action, 0.0) 

    # in env.py -> update_q_value_compete_ch
    def update_q_value_compete_ch(self, node_id, state_tuple, action, reward, next_state_tuple):
        node = self.nodes[node_id]
        q_table = node["q_table_compete_ch"]
        alpha = node["alpha_compete"]
        
        # [核心修改] 引入动态Gamma
        base_gamma = node["gamma_compete"]
        # 如果奖励是一个巨大的负数（意味着发生了严重失败），我们就降低gamma，
        # 让节点更关注这次的惩罚，而不是未来的不确定收益。
        # -25 是一个阈值，可以根据 routing_failure_penalty (-50) 来调整。
        dynamic_gamma = base_gamma * 0.1 if reward < -25 else base_gamma

        old_q_value = self.get_q_value_compete_ch(node_id, state_tuple, action)
        
        next_max_q = 0.0
        if next_state_tuple is not None:
            q_next_action0 = self.get_q_value_compete_ch(node_id, next_state_tuple, 0)
            q_next_action1 = self.get_q_value_compete_ch(node_id, next_state_tuple, 1)
            next_max_q = max(q_next_action0, q_next_action1)

        # 使用动态gamma进行更新
        new_q_value = old_q_value + alpha * (reward + dynamic_gamma * next_max_q - old_q_value)
        
        if state_tuple not in q_table:
            q_table[state_tuple] = {0: 0.0, 1: 0.0}
        q_table[state_tuple][action] = new_q_value

    # in env.py
    def calculate_reward_for_ch_competition(self, node_id, action_taken, 
                                                actual_members_joined=0, 
                                                is_uncovered_after_all_selections=False,
                                                was_connected_to_bs=False):
        """
        [最终融合版] 计算节点竞争CH后的奖励。
        结合了“贡献导向”（处理的数据包总量）和“能效导向”（能量消耗）的核心思想，
        并保留了对自身能量、轮换、位置、扎堆等多维因素的精细考量。
        """
        node = self.nodes[node_id]
        raw_state = self.get_node_state_for_ch_competition(node_id)
        if raw_state is None: return 0.0

        # --- 步骤 1: 获取模糊逻辑的动态权重 (这部分逻辑不变) ---
        current_total_energy = sum(n['energy'] for n in self.nodes if n['status'] == 'active')
        current_total_initial_energy = sum(n['initial_energy'] for n in self.nodes if n['status'] == 'active' and n['initial_energy'] > 0)
        net_energy_level_normalized = current_total_energy / current_total_initial_energy if current_total_initial_energy > 0 else 0
        node_self_energy_normalized = raw_state["e_self"]
        num_alive = self.get_alive_nodes()
        ch_density_global_val = len(self.confirmed_cluster_heads_for_epoch) / num_alive if num_alive > 0 else 0
        ch_to_bs_dis_normalized = raw_state["d_bs"] / self.network_diagonal if self.network_diagonal > 0 else 0
        

        
        if not hasattr(self, 'reward_weights_adjuster'):
            from utils.fuzzy import RewardWeightsFuzzySystemForCHCompetition
            self.reward_weights_adjuster = RewardWeightsFuzzySystemForCHCompetition(self.config)
        
        fuzzy_reward_weights = self.reward_weights_adjuster.compute_reward_weights(
            current_net_energy_level=net_energy_level_normalized,
            current_node_self_energy=node_self_energy_normalized,
            current_ch_density_global=ch_density_global_val,
            current_ch_to_bs_dis_normalized=ch_to_bs_dis_normalized
        )

        # --- 步骤 2: 定义所有奖励/惩罚的基础单位值 (从config加载) ---
        reward_cfg = self.config.get('rewards', {}).get('ch_competition', {}) # 使用 ch_competition 键
        base_reward_forwarding_unit = reward_cfg.get('forwarding_unit', 0.5)
        base_penalty_energy_drain = reward_cfg.get('energy_drain_penalty', 20.0)
        base_reward_energy_self_unit = reward_cfg.get('self_energy_unit', 5.0)
        base_reward_rotation_unit = reward_cfg.get('rotation_unit', 0.1)
        base_penalty_crowding = reward_cfg.get('crowding_penalty', 15.0) # 使用新名称
        base_distance_impact = reward_cfg.get('distance_impact_unit', 5.0)
        base_reward_conserve_energy_low_self = reward_cfg.get('conserve_energy_low_self', 3.0)
        base_reward_passivity_ch_enough = reward_cfg.get('passivity_ch_enough', 2.0)
        base_penalty_missed_opportunity = reward_cfg.get('missed_opportunity', 10.0)
        base_penalty_uncovered = reward_cfg.get('self_uncovered', 8.0)
        optimal_ch_nearby_threshold = self.config.get('deec',{}).get('optimal_ch_nearby_threshold', 2)
        routing_failure_penalty = reward_cfg.get('routing_failure_penalty', -50.0)

        # --- 步骤 3: 根据行动计算最终奖励 ---
        reward = 0.0

        if action_taken == 1: # 决策：成为CH
            if was_connected_to_bs:
                # [核心修改 1] 贡献奖励：基于总处理数据量（成员数 + 转发数）
                packets_forwarded = node.get("packets_forwarded_this_epoch", 0)
                total_packets_processed = actual_members_joined + packets_forwarded
                reward += fuzzy_reward_weights['w_members_factor'] * base_reward_forwarding_unit * total_packets_processed
                
                # [核心修改 2] 能效惩罚：基于单位贡献的能量消耗
                initial_energy_as_ch = node.get('initial_energy_as_ch_this_epoch', node['energy'])
                energy_consumed = initial_energy_as_ch - node['energy']
                # 计算能效：每处理一个包消耗多少能量。值越小越好。
                # 为避免除零，如果未处理包，则能效惩罚为0。
                energy_efficiency_penalty = 0
                if total_packets_processed > 0:
                    cost_per_packet = energy_consumed / total_packets_processed
                    energy_efficiency_penalty = cost_per_packet * base_penalty_energy_drain
                
                reward -= fuzzy_reward_weights['w_cost_ch_factor'] * energy_efficiency_penalty

                # [保留并融合] 自身高能量的额外奖励
                if raw_state["e_self"] > 0.6:
                    reward += fuzzy_reward_weights['w_energy_self_factor'] * base_reward_energy_self_unit * raw_state["e_self"]
                
                # [保留并融合] 轮换收益
                reward += fuzzy_reward_weights['w_rotation_factor'] * base_reward_rotation_unit * raw_state["t_last_ch"]

                # [保留并融合] 扎堆惩罚
                if raw_state["n_ch_nearby"] > optimal_ch_nearby_threshold:
                    reward -= fuzzy_reward_weights['w_cost_ch_factor'] * base_penalty_crowding * (raw_state["n_ch_nearby"] - optimal_ch_nearby_threshold)
                
                # [保留并融合] 距离影响
                reward_adjustment_from_distance = (1.0 - fuzzy_reward_weights['w_dis']) * base_distance_impact
                reward += reward_adjustment_from_distance
            else:
            # --- 如果路由失败，则施加巨大的、一票否决式的惩罚 ---
                reward += routing_failure_penalty

        else: # 决策：不成为CH
            # [保留] 这部分逻辑与您现有代码完全一致
            if raw_state["e_self"] < 0.3:
                reward += base_reward_conserve_energy_low_self
            
            if raw_state["n_ch_nearby"] >= optimal_ch_nearby_threshold:
                reward += base_reward_passivity_ch_enough * raw_state["n_ch_nearby"]
            
            if raw_state["n_ch_nearby"] < optimal_ch_nearby_threshold and \
            raw_state["e_self"] > 0.7 and \
            raw_state["e_neighbor_avg"] < raw_state["e_self"]:
                reward -= base_penalty_missed_opportunity

            if is_uncovered_after_all_selections:
                reward -= base_penalty_uncovered
        
        # logger.debug(f"Node {node_id} CompeteCH Reward: Action={action_taken}, Members={actual_members_joined}, Fwd={packets_forwarded}, Final_R={reward:.2f}")
        return reward

    def run_pretraining(self):
        """
        [V6.1 Pro核心] 运行完整的预训练流程。
        """
        pretrain_config = self.config.get('pre_training', {})
        if not pretrain_config.get('enabled', False):
            return

        logger.info("="*15 + " 开始Q-tables预训练 " + "="*15)
        #logger.info("  为预训练构建初始空间索引...")
        self._build_spatial_index()
        # 1. 预训练CH竞争Q-table
        rounds_compete = pretrain_config.get('rounds_compete_ch', 50000)
        if rounds_compete > 0:
            self._pretrain_q_compete_ch(rounds_compete)

        # 2. 预训练普通节点选择CH的Q-table
        rounds_select = pretrain_config.get('rounds_select_ch', 20000)
        if rounds_select > 0:
            self._pretrain_q_select_ch(rounds_select)

        logger.info("="*15 + " Q-tables预训练结束 " + "="*15)


    def _pretrain_q_compete_ch(self, num_rounds):
        """预训练CH竞争Q-table (`q_table_compete_ch`)。"""
        logger.info(f"--- 预训练CH竞争Q-table，共 {num_rounds} 轮... ---")
        
        # 获取所有节点ID，用于随机抽样
        all_node_ids = [n["id"] for n in self.nodes]
        
        for i in range(num_rounds):
            if i > 0 and i % (num_rounds // 10) == 0:
                logger.info(f"  竞争表预训练进度: {i / num_rounds * 100:.0f}%")

            # 随机选择一个节点进行“思考”
            node_id = random.choice(all_node_ids)
            node = self.nodes[node_id]

            # --- 1. 创建一个完整的、随机的“虚拟世界”状态 ---
            # a. 随机化自身能量
            original_energy = node["energy"]
            node["energy"] = random.uniform(0.05 * node["initial_energy"], node["initial_energy"])

            # b. 随机化上次当选CH的时间
            original_time_since = node["time_since_last_ch"]
            node["time_since_last_ch"] = random.randint(0, self.epoch_length * 3)

            # c. 随机化周围的“旧CH”环境
            original_prev_chs = self.confirmed_cluster_heads_previous_epoch
            other_node_ids = [n_id for n_id in all_node_ids if n_id != node_id]
            num_virtual_chs = random.randint(0, 8)
            self.confirmed_cluster_heads_previous_epoch = random.sample(other_node_ids, num_virtual_chs)

            # --- 2. 获取该虚拟状态下的离散State元组 ---
            state_tuple = self.get_discrete_state_tuple_for_competition(node_id)
            
            # --- 3. 恢复节点的真实状态，避免污染 ---
            node["energy"] = original_energy
            node["time_since_last_ch"] = original_time_since
            self.confirmed_cluster_heads_previous_epoch = original_prev_chs
            
            if state_tuple is None: continue

            # --- 4. 使用专家函数进行评估 ---
            # 评估 action=1 (宣告) 的价值。假设宣告后最坏情况：当选了但没成员
            reward_as_ch = self.calculate_reward_for_ch_competition(
                node_id, action_taken=1, actual_members_joined=0, is_uncovered_after_all_selections=False
            )
            
            # 评估 action=0 (不宣告) 的价值。假设不宣告后最坏情况：自己未被覆盖
            reward_as_normal = self.calculate_reward_for_ch_competition(
                node_id, action_taken=0, actual_members_joined=0, is_uncovered_after_all_selections=True
            )

            # --- 5. 更新Q-table ---
            q_table = node["q_table_compete_ch"]
            # 使用加权平均来平滑地更新Q值，而不是简单覆盖
            # 这使得多次遇到相似状态时，Q值更稳定
            alpha_pretrain = 0.1 # 预训练专用的小学习率
            
            old_q0 = q_table.get(state_tuple, {}).get(0, 0.0)
            old_q1 = q_table.get(state_tuple, {}).get(1, 0.0)
            
            new_q0 = old_q0 + alpha_pretrain * (reward_as_normal - old_q0)
            new_q1 = old_q1 + alpha_pretrain * (reward_as_ch - old_q1)

            if state_tuple not in q_table:
                q_table[state_tuple] = {}
            q_table[state_tuple][0] = new_q0
            q_table[state_tuple][1] = new_q1

    def _pretrain_q_select_ch(self, num_rounds):
        """
        [V6.1 Pro 完整版] 预训练普通节点选择CH的Q-table (`q_table_select_ch`)。
        通过模拟大量的“CH市场”场景，并使用模糊逻辑系统作为专家来为选择打分。
        """
        logger.info(f"--- 预训练CH选择Q-table，共 {num_rounds} 轮... ---")

        all_node_ids = [n["id"] for n in self.nodes]
        
        # 获取用于模糊逻辑归一化的参考值
        avg_load_per_ch_ref = len(all_node_ids) * self.p_opt_initial
        avg_e_send_ref = self.calculate_transmission_energy(distance=75, packet_size_bits=4000) # 假设一个中等距离的能耗作为参考

        for i in range(num_rounds):
            if i > 0 and i % (num_rounds // 10) == 0:
                logger.info(f"  选择表预训练进度: {i / num_rounds * 100:.0f}%")

            # --- 1. 创建一个随机的“CH市场” ---
            # a. 随机选择一个普通节点作为决策者
            try:
                normal_node_id = random.choice(all_node_ids)
                normal_node = self.nodes[normal_node_id]
            except IndexError:
                continue # 如果节点列表为空

            # b. 随机选择一批节点扮演CH
            num_virtual_chs = random.randint(1, 15)
            # 确保不把决策节点自己选为CH
            other_node_ids = [n_id for n_id in all_node_ids if n_id != normal_node_id]
            if len(other_node_ids) < num_virtual_chs:
                continue
            virtual_ch_ids = random.sample(other_node_ids, num_virtual_chs)

            # c. 为这些虚拟CH赋予随机状态，并保存它们的原始状态
            original_states = {} 
            for ch_id in virtual_ch_ids:
                ch_node = self.nodes[ch_id]
                original_states[ch_id] = {'energy': ch_node['energy']}
                # 随机化能量
                ch_node['energy'] = random.uniform(0.05 * ch_node['initial_energy'], ch_node['initial_energy'])
                
            # --- 2. 让决策节点对所有可达的虚拟CH进行评估 ---
            for ch_id in virtual_ch_ids:
                dist_to_ch = self.calculate_distance(normal_node_id, ch_id)
                # 只评估通信范围内的CH
                if dist_to_ch > normal_node["base_communication_range"]:
                    continue

                ch_node = self.nodes[ch_id]

                # a. 准备所有模糊逻辑输入
                # 输入1: CH到BS的距离
                current_dc_base = self.calculate_distance_to_base_station(ch_id)
                # 输入2: CH的归一化能量
                current_e_cluster_normalized = ch_node['energy'] / ch_node['initial_energy']
                # 输入3: CH的负载率 (模拟)
                # 假设负载与CH到BS的距离成反比（离得近的负载高）
                simulated_load = avg_load_per_ch_ref * (1.5 - (current_dc_base / self.network_diagonal))
                current_p_cluster_ratio_val = simulated_load / avg_load_per_ch_ref if avg_load_per_ch_ref > 0 else 0
                # 输入4: 通信成功率 (模拟)
                # 假设成功率与距离成反比
                current_r_success_normalized = np.clip(1.0 - (dist_to_ch / (normal_node["base_communication_range"] * 1.2)), 0.1, 1.0)
                # 输入5: 发送能耗率 (模拟)
                e_send = self.calculate_transmission_energy(dist_to_ch, 4000)
                current_e_send_total_ratio_val = e_send / avg_e_send_ref if avg_e_send_ref > 0 else 0

                # b. 调用模糊逻辑“专家”
                fuzzy_weights = self.normal_node_fuzzy_logic.compute_weights(
                    current_dc_base=current_dc_base,
                    current_e_cluster_normalized=current_e_cluster_normalized,
                    current_p_cluster_ratio_val=current_p_cluster_ratio_val,
                    current_r_success_normalized=current_r_success_normalized,
                    current_e_send_total_ratio_val=current_e_send_total_ratio_val
                )

                # c. 将专家的权重转化为一个综合评估分 (作为reward)
                # 这是一个启发式公式：我们奖励高能量和好的路径，惩罚高负载和远距离
                # 权重本身就反映了“重要性”，所以可以直接加权求和
                score = (
                    fuzzy_weights['w_e_ch'] * 1.0 +      # 能量权重越高越好
                    (1 - fuzzy_weights['w_path']) * 1.0 + # 路径权重越低越好（代表路径质量高）
                    (1 - fuzzy_weights['w_load']) * 0.5 + # 负载权重越低越好
                    (1 - fuzzy_weights['w_dist_bs']) * 0.5 # 距离权重越低越好
                )
                
                # --- 3. 更新Q-table ---
                q_table = normal_node["q_table_select_ch"]
                alpha_pretrain = 0.1
                old_q = q_table.get(ch_id, 0.0)
                # 使用标准的Q-learning更新公式，目标值就是我们的专家评估分
                new_q = old_q + alpha_pretrain * (score - old_q)
                q_table[ch_id] = new_q

            # --- 4. 恢复所有被修改节点的原始状态 ---
            for ch_id, state in original_states.items():
                self.nodes[ch_id]['energy'] = state['energy']

    def _update_node_roles_and_timers(self, ch_list_to_update):
        """
        [V2.5版 辅助函数] 根据传入的CH列表，更新这些特定节点的角色为CH。
        """
        # 1. 将传入的列表转换为集合，以便快速查找
        ch_set_to_update = set(ch_list_to_update)

        ch_range_enhancement_factor = self.config.get('network', {}).get('ch_range_enhancement_factor', 1.0)
        if ch_range_enhancement_factor != 1.0:
            logger.debug(f"本轮CH通信范围增强因子为: {ch_range_enhancement_factor}")
        # 2. 只遍历需要被更新为CH的节点
        for node_id in ch_set_to_update:
        # 安全性检查
            if not (0 <= node_id < len(self.nodes)):
                logger.warning(f"在 _update_node_roles_and_timers 中遇到无效的节点ID: {node_id}")
                continue

            node_data = self.nodes[node_id]
            
            # 确保节点是活跃的才能被更新
            if node_data["status"] == "active":
                # a. 更新角色和计时器
                node_data["role"] = "cluster_head"
                node_data["cluster_id"] = node_data["id"] # CH的cluster_id是它自己
                node_data["time_since_last_ch"] = 0       # 当选CH，计时器清零
                
                # b. [*** 核心修改 ***] 动态增强CH的通信范围
                #    current_communication_range = base_range * factor
                node_data["current_communication_range"] = node_data["base_communication_range"] * ch_range_enhancement_factor
                
                # c. [重要] 记录成为CH时的初始能量，用于Q表更新时的奖励计算
                node_data['initial_energy_as_ch_this_epoch'] = node_data["energy"]
                
                logger.debug(f"节点 {node_id} 已被更新为CH。基础范围: {node_data['base_communication_range']:.2f}m, "
                            f"增强后当前范围: {node_data['current_communication_range']:.2f}m。")
    
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
            was_a_successful_ch = False
            if self.nodes[node_id]["role"] == "cluster_head" and node_id in self.confirmed_cluster_heads_for_epoch: # 确认它真的是本epoch的CH
                actual_members_this_epoch = len([
                    m_node for m_node in self.nodes 
                    if m_node.get("cluster_id") == node_id and m_node["status"] == 'active'
                ])
                if node.get("can_route_to_bs_this_epoch", False): # 假设有这个标志
                    was_a_successful_ch = True
            
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
                is_uncovered_at_epoch_end,
                was_a_successful_ch
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

            if self.ch_behavior_log_path:
                try:
                    with open(self.ch_behavior_log_path, "a", encoding="utf-8") as f_ch_log:
                        for ch_id_log in self.confirmed_cluster_heads_for_epoch: # 使用本epoch的CH列表
                            ch_node_log = self.nodes[ch_id_log]
                            if ch_node_log["status"] == "active" and ch_node_log["role"] == "cluster_head": # 确保是活跃CH
                                
                                # 获取选举时的状态 (需要从 competition_log_for_current_epoch 中获取)
                                # 注意：competition_log 中的 "raw_state_for_log" 存的是宣告时的状态
                                election_time_energy = ch_node_log["initial_energy"] # 简化：假设选举时是满能量，或需要回溯
                                time_since_last_ch_at_election = 0 # 简化：或从log获取
                                neighbors_at_election = 0 # 简化：或从log获取
                                dist_to_bs_val = self.calculate_distance_to_base_station(ch_id_log)

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
                except Exception as e:
                    logger.error(f"写入CH行为日志到 {self.ch_behavior_log_path} 失败: {e}")
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
        d_c_base_j = self.calculate_distance_to_base_station(chosen_ch_id)
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

# in env.py
    def update_q_value_select_next_hop(self, 
                                        ch_node_id, 
                                        chosen_next_hop_id,
                                        reward,
                                        max_q_for_next_state, # 新参数：预先计算好的下一状态的最大Q值
                                        is_terminal_next_hop):
        """
        [最终修复版] 更新簇头选择下一跳的Q值。
        直接使用传入的 max_q_for_next_state，而不是自己计算。
        """
        if ch_node_id == -1 or not (0 <= ch_node_id < len(self.nodes)):
            logger.error(f"update_q_value_select_next_hop: 尝试为一个无效的ID({ch_node_id})更新Q表。")
            return
        
        ch_node_data = self.nodes[ch_node_id]
        if ch_node_data["status"] == "dead" or ch_node_data["role"] != "cluster_head":
            return 

        if "q_table_select_next_hop" not in ch_node_data:
            ch_node_data["q_table_select_next_hop"] = {}
        
        q_table = ch_node_data["q_table_select_next_hop"]
        
        q_cfg = self.config.get('q_learning', {})
        alpha = ch_node_data.get("alpha_select_next_hop", float(q_cfg.get('alpha_ch_hop', 0.1)))
        gamma = ch_node_data.get("gamma_select_next_hop", float(q_cfg.get('gamma_ch_hop', 0.9)))

        old_q_value = q_table.get(chosen_next_hop_id, 0.0)

        # [核心修改] 直接使用传入的 max_q_for_next_state。
        # 如果是终止状态，max_q_for_next_state 应该为 0。
        max_q_next = 0.0
        if not is_terminal_next_hop:
            max_q_next = max_q_for_next_state

        new_q_value = old_q_value + alpha * (reward + gamma * max_q_next - old_q_value)
        
        q_table[chosen_next_hop_id] = new_q_value

        logger.debug(f"CH {ch_node_id} Q_select_next_hop updated: NextHop={chosen_next_hop_id}, R={reward:.2f}, OldQ={old_q_value:.3f}, NewQ={new_q_value:.3f}, NextMaxQ={max_q_next:.3f}")

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

        # [*** 核心修改 ***]
        # 阶段 3.5: 验证CH路由路径的连通性
        # 这个验证只在每个Epoch的最后一轮执行，因为Q-table的更新也只在此时发生
        if (self.current_round + 1) % self.epoch_length == 0 or \
        self.current_round == self.config.get('simulation', {}).get('total_rounds', 1) - 1:
            self._validate_ch_routing_paths()
        
        # 阶段 4: 执行本轮所有暂存的能量消耗
        self._apply_energy_consumption()

        # 阶段 5: 更新并记录本轮的性能指标
        self._update_and_log_performance_metrics()
        
        logger.info(f"--- 第 {self.current_round} 轮结束 ---")
        
        # 检查仿真是否应结束 (例如, 所有节点都死亡)
        if self.get_alive_nodes() == 0:
            logger.info("网络中所有节点均已死亡，仿真结束。")
            return False
            
        return True


    # in env.py
    def _prepare_for_new_round(self):
        """
        [最终修复版] 阶段 0: 为新一轮仿真做准备。
        确保 chosen_next_hop_id 初始化为 NO_PATH_ID 而不是 None。
        """
        self._build_spatial_index()
        
        # 重置每轮的统计数据
        self.sim_packets_generated_this_round = 0
        self.sim_packets_delivered_bs_this_round = 0
        self.sim_total_delay_this_round = 0.0
        self.sim_num_packets_for_delay_this_round = 0

        # 在每个Epoch的第0轮，重置epoch级别的计数器
        if self.current_round % self.epoch_length == 0:
            for node in self.nodes:
                node["packets_forwarded_this_epoch"] = 0

        for node in self.nodes:
            if node["status"] == "active":
                # 1. 统一的角色和状态初始化
                node["is_overloaded"] = False
                
                # [*** 核心修复 ***] 初始化为 NO_PATH_ID，而不是 None
                node["chosen_next_hop_id"] = self.NO_PATH_ID
                
                node["current_communication_range"] = node["base_communication_range"]
                
                # 3. 清空上一轮的待消耗能量
                node["pending_tx_energy"] = 0.0
            node["pending_rx_energy"] = 0.0
            node["pending_aggregation_energy"] = 0.0
            
            # 4. 数据包生成
            self.sim_packets_generated_this_round += 1
            self.sim_packets_generated_total += 1
            node["has_data_to_send"] = True
            
            # 5. Epsilon 衰减 (普通节点选CH)
            min_eps = self.config.get('q_learning',{}).get('epsilon_select_ch_min', 0.01)
            decay = self.config.get('q_learning',{}).get('epsilon_select_ch_decay_per_round', 0.998)
            current_eps = node.get("epsilon_select_ch", self.config.get('q_learning',{}).get('epsilon_select_ch_initial', 0.3))
            node["epsilon_select_ch"] = max(min_eps, current_eps * decay)
    
    def _run_epoch_start_phase(self):
        """
        [V12.0 最终版] 采用“初选-诊断-微调”三阶段选举，确保角色、分布和连通性。
        """
        logger.info(f"--- ***** 新 Epoch 开始 (轮次 {self.current_round}) ***** ---")

        # 1. 角色洗牌与Q表更新
        # ... (这部分不变) ...
        for node in self.nodes:
            if node["status"] == "active":
                node["role"] = "normal"
                node["cluster_id"] = -1
                node["is_gateway_ch"] = False
                node["can_connect_bs_directly"] = False
                if "time_since_last_ch" not in node:
                    node["time_since_last_ch"] = random.randint(0, self.epoch_length)
                node["time_since_last_ch"] += 1

        if self.current_round > 0:
            self._update_ch_competition_q_tables_at_epoch_end()
            self._update_select_ch_q_tables()
        self.confirmed_cluster_heads_previous_epoch = list(self.confirmed_cluster_heads_for_epoch)

        # --- [核心修改] 三阶段选举流程 ---

        # 阶段 1: 初选 (基于分区差异化策略)
        logger.info("选举阶段1：执行分区差异化初选...")
        gateway_ch_ids = self.promote_gateway_chs()
        initial_regular_chs = self._elect_regular_chs(gateway_ch_ids)
        
        # 初步的CH集合，用于连通性诊断
        preliminary_ch_set = set(gateway_ch_ids + initial_regular_chs)
        logger.info(f"初选完成，产生 {len(preliminary_ch_set)} 个初步CH。")

        # 阶段 2: 连通性诊断与桥接 (确保所有CH都能接入骨干网)
        logger.info("选举阶段2：诊断连通性并增选桥接CH...")
        # [新增] 调用新的辅助函数
        bridge_chs, final_ch_set = self._ensure_ch_connectivity(preliminary_ch_set)
        if bridge_chs:
            logger.info(f"为保证连通性，已增选 {len(bridge_chs)} 个桥接CH: {bridge_chs}")

        # 阶段 3: 最终确认与角色更新
        logger.info("选举阶段3：最终确认CH列表并更新角色...")
        self.gateway_chs_for_epoch = gateway_ch_ids
        # 常规CH = 初始选的 + 桥接的
        self.regular_chs_for_epoch = list(set(initial_regular_chs + bridge_chs))
        self.confirmed_cluster_heads_for_epoch = list(final_ch_set)

        # 为所有最终的CH更新角色和通信范围
        self._update_node_roles_and_timers(self.confirmed_cluster_heads_for_epoch)

        logger.info(f"本Epoch最终CH列表确认: {len(self.confirmed_cluster_heads_for_epoch)}个 "
                    f"({len(self.gateway_chs_for_epoch)}个网关, {len(self.regular_chs_for_epoch) - len(bridge_chs)}个常规, {len(bridge_chs)}个桥接).")

    def _elect_regular_chs(self, gateway_ch_ids):
        """
        [V11.0 最终版] 采用“分区差异化选举策略”，结合宏观调控和拓扑抑制。
        """
        # 1. 确定总选举池和合格节点数
        # ... (这部分代码不变) ...
        nodes_for_election = [
            n for n in self.nodes 
            if n["status"] == "active" and not n.get("is_gateway_ch", False)
        ]
        if not nodes_for_election:
            return []
        num_alive_eligible = len(nodes_for_election)

        # 2. [宏观调控] 计算本轮总体的 p_opt_current
        # ... (这部分代码不变) ...
        # ... (得到 total_ideal_ch_count) ...
        pdr_ma = self.get_pdr_moving_average()
        avg_energy_norm = self._calculate_current_average_energy() / self.E0
        isolated_rate = getattr(self, 'isolated_node_rate_last_epoch', 0.0)
        congestion = getattr(self, 'congestion_level_last_epoch', 0.0)

        adjustment_factor = self.strategy_fuzzy_logic.compute_p_opt_factor(
            pdr=pdr_ma,
            energy=avg_energy_norm,
            isolated_rate=isolated_rate,
            congestion=congestion
        )
        
        self.p_opt_current = self.p_opt_initial * adjustment_factor
        
        p_opt_min_cap = self.config.get('deec', {}).get('p_opt_min_cap', 0.03)
        p_opt_max_cap = self.config.get('deec', {}).get('p_opt_max_cap', 0.25)
        self.p_opt_current = np.clip(self.p_opt_current, p_opt_min_cap, p_opt_max_cap)
        
        logger.info(f"高阶模糊策略决策 -> 最终 p_opt_current={self.p_opt_current:.3f}")
        total_ideal_ch_count = int(num_alive_eligible * self.p_opt_current)


        # 3. [分区] 将所有候选节点划分到四个象限
        # ... (这部分代码不变) ...
        center_x = self.config['network']['area_size'][0] / 2
        center_y = self.config['network']['area_size'][1] / 2
        quadrants = {
            'top_right': [], 'top_left': [],
            'bottom_right': [], 'bottom_left': []
        }
        for node in nodes_for_election:
            x, y = node['position']
            if x >= center_x and y >= center_y:
                quadrants['top_right'].append(node)
            elif x < center_x and y >= center_y:
                quadrants['top_left'].append(node)
            elif x >= center_x and y < center_y:
                quadrants['bottom_right'].append(node)
            else:
                quadrants['bottom_left'].append(node)

        # 4. [分区差异化选举]
        final_regular_chs = []
        suppression_factor = self.config.get('deec', {}).get('ch_election', {}).get('suppression_factor', 0.5)
        
        # 定义中心区域的边界，例如占整个区域的40%
        relay_zone_radius_sq = (self.network_diagonal * 0.4)**2 

        for quadrant_name, nodes_in_quadrant in quadrants.items():
            if not nodes_in_quadrant:
                continue

            num_nodes_in_quadrant = len(nodes_in_quadrant)
            
            # a. 计算本区域的弹性配额
            quadrant_node_ratio = num_nodes_in_quadrant / num_alive_eligible if num_alive_eligible > 0 else 0
            ideal_ch_count_for_quadrant = round(total_ideal_ch_count * quadrant_node_ratio)
            
            # b. [核心修改] 在本区域内，根据节点位置采用不同评分标准
            candidate_scores_in_quadrant = {}
            ideal_members = 1 / self.p_opt_current if self.p_opt_current > 0 else 10
            
            for node in nodes_in_quadrant:
                node_id = node["id"]
                state_tuple = self.get_discrete_state_tuple_for_competition(node_id)
                if state_tuple is None: continue
                
                q0, q1 = self.get_q_value_compete_ch(node_id, state_tuple, 0), self.get_q_value_compete_ch(node_id, state_tuple, 1)
                q_advantage = q1 - q0
                energy_score = node["energy"] / node["initial_energy"]
                
                # 判断节点属于哪个区域
                dist_to_bs_sq = self.calculate_distance_to_base_station(node_id)**2

                if dist_to_bs_sq <= relay_zone_radius_sq:
                    # --- 中继区 (Relay Zone) 选举标准 ---
                    # 目标：选出强大的骨干路由器
                    # 优先考虑：1. 离BS近  2. 能量高
                    
                    # 1. 距离评分 (越近越好)
                    distance_score = 1.0 - (math.sqrt(dist_to_bs_sq) / (self.network_diagonal * 0.4))
                    
                    # 最终评分：距离(50%) + 能量(40%) + Q学习(10%)
                    score = (0.5 * distance_score) + (0.4 * energy_score) + (0.1 * q_advantage)

                else:
                    # --- 接入区 (Access Zone) 选举标准 ---
                    # 目标：选出优秀的社区服务中心
                    # 优先考虑：1. 覆盖潜力  2. 能量充足
                    
                    # 1. 覆盖潜力评分 (有足够邻居来形成簇)
                    neighbor_count = len(self.get_node_neighbors(node_id, node["base_communication_range"]))
                    coverage_potential = min(neighbor_count, ideal_members) / (ideal_members + 1e-6)
                    
                    # 最终评分：覆盖潜力(50%) + 能量(40%) + Q学习(10%)
                    score = (0.5 * coverage_potential) + (0.4 * energy_score) + (0.1 * q_advantage)
                
                candidate_scores_in_quadrant[node_id] = score
            
            if not candidate_scores_in_quadrant: continue

            # c. 在本区域内进行拓扑抑制选举 (这部分逻辑不变)
            chs_in_quadrant = []
            potential_candidates_in_quadrant = dict(candidate_scores_in_quadrant)
            while len(chs_in_quadrant) < ideal_ch_count_for_quadrant and potential_candidates_in_quadrant:
                best_candidate_id = max(potential_candidates_in_quadrant, key=lambda k: potential_candidates_in_quadrant[k])
                chs_in_quadrant.append(best_candidate_id)
                del potential_candidates_in_quadrant[best_candidate_id]
                
                neighbors_to_suppress = self.get_node_neighbors(best_candidate_id, self.nodes[best_candidate_id]["base_communication_range"])
                for neighbor_id in neighbors_to_suppress:
                    if neighbor_id in potential_candidates_in_quadrant:
                        potential_candidates_in_quadrant[neighbor_id] *= suppression_factor
            
            # d. 最低配额保障 (MSL for Quadrant) (这部分逻辑不变)
            if not chs_in_quadrant and nodes_in_quadrant: # 确保分区内有节点才触发
                best_in_quadrant = max(candidate_scores_in_quadrant, key=lambda k: candidate_scores_in_quadrant[k])
                chs_in_quadrant.append(best_in_quadrant)
                logger.warning(f"分区配额保障触发：为区域 {quadrant_name} 强制增补了CH {best_in_quadrant}。")
                
            final_regular_chs.extend(chs_in_quadrant)

        return final_regular_chs

    def _run_normal_node_selection_phase(self):
        """
        [V12.0 最终版] 普通节点决策并记录CH选择。
        该版本引入了“能耗感知”决策，即在Q值的基础上，惩罚连接成本高的CH，
        以避免节点为连接一个遥远的“好”CH而耗尽自身能量。
        """        
        logger.info("开始阶段2：普通节点进行CH关联决策...")
        
        # 1. 安全性检查：确保有常规CH可供选择
        if not hasattr(self, 'regular_chs_for_epoch') or not self.regular_chs_for_epoch:
            logger.warning("没有任何常规CH可供选择 (self.regular_chs_for_epoch为空)，所有普通节点将尝试补救或保持孤立。")
            # 确保所有普通节点的cluster_id都是-1，以便后续补救
            for node_data in self.nodes:
                if node_data["role"] == "normal":
                    node_data["cluster_id"] = -1
            return

        num_nodes_switched = 0
        num_nodes_newly_assigned = 0
        relay_zone_radius = self.network_diagonal * 0.4

        # 从config获取能耗惩罚的权重
        # 如果config中没有，则默认为0，即不启用能耗惩罚
        energy_cost_penalty_factor = self.config.get('rewards', {}).get('select_ch', {}).get('energy_cost_penalty_factor', 0)
        if energy_cost_penalty_factor > 0:
            logger.debug(f"本轮普通节点选择CH时，启用能耗惩罚，因子为: {energy_cost_penalty_factor}")

        for node_data in self.nodes:
            # 只处理活跃的、未分配角色的普通节点
            if not (node_data["status"] == "active" and node_data["role"] == "normal"):
                continue

            node_id = node_data["id"]
            previous_ch_id = node_data.get("cluster_id", -1) # 使用.get()更安全
            
            # 2. 寻找并分类可达的常规CH
            reachable_access_chs = []
            reachable_relay_chs = []
            
            # 使用节点的当前通信范围进行判断
            node_comm_range = node_data.get("current_communication_range", node_data["base_communication_range"])

            for ch_id in self.regular_chs_for_epoch:
                # 确保CH是活跃的
                if self.nodes[ch_id]["status"] == "active":
                    dist = self.calculate_distance(node_id, ch_id)
                    if dist <= node_comm_range:
                        if self.calculate_distance_to_base_station(ch_id) <= relay_zone_radius:
                            reachable_relay_chs.append(ch_id)
                        else:
                            reachable_access_chs.append(ch_id)
            
            # 3. 根据“接入优先”原则确定最终候选池
            candidate_ch_ids = []
            if reachable_access_chs:
                candidate_ch_ids = reachable_access_chs
            elif reachable_relay_chs:
                logger.debug(f"节点 {node_id} 找不到接入CH，尝试连接中继CH。")
                candidate_ch_ids = reachable_relay_chs
            
            # 4. 如果没有候选者，则标记为孤立，等待补救
            if not candidate_ch_ids:
                node_data["cluster_id"] = -1
                node_data["last_epoch_choice"] = {'ch_id': -1, 'is_successful': False}
                continue

            # 5. [核心] 在候选池中进行“能耗感知”的Q学习决策
            decision_scores = {}
            for ch_id in candidate_ch_ids:
                # a. 获取Q学习的长期价值
                q_value = self.get_q_value_select_ch(node_id, ch_id)
                
                # b. 计算连接到这个CH的短期能耗成本
                dist_to_ch = self.calculate_distance(node_id, ch_id)
                # 使用配置中的包大小
                packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
                estimated_tx_energy = self.calculate_transmission_energy(dist_to_ch, packet_size)
                
                # c. 计算能耗惩罚
                energy_penalty = estimated_tx_energy * energy_cost_penalty_factor
                
                # d. 最终决策分数 = Q学习价值 - 能耗惩罚
                final_score = q_value - energy_penalty
                decision_scores[ch_id] = final_score

            # 6. 考虑CH过载惩罚
            overload_penalty = self.config.get('rewards', {}).get('select_ch', {}).get('overload_penalty', -100.0)
            for ch_id in candidate_ch_ids:
                if self.nodes[ch_id].get("is_overloaded", False):
                    # 直接施加一个巨大的惩罚，使节点极力避免选择过载的CH
                    decision_scores[ch_id] += overload_penalty 

            # 7. Epsilon-Greedy 决策
            chosen_ch_id = -1
            epsilon = node_data.get("epsilon_select_ch", 0.3)
            if random.random() < epsilon:
                chosen_ch_id = random.choice(candidate_ch_ids)
                logger.debug(f"节点 {node_id} (epsilon={epsilon:.2f}) 随机探索，选择了CH {chosen_ch_id}")
            else:
                # 基于新的、考虑了能耗和过载的决策分数来选择
                chosen_ch_id = max(decision_scores, key=lambda k: decision_scores[k])
                logger.debug(f"节点 {node_id} (epsilon={epsilon:.2f}) 利用价值，选择了CH {chosen_ch_id} (最高分: {decision_scores[chosen_ch_id]:.3f})")

            # 8. 更新本轮状态并记录决策
            node_data["cluster_id"] = chosen_ch_id
            node_data["last_epoch_choice"] = {
                'ch_id': chosen_ch_id,
                'is_successful': True # 先乐观地假设成功，在epoch结束时根据真实结果更新Q表
            }

            # 9. 统计切换情况
            if chosen_ch_id != previous_ch_id:
                if previous_ch_id == -1:
                    num_nodes_newly_assigned += 1
                else:
                    num_nodes_switched += 1

        logger.info(f"CH关联决策完成：{num_nodes_newly_assigned} 个节点初次分配，{num_nodes_switched} 个节点切换了CH。")
        
    def _run_ch_capacity_check_phase(self):
        """阶段 2.5: 检查并处理 CH 容量超限的问题。"""
        logger.info("开始阶段 2.5：CH 容量限制检查...")
        if not self.enable_ch_capacity_limit and self.confirmed_cluster_heads_for_epoch:return
        num_rejections_this_round = 0

        overload_tolerance = self.config.get('q_learning', {}).get('ch_management', {}).get('overload_tolerance', 2)
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
        
        rejection_threshold = max_members_for_ch_calculated + overload_tolerance

        for ch_id in self.confirmed_cluster_heads_for_epoch:
            ch_node_data = self.nodes[ch_id] # 使用不同变量名
            if ch_node_data["status"] != "active" or ch_node_data["role"] != "cluster_head":
                continue

            # 找出所有选择了这个CH的普通节点
            current_members_of_ch = [
                node for node in self.nodes 
                if node.get("cluster_id") == ch_id and node.get("role") == "normal"
            ]
            
            num_current_members = len(current_members_of_ch)
            # logger.debug(f"  CH {ch_id} 当前有 {num_current_members} 个成员，容量上限 {max_members_for_ch_calculated}.")

            if num_current_members > max_members_for_ch_calculated:
            # 标记自己为过载，以便下一轮普通节点可以感知到
                ch_node_data["is_overloaded"] = True
                logger.info(f"  CH {ch_id} 过载 (成员数 {num_current_members} > 上限 {max_members_for_ch_calculated})，已广播过载状态。")
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
                            if "collected_raw_packets" in ch_node_data and ch_node_data["collected_raw_packets"] > 0:
                                ch_node_data["collected_raw_packets"] -= 1
                            
                            # 将被拒绝节点的数据重新标记为“待发送”
                            rejected_node["has_data_to_send"] = True
                            num_rejections_this_round += 1
                        else:
                            break # 如果排序后的成员列表已空
            else:
                ch_node_data["is_overloaded"] = False # 如果没过载，确保状态是False

        if num_rejections_this_round > 0:
            logger.info(f"CH容量限制阶段：共拒绝了 {num_rejections_this_round} 个节点的加入请求。")


    def _run_ch_routing_phase(self):
        """
        [V-Final 最终版] CH路由与数据传输总指挥。
        整合了数据交接、融合、能耗计算、动态成本路由和智能流控批量发送。
        """
        logger.info("开始阶段 3：CH路由与数据传输...")
        
        # --- 阶段 1: 优雅的数据交接 ---
        current_ch_set = set(self.confirmed_cluster_heads_for_epoch)
        for holder_id, buffer in list(self.packets_in_transit.items()):
            if holder_id not in current_ch_set and self.nodes[holder_id]["status"] == "active":
                logger.warning(f"节点 {holder_id} 在本轮落选CH，其持有的 {len(buffer)} 个数据包需要处理。")
                packets_remained_after_handover = []
                for packet in buffer:
                    best_new_holder, min_dist = None, float('inf')
                    for ch_id in current_ch_set:
                        if ch_id not in self.packets_in_transit: self.packets_in_transit[ch_id] = []
                        if len(self.packets_in_transit[ch_id]) < self.ch_forwarding_buffer_size:
                            dist = self.calculate_distance(holder_id, ch_id)
                            if dist < min_dist:
                                min_dist, best_new_holder = dist, ch_id
                    if best_new_holder:
                        self._add_packet_to_queue_with_aging(best_new_holder, packet)
                    else:
                        packets_remained_after_handover.append(packet)
                
                if not packets_remained_after_handover:
                    if holder_id in self.packets_in_transit:
                        del self.packets_in_transit[holder_id]
                else:
                    self.packets_in_transit[holder_id] = packets_remained_after_handover
                    logger.error(f"交接失败：落选CH {holder_id} 仍有 {len(packets_remained_after_handover)} 个包无法交接！网络在交接阶段已饱和。")

        # --- 阶段 2: CH融合数据，并计算成员->CH的传输能耗 ---
        packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            if self.nodes[ch_id]["status"] == "active":
                if ch_id not in self.packets_in_transit: self.packets_in_transit[ch_id] = []
                
                # 找到所有选择此CH的成员（包括CH自己）中，有数据待发送的
                raw_packets_sources = [node['id'] for node in self.nodes if (node.get("cluster_id") == ch_id or node["id"] == ch_id) and node.get("has_data_to_send")]
                
                if raw_packets_sources:
                    # a. 计算成员->CH的传输能耗 (TX和RX)
                    for member_id in raw_packets_sources:
                        if member_id != ch_id: # CH自己的数据不产生TX/RX
                            member_node = self.nodes[member_id]
                            dist = self.calculate_distance(member_id, ch_id)
                            tx_energy = self.calculate_transmission_energy(dist, packet_size)
                            member_node["pending_tx_energy"] += tx_energy
                            member_node["tx_count"] += 1
                            
                            rx_energy = self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)
                            self.nodes[ch_id]["pending_rx_energy"] += rx_energy
                            self.nodes[ch_id]["rx_count"] += 1

                    # b. 检查缓冲区，融合数据包并计算聚合能耗
                    if len(self.packets_in_transit[ch_id]) < self.ch_forwarding_buffer_size:
                        uid = f"{self.current_round}-{ch_id}"
                        new_packet = {"source_ch": ch_id, "gen_round": self.current_round, "path": [ch_id], "uid": uid, "num_raw_packets": len(raw_packets_sources), "original_sources": raw_packets_sources}
                        self._add_packet_to_queue_with_aging(ch_id, new_packet)
                        
                        agg_cost_per_bit = self.config.get('energy', {}).get('aggregation_cost_per_bit', 5e-9)
                        total_raw_size = packet_size * len(raw_packets_sources)
                        agg_energy = agg_cost_per_bit * total_raw_size
                        self.nodes[ch_id]["pending_aggregation_energy"] += agg_energy
                        
                        # c. 标记原始数据已被处理
                        for node_id in raw_packets_sources: 
                            self.nodes[node_id]["has_data_to_send"] = False
                    else:
                        logger.warning(f"CH {ch_id} 的发送队列已满，本轮新融合的 {len(raw_packets_sources)} 个原始包被丢弃。")
                        for node_id in raw_packets_sources:
                            if node_id != ch_id:
                                self.nodes[node_id]["last_epoch_choice"]['is_successful'] = False

        # --- 步骤 3: [核心升级] 构建由模糊逻辑驱动的动态成本路由图 ---
        G_ch_routing = nx.DiGraph()
        routing_nodes = self.confirmed_cluster_heads_for_epoch
        G_ch_routing.add_nodes_from(routing_nodes)
        G_ch_routing.add_node(self.BS_ID)
        
        # 获取一些参考值
        avg_load_for_nh_ref = self.get_alive_nodes() / len(routing_nodes) if routing_nodes else 10

        for u_id in routing_nodes:
            u_node = self.nodes[u_id]
            if u_node["status"] != "active": continue
            
            # a. u -> v (v是另一个CH)
            # 找到所有可达的邻居CH
            candidate_neighbors = self._find_candidate_next_hops(u_id)
            for nh_id, nh_type, dist in candidate_neighbors:
                if nh_type != "CH": continue # 只处理CH到CH的链路
                
                # --- 使用模糊逻辑计算成本 ---
                v_node = self.nodes[nh_id]
                
                # 准备模糊输入
                e_v_norm = v_node["energy"] / v_node["initial_energy"] if v_node["initial_energy"] > 0 else 0
                load_v_actual = len(self.packets_in_transit.get(nh_id, [])) # 使用实时缓冲区负载
                r_success_norm = u_node.get("history_success_with_nh", {}).get(nh_id, 0.9) # 假设的历史成功率
                e_cost_norm = self.calculate_transmission_energy(dist, 4000) / self.calculate_transmission_energy(u_node["current_communication_range"], 4000)
                
                fuzzy_weights = self.ch_path_fuzzy_logic.compute_weights(
                    current_dc_bs_neighbor=self.calculate_distance_to_base_station(nh_id),
                    current_e_c_neighbor=e_v_norm,
                    current_load_c_actual=load_v_actual,
                    current_r_c_success=r_success_norm,
                    current_e_ctx_cost_normalized=np.clip(e_cost_norm, 0, 1),
                    avg_load_for_neighbor_ch=avg_load_for_nh_ref
                )
                
                # 将模糊权重转化为一个综合成本值
                # 成本 = 路径差的权重 + 能耗权重 + 负载权重 + (1 - 可持续性权重)
                # 我们希望选择权重组合最优的路径
                cost = fuzzy_weights.get('w_path', 0.5) + \
                    fuzzy_weights.get('w_e_cost', 0.5) + \
                    fuzzy_weights.get('w_load_neighbor', 0.5) + \
                    (1 - fuzzy_weights.get('w_fur', 0.5))
                
                # 为骨干网CH提供成本折扣
                if v_node.get("is_gateway_ch", False):
                    cost *= self.config.get('routing', {}).get('gateway_hop_discount', 0.7)
                elif self.calculate_distance_to_base_station(nh_id) <= (self.network_diagonal * 0.4):
                    cost *= self.config.get('routing', {}).get('relay_hop_discount', 0.85)

                G_ch_routing.add_edge(u_id, nh_id, weight=cost)

            # b. u -> BS
            if self.calculate_distance_to_base_station(u_id) <= u_node.get("current_communication_range"):
                dist_to_bs = self.calculate_distance_to_base_station(u_id)
                e_cost_norm_bs = self.calculate_transmission_energy(dist_to_bs, 4000) / self.calculate_transmission_energy(u_node["current_communication_range"], 4000)
                
                # 模拟一个“下一跳是BS”的场景来获取模糊权重
                fuzzy_weights_for_bs = self.ch_path_fuzzy_logic.compute_weights(
                    current_dc_bs_neighbor=0,
                    current_e_c_neighbor=1.0,
                    current_load_c_actual=0,
                    current_r_c_success=1.0,
                    current_e_ctx_cost_normalized=np.clip(e_cost_norm_bs, 0, 1),
                    avg_load_for_neighbor_ch=avg_load_for_nh_ref
                )
                cost_bs = fuzzy_weights_for_bs.get('w_path', 0.5) + \
                        fuzzy_weights_for_bs.get('w_e_cost', 0.5)
                
                # 自身能量低的节点，直连成本应该更高
                if (u_node['energy'] / u_node['initial_energy']) < 0.3:
                    cost_bs *= 2.0
                
                G_ch_routing.add_edge(u_id, self.BS_ID, weight=cost_bs)

        # --- 阶段 4: 计算最小成本路径 (Dijkstra) ---
        min_costs_to_bs = {}
        try:
            reversed_G = G_ch_routing.reverse(copy=True)
            min_costs_to_bs_raw = nx.single_source_dijkstra_path_length(reversed_G, self.BS_ID, weight='weight')
            for node_id in routing_nodes: min_costs_to_bs[node_id] = min_costs_to_bs_raw.get(node_id, float('inf'))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            for node_id in routing_nodes: min_costs_to_bs[node_id] = float('inf')

        # --- 阶段 5: 批量传输决策与意图生成 ---
        transfer_intentions = []
        active_ch_list_sorted = sorted(routing_nodes, key=lambda cid: self.nodes[cid]["energy"], reverse=True)
        avg_packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        q_cfg = self.config.get('q_learning', {}).get('ch_management', {})
        energy_budget_ratio = q_cfg.get('max_energy_budget_per_round_for_tx', 0.1)
        abs_max_packets = q_cfg.get('max_packets_per_round_absolute', 5)

        for ch_id in active_ch_list_sorted:
            ch_node_data = self.nodes[ch_id]
            if not self.packets_in_transit.get(ch_id): continue

            candidates_info = self._find_candidate_next_hops(ch_id)
            path_history = self.packets_in_transit[ch_id][0].get("path", [])
            chosen_next_hop_id = self._decide_next_hop_with_q_learning(ch_id, candidates_info, path_history, min_costs_to_bs)
            
            if chosen_next_hop_id == -100: continue
            ch_node_data["chosen_next_hop_id"] = chosen_next_hop_id

            dist_to_nh = self.calculate_distance(ch_id, chosen_next_hop_id) if chosen_next_hop_id != self.BS_ID else self.calculate_distance_to_base_station(ch_id)
            energy_per_packet_tx = self.calculate_transmission_energy(dist_to_nh, avg_packet_size)
            
            energy_budget = ch_node_data["energy"] * energy_budget_ratio
            max_packets_by_energy = int(energy_budget / energy_per_packet_tx) if energy_per_packet_tx > 0 else 0

            available_slots_in_nh = float('inf')
            if chosen_next_hop_id != self.BS_ID:
                nh_buffer = self.packets_in_transit.get(chosen_next_hop_id, [])
                available_slots_in_nh = self.ch_forwarding_buffer_size - len(nh_buffer)

            num_packets_in_buffer = len(self.packets_in_transit[ch_id])
            num_packets_to_send = max(0, min(num_packets_in_buffer, max_packets_by_energy, available_slots_in_nh, abs_max_packets))
            
            if num_packets_to_send > 0:
                packet_batch = self.packets_in_transit[ch_id][:num_packets_to_send]
                transfer_intentions.append((ch_id, chosen_next_hop_id, packet_batch))

        # --- 阶段 6: 批量传输执行与冲突处理 ---
        delivery_attempts = {}
        for sender_id, receiver_id, packet_batch in transfer_intentions:
            if receiver_id not in delivery_attempts: delivery_attempts[receiver_id] = []
            delivery_attempts[receiver_id].append((sender_id, packet_batch))

        for receiver_id, attempts in delivery_attempts.items():
            if receiver_id == self.BS_ID:
                for sender_id, packet_batch in attempts:
                    self._execute_successful_batch_transfer(sender_id, receiver_id, packet_batch)
                continue
            
            sorted_attempts = sorted(attempts, key=lambda item: self.nodes[item[0]]["energy"], reverse=True)
            available_slots = self.ch_forwarding_buffer_size - len(self.packets_in_transit.get(receiver_id, []))
            
            for sender_id, packet_batch in sorted_attempts:
                if len(packet_batch) <= available_slots:
                    self._execute_successful_batch_transfer(sender_id, receiver_id, packet_batch)
                    available_slots -= len(packet_batch)
                else:
                    self._penalize_failed_routing_action(sender_id, receiver_id)

        # --- 阶段 7: 清理和更新Epsilon ---
        for holder_id, buffer in list(self.packets_in_transit.items()):
            if self.nodes[holder_id]["status"] == "dead":
                del self.packets_in_transit[holder_id]

        for ch_id in self.confirmed_cluster_heads_for_epoch:
            if self.nodes[ch_id]["status"] == "active":
                min_eps = self.config.get('q_learning',{}).get('epsilon_ch_hop_min',0.01)
                decay = self.config.get('q_learning',{}).get('epsilon_ch_hop_decay_per_round',0.998)
                current_eps = self.nodes[ch_id].get("epsilon_select_next_hop", self.config.get('q_learning',{}).get('epsilon_ch_hop_initial',0.2))
                self.nodes[ch_id]["epsilon_select_next_hop"] = max(min_eps, current_eps * decay)


    def _apply_energy_consumption(self):
        """在每轮结束时，统一扣除所有暂存的能量，并计算背景能耗。"""
        energy_cfg = self.config.get('energy', {})
        idle_listening_cost = float(energy_cfg.get('idle_listening_per_round', 1e-6))
        sensing_cost = float(energy_cfg.get('sensing_per_round', 5e-7))

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

    def _decide_next_hop_with_q_learning(self, ch_id, candidates_info, path_history, min_costs_to_bs):
        ch_node_data = self.nodes[ch_id]
        
        # ... (自适应Epsilon的计算逻辑保持不变) ...
        base_epsilon = ch_node_data.get("epsilon_select_next_hop", 0.2)
        energy_level = ch_node_data["energy"] / ch_node_data["initial_energy"] if ch_node_data["initial_energy"] > 0 else 0
        buffer_load = len(self.packets_in_transit.get(ch_id, [])) / self.ch_forwarding_buffer_size if self.ch_forwarding_buffer_size > 0 else 0
        adaptive_epsilon = base_epsilon * min(1.0, (energy_level / 0.5)) * (1.0 - buffer_load)

        q_cfg = self.config.get('q_learning', {})
        q_factor = q_cfg.get('q_value_factor', 1.0)
        path_reward_factor = q_cfg.get('dynamic_path_reward_factor', 10.0)
        congestion_penalty_factor = q_cfg.get('congestion_penalty_factor', 5.0) 

        # [核心修复1] 在循环外提前获取 current_cost，解决 Pylance 警告
        current_cost = min_costs_to_bs.get(ch_id, float('inf'))

        # [核心重构] 创建一个字典列表，存储每个候选者的所有决策信息
        candidate_decision_info = []
        
        for nh_id, nh_type, dist in candidates_info:
            if nh_id in path_history:
                continue
            
            q_original = self.get_q_value_select_next_hop(ch_id, nh_id)
            next_hop_cost = min_costs_to_bs.get(nh_id, float('inf')) if nh_id != self.BS_ID else 0.0
            
            if current_cost == float('inf'):
                path_reward = -10
            else:
                path_reward = (current_cost - next_hop_cost) * path_reward_factor
            
            congestion_penalty = 0.0
            if nh_id != self.BS_ID:
                nh_buffer = self.packets_in_transit.get(nh_id, [])
                congestion_penalty = (len(nh_buffer) / self.ch_forwarding_buffer_size) * congestion_penalty_factor if self.ch_forwarding_buffer_size > 0 else 0

            q_for_decision = (q_factor  * q_original) + path_reward - congestion_penalty
            
            # 将所有信息存入列表
            candidate_decision_info.append({
                "id": nh_id,
                "q_decision": q_for_decision,
                "q_base": (q_factor * q_original) + path_reward # 不含拥塞惩罚的基础分
            })
        
        if not candidate_decision_info:
            if self.calculate_distance_to_base_station(ch_id) <= ch_node_data.get("current_communication_range", ch_node_data["base_communication_range"]):
                return self.BS_ID
            else:
                return self.NO_PATH_ID

        # 判断当前CH是否为边缘CH
        is_edge_ch = self.calculate_distance_to_base_station(ch_id) > (self.network_diagonal * 0.4)

        # Epsilon-Greedy决策
        if random.random() < adaptive_epsilon:
            logger.debug(f"CH {ch_id} (adaptive_epsilon={adaptive_epsilon:.3f}) is exploring.")
            
            # 使用稳定的Softmax进行带权随机选择 (基于完整的决策Q值)
            q_values = np.array([c["q_decision"] for c in candidate_decision_info])
            
            if q_values.size == 0 or np.isnan(q_values).any() or np.isinf(q_values).any():
                return random.choice([c["id"] for c in candidate_decision_info])

            q_values_stable = q_values - np.max(q_values)
            temperature = 0.1 
            exp_q = np.exp(q_values_stable / temperature)
            sum_exp_q = np.sum(exp_q)
            if sum_exp_q <= 0 or np.isinf(sum_exp_q):
                return random.choice([c["id"] for c in candidate_decision_info])
                
            probs = exp_q / sum_exp_q
            probs /= np.sum(probs)
            
            candidate_ids = [c["id"] for c in candidate_decision_info]
            return np.random.choice(candidate_ids, p=probs)
        else:
            # --- 利用（Exploitation）逻辑 ---
            if is_edge_ch:
                # 对于边缘CH，忽略拥塞，基于 "q_base" 选择
                logger.debug(f"Edge CH {ch_id} is making a connection-focused decision.")
                best_candidate = max(candidate_decision_info, key=lambda c: c["q_base"])
                return best_candidate["id"]
            else:
                # 对于中心CH，考虑拥塞，基于 "q_decision" 选择
                logger.debug(f"Central CH {ch_id} is making a congestion-aware decision.")
                best_candidate = max(candidate_decision_info, key=lambda c: c["q_decision"])
                return best_candidate["id"]


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

        current_round_pdr = self.sim_packets_delivered_bs_this_round / self.sim_packets_generated_this_round if self.sim_packets_generated_this_round > 0 else 0
        self.pdr_history.append(current_round_pdr)
        if len(self.pdr_history) > self.pdr_ma_window:
            self.pdr_history.pop(0) # 保持窗口大小

        # 写入性能日志
        if self.performance_log_path:
            try:
                with open(self.performance_log_path, "a", encoding="utf-8") as f_perf:
                    f_perf.write(f"{self.current_round},{num_alive},{total_energy_now:.4f},{avg_energy_now:.4f},"
                                 f"{num_ch_now},{avg_ch_energy_now:.4f},{avg_members_now:.2f},{ch_load_variance_now:.2f},"
                                 f"{self.sim_packets_generated_this_round},{self.sim_packets_delivered_bs_this_round},{avg_delay_this_round:.4f}\n")
            except Exception as e:
                logger.error(f"写入性能日志到 {self.performance_log_path} 失败: {e}")
        
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
            self.isolated_node_rate_last_epoch = self._calculate_isolated_node_rate()
            self.congestion_level_last_epoch = self._calculate_network_congestion_level()
            self.coverage_gini_last_epoch = self._calculate_ch_coverage_gini() # [新增] 计算基尼系数
            logger.info(
                f"Epoch {self.current_round // self.epoch_length} 结束: "
                f"孤立普通节点数 = {isolated_normal_nodes_at_epoch_end}, "
                f"CH数 = {len(self.confirmed_cluster_heads_for_epoch)}, "
                f"孤立率 = {self.isolated_node_rate_last_epoch:.2f}, "
                f"拥塞水平 = {self.congestion_level_last_epoch:.2f}, "
                f"覆盖基尼系数 = {self.coverage_gini_last_epoch:.3f}"
            )
            

    def _remedy_isolated_nodes(self):
        """
        [V10.0 适配版] 对孤立节点进行补救，优先选择接入型常规CH。
        """
        logger.info("开始对孤立普通节点进行补救...")
        
        # 1. 找到所有需要被补救的孤立节点
        isolated_nodes = [
            n for n in self.nodes 
            if n["status"] == "active" and n["role"] == "normal" and n["cluster_id"] == -1
        ]
        if not isolated_nodes:
            return # 没有孤立节点，直接返回

        # 2. 找到所有可供选择的、未满员的常规CH，并按角色分类
        available_access_chs = []
        available_relay_chs = []
        relay_zone_radius = self.network_diagonal * 0.4
        
        # 计算容量上限
        num_alive_non_direct_bs_nodes = len([n for n in self.nodes if n["status"] == "active" and n["role"] == "normal"])
        num_active_regular_chs = len(self.regular_chs_for_epoch)
        avg_members_per_ch_ideal = num_alive_non_direct_bs_nodes / num_active_regular_chs if num_active_regular_chs > 0 else 10
        max_members_for_ch = max(1, int(avg_members_per_ch_ideal * self.ch_max_members_factor))

        # [核心修改] 只在常规CH中寻找空位
        for ch_id in self.regular_chs_for_epoch:
            ch_node = self.nodes[ch_id]
            if ch_node["status"] != "active":
                continue
                
            current_members = len([m for m in self.nodes if m.get("cluster_id") == ch_id])
            if current_members < max_members_for_ch:
                # 判断角色
                if self.calculate_distance_to_base_station(ch_id) <= relay_zone_radius:
                    available_relay_chs.append(ch_id)
                else:
                    available_access_chs.append(ch_id)
        
        available_gateway_chs = []
        if hasattr(self, 'gateway_chs_for_epoch'):
            for ch_id in self.gateway_chs_for_epoch:
                ch_node = self.nodes[ch_id]
                if ch_node["status"] != "active":
                    continue
                # 网关CH没有容量限制，或有一个非常宽松的限制
                available_gateway_chs.append(ch_id)

        # 3. [核心修改] 分层补救逻辑
        num_remedied = 0
        for node in isolated_nodes:
            best_ch_id = -1
            min_dist = float('inf')

            # a. 优先尝试在“接入CH”中寻找最近的
            if available_access_chs:
                for ch_id in available_access_chs:
                    dist = self.calculate_distance(node["id"], ch_id)
                    if dist < node["base_communication_range"] and dist < min_dist:
                        min_dist = dist
                        best_ch_id = ch_id
            
            # b. 如果在接入CH中找不到，才退而求其次，尝试“中继CH”
            if best_ch_id == -1 and available_relay_chs:
                logger.warning(f"孤立节点 {node['id']} 在接入CH中找不到归属，尝试连接到中继CH。")
                min_dist = float('inf') # 重置最小距离
                for ch_id in available_relay_chs:
                    dist = self.calculate_distance(node["id"], ch_id)
                    if dist < node["base_communication_range"] and dist < min_dist:
                        min_dist = dist
                        best_ch_id = ch_id
            
            if best_ch_id == -1 and available_gateway_chs:
                min_dist = float('inf')
                for ch_id in available_gateway_chs:
                    dist = self.calculate_distance(node["id"], ch_id)
                    if dist < node["base_communication_range"] and dist < min_dist:
                        min_dist = dist
                        best_ch_id = ch_id
                        choice_type = "Gateway"
            
            # c. 如果找到了最佳CH，则进行关联
            if best_ch_id != -1:
                node["cluster_id"] = best_ch_id
                num_remedied += 1
                # 从可用列表中移除，以防止它被过度填充（虽然不太可能，但更健壮）
                # 注意：这个移除操作会影响后续孤立节点的选择，可以根据需要决定是否保留
                # if best_ch_id in available_access_chs:
                #     # ... (更复杂的负载更新和列表移除逻辑)
                # elif best_ch_id in available_relay_chs:
                #     # ...
        
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
        
        ch_current_comm_range = ch_node_data.get("current_communication_range", ch_node_data["base_communication_range"])
        candidate_next_hops_info = []
        
        for other_ch_id in self.confirmed_cluster_heads_for_epoch:
            if other_ch_id != ch_id and self.nodes[other_ch_id]["status"] == "active":
                dist = self.calculate_distance(ch_id, other_ch_id)
                # [确保] 使用正确的范围进行判断
                if dist <= ch_current_comm_range:
                    candidate_next_hops_info.append((other_ch_id, "CH", dist))
        # b. 基站BS
        dist_to_bs = self.calculate_distance_to_base_station(ch_id)
        if dist_to_bs <= ch_current_comm_range:
            candidate_next_hops_info.append((bs_id_for_routing, "BS", dist_to_bs))
               
        return candidate_next_hops_info
    
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
                dc_bs_of_nh = self.calculate_distance_to_base_station(chosen_nh_id)
                e_of_nh_norm = (next_hop_node_obj["energy"] / next_hop_node_obj["initial_energy"]) if next_hop_node_obj["initial_energy"] > 0 else 0
                load_actual_of_nh = len([m for m in self.nodes if m.get("cluster_id") == chosen_nh_id and m.get("status") == "active"])
        else: # 如果是直连BS节点
            if chosen_nh_type == DIRECT_BS_NODE_TYPE_STR and 0 <= chosen_nh_id < len(self.nodes):
                 next_hop_node_obj = self.nodes[chosen_nh_id]
                 dc_bs_of_nh = self.calculate_distance_to_base_station(chosen_nh_id)
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
        dist_ch_to_bs = self.calculate_distance_to_base_station(ch_id)
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

    def _execute_successful_batch_transfer(self, sender_id, receiver_id, packet_batch):
        """
        [最终修复版] 处理一批数据包的成功传递，并使用“下游预估价值”更新Q表。
        """
        if not packet_batch: return

        batch_size = len(packet_batch)
        logger.debug(f"执行批量传输: {sender_id} -> {receiver_id}, 包含 {batch_size} 个包。")

        # --- 步骤 1: 暂存能耗 (逻辑不变) ---
        dist = self.calculate_distance(sender_id, receiver_id) if receiver_id != self.BS_ID else self.calculate_distance_to_base_station(sender_id)
        avg_packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        # 简化能耗计算：认为批量发送的能耗是单次发送的N倍
        total_tx_energy = self.calculate_transmission_energy(dist, avg_packet_size) * batch_size
        self.nodes[sender_id]["pending_tx_energy"] += total_tx_energy
        self.nodes[sender_id]["tx_count"] += batch_size

        if receiver_id != self.BS_ID:
            total_rx_energy = self.calculate_transmission_energy(0, avg_packet_size, is_tx_operation=False) * batch_size
            self.nodes[receiver_id]["pending_rx_energy"] += total_rx_energy
            self.nodes[receiver_id]["rx_count"] += batch_size
            num_raw_packets_in_batch = sum(p.get("num_raw_packets", 1) for p in packet_batch)
            if "packets_forwarded_this_epoch" in self.nodes[receiver_id]:
                self.nodes[receiver_id]["packets_forwarded_this_epoch"] += num_raw_packets_in_batch
            else:
                self.nodes[receiver_id]["packets_forwarded_this_epoch"] = num_raw_packets_in_batch

        # --- 步骤 2: 更新Q-table (核心修改) ---
        
        # a. 计算即时奖励 (Reward)
        # 您可以在这里调用一个更复杂的奖励函数，例如我们之前讨论的 calculate_reward_for_selecting_next_hop
        # 为确保代码能直接运行，这里使用一个简化的、但逻辑正确的奖励
           # a. [核心] 调用完整的、基于模糊逻辑的奖励计算函数
        #    首先需要确定 receiver_id 的类型
        receiver_type = "BS" if receiver_id == self.BS_ID else "CH"
        
        #    然后获取所有候选者信息，以便传入函数
        all_candidates_info = self._find_candidate_next_hops(sender_id)

        #    调用函数计算每个包的奖励
        reward_per_packet, _, is_terminal = self._calculate_routing_reward_and_next_state(
            ch_id=sender_id,
            chosen_nh_id=receiver_id,
            chosen_nh_type=receiver_type,
            dist_to_nh=dist,
            success_flag=True, # 因为这是在“successful_batch_transfer”中
            all_candidates_info=all_candidates_info
        )
        # 将单个包的奖励乘以批次大小，作为本次动作的总奖励
        total_reward = reward_per_packet * batch_size


        # b. 计算下一状态的最大Q值 (max_Q(S', A'))
        max_q_for_next_state = 0.0
        if not is_terminal:
            # S' 就是 receiver_id 接收了数据包之后的状态。
            # 我们需要估算 receiver_id 在它的下一步能获得的最大Q值。
            
            # 1. 找到 receiver_id 的所有可用下一跳
            receiver_candidates = self._find_candidate_next_hops(receiver_id)
            
            # 2. 从 receiver_id 的Q-table中，找到它所有未来选择的Q值
            q_values_for_receiver = []
            if receiver_candidates:
                for next_hop_of_receiver_info in receiver_candidates:
                    nh_id_of_receiver = next_hop_of_receiver_info[0]
                    q_val = self.get_q_value_select_next_hop(receiver_id, nh_id_of_receiver)
                    q_values_for_receiver.append(q_val)
            
            # 3. receiver_id 的最优未来收益，就是这些Q值中的最大者
            if q_values_for_receiver:
                max_q_for_next_state = max(q_values_for_receiver)

        # c. 调用修改后的 update 函数，更新 sender_id 的Q-table
        self.update_q_value_select_next_hop(
            sender_id, 
            receiver_id, 
            total_reward, 
            max_q_for_next_state, # 传递预先计算好的“未来价值”
            is_terminal
        )

        # --- 步骤 3: 更新缓冲区和PDR统计 (逻辑不变) ---
        if sender_id in self.packets_in_transit:
            self.packets_in_transit[sender_id] = self.packets_in_transit[sender_id][batch_size:]

        if is_terminal:
            for packet in packet_batch:
                num_delivered = packet.get("num_raw_packets", 1)
                self.sim_packets_delivered_bs_this_round += num_delivered
                self.sim_packets_delivered_bs_total += num_delivered
                delay = self.current_round - packet["gen_round"] + 1
                self.sim_total_delay_this_round += delay * num_delivered
                self.sim_num_packets_for_delay_this_round += num_delivered
        else:
            for packet in packet_batch:
                packet["path"].append(receiver_id)
                self._add_packet_to_queue_with_aging(receiver_id, packet)
    

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


    def promote_gateway_chs(self):
        """
        [V2.5核心] 识别能直连BS的节点，并将它们强制提升为特殊的“网关CH”。
        """
        logger.info("开始提拔网关CH...")
        direct_comm_threshold = self.config.get('network', {}).get('direct_bs_comm_threshold', 50)
        initial_e = self.E0 if hasattr(self, 'E0') and self.E0 > 0 else self.config.get('energy', {}).get('initial', 1.0)
        min_energy_for_gateway = self.config.get('energy', {}).get('min_energy_direct_bs_factor', 0.1) * initial_e

        promoted_gateway_chs = []
        for node in self.nodes:
            if node["status"] == "active":
                d_to_bs = self.calculate_distance_to_base_station(node["id"])
                # 这里的 can_connect_bs_directly 标记也一并设置
                node["can_connect_bs_directly"] = (d_to_bs <= direct_comm_threshold)

                if node["can_connect_bs_directly"] and node["energy"] > min_energy_for_gateway:
                    node["role"] = "cluster_head"
                    node["is_gateway_ch"] = True
                    node["cluster_id"] = node["id"]
                    node["time_since_last_ch"] = 0 
                    promoted_gateway_chs.append(node["id"])

        if promoted_gateway_chs:
            logger.info(f"已成功提拔 {len(promoted_gateway_chs)} 个节点为固定的网关CH: {promoted_gateway_chs}")
        
        return promoted_gateway_chs 
    
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
                distance_to_bs = self.calculate_distance_to_base_station(current_node_id)
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
                distance_to_bs_for_ch = self.calculate_distance_to_base_station(current_node_id)
                # 假设CH发送一个（可能更大的）聚合数据包
                aggregated_packet_size_factor = float(self.config.get("simulation", {}).get("ch_aggregated_packet_factor", 1.0)) # 聚合后数据包大小因子
                tx_c_ch = self.calculate_transmission_energy(distance_to_bs_for_ch, packet_size_bits * aggregated_packet_size_factor, is_tx_operation=True)
                total_cost_this_round += tx_c_ch
                node_data["tx_count"] += 1

            # 最终从节点扣除本轮总成本
            self.consume_node_energy(current_node_id, total_cost_this_round)
        
    
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

    def get_pdr_moving_average(self):
        """计算PDR的移动平均值。"""
        if not self.pdr_history:
            return 0.5 # 在仿真初期，返回一个中性的值
        return np.mean(self.pdr_history)
    
    def _calculate_isolated_node_rate(self):
        """[V6] 计算当前轮次结束时的孤立节点率（不含直连BS节点）。"""
        # 筛选出所有应该被分簇的活跃节点
        eligible_nodes = [
            n for n in self.nodes 
            if n["status"] == "active" and not n.get("can_connect_bs_directly", False)
        ]
        if not eligible_nodes:
            return 0.0
        
        # 从这些节点中，计算有多少是孤立的（未加入任何簇）
        isolated_count = sum(
            1 for n in eligible_nodes if n.get("cluster_id") == -1
        )
        
        return isolated_count / len(eligible_nodes)

    def _calculate_network_congestion_level(self):
        """[V6] 计算当前活跃CH的平均发送队列占用率作为拥塞指标。"""
        if not self.confirmed_cluster_heads_for_epoch or self.ch_forwarding_buffer_size <= 0:
            return 0.0

        total_occupancy_ratio = 0
        num_active_chs_with_buffer = 0
        
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            if self.nodes[ch_id]["status"] == "active":
                # 即使CH不在packets_in_transit中，也应将其视为占用率为0，计入平均值
                num_active_chs_with_buffer += 1
                if ch_id in self.packets_in_transit:
                    buffer = self.packets_in_transit[ch_id]
                    total_occupancy_ratio += len(buffer) / self.ch_forwarding_buffer_size

        return total_occupancy_ratio / num_active_chs_with_buffer if num_active_chs_with_buffer > 0 else 0.0
    
    def _add_packet_to_queue_with_aging(self, ch_id, packet):
            """[新辅助函数] 将数据包加入队列，如果队列满，则淘汰最老的包。"""
            if ch_id not in self.packets_in_transit:
                self.packets_in_transit[ch_id] = []
            
            buffer = self.packets_in_transit[ch_id]
            
            # 给包打上时间戳（年龄）
            packet['timestamp'] = self.current_round

            if len(buffer) < self.ch_forwarding_buffer_size:
                buffer.append(packet)
            else:
                # 队列已满，找到最老的包（时间戳最小）并替换
                buffer.sort(key=lambda p: p.get('timestamp', 0))
                oldest_packet = buffer.pop(0)
                buffer.append(packet)
                logger.warning(f"CH {ch_id} 队列满，新包挤掉了旧包 (UID: {oldest_packet.get('uid')})。")

    def _calculate_ch_coverage_gini(self):
        """计算当前活跃CH所覆盖成员数量的基尼系数。"""
        ch_members_counts = []
        if not self.confirmed_cluster_heads_for_epoch:
            return 0.0 # 没有CH，不存在不均衡

        for ch_id in self.confirmed_cluster_heads_for_epoch:
            if self.nodes[ch_id]["status"] == "active":
                count = len([node for node in self.nodes if node.get("cluster_id") == ch_id])
                ch_members_counts.append(count)
        
        if len(ch_members_counts) < 2:
            return 0.0 # 只有一个或没有CH，不存在不均衡

        # Gini coefficient calculation
        counts = np.array(sorted(ch_members_counts))
        n = len(counts)
        cum_counts = np.cumsum(counts)
        # G = (n + 1 - 2 * np.sum(cum_counts) / cum_counts[-1]) / n
        # 修正：避免在所有成员都为0时除以0
        total_members = cum_counts[-1]
        if total_members == 0:
            return 0.0
        
        G = (n + 1 - 2 * np.sum(cum_counts) / total_members) / n
        return G
    
    def _update_select_ch_q_tables(self):
        """
        [V11.0 最终版] 在Epoch结束时，根据上一个Epoch的真实交互结果，
        使用完整的模糊奖励机制，来更新普通节点的q_table_select_ch。
        """
        logger.info("Epoch结束，更新普通节点CH选择Q-table...")

        # 1. 预先计算一些在循环中会用到的、全局性的参考值
        avg_load_per_ch_ref = self.get_alive_nodes() / len(self.regular_chs_for_epoch) if self.regular_chs_for_epoch else 10
        AVG_E_SEND_REF_NORMAL_NODE = 0.001 # 这个可以保持为配置的参考值

        for node_data in self.nodes:
            # 只处理上一个Epoch做出了选择的普通节点
            if (node_data["status"] == "active" and 
                "last_epoch_choice" in node_data and 
                node_data["last_epoch_choice"]['ch_id'] != -1):
                
                node_id = node_data["id"]
                choice = node_data["last_epoch_choice"]
                chosen_ch_id = choice['ch_id']
                was_successful = choice['is_successful'] # 这是最重要的真实结果

                # 安全性检查：确保CH ID仍然有效
                if not (0 <= chosen_ch_id < len(self.nodes)):
                    continue
                
                ch_node_chosen = self.nodes[chosen_ch_id]

                # 2. [核心] 重新准备调用奖励函数所需的所有参数
                #    这些参数反映了那个CH在Epoch结束时的状态，或整个过程的真实消耗
                
                # a. 重新计算与该CH相关的模糊输入
                dist_to_chosen_ch = self.calculate_distance(node_id, chosen_ch_id)
                
                dc_base_chosen_ch = self.calculate_distance_to_base_station(chosen_ch_id)
                # 使用CH在Epoch结束时的能量状态，这更能反映它的可持续性
                e_cluster_chosen_ch_norm = ch_node_chosen["energy"] / ch_node_chosen["initial_energy"] if ch_node_chosen["initial_energy"] > 0 else 0
                
                # 使用CH在Epoch结束时的真实负载
                load_actual_chosen_ch = len([m for m in self.nodes if m.get("cluster_id") == chosen_ch_id])
                p_cluster_ratio_for_fuzzy = load_actual_chosen_ch / avg_load_per_ch_ref if avg_load_per_ch_ref > 0 else 0
                
                # 成功率和能耗可以是预估值，因为它们是物理特性
                r_success_with_chosen_ch_norm = node_data.get("history_success_with_ch", {}).get(chosen_ch_id, 0.9)
                expected_e_send = self.calculate_transmission_energy(dist_to_chosen_ch, 4000)
                e_send_total_ratio_for_fuzzy = expected_e_send / AVG_E_SEND_REF_NORMAL_NODE if AVG_E_SEND_REF_NORMAL_NODE > 0 else 0

                # b. 重新调用模糊逻辑专家，获取当时的决策权重
                fuzzy_weights = self.normal_node_fuzzy_logic.compute_weights(
                    current_dc_base=dc_base_chosen_ch,
                    current_e_cluster_normalized=e_cluster_chosen_ch_norm,
                    current_p_cluster_ratio_val=p_cluster_ratio_for_fuzzy, 
                    current_r_success_normalized=r_success_with_chosen_ch_norm,
                    current_e_send_total_ratio_val=e_send_total_ratio_for_fuzzy 
                )

                # c. [核心] 调用完整的奖励函数，传入真实结果
                final_reward = self.calculate_reward_for_selecting_ch(
                    node_id, chosen_ch_id, fuzzy_weights,
                    transmission_successful=was_successful, # <--- 使用真实的成功/失败结果
                    actual_energy_spent_tx=expected_e_send
                )
                
                # 3. 使用这个高质量的、基于真实结果的奖励来更新Q-table
                self.update_q_value_select_ch(node_id, chosen_ch_id, final_reward)

    def _ensure_ch_connectivity(self, preliminary_ch_set):
        """
        [V12.0 新增] 诊断初步CH集的连通性，并增选“桥接CH”来连接孤立的CH。
        返回: (增选的桥接CH列表, 最终的完整CH集合)
        """
        if not preliminary_ch_set:
            return [], set()

        # 1. 构建初步CH网络的图
        ch_graph = nx.Graph()
        ch_list = list(preliminary_ch_set)
        ch_graph.add_nodes_from(ch_list)
        for i in range(len(ch_list)):
            for j in range(i + 1, len(ch_list)):
                u, v = ch_list[i], ch_list[j]
                dist = self.calculate_distance(u, v)
                # 使用增强后的范围进行双向判断
                u_range = self.nodes[u].get("base_communication_range") * self.config.get('network', {}).get('ch_range_enhancement_factor', 1.0)
                v_range = self.nodes[v].get("base_communication_range") * self.config.get('network', {}).get('ch_range_enhancement_factor', 1.0)
                if dist <= u_range and dist <= v_range:
                    ch_graph.add_edge(u, v)

        # 2. 定义骨干网 (能直连BS的CH)
        backbone_chs = {ch_id for ch_id in ch_list if self.calculate_distance_to_base_station(ch_id) <= self.nodes[ch_id].get("base_communication_range") * self.config.get('network', {}).get('ch_range_enhancement_factor', 1.0)}
        if not backbone_chs:
            # 如果没有任何CH能直连BS，则认为离BS最近的那个是骨干网的起点
            if ch_list:
                closest_ch_to_bs = min(ch_list, key=lambda cid: self.calculate_distance_to_base_station(cid))
                backbone_chs.add(closest_ch_to_bs)
            else:
                return [], set() # 没有CH，无法操作

        # 3. 寻找所有孤立的CH（无法通过CH网络到达任何一个骨干CH）
        isolated_chs = []
        for ch_id in ch_list:
            is_connected_to_backbone = False
            for backbone_node in backbone_chs:
                if nx.has_path(ch_graph, ch_id, backbone_node):
                    is_connected_to_backbone = True
                    break
            if not is_connected_to_backbone:
                isolated_chs.append(ch_id)
        
        if not isolated_chs:
            return [], preliminary_ch_set # 网络已完全连通

        # 4. 为每个孤立的CH寻找并增选“桥接CH”
        bridge_chs_to_add = set()
        final_ch_set = set(preliminary_ch_set)

        # 找到所有非CH的活跃节点作为桥接候选人
        bridge_candidates = [
            n for n in self.nodes 
            if n["status"] == "active" and n["id"] not in final_ch_set
        ]

        for isolated_ch_id in isolated_chs:
            # 找到离这个孤立CH最近的骨干CH
            closest_backbone_ch = min(backbone_chs, key=lambda b_id: self.calculate_distance(isolated_ch_id, b_id))
            
            # 在孤立CH和最近的骨干CH之间的连线上寻找最佳“桥接点”
            pos_isolated = self.nodes[isolated_ch_id]["position"]
            pos_backbone = self.nodes[closest_backbone_ch]["position"]
            
            best_bridge_candidate = -1
            min_bridge_score = float('inf')

            for candidate in bridge_candidates:
                # a. 候选点必须在两者之间的一个大致区域内
                pos_candidate = candidate["position"]
                # (简单的几何判断，可以优化)
                
                # b. 候选点的评分 = 到孤立CH的距离 + 到骨干CH的距离
                # 我们希望选一个能同时连接两者的节点
                dist_to_iso = self.calculate_distance(candidate["id"], isolated_ch_id)
                dist_to_backbone = self.calculate_distance(candidate["id"], closest_backbone_ch)
                
                # 候选点必须有能力连接两者
                cand_range = candidate.get("base_communication_range") * self.config.get('network', {}).get('ch_range_enhancement_factor', 1.0)
                if dist_to_iso <= cand_range and dist_to_backbone <= cand_range:
                    bridge_score = dist_to_iso + dist_to_backbone
                    if bridge_score < min_bridge_score:
                        min_bridge_score = bridge_score
                        best_bridge_candidate = candidate["id"]

            if best_bridge_candidate != -1:
                bridge_chs_to_add.add(best_bridge_candidate)
                final_ch_set.add(best_bridge_candidate)
                # 将新加入的桥接CH也视为骨干网的一部分，以便后续的孤立CH可以连接到它
                backbone_chs.add(best_bridge_candidate) 
                # 从候选人中移除，防止被重复选择
                bridge_candidates = [c for c in bridge_candidates if c["id"] != best_bridge_candidate]

        return list(bridge_chs_to_add), final_ch_set

    def _validate_ch_routing_paths(self):
        """
        [FQ-DEEC 新增] 验证本轮所有CH的路由路径是否能最终到达BS。
        为每个CH节点设置一个新的标志 'can_route_to_bs_this_epoch'。
        """
        logger.debug("开始验证FQ-DEEC的CH路由路径...")
        
        # 遍历所有本轮的CH
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            ch_node = self.nodes[ch_id]
            
            # 默认都设置为False
            ch_node["can_route_to_bs_this_epoch"] = False

            if ch_node["status"] != "active":
                continue

            path = [ch_id]
            current_node_id = ch_id
            can_reach_bs = False
            
            # 沿着 chosen_next_hop_id 路径走，最多走CH的数量次（防止无限循环）
            for _ in range(len(self.confirmed_cluster_heads_for_epoch) + 2):
                if not (0 <= current_node_id < len(self.nodes)):
                    logger.warning(f"路径验证中遇到无效的节点ID: {current_node_id}")
                    break
                
                # 使用我们定义的NO_PATH_ID作为默认值
                next_hop_id = self.nodes[current_node_id].get("chosen_next_hop_id", self.NO_PATH_ID)
                
                if next_hop_id == self.BS_ID:
                    can_reach_bs = True
                    break
                
                if next_hop_id == self.NO_PATH_ID or next_hop_id in path:
                    break
                
                path.append(next_hop_id)
                current_node_id = next_hop_id
            
            # 设置标志
            ch_node["can_route_to_bs_this_epoch"] = can_reach_bs
            
            if can_reach_bs:
                logger.debug(f"  路径验证成功: CH {ch_id} 可以到达 BS。")
            else:
                logger.debug(f"  路径验证失败: CH {ch_id} 无法到达 BS。")

    def calculate_distance_to_point(self, node_id, point_coords):
        """计算节点到指定坐标点的距离。"""
        node = self.nodes[node_id]
        dx = node['position'][0] - point_coords[0]
        dy = node['position'][1] - point_coords[1]
        return math.sqrt(dx*dx + dy*dy)