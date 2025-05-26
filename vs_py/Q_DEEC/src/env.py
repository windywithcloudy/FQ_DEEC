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
    
    def _calculate_location_factor(self, distance_to_bs):
        """
        计算基于节点到基站距离的位置因子。
        Args:
            distance_to_bs (float): 节点到基站的距离。
        Returns:
            float: 位置因子 (0-1]。
        """
        if not self.location_factor_enabled:
            return 1.0 # 如果未启用，则不影响概率

        factor = 1.0
        # 归一化距离或使用绝对距离进行判断
        # 这里使用绝对距离判断，参数来自config

        if distance_to_bs < self.optimal_bs_dist_min:
            # 线性增加：从 penalty_factor_too_close (在d=0时) 到 1 (在 optimal_bs_dist_min 时)
            # factor = self.penalty_factor_too_close + \
            #          (1.0 - self.penalty_factor_too_close) * (distance_to_bs / self.optimal_bs_dist_min) \
            #          if self.optimal_bs_dist_min > 0 else self.penalty_factor_too_close
            # 或者简单惩罚
            factor = self.penalty_factor_too_close
        elif distance_to_bs > self.optimal_bs_dist_max:
            # 线性减少：从 1 (在 optimal_bs_dist_max 时) 到 penalty_factor_too_far (在网络最大距离时)
            # network_diagonal = 250 * np.sqrt(2) # 需要获取或定义网络最大可能距离
            # if distance_to_bs < network_diagonal and network_diagonal > self.optimal_bs_dist_max:
            #     factor = self.penalty_factor_too_far + \
            #              (1.0 - self.penalty_factor_too_far) * \
            #              max(0, (network_diagonal - distance_to_bs) / (network_diagonal - self.optimal_bs_dist_max))
            # else: # 如果超出网络范围或无法计算斜率
            #     factor = self.penalty_factor_too_far
            # 或者简单惩罚
            factor = self.penalty_factor_too_far
        
        # 确保因子在合理范围内 (例如，不小于0，不大于1)
        return max(0.1, min(1.0, factor)) # 最小给个0.1避免完全抑制
    
    def deec_candidate_election(self):
        """
        执行DEEC协议选举候选簇头，并进行初步的数量和空间分布优化。
        此阶段不进行成员分配，只确定哪些节点有潜力成为CH。
        """
        logger.info(f"第 {self.current_round} 轮：开始DEEC候选簇头选举 (p_opt_current: {self.p_opt_current:.3f})...")
        self.candidate_cluster_heads = [] # 清空上一轮的候选
        
        for node in self.nodes:
            if node["status"] == "active":
                 node["cluster_id"] = -1 # 清除上一轮的分配信息

        num_alive_nodes = self.get_alive_nodes()
        if num_alive_nodes == 0:
            logger.info("没有存活节点，无法选举候选簇头。")
            return

        current_avg_energy = self._calculate_current_average_energy()
        if current_avg_energy <= 0: # 应该在有存活节点时 > 0，但以防万一
            logger.warning("当前网络平均能量为0或负，无法计算DEEC概率。")
            # 强制选择 (如果需要)
            if num_alive_nodes > 0:
                highest_energy_node = max((n for n in self.nodes if n["status"] == "active"), key=lambda x: x["energy"], default=None)
                if highest_energy_node:
                    self.candidate_cluster_heads.append(highest_energy_node["id"])
                    logger.info(f"由于平均能量问题，强制选择能量最高的节点 {highest_energy_node['id']} 作为候选簇头。")
            return

        # 期望的候选簇头数量
        num_expected_candidates = self.p_opt_current * num_alive_nodes
        
        # 1. 识别不参与CH选举的节点 (例如直连BS的)
        nodes_for_ch_election_pool = [
            n for n in self.nodes 
            if n["status"] == "active" and not n.get("can_connect_bs_directly", False)
        ]
        if not nodes_for_ch_election_pool:
            logger.info("没有符合CH选举资格的节点 (可能都直连BS或死亡)。")
            self.candidate_cluster_heads = []
            return

        temp_ch_candidates_ids_prob = []
        for node in nodes_for_ch_election_pool:
            base_prob_i = self.p_opt_current * (node["energy"] / current_avg_energy) if current_avg_energy > 0 else 0
            
            # 计算并应用位置因子
            distance_to_bs = self.calculate_distance2_bs(node["id"])
            location_f = self._calculate_location_factor(distance_to_bs)
            
            final_prob_i = base_prob_i * location_f
            final_prob_i = min(1.0, max(0.0, final_prob_i)) # 确保概率在 [0,1]
            
            logger.debug(f"节点 {node['id']}: E={node['energy']:.2f}, d_bs={distance_to_bs:.1f}, base_P={base_prob_i:.3f}, loc_F={location_f:.2f}, final_P={final_prob_i:.3f}")

            if random.random() < final_prob_i:
                temp_ch_candidates_ids_prob.append(node["id"])
        
        logger.debug(f"DEEC概率选举 (含位置因子)：初步产生 {len(temp_ch_candidates_ids_prob)} 个候选CH意向: {temp_ch_candidates_ids_prob}")

        # 2. 如果有候选，进行数量和能量筛选
        current_candidates_after_initial_filter = []
        if temp_ch_candidates_ids_prob: 
            temp_ch_candidates_ids_prob.sort(key=lambda id_val: self.nodes[id_val]["energy"], reverse=True)
            # 调整期望候选数量的计算基数，应为参与选举的节点数
            num_eligible_for_election = len(nodes_for_ch_election_pool)
            effective_expected_candidates = self.p_opt_current * num_eligible_for_election

            max_chs_to_select = max(1, int(effective_expected_candidates * 1.5)) 
            current_candidates_after_initial_filter = temp_ch_candidates_ids_prob[:max_chs_to_select]
            if len(temp_ch_candidates_ids_prob) > max_chs_to_select:
                 logger.debug(f"候选CH过多，筛选至 {len(current_candidates_after_initial_filter)} 个 (按能量)。")


        # 3. 基于距离的冗余候选CH移除 (空间分布优化)
        final_refined_candidates = []
        if current_candidates_after_initial_filter:
            # d_min_ch_dist 可以从配置读取或设为固定值
            # 确保 communication_range 是数字
            comm_range = self.config.get('network', {}).get('communication_range', 100.0)
            if not isinstance(comm_range, (int, float)): comm_range = 100.0 # fallback
            
            min_ch_dist_factor = self.config.get('deec', {}).get('min_inter_ch_distance_factor', 0.5)
            d_min_ch_dist = min_ch_dist_factor * comm_range
            logger.debug(f"空间分布优化：最小CH间距 d_min_ch_dist = {d_min_ch_dist:.2f}m")

            # 候选者已经按能量排序过了，所以能量高的会先进入 final_refined_candidates
            for cand_id in current_candidates_after_initial_filter:
                node_cand = self.nodes[cand_id]
                # 确保节点仍是活跃的 (虽然eligible_nodes_for_ch应该已经筛选过了)
                if node_cand["status"] != "active": 
                    continue

                is_too_close = False
                for final_cand_id in final_refined_candidates:
                    # final_cand_id 已经是活跃的，因为它被加入了 final_refined_candidates
                    dist = self.calculate_distance(cand_id, final_cand_id)
                    if dist < d_min_ch_dist:
                        is_too_close = True
                        logger.debug(f"候选CH {cand_id} (能量 {node_cand['energy']:.2f}) 因与已选最终候选 {final_cand_id} 距离 ({dist:.2f}m) 过近 (<{d_min_ch_dist:.2f}m) 而被考虑移除。")
                        break
                if not is_too_close:
                    final_refined_candidates.append(cand_id)
            
            self.candidate_cluster_heads = final_refined_candidates
            logger.info(f"DEEC选举：初步筛选剩 {len(current_candidates_after_initial_filter)} 个，距离优化后最终 {len(self.candidate_cluster_heads)} 个候选CH: {self.candidate_cluster_heads}")
        else: # 如果 initial filter 后就没有候选了
            self.candidate_cluster_heads = []


        # 4. 如果最终没有选出任何候选CH，则强制选择一个
        if not self.candidate_cluster_heads and num_alive_nodes > 0:
            logger.warning("本轮未能通过DEEC（包括优化）选出任何候选簇头。")
            highest_energy_node = None
            max_energy = -1
            for node_data_val in self.nodes: # 使用不同的变量名
                if node_data_val["status"] == "active" and node_data_val["energy"] > max_energy:
                    max_energy = node_data_val["energy"]
                    highest_energy_node = node_data_val
            
            if highest_energy_node:
                self.candidate_cluster_heads.append(highest_energy_node["id"])
                logger.info(f"强制选择能量最高的节点 {highest_energy_node['id']} (能量 {highest_energy_node['energy']:.2f}J) 作为候选簇头。")
            else:
                logger.error("严重错误：有存活节点但无法找到能量最高的节点进行强制CH选择。")

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

                d_to_bs = self.calculate_distance2_bs(node["id"])
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
    
    def simulate_round_energy_consumption(self):
        """模拟一轮中由于基本操作（如感知、空闲监听）产生的能量消耗"""
        # 这是一个非常简化的模型，你可以根据需要调整
        energy_cfg = self.config.get('energy', {})
        try:
            idle_listening_cost_per_round = float(energy_cfg.get('idle_listening_per_round', 1e-5))
            sensing_cost_per_round = float(energy_cfg.get('sensing_per_round', 5e-6))
        except (ValueError, TypeError):
            logger.error("配置文件中的 idle_listening_per_round 或 sensing_per_round 不是有效的数字。请检查 config.yml。")
            idle_listening_cost_per_round = 1e-5
            sensing_cost_per_round = 5e-6
        
        nodes_to_kill_this_round = []
        for node in self.nodes:
            if node["status"] == "active":
                cost = idle_listening_cost_per_round + sensing_cost_per_round
                if not isinstance(node["energy"], (int, float)): 
                    logger.warning(f"节点 {node['id']} 的能量值类型不正确: {node['energy']} (类型: {type(node['energy'])}). 跳过能耗计算。")
                    continue
                if not isinstance(cost, (int, float)): 
                    logger.warning(f"计算得到的能耗 cost 类型不正确: {cost} (类型: {type(cost)}). 跳过能耗计算。")
                    continue
                
                node["energy"] -= cost
                if node["energy"] < 0:
                    node["energy"] = 0
                    nodes_to_kill_this_round.append(node["id"])
        
        for node_id in nodes_to_kill_this_round:
            if self.nodes[node_id]["status"] == "active":
                 self.kill_node(node_id)
        
    def calculate_distance(self, node1_idx, node2_idx):
        """计算两个节点之间的欧氏距离 (基于节点ID)"""
        # 修改为接受节点ID，或直接接受节点字典
        node1 = self.nodes[node1_idx]
        node2 = self.nodes[node2_idx]
        x1, y1 = node1["position"]
        x2, y2 = node2["position"]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def calculate_distance2_bs(self, node1_idx):
        node1 = self.nodes[node1_idx]
        x1, y1 = node1["position"]
        return ((x1 - 250) ** 2 + (y1 - 250) ** 2) ** 0.5

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

    def step(self, current_round_num):
        """执行一个仿真轮次的逻辑"""
        self.current_round = current_round_num
        logger.info(f"--- 开始第 {self.current_round} 轮 ---")
        self.identify_direct_bs_nodes()
        for node in self.nodes:
            if node["status"] == "active":
                 node["current_communication_range"] = node["base_communication_range"]
        
        
        # 1. DEEC簇头选举
        self.deec_candidate_election()

        # === 阶段2: 普通节点使用Q学习选择簇头 ===
        logger.info("开始阶段2：普通节点Q学习选择簇头...")
        num_nodes_assigned_by_q_learning = 0
        if self.candidate_cluster_heads:
            for node_data in self.nodes:
                if node_data["status"] == "active" and node_data["role"] == "normal":
                    # TODO: 在这里集成普通节点的Q学习选择CH的逻辑
                    # 1. 获取该普通节点的Q学习Agent (或模糊Q学习系统实例)
                    # 2. 普通节点感知 self.candidate_cluster_heads 列表中的候选CH
                    # 3. 对每个候选CH，收集其状态信息，调用模糊逻辑得到权重
                    # 4. 计算选择该候选CH的Q值或即时奖励
                    # 5. 根据Q值和探索策略选择一个CH
                    # 6. 如果选择了CH，则 node_data["cluster_id"] = chosen_ch_id
                    #    num_nodes_assigned_by_q_learning += 1
                    #    被选中的CH的负载信息需要更新 (如果Q学习或模糊逻辑的输入包含负载)
                    
                    # ---- 临时占位符：随机选择一个可达的候选CH ----
                    # ---- 你需要用你的Q学习逻辑替换这里 ----
                    eligible_chs_for_node = []
                    for ch_id in self.candidate_cluster_heads:
                        if self.nodes[ch_id]["status"] == "active": #确保候选CH是活的
                            distance = self.calculate_distance(node_data["id"], ch_id)
                            if distance <= node_data["current_communication_range"] and \
                               distance <= self.nodes[ch_id]["current_communication_range"]:
                                eligible_chs_for_node.append(ch_id)
                    if eligible_chs_for_node:
                        chosen_ch_id = random.choice(eligible_chs_for_node)
                        node_data["cluster_id"] = chosen_ch_id
                        num_nodes_assigned_by_q_learning +=1
                        logger.debug(f"节点 {node_data['id']} (Q学习占位符-随机)选择了候选CH {chosen_ch_id}")
                    else:
                        logger.debug(f"节点 {node_data['id']} (Q学习占位符) 未能找到可达的候选CH。")
                    # ---- Q学习占位符结束 ----
            logger.info(f"Q学习选择阶段：{num_nodes_assigned_by_q_learning} 个节点尝试了选择CH。")
        else:
            logger.warning(f"第 {self.current_round} 轮：没有候选簇头，普通节点无法通过Q学习选择。")

        # 确定最终的CH列表 (那些至少有一个成员通过Q学习加入的候选CH，或者所有候选CH都算？)
        # 简单起见，我们先假设所有被选为候选的，并且仍然存活的，就是本轮的活跃CH
        # 但实际上，如果一个候选CH没有吸引到任何成员，它可能不应该扮演CH的角色消耗能量
        # 这一步的逻辑需要根据你的协议设计来确定。
        # 我们可以统计哪些候选CH被普通节点选择了：
        final_active_chs_ids = set()
        for node_data in self.nodes:
            if node_data["status"] == "active" and node_data["role"] == "normal" and node_data["cluster_id"] != -1:
                final_active_chs_ids.add(node_data["cluster_id"])
        
        # 更新节点的role，只有被选中的候选CH才最终成为"cluster_head"
        for node_data in self.nodes:
            if node_data["id"] in final_active_chs_ids and node_data["status"] == "active":
                node_data["role"] = "cluster_head"
            elif node_data["status"] == "active": # 其他活节点（包括未被选为CH的候选者）都是普通节点
                node_data["role"] = "normal"
        
        logger.info(f"Q学习分配后，最终活跃簇头 ({len(final_active_chs_ids)}个): {list(final_active_chs_ids)}")


        # === 阶段2之后: 处理仍然孤立的普通节点 ===
        isolated_normal_nodes_after_q_learning = 0
        nodes_needing_assignment_after_q = []
        for node_data in self.nodes:
            if node_data["status"] == "active" and node_data["role"] == "normal" and node_data["cluster_id"] == -1:
                isolated_normal_nodes_after_q_learning += 1
                nodes_needing_assignment_after_q.append(node_data)
        
        logger.info(f"Q学习选择后，有 {isolated_normal_nodes_after_q_learning} 个普通节点仍然孤立。")

        if isolated_normal_nodes_after_q_learning > 0 and final_active_chs_ids: # 只有当有孤立节点且有活跃CH时才尝试
            # 尝试使用增加通信范围的 assign_nodes_to_clusters 作为备用策略
            max_backup_attempts = 2 # 最多额外尝试2次
            for attempt in range(1, max_backup_attempts + 1):
                logger.info(f"对Q学习后孤立的节点进行第 {attempt} 次备用分配尝试...")
                # 只对仍然孤立的节点进行操作，使用最终活跃的CH作为候选
                newly_isolated_after_this_attempt = self.assign_nodes_to_clusters(
                    attempt=attempt + 1, # attempt参数控制通信范围增加，所以从2开始
                    nodes_to_assign=[n for n in nodes_needing_assignment_after_q if n["cluster_id"] == -1],
                    candidate_chs=list(final_active_chs_ids)
                )
                if newly_isolated_after_this_attempt == 0:
                    logger.info("所有Q学习后孤立的节点已通过备用策略分配。")
                    break
            isolated_normal_nodes_after_q_learning = newly_isolated_after_this_attempt


        # === 阶段3: CH使用Q学习选择下一跳 (TODO) ===
        # for ch_id in final_active_chs_ids:
        #     # CH节点执行其Q学习逻辑选择下一跳...
        #     pass

        # --- 数据传输模拟 (TODO) ---
        
        # 调整下一轮的p_opt
        self.adjust_p_opt_for_next_round(isolated_normal_nodes_after_q_learning)
        
        # 模拟本轮的基础能量消耗
        self.simulate_round_energy_consumption()
        
        logger.info(f"--- 第 {self.current_round} 轮结束 ---")
        alive_nodes = self.get_alive_nodes()
        final_ch_count_this_round = len([n for n in self.nodes if n["status"]=="active" and n["role"]=="cluster_head"])
        logger.info(f"轮次结束时存活节点数: {alive_nodes}, 最终活跃CH数: {final_ch_count_this_round}")
        if alive_nodes == 0:
            logger.info("所有节点已死亡，仿真可以提前结束。")
            return False 
        return True

    def _get_packet_loss_rate(self, distance):
        """基于距离的Log-normal阴影模型"""
        PL_d0 = 55  # 参考距离d0=1m时的路径损耗(dB)
        path_loss = PL_d0 + 10 * 3.0 * np.log10(distance) + np.random.normal(0, 4)
        snr = 10 - path_loss  # 假设发射功率10dBm
        return 1 / (1 + np.exp(snr - 5))  # Sigmoid模拟丢包率
