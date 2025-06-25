# src/env_q_deec.py

import logging
import random
import numpy as np
from env_deec import WSNEnvDEEC # 关键：继承我们修改好的、逐轮选举的DEEC环境

logger = logging.getLogger("WSN_Simulation")

class WSNEnvQDEEC(WSNEnvDEEC):
    """
    一个实现了经典分布式Q-learning路由协议 (Q-DEEC) 的环境。
    - CH选举: 与DEEC相同，采用每轮基于能量的概率选举。
    - CH路由: 每个CH作为独立的智能体，在每一轮使用Q-learning从邻居CH和BS中选择下一跳。
    """
    def __init__(self, config_path=None, performance_log_path=None, ch_behavior_log_path=None):
        # 首先调用父类的构造函数(WSNEnvDEEC -> WSNEnv)
        super().__init__(config_path, performance_log_path, ch_behavior_log_path)
        self.is_intelligent_agent = True # 标记这是一个智能体协议
        
        logger.info("="*20)
        logger.info("Q-DEEC (纯分布式RL) 环境已初始化。")
        logger.info(" - CH选举: 继承DEEC的逐轮概率选举。")
        logger.info(" - CH路由: 每轮使用Q-learning独立决策。")
        logger.info("="*20)

    # _prepare_for_new_round 和 _run_deec_election_and_assignment 方法将从 WSNEnvDEEC 继承，无需重写
    # step 方法也从 WSNEnvDEEC 继承，但我们需要重写路由和能耗计算部分

    def step(self, current_round_num):
        """
        [Q-DEEC专属] 重写 step 函数，以调用Q-DEEC的特定逻辑流程。
        """
        self.current_round = current_round_num
        logger.debug(f"--- 开始第 {self.current_round} 轮 (Q-DEEC模式) ---")

        # 阶段 0: 准备工作 (继承自DEEC，每轮重置角色)
        self._prepare_for_new_round()

        # 阶段 1: 选举和分配 (继承自DEEC，每轮执行)
        self._run_deec_election_and_assignment()
        
        # 阶段 2: 路由决策与数据传输 (Q-DEEC的核心)
        # 我们将路由、能耗、PDR统计都整合到这个函数中
        self._run_q_learning_routing_phase()

        # 阶段 3: 更新并记录本轮的性能指标
        self._update_and_log_performance_metrics()
        
        logger.debug(f"--- 第 {self.current_round} 轮结束 (Q-DEEC模式) ---")
        
        if self.get_alive_nodes() == 0:
            logger.info("Q-DEEC网络中所有节点均已死亡，仿真结束。")
            return False
            
        return True

    def _run_q_learning_routing_phase(self):
        """
        [Q-DEEC核心] 实现数据融合、Q-learning决策、传输、奖励计算和能耗统计。
        """
        packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        agg_cost_per_bit = self.config.get('energy', {}).get('aggregation_cost_per_bit', 5e-9)
        idle_listening_cost = float(self.config.get('energy', {}).get('idle_listening_per_round', 1e-6))
        sensing_cost = float(self.config.get('energy', {}).get('sensing_per_round', 5e-7))

        # 初始化本轮能耗字典
        energy_costs = {node["id"]: idle_listening_cost + sensing_cost for node in self.nodes if node["status"] == "active"}

        # 1. 成员向CH发送数据并计算能耗
        for node in self.nodes:
            if node['status'] == 'active' and node['role'] == 'normal' and node['cluster_id'] != -1:
                ch_id = node['cluster_id']
                if ch_id < len(self.nodes) and self.nodes[ch_id]['status'] == 'active':
                    # 普通节点发送能耗
                    dist = self.calculate_distance(node['id'], ch_id)
                    energy_costs[node['id']] = energy_costs.get(node['id'], 0) + self.calculate_transmission_energy(dist, packet_size)
                    
                    # CH接收能耗
                    rx_energy = self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)
                    energy_costs[ch_id] = energy_costs.get(ch_id, 0) + rx_energy
                    
                    # 准备数据包
                    if ch_id not in self.packets_in_transit:
                        self.packets_in_transit[ch_id] = []
                    new_packet = {"source_node": node['id'], "gen_round": self.current_round}
                    self.packets_in_transit[ch_id].append(new_packet)

        # 2. CH自身生成数据包，并进行数据融合
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            ch_node = self.nodes[ch_id]
            if ch_node['status'] == 'active':
                if ch_id not in self.packets_in_transit:
                    self.packets_in_transit[ch_id] = []
                # CH自身的数据
                self.packets_in_transit[ch_id].append({"source_node": ch_id, "gen_round": self.current_round})
                
                # 融合能耗
                num_packets = len(self.packets_in_transit.get(ch_id, []))
                if num_packets > 0:
                    energy_costs[ch_id] = energy_costs.get(ch_id, 0) + num_packets * agg_cost_per_bit * packet_size

        # 3. CH进行Q-learning决策和传输 (按离BS从远到近的顺序)
        sorted_chs = sorted(
            self.confirmed_cluster_heads_for_epoch,
            key=lambda cid: self.calculate_distance_to_base_station(cid),
            reverse=True
        )

        for ch_id in sorted_chs:
            ch_node = self.nodes[ch_id]
            # 必须是活跃的，并且有数据要发
            if ch_node['status'] != 'active' or not self.packets_in_transit.get(ch_id):
                continue
            
            # a. 寻找所有可能的下一跳 (邻居CH + BS)
            # 注意：这里的邻居CH必须是本轮当选的CH
            candidate_next_hops_info = self._find_candidate_next_hops(ch_id) # 复用父类的方法

            if not candidate_next_hops_info:
                # 没有下一跳，数据包丢失
                self.packets_in_transit[ch_id] = [] # 清空缓冲区
                continue

            # b. Epsilon-Greedy决策
            chosen_next_hop_id = self.NO_PATH_ID
            epsilon_cfg = self.config.get('q_learning', {}).get('ch_select_next_hop', {})
            epsilon = ch_node.get("epsilon_select_next_hop", epsilon_cfg.get('initial', 0.2))

            if random.random() > epsilon:
                # Greed: 选择Q值最高的
                q_values = {nh_id: self.get_q_value_select_next_hop(ch_id, nh_id) for nh_id, _, _ in candidate_next_hops_info}
                chosen_next_hop_id = max(q_values, key=lambda k:q_values[k])
            else:
                # Exploration: 随机选择一个
                candidate_ids = [nh_id for nh_id, _, _ in candidate_next_hops_info]
                chosen_next_hop_id = random.choice(candidate_ids)
            
            # 更新epsilon
            decay = epsilon_cfg.get('decay_per_round', 0.998)
            min_eps = epsilon_cfg.get('min', 0.01)
            ch_node["epsilon_select_next_hop"] = max(min_eps, epsilon * decay)

            # c. 执行传输
            num_packets_to_send = len(self.packets_in_transit[ch_id])
            is_terminal = (chosen_next_hop_id == self.BS_ID)
            
            # 计算发送能耗
            dist_to_nh = self.calculate_distance(ch_id, chosen_next_hop_id) if not is_terminal else self.calculate_distance_to_base_station(ch_id)
            tx_energy = self.calculate_transmission_energy(dist_to_nh, packet_size) * num_packets_to_send
            energy_costs[ch_id] = energy_costs.get(ch_id, 0) + tx_energy
            
            # PDR 和 下一跳接收处理
            if is_terminal:
                self.sim_packets_delivered_bs_this_round += num_packets_to_send
                self.sim_packets_delivered_bs_total += num_packets_to_send
            else: # 如果下一跳是另一个CH
                if chosen_next_hop_id < len(self.nodes) and self.nodes[chosen_next_hop_id]['status'] == 'active':
                    # 接收能耗
                    rx_energy = self.calculate_transmission_energy(0, packet_size, is_tx_operation=False) * num_packets_to_send
                    energy_costs[chosen_next_hop_id] = energy_costs.get(chosen_next_hop_id, 0) + rx_energy
                    # 将数据包放入下一跳的缓冲区
                    if chosen_next_hop_id not in self.packets_in_transit: self.packets_in_transit[chosen_next_hop_id] = []
                    self.packets_in_transit[chosen_next_hop_id].extend(self.packets_in_transit[ch_id])
            
            # d. 计算奖励和更新Q-table
            # 使用父类(WSNEnv)的复杂奖励计算函数，无需模糊逻辑
            reward, _, _ = self._calculate_routing_reward_and_next_state(
                ch_id, chosen_next_hop_id, "CH" if not is_terminal else "BS",
                dist_to_nh, True, candidate_next_hops_info
            )
            
            # 获取下一状态的最大Q值
            max_q_for_next_state = 0.0
            if not is_terminal and chosen_next_hop_id < len(self.nodes) and self.nodes[chosen_next_hop_id]['status'] == 'active':
                next_hop_candidates = self._find_candidate_next_hops(chosen_next_hop_id)
                if next_hop_candidates:
                    q_values_for_next_hop = [self.get_q_value_select_next_hop(chosen_next_hop_id, nh_id) for nh_id, _, _ in next_hop_candidates]
                    if q_values_for_next_hop:
                        max_q_for_next_state = max(q_values_for_next_hop)
            
            # 更新Q-table
            self.update_q_value_select_next_hop(
                ch_id, chosen_next_hop_id, reward, max_q_for_next_state, is_terminal
            )
            
            # 清空当前CH的缓冲区
            self.packets_in_transit[ch_id] = []

        # 4. 统一扣除所有能耗
        for node_id, cost in energy_costs.items():
            if cost > 0:
                self.consume_node_energy(node_id, cost)
    
    # in src/env_q_deec.py -> class WSNEnvQDEEC
    def _calculate_routing_reward_and_next_state(self, ch_id, chosen_nh_id, chosen_nh_type, 
                                                 dist_to_nh, success_flag, all_candidates_info):
        """
        [Q-DEEC专属-重写] 计算一个简化的、不依赖模糊逻辑的奖励。
        """
        # --- 获取基础奖励/惩罚单位值 (从config) ---
        reward_cfg = self.config.get('rewards', {}).get('ch_select_next_hop_simple', {})
        R_REACH_BS = float(reward_cfg.get('reach_bs_bonus', 100))
        R_FAIL_TRANSMISSION = float(reward_cfg.get('transmission_fail_penalty', -100))
        K_PROGRESS = float(reward_cfg.get('data_progress_unit', 1.0))
        K_ENERGY_COST = float(reward_cfg.get('energy_cost_penalty_unit', 2000))
        K_NH_LOW_ENERGY = float(reward_cfg.get('next_hop_low_energy_penalty', -20))
        K_NH_HIGH_LOAD = float(reward_cfg.get('next_hop_high_load_penalty', -15))

        # --- 1. 传输结果奖励/惩罚 ---
        if not success_flag:
            return R_FAIL_TRANSMISSION, [], False # 传输失败，直接返回大惩罚

        total_reward = 0
        if chosen_nh_type == "BS":
            total_reward += R_REACH_BS

        # --- 2. 路径进展奖励 ---
        dist_ch_to_bs = self.calculate_distance_to_base_station(ch_id)
        dc_bs_of_nh = self.calculate_distance_to_base_station(chosen_nh_id) if chosen_nh_type != "BS" else 0
        data_advanced_amount = dist_ch_to_bs - dc_bs_of_nh
        total_reward += data_advanced_amount * K_PROGRESS

        # --- 3. 能耗惩罚 ---
        packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        actual_e_tx_to_nh = self.calculate_transmission_energy(dist_to_nh, packet_size)
        total_reward -= actual_e_tx_to_nh * K_ENERGY_COST

        # --- 4. 下一跳状态惩罚 ---
        if chosen_nh_type == "CH":
            nh_node = self.nodes[chosen_nh_id]
            # a. 能量惩罚
            nh_energy_norm = nh_node['energy'] / nh_node['initial_energy'] if nh_node['initial_energy'] > 0 else 0
            if nh_energy_norm < 0.3: # 如果下一跳能量低
                total_reward += K_NH_LOW_ENERGY * (1 - nh_energy_norm / 0.3)
            
            # b. 负载惩罚
            nh_load = len(self.packets_in_transit.get(chosen_nh_id, []))
            if nh_load > self.ch_forwarding_buffer_size * 0.7: # 如果下一跳缓存满
                total_reward += K_NH_HIGH_LOAD * (nh_load / self.ch_forwarding_buffer_size)

        is_terminal = (chosen_nh_type == "BS")
        return total_reward, [], is_terminal