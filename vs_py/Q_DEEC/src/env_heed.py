# src/env_heed.py

import random
import math
import logging
from env import WSNEnv  # 从你的核心env文件继承

logger = logging.getLogger("WSN_Simulation")

class WSNEnvHEED(WSNEnv):
    """
    一个实现经典HEED分簇协议的环境。
    它继承了WSNEnv的基础设施，但重写了核心决策方法。
    """
    def __init__(self, config_path=None, performance_log_path=None, ch_behavior_log_path=None):
        super().__init__(config_path, performance_log_path, ch_behavior_log_path)
        self.is_intelligent_agent = False
        logger.info("="*20)
        logger.info("HEED 环境已初始化。将使用基于能量和通信成本的迭代选举逻辑。")
        logger.info("="*20)


    def _run_heed_election(self):
        """实现经典的、迭代式的HEED簇头选举算法。"""
        #logger.info("开始HEED迭代选举...")
        
        # --- 1. 初始化阶段 ---
        heed_config = self.config.get('heed', {})
        p_min = heed_config.get('p_min', 1e-4) # 最小CH概率
        max_iterations = heed_config.get('max_iterations', 20) # 设置最大迭代次数防止死循环
        
        # 初始化每个节点的状态
        for node in self.nodes:
            if node['status'] == 'active':
                node['is_final_ch'] = False
                node['my_final_ch'] = -1
                # 计算初始CH_prob
                prob = self.p_opt_initial * (node['energy'] / node['initial_energy'])
                node['ch_prob'] = max(prob, p_min)

        # --- 2. 主迭代阶段 ---
        for i in range(max_iterations):
            logger.debug(f"HEED 迭代 {i+1}/{max_iterations}")
            
            # a. 宣告阶段
            tentative_ch_ids = set()
            for node in self.nodes:
                if node['status'] == 'active' and node['my_final_ch'] == -1 and not node['is_final_ch']:
                    if node['ch_prob'] >= 1.0:
                        node['is_final_ch'] = True
                        # 最终CH也广播自己
                        tentative_ch_ids.add(node['id'])
                    elif random.random() < node['ch_prob']:
                        tentative_ch_ids.add(node['id'])

            if not tentative_ch_ids:
                logger.debug("本轮迭代无暂定CH，可能所有节点已入簇。")
            
            # b. 节点选择阶段
            nodes_still_undecided = 0
            for node in self.nodes:
                if node['status'] == 'active' and node['my_final_ch'] == -1 and not node['is_final_ch']:
                    
                    # 寻找通信范围内的所有宣告者（最终的或暂定的）
                    best_ch_id = -1
                    min_cost = float('inf')
                    
                    neighbors = self.get_node_neighbors(node['id'], node['base_communication_range'])
                    # 将邻居和自身都加入潜在的CH检查列表
                    candidate_ch_sources = tentative_ch_ids.intersection(set(neighbors + [node['id']]))

                    for ch_id in candidate_ch_sources:
                        # HEED的成本函数：这里我们使用最经典的“度”作为成本，度越高成本越低
                        # 也可以使用距离，但度更能反映通信开销
                        cost = 1 / (1 + len(self.get_node_neighbors(ch_id, self.nodes[ch_id]['base_communication_range'])))
                        if cost < min_cost:
                            min_cost = cost
                            best_ch_id = ch_id
                    
                    if best_ch_id != -1:
                        node['my_final_ch'] = best_ch_id

                    nodes_still_undecided += 1

            # c. 检查是否所有节点都已完成决策
            if nodes_still_undecided == 0:
                #logger.info(f"HEED选举在第 {i+1} 轮迭代后收敛。")
                break

            # d. 更新概率
            for node in self.nodes:
                if node['status'] == 'active' and node['my_final_ch'] == -1 and not node['is_final_ch']:
                    node['ch_prob'] = min(node['ch_prob'] * 2, 1.0)
        else: # for-else循环，如果循环正常结束（未被break），则执行
            logger.warning(f"HEED选举达到最大迭代次数 {max_iterations}，可能部分节点仍未入簇。")

        # --- 3. 最终化阶段 ---
        final_ch_ids = []
        for node in self.nodes:
            if node['status'] == 'active':
                if node['is_final_ch']:
                    final_ch_ids.append(node['id'])
                # 将自己选为CH的节点也成为最终CH
                elif node['my_final_ch'] == node['id']:
                    final_ch_ids.append(node['id'])
        
        #logger.info(f"HEED选举结束，最终选出 {len(final_ch_ids)} 个CH。")
        return list(set(final_ch_ids)) #去重以防万一

    # HEED的节点分配和路由逻辑与DEEC类似，我们可以直接复用DEEC的简化实现
    # 或者，为了代码清晰，我们在这里也重写它们
    def _run_normal_node_selection_phase(self):
        """重写节点选择阶段，节点已经通过HEED迭代找到了自己的CH。"""
        #logger.info("HEED模式：根据选举结果更新节点cluster_id...")
        for node in self.nodes:
            if node['status'] == 'active' and node['role'] == 'normal':
                # 在HEED选举中，my_final_ch已经确定了归属
                node['cluster_id'] = node.get('my_final_ch', -1)

    def _run_heed_election_and_assignment(self):
        """
        [HEED专属-新增] 整合HEED的选举、角色更新和分配。
        """
        # 1. 执行HEED迭代选举
        final_ch_ids = self._run_heed_election()
        
        # 2. 更新最终的CH列表和节点角色
        self.confirmed_cluster_heads_for_epoch = final_ch_ids
        self._update_node_roles_and_timers(self.confirmed_cluster_heads_for_epoch)

        # 3. 分配普通节点
        self._run_normal_node_selection_phase()

    def _prepare_for_new_round(self):
        """
        [HEED专属-新增] 为新一轮做准备，重置节点状态。
        """
        self._build_spatial_index()
        
        self.sim_packets_generated_this_round = 0
        self.sim_packets_delivered_bs_this_round = 0
        self.sim_total_delay_this_round = 0.0
        self.sim_num_packets_for_delay_this_round = 0

        for node in self.nodes:
            if node["status"] == "active":
                # 重置HEED选举所需的状态
                node["is_final_ch"] = False
                node["my_final_ch"] = -1
                node["role"] = "normal"
                node["cluster_id"] = -1
                # 数据包生成
                self.sim_packets_generated_this_round += 1
                self.sim_packets_generated_total += 1
                node["has_data_to_send"] = True


    def _run_heed_routing_and_energy_consumption(self):
        """
        [HEED专属-新增] 合并路由决策、PDR统计和能耗计算。
        """
        # 1. 成员向CH发送数据并计算能耗
        packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        agg_cost_per_bit = self.config.get('energy', {}).get('aggregation_cost_per_bit', 5e-9)
        idle_listening_cost = float(self.config.get('energy', {}).get('idle_listening_per_round', 1e-6))
        sensing_cost = float(self.config.get('energy', {}).get('sensing_per_round', 5e-7))

        energy_costs = {node["id"]: idle_listening_cost + sensing_cost for node in self.nodes if node["status"] == "active"}
        packets_managed_by_ch = {ch_id: 0 for ch_id in self.confirmed_cluster_heads_for_epoch}

        # 成员 -> CH
        for node in self.nodes:
            if node['status'] == 'active' and node['role'] == 'normal' and node['cluster_id'] != -1:
                ch_id = node['cluster_id']
                if ch_id in packets_managed_by_ch:
                    packets_managed_by_ch[ch_id] += 1
                    dist = self.calculate_distance(node['id'], ch_id)
                    energy_costs[node['id']] = energy_costs.get(node['id'], 0) + self.calculate_transmission_energy(dist, packet_size)
                    energy_costs[ch_id] = energy_costs.get(ch_id, 0) + self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)

        # CH自身的数据包
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            packets_managed_by_ch[ch_id] += 1
        
        # 2. CH路由决策与能耗计算
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            ch_node = self.nodes[ch_id]
            if ch_node['status'] != 'active': continue

            # 融合能耗
            num_packets = packets_managed_by_ch.get(ch_id, 0)
            if num_packets > 0:
                energy_costs[ch_id] = energy_costs.get(ch_id, 0) + num_packets * agg_cost_per_bit * packet_size

            # 贪婪地理路由决策
            my_dist_to_bs = self.calculate_distance_to_base_station(ch_id)
            best_next_hop_id = self.BS_ID
            min_dist_to_bs_of_nh = my_dist_to_bs
            
            neighbors = self.get_node_neighbors(ch_id, ch_node['base_communication_range'])
            for neighbor_id in neighbors:
                if neighbor_id in self.confirmed_cluster_heads_for_epoch:
                    neighbor_dist_to_bs = self.calculate_distance_to_base_station(neighbor_id)
                    if neighbor_dist_to_bs < min_dist_to_bs_of_nh:
                        min_dist_to_bs_of_nh = neighbor_dist_to_bs
                        best_next_hop_id = neighbor_id

            # PDR统计和路由能耗计算
            can_reach_bs = (best_next_hop_id == self.BS_ID and my_dist_to_bs <= ch_node['base_communication_range']) or (best_next_hop_id != self.BS_ID)
            
            if can_reach_bs:
                if num_packets > 0:
                    self.sim_packets_delivered_bs_this_round += num_packets
                    self.sim_packets_delivered_bs_total += num_packets
                
                # 计算发送能耗
                dist_to_nh = self.calculate_distance(ch_id, best_next_hop_id) if best_next_hop_id != self.BS_ID else my_dist_to_bs
                energy_costs[ch_id] = energy_costs.get(ch_id, 0) + self.calculate_transmission_energy(dist_to_nh, packet_size)
                # 计算下一跳接收能耗
                if best_next_hop_id != self.BS_ID:
                    energy_costs[best_next_hop_id] = energy_costs.get(best_next_hop_id, 0) + self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)

        # 3. 统一扣除能量
        for node_id, cost in energy_costs.items():
            if cost > 0:
                self.consume_node_energy(node_id, cost)


    def step(self, current_round_num):
        """
        [HEED专属-修改版] 重写 step 函数，让HEED选举每轮都发生。
        """
        self.current_round = current_round_num
        logger.info(f"--- 开始第 {self.current_round} 轮 (HEED模式-逐轮选举) ---")

        self._prepare_for_new_round()
        
        # 选举和分配
        self._run_heed_election_and_assignment()

        # 路由
        # HEED的路由逻辑比较简单，可以合并到能耗计算中
        self._run_heed_routing_and_energy_consumption()

        # 日志
        self._update_and_log_performance_metrics()
        
        logger.info(f"--- 第 {self.current_round} 轮结束 (HEED模式-逐轮选举) ---")
        
        if self.get_alive_nodes() == 0:
            logger.info("HEED网络中所有节点均已死亡，仿真结束。")
            return False
        
        return True