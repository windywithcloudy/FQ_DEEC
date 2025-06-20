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

    def _run_epoch_start_phase(self):
        """重写Epoch开始阶段，执行完整的HEED选举流程。"""
        logger.info(f"--- ***** 新 Epoch 开始 (轮次 {self.current_round}) - HEED模式 ***** ---")
        
        # 1. HEED选举逻辑
        final_ch_ids = self._run_heed_election()
        
        # 2. 更新最终的CH列表和节点角色
        self.confirmed_cluster_heads_for_epoch = final_ch_ids
        self._update_node_roles_and_timers(self.confirmed_cluster_heads_for_epoch)

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

    # in env_heed.py -> class WSNEnvHEED

    def _run_ch_routing_phase(self):
        """
        [HEED专用 V3 - 最终修复版]
        先用贪婪地理原则决策，然后调用父类的通用执行框架。
        """
        # --- 阶段1 & 2 (数据融合) - 保持不变 ---
        # (这部分代码是正确的，无需修改)
        packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            if self.nodes[ch_id]["status"] == "active":
                if ch_id not in self.packets_in_transit: self.packets_in_transit[ch_id] = []
                members = [n for n in self.nodes if n.get("cluster_id") == ch_id and n.get("has_data_to_send")]
                if not members and not self.nodes[ch_id].get("has_data_to_send"): continue
                for member_node in members:
                    if member_node['id'] != ch_id:
                        dist = self.calculate_distance(member_node['id'], ch_id)
                        member_node["pending_tx_energy"] += self.calculate_transmission_energy(dist, packet_size)
                        self.nodes[ch_id]["pending_rx_energy"] += self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)
                num_raw_packets = len(members) + (1 if self.nodes[ch_id].get("has_data_to_send") else 0)
                if num_raw_packets > 0:
                    new_packet = {
                        "gen_round": self.current_round, 
                        "num_raw_packets": num_raw_packets,
                        "path": [ch_id]  # 添加初始路径
                    }
                    self.packets_in_transit[ch_id].append(new_packet)
                    for node in members: node["has_data_to_send"] = False
                    if self.nodes[ch_id].get("has_data_to_send"): self.nodes[ch_id]["has_data_to_send"] = False

        # --- HEED的决策阶段 ---
        logger.debug("HEED开始贪婪地理路由决策...")
        active_ch_list = [ch_id for ch_id in self.confirmed_cluster_heads_for_epoch if self.nodes[ch_id]["status"] == "active"]
        for ch_id in active_ch_list:
            my_dist_to_bs = self.calculate_distance_to_base_station(ch_id)
            best_next_hop = self.BS_ID # 默认下一跳是BS
            min_dist_to_bs_of_nh = my_dist_to_bs
            
            # 使用基础通信范围进行决策，更符合HEED原始意图
            neighbors = self.get_node_neighbors(ch_id, self.nodes[ch_id]["base_communication_range"])
            
            for neighbor_id in neighbors:
                if neighbor_id in self.confirmed_cluster_heads_for_epoch:
                    neighbor_dist_to_bs = self.calculate_distance_to_base_station(neighbor_id)
                    if neighbor_dist_to_bs < min_dist_to_bs_of_nh:
                        min_dist_to_bs_of_nh = neighbor_dist_to_bs
                        best_next_hop = neighbor_id
            
            # 设置节点的决策结果
            # 只有在找到了一个更近的邻居时，才会更新下一跳
            if best_next_hop != self.BS_ID or self.calculate_distance_to_base_station(ch_id) <= self.nodes[ch_id]["base_communication_range"]:
                 self.nodes[ch_id]['chosen_next_hop_id'] = best_next_hop
            else:
                 self.nodes[ch_id]['chosen_next_hop_id'] = self.NO_PATH_ID

        # --- 调用通用的传输执行函数 ---
        self._execute_routing_and_transmission()