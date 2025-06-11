# src/env_deec.py

import random
import math
from env import WSNEnv # 从你的核心env文件继承
import logging # 导入标准的logging库
logger = logging.getLogger("WSN_Simulation")

class WSNEnvDEEC(WSNEnv):
    """
    一个只实现经典DEEC协议及其简化多跳路由的环境。
    它继承了WSNEnv的基础设施，但重写了核心决策方法。
    """
    def __init__(self, config_path=None, performance_log_path=None, ch_behavior_log_path=None):
        # 首先调用父类的构造函数，加载所有通用配置和基础设施
        super().__init__(config_path, performance_log_path, ch_behavior_log_path)
        logger.info("DEEC环境已初始化。将使用经典的、基于概率的选举和分配逻辑。")
        # DEEC不需要复杂的Q表和模糊逻辑系统，但为了保持对象结构一致，我们保留这些属性
        # 在实际决策中，它们不会被使用
    
    # in src/env_deec.py -> class WSNEnvDEEC

    def _prepare_for_new_round(self):
        """
        [DEEC专属] 为新一轮做准备。
        这个版本不重置节点角色，因为DEEC的角色在一个Epoch内是固定的。
        """
        self._build_spatial_index()
        
        # 只重置每轮的统计数据
        self.sim_packets_generated_this_round = 0
        self.sim_packets_delivered_bs_this_round = 0
        self.sim_total_delay_this_round = 0.0
        self.sim_num_packets_for_delay_this_round = 0

        # 在DEEC中，节点状态相对简单，我们甚至可以不在每轮重置太多东西
        # 只需要确保数据包生成即可
        for node in self.nodes:
            if node["status"] == "active":
                self.sim_packets_generated_this_round += 1
                self.sim_packets_generated_total += 1
                node["has_data_to_send"] = True

    # in src/env_deec.py -> class WSNEnvDEEC

    def _run_epoch_start_phase(self):
        """
        [DEEC专属] 重写Epoch开始阶段，使用原始DEEC选举逻辑。
        """
        logger.info(f"--- ***** 新 Epoch 开始 (轮次 {self.current_round}) - DEEC模式 ***** ---")
        
        # 1. 在新Epoch开始时，将所有节点重置为normal
        for node in self.nodes:
            if node["status"] == "active":
                node["role"] = "normal"
                node["cluster_id"] = -1

        # 2. DEEC选举逻辑
        ch_declarations = self._run_deec_election()
        
        # 3. 宣告即当选，并更新角色
        self.confirmed_cluster_heads_for_epoch = ch_declarations
        self._update_node_roles_and_timers(ch_declarations) # 这个父类方法可以继续使用

    def _run_deec_election(self):
        """
        实现经典的DEEC协议中基于剩余能量的簇头选举概率模型。
        """
        ch_declarations = []
        p = self.p_opt_initial # DEEC使用固定的期望概率
        num_alive = self.get_alive_nodes()
        if num_alive == 0: return []
        
        # 1. 计算网络当前总能量和平均能量
        total_energy = sum(n['energy'] for n in self.nodes if n['status'] == 'active')
        avg_energy = total_energy / num_alive if num_alive > 0 else 0
        if avg_energy == 0: return []

        # 2. 遍历每个节点，根据DEEC公式计算其成为CH的概率并进行随机选择
        for node in self.nodes:
            if node["status"] == "active" and not node.get("can_connect_bs_directly", False):
                # 节点能量必须高于0
                if node['energy'] > 0:
                    # 计算该节点的概率 T(n)
                    prob_n = p * (node['energy'] / avg_energy)
                    
                    # 确保概率不会超过1（虽然理论上在p很小时不太可能）
                    prob_n = min(prob_n, 1.0)
                    
                    if random.random() <= prob_n:
                        ch_declarations.append(node["id"])
        
        #logger.info(f"DEEC选举：{len(ch_declarations)} 个节点宣告成为CH。")
        return ch_declarations

    def _run_normal_node_selection_phase(self):
        """
        重写节点选择阶段，使用经典分簇协议中最简单的逻辑：
        普通节点选择信号最强（即距离最近）的CH加入。
        """
        #logger.info("DEEC模式：普通节点选择最近的CH...")
        
        # 清空上一轮的分配结果
        for node in self.nodes:
            if node['role'] == 'normal':
                node['cluster_id'] = -1

        # 如果本轮没有选出CH，则所有节点都无法入簇
        if not self.confirmed_cluster_heads_for_epoch:
            #logger.warning("DEEC模式：本轮没有选举出任何CH，所有普通节点将保持孤立。")
            return

        for node_data in self.nodes:
            # 只处理活跃的、非直连的普通节点
            if node_data["status"] == "active" and node_data["role"] == "normal" and not node_data.get("can_connect_bs_directly", False):
                min_dist = float('inf')
                chosen_ch_id = -1
                
                # 遍历所有已确认的CH，找到最近的一个
                for ch_id in self.confirmed_cluster_heads_for_epoch:
                    # 确保CH是活跃的
                    if self.nodes[ch_id]["status"] == "active":
                        dist = self.calculate_distance(node_data["id"], ch_id)
                        # 检查是否在通信范围内
                        if dist <= node_data["base_communication_range"]:
                            if dist < min_dist:
                                min_dist = dist
                                chosen_ch_id = ch_id
                
                node_data["cluster_id"] = chosen_ch_id
                if chosen_ch_id == -1:
                    logger.debug(f"节点 {node_data['id']} 在DEEC模式下找不到可达的CH。")

    # in src/env_deec.py

    def _run_ch_routing_phase(self):
        """
        [DEEC专属-PDR修复版] CH进行数据路由。
        除了确定下一跳，还检查整条链路是否能通到BS。
        """
        # logger.info("DEEC模式：CH进行数据路由...")

        # 1. 为每个CH确定其直接的下一跳
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            ch_node = self.nodes[ch_id]
            if ch_node["status"] != "active": 
                continue

            ch_node["chosen_next_hop_id"] = -1 # 默认无路可走
            dist_to_bs = self.calculate_distance_to_bs(ch_id)

            # 优先直连BS
            if dist_to_bs <= ch_node["base_communication_range"]:
                ch_node["chosen_next_hop_id"] = self.BS_ID
                continue

            # 寻找最佳中继CH
            best_relay_ch_id = -1
            min_relay_dist = float('inf')
            
            for other_ch_id in self.confirmed_cluster_heads_for_epoch:
                if ch_id == other_ch_id: continue
                other_ch_node = self.nodes[other_ch_id]
                if other_ch_node["status"] != "active": continue

                dist_to_other = self.calculate_distance(ch_id, other_ch_id)
                if dist_to_other <= ch_node["base_communication_range"]:
                    if self.calculate_distance_to_bs(other_ch_id) < dist_to_bs:
                        if dist_to_other < min_relay_dist:
                            min_relay_dist = dist_to_other
                            best_relay_ch_id = other_ch_id
            
            ch_node["chosen_next_hop_id"] = best_relay_ch_id

        # 2. [新增] 验证每个CH的完整路径，并标记其是否能成功送达
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            ch_node = self.nodes[ch_id]
            if ch_node["status"] != "active":
                ch_node["can_route_to_bs"] = False
                continue

            path = [ch_id]
            current_node_id = ch_id
            can_reach_bs = False
            
            # 沿着下一跳路径走，最多走CH的数量次（防止无限循环）
            for _ in range(len(self.confirmed_cluster_heads_for_epoch) + 1):
                next_hop_id = self.nodes[current_node_id].get("chosen_next_hop_id", -1)
                
                if next_hop_id == self.BS_ID:
                    can_reach_bs = True
                    break
                
                if next_hop_id == -1 or next_hop_id in path: # 遇到死胡同或循环
                    break
                
                # 移动到下一跳
                path.append(next_hop_id)
                current_node_id = next_hop_id
            
            ch_node["can_route_to_bs"] = can_reach_bs
            if not can_reach_bs:
                logger.warning(f"DEEC模式：CH {ch_id} 的路由链无法到达BS。路径尝试: {path}")

    # in src/env_deec.py

    def _execute_energy_consumption_for_round(self):
        """
        [DEEC专属-PDR修复版] 为DEEC模式重写能量消耗和数据包送达统计。
        """
        energy_cfg = self.config.get('energy', {})
        packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        agg_cost_per_bit = energy_cfg.get('aggregation_cost_per_bit', 5e-9)
        idle_listening_cost = float(energy_cfg.get('idle_listening_per_round', 1e-6))
        sensing_cost = float(energy_cfg.get('sensing_per_round', 5e-7))

        energy_costs = {node["id"]: idle_listening_cost + sensing_cost for node in self.nodes}

        # 1. 普通节点 -> CH
        for node in self.nodes:
            if node["status"] == "active" and node["role"] == "normal" and node["cluster_id"] >= 0:
                ch_id = node["cluster_id"]
                if self.nodes[ch_id]["status"] == "active":
                    dist = self.calculate_distance(node["id"], ch_id)
                    tx_energy = self.calculate_transmission_energy(dist, packet_size)
                    energy_costs[node["id"]] += tx_energy
                    rx_energy = self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)
                    energy_costs[ch_id] += rx_energy

        # 2. CH 融合 & 路由 & 数据送达统计
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            ch_node = self.nodes[ch_id]
            if ch_node["status"] != "active": continue

            # a. 融合能耗
            members = [n for n in self.nodes if n.get("cluster_id") == ch_id]
            num_fusion_packets = len(members) + 1
            agg_energy = num_fusion_packets * agg_cost_per_bit * packet_size
            energy_costs[ch_id] += agg_energy

            # b. 路由链能耗计算 和 PDR统计
            current_node_id = ch_id
            # 沿着之前计算好的路径走
            for _ in range(len(self.confirmed_cluster_heads_for_epoch) + 1):
                sender_node = self.nodes[current_node_id]
                next_hop_id = sender_node.get("chosen_next_hop_id", -1)

                if next_hop_id == -1: break # 路径中断

                # 计算发送能耗
                if next_hop_id == self.BS_ID:
                    dist = self.calculate_distance_to_bs(current_node_id)
                else:
                    dist = self.calculate_distance(current_node_id, next_hop_id)
                
                tx_energy = self.calculate_transmission_energy(dist, packet_size)
                energy_costs[current_node_id] += tx_energy

                # 如果下一跳不是BS，计算接收能耗
                if next_hop_id != self.BS_ID:
                    if self.nodes[next_hop_id]["status"] == "active":
                        rx_energy = self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)
                        energy_costs[next_hop_id] += rx_energy
                    else:
                        break # 下一跳死亡，路径中断
                
                current_node_id = next_hop_id
                if current_node_id == self.BS_ID: break # 到达BS

            # [核心] PDR统计
            if ch_node.get("can_route_to_bs", False):
                # 这个CH和它所有成员的数据包都算成功送达
                num_delivered_packets = num_fusion_packets
                self.sim_packets_delivered_bs_this_round += num_delivered_packets
                # 可以在这里简化延迟计算，比如跳数
                # self.sim_total_delay_this_round += len(path) * num_delivered_packets
                # self.sim_num_packets_for_delay_this_round += num_delivered_packets

        # 3. 直连BS节点（如果有的话）
        # 在你的DEEC模型中，这个逻辑可以被简化或合并，因为can_connect_bs_directly的节点
        # 在_run_ch_routing_phase中会被直接分配BS_ID作为下一跳
        # 它们的能量消耗和PDR已经在上面的循环中被正确处理了。

        # 4. 统一扣除能量
        for node_id, cost in energy_costs.items():
            if cost > 0:
                self.consume_node_energy(node_id, cost)