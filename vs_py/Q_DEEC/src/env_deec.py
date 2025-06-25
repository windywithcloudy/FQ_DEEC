# src/env_deec.py

import random
import math
from env import WSNEnv # 从你的核心env文件继承
import logging # 导入标准的logging库
logger = logging.getLogger("WSN_Simulation")
import networkx as nx

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
    

    def _prepare_for_new_round(self):
        """
        [DEEC专属-修改版] 为新一轮做准备。
        每轮开始前重置所有节点为 normal，以进行新一轮选举。
        """
        self._build_spatial_index()
        
        # 重置每轮的统计数据
        self.sim_packets_generated_this_round = 0
        self.sim_packets_delivered_bs_this_round = 0
        self.sim_total_delay_this_round = 0.0
        self.sim_num_packets_for_delay_this_round = 0

        # 每轮开始时，所有活跃节点都回归普通身份，准备新一轮选举
        for node in self.nodes:
            if node["status"] == "active":
                node["role"] = "normal"
                node["cluster_id"] = -1
                self.sim_packets_generated_this_round += 1
                self.sim_packets_generated_total += 1
                node["has_data_to_send"] = True


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

    # in env_deec.py
    def _run_ch_routing_phase(self):
        """
        [DEEC专属-最终修复V3] CH进行数据路由。
        使用唯一的 NO_PATH_ID 来区分“无路可走”和“下一跳是BS”，解决逻辑歧义。
        """
        # logger.info("DEEC模式：CH进行数据路由...")
        #logger.info("--- DEEC ROUTING DEBUG START ---") # 您可以保留或删除调试日志
        num_potential_direct_linkers = 0

        # 1. 为每个CH确定其直接的下一跳
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            ch_node = self.nodes[ch_id]
            if ch_node["status"] != "active": 
                continue

            # [核心修改] 使用新的 NO_PATH_ID 作为默认值
            ch_node["chosen_next_hop_id"] = self.NO_PATH_ID 
            
            ch_current_range = ch_node.get("current_communication_range", ch_node["base_communication_range"])
            dist_to_bs = self.calculate_distance_to_base_station(ch_id)
            
            #logger.info(f"  CH {ch_id}: dist_to_bs={dist_to_bs:.2f}, current_range={ch_current_range:.2f}")

            # 决策优先级调整：直连BS是最高优先级！
            if dist_to_bs <= ch_current_range:
                ch_node["chosen_next_hop_id"] = self.BS_ID
                num_potential_direct_linkers += 1
                #logger.info(f"    DECISION: CH {ch_id} WILL connect to BS. ID set to {self.BS_ID}.")
                continue

            # 如果不能直连BS，才开始寻找最佳中继CH
            candidate_relays = {}
            for other_ch_id in self.confirmed_cluster_heads_for_epoch:
                if ch_id == other_ch_id: continue
                other_ch_node = self.nodes[other_ch_id]
                if other_ch_node["status"] != "active": continue

                dist_to_other = self.calculate_distance(ch_id, other_ch_id)
                other_current_range = other_ch_node.get("current_communication_range", other_ch_node["base_communication_range"])

                if dist_to_other <= ch_current_range and dist_to_other <= other_current_range:
                    if self.calculate_distance_to_base_station(other_ch_id) < dist_to_bs:
                        candidate_relays[other_ch_id] = dist_to_other
            
            if candidate_relays:
                best_relay_ch_id = min(candidate_relays, key=lambda k:candidate_relays[k])
                ch_node["chosen_next_hop_id"] = best_relay_ch_id
        
        #logger.info(f"--- DEEC ROUTING DEBUG END --- Found {num_potential_direct_linkers} CHs that can directly link to BS.")

        # 2. 验证每个CH的完整路径
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            ch_node = self.nodes[ch_id]
            if ch_node["status"] != "active":
                ch_node["can_route_to_bs"] = False
                continue

            path = [ch_id]
            current_node_id = ch_id
            can_reach_bs = False
            
            for _ in range(len(self.confirmed_cluster_heads_for_epoch) + 2):
                if not (0 <= current_node_id < len(self.nodes)): break
                
                # [核心修改] 使用 NO_PATH_ID 作为 get 的默认值
                next_hop_id = self.nodes[current_node_id].get("chosen_next_hop_id", self.NO_PATH_ID)
                
                if next_hop_id == self.BS_ID:
                    can_reach_bs = True
                    break
                
                # [核心修改] 判断路径中断的条件
                if next_hop_id == self.NO_PATH_ID or next_hop_id in path:
                    break
                
                path.append(next_hop_id)
                current_node_id = next_hop_id
            
            ch_node["can_route_to_bs"] = can_reach_bs
        
            #if can_reach_bs:
             #   logger.info(f"    PATH VALIDATION: CH {ch_id} can reach BS. Path: {path + [self.BS_ID]}")
            #else:
            #    logger.debug(f"    PATH VALIDATION: CH {ch_id} CANNOT reach BS. Path attempt: {path}")

    # in env_deec.py
    # in env_deec.py
    def _apply_energy_consumption(self):
        """
        [DEEC专属-最终修复V6.1] 使用清晰的步骤，分离PDR统计与能耗计算，并使用 NO_PATH_ID。
        """
        energy_cfg = self.config.get('energy', {})
        packet_size = self.config.get("simulation", {}).get("packet_size", 4000)
        agg_cost_per_bit = energy_cfg.get('aggregation_cost_per_bit', 5e-9)
        idle_listening_cost = float(energy_cfg.get('idle_listening_per_round', 1e-6))
        sensing_cost = float(energy_cfg.get('sensing_per_round', 5e-7))

        energy_costs = {node["id"]: idle_listening_cost + sensing_cost for node in self.nodes if node["status"] == "active"}
        
        # 步骤 1: 统计每个CH初始负责的数据包总数，并计算普通节点->CH的能耗
        packets_managed_by_ch = {ch_id: 0 for ch_id in self.confirmed_cluster_heads_for_epoch if self.nodes[ch_id]["status"] == "active"}
        
        for node in self.nodes:
            if node["status"] != "active": continue
            
            if node["role"] == "cluster_head":
                if node["id"] in packets_managed_by_ch:
                    packets_managed_by_ch[node["id"]] += 1
            elif node["role"] == "normal" and node.get("cluster_id", -1) >= 0:
                ch_id = node["cluster_id"]
                if ch_id < len(self.nodes) and self.nodes[ch_id]["status"] == "active" and ch_id in packets_managed_by_ch:
                    packets_managed_by_ch[ch_id] += 1
                    # 计算普通节点 -> CH 的能耗
                    dist = self.calculate_distance(node["id"], ch_id)
                    tx_energy = self.calculate_transmission_energy(dist, packet_size)
                    if node["id"] in energy_costs: energy_costs[node["id"]] += tx_energy
                    rx_energy = self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)
                    if ch_id in energy_costs: energy_costs[ch_id] += rx_energy

        # 步骤 2: 计算所有CH的融合和路由能耗
        for ch_id in self.confirmed_cluster_heads_for_epoch:
            ch_node = self.nodes[ch_id]
            if ch_node["status"] != "active": continue

            # a. 融合能耗
            num_initial_packets = packets_managed_by_ch.get(ch_id, 0)
            if num_initial_packets > 0 and ch_id in energy_costs:
                energy_costs[ch_id] += num_initial_packets * agg_cost_per_bit * packet_size

            # b. 路由能耗 (每个CH只负责发送自己的聚合包一次)
            next_hop = ch_node.get("chosen_next_hop_id", self.NO_PATH_ID)
            
            # [核心修改] 只要有下一跳（不是无路可走），就要消耗发送能量
            if next_hop != self.NO_PATH_ID:
                if next_hop == self.BS_ID:
                    dist = self.calculate_distance_to_base_station(ch_id)
                else:
                    dist = self.calculate_distance(ch_id, next_hop)
                
                # 发送一个聚合包的能耗
                tx_energy = self.calculate_transmission_energy(dist, packet_size)
                if ch_id in energy_costs: energy_costs[ch_id] += tx_energy

                # 下一跳的接收能耗
                if next_hop != self.BS_ID and next_hop < len(self.nodes) and self.nodes[next_hop]["status"] == "active" and next_hop in energy_costs:
                    rx_energy = self.calculate_transmission_energy(0, packet_size, is_tx_operation=False)
                    energy_costs[next_hop] += rx_energy

        # 步骤 3: PDR统计 (完全独立的逻辑)
        for ch_id, num_packets in packets_managed_by_ch.items():
            ch_node = self.nodes[ch_id]
            if ch_node["status"] == "active" and ch_node.get("can_route_to_bs", False):
                if num_packets > 0:
                    self.sim_packets_delivered_bs_this_round += num_packets
                    self.sim_packets_delivered_bs_total += num_packets
                    #logger.info(f"    PDR LOG: CH {ch_id} and its {num_packets - 1} members' packets delivered.")

        # 步骤 4: 统一扣除能量
        for node_id, cost in energy_costs.items():
            if cost > 0:
                self.consume_node_energy(node_id, cost)

    def step(self, current_round_num):
        """
        [DEEC专属-修改版] 重写 step 函数，以调用DEEC的特定逻辑流程。
        选举和分配改为每轮执行。
        """
        self.current_round = current_round_num
        logger.info(f"--- 开始第 {self.current_round} 轮 (DEEC模式-逐轮选举) ---")

        # 阶段 0: 准备工作 (每轮重置必要的节点状态)
        self._prepare_for_new_round()

        # 阶段 1: 选举和分配 (每轮都执行)
        self._run_deec_election_and_assignment()

        # 阶段 2: CH 选择下一跳进行路由
        self._run_ch_routing_phase()
        
        # 阶段 3: 执行本轮所有暂存的能量消耗，并统计PDR
        self._apply_energy_consumption()

        # 阶段 4: 更新并记录本轮的性能指标
        self._update_and_log_performance_metrics()
        
        logger.info(f"--- 第 {self.current_round} 轮结束 (DEEC模式-逐轮选举) ---")
        
        if self.get_alive_nodes() == 0:
            logger.info("DEEC网络中所有节点均已死亡，仿真结束。")
            return False
            
        return True
    

    def _run_deec_election_and_assignment(self):
        """
        [DEEC专属-新增] 整合了选举、角色更新和节点分配的函数。
        """
        # 1. DEEC选举逻辑
        ch_declarations = self._run_deec_election()
        
        # 2. 宣告即当选，并更新角色
        self.confirmed_cluster_heads_for_epoch = ch_declarations # 变量名可以不改，但它现在是本轮的CH
        self._update_node_roles_and_timers(ch_declarations)

        # 3. 普通节点分配
        self._run_normal_node_selection_phase()