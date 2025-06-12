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

    def _run_ch_routing_phase(self):
        """重写CH路由阶段，使用简化的地理路由。"""
        # 这部分逻辑可以与env_deec.py中的实现保持一致
        super()._run_ch_routing_phase()