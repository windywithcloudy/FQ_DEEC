# src/env_q_deec.py (V12 - 最终正确版)

import logging
from env import WSNEnv  # 从你的核心env文件继承

# 确保我们能获取到正确的logger实例
logger = logging.getLogger("WSN_Simulation")

class WSNEnvQDEEC(WSNEnv):
    """
    一个只在CH路由阶段禁用模糊逻辑的Q-DEEC对比算法环境。
    它继承了FQ-DEEC所有先进的选举和分配框架，只在路由成本计算上
    退回到基于固定权重的物理模型，以进行公平的“消融实验”。
    """
    def __init__(self, config_path=None, performance_log_path=None, ch_behavior_log_path=None):
        # 首先调用父类的构造函数，加载所有通用配置和基础设施
        super().__init__(config_path, performance_log_path, ch_behavior_log_path)
        self.is_intelligent_agent = True
        
        logger.info("="*20)
        logger.info("Q-DEEC 环境已初始化。")
        logger.info("CH选举和节点分配逻辑与FQ-DEEC完全相同。")
        logger.info("仅在CH路由成本计算中，用固定权重模型替代模糊逻辑。")
        logger.info("="*20)

    def _get_fuzzy_routing_cost(self, u_id, v_id, dist, use_dynamic_load=True):
        """
        [重写] CH路由成本计算。
        用一个简单的、基于物理属性的固定权重模型来替代模糊逻辑。
        """
        # 从config获取动态成本的权重因子
        cost_weights = self.config.get('routing', {}).get('dynamic_cost_weights', {})
        w_dist = cost_weights.get('distance', 0.2)
        w_energy = cost_weights.get('energy', 0.4)
        w_load = cost_weights.get('load', 0.4)
        
        u_node = self.nodes[u_id]
        v_node = self.nodes[v_id]
        
       # 计算下一跳的物理状态
        # 能量越高越好，所以成本应该与 (1 - 归一化能量) 成正比
        energy_cost_v = 1.0 - (v_node['energy'] / v_node['initial_energy']) if v_node['initial_energy'] > 0 else 1.0
        
        load_cost_v = 0
        # 只有在明确指示使用动态负载时，才计算它
        # 这使得Q-DEEC在构建路由图时（use_dynamic_load=False）和FQ-DEEC的行为完全一致
        if use_dynamic_load:
            buffer_v = self.packets_in_transit.get(v_id, [])
            load_cost_v = len(buffer_v) / self.ch_forwarding_buffer_size if self.ch_forwarding_buffer_size > 0 else 0
        
        # 距离成本
        dist_cost = dist / u_node["base_communication_range"] if u_node["base_communication_range"] > 0 else 1.0

        # 计算一个线性的、非模糊的成本值
        cost = (w_dist * dist_cost + 
                w_energy * energy_cost_v + 
                w_load * load_cost_v)
        
        # 对骨干网节点的能量惩罚逻辑保持不变，以确保路由的稳定性
        relay_zone_radius = self.network_diagonal * 0.4
        energy_threshold = 0.4
        current_energy_norm = v_node['energy'] / v_node['initial_energy'] if v_node['initial_energy'] > 0 else 0
        if self.calculate_distance_to_base_station(v_id) <= relay_zone_radius and current_energy_norm < energy_threshold:
            penalty_factor = 1.0 + 2.0 * ((energy_threshold - current_energy_norm) / energy_threshold)**2
            cost *= penalty_factor
        
        return cost

    # 其他所有方法，包括 _run_ch_routing_phase, _decide_next_hop_with_q_learning 等
    # 都不需要重写，因为它们会通过继承自动调用这个被重写了的 _get_fuzzy_routing_cost 方法。