import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt # Keep for viewing, can be commented out for production

import logging # 使用标准logging
logger = logging.getLogger(__name__) # 创建一个本地logger
if not logger.handlers: # 防止重复添加handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# =============================================================================
# Fuzzy System for Normal Node Selecting a Cluster Head (CH)
# =============================================================================
class NormalNodeCHSelectionFuzzySystem:
    def __init__(self, main_sim_config): # 接收主config
        self.main_config = main_sim_config
        # self.fuzzy_specific_config = self.main_config.get('fuzzy', {}).get('normal_node_select_ch', {}) # 不再需要这个子配置

        # --- Antecedents (Inputs) ---
        self.d_c_base = None           # Distance: CH to Base Station (实际距离)
        self.e_cluster = None          # Energy: CH's Normalized Remaining Energy [0,1]
        self.p_cluster_ratio = None    # Load: CH's Current Load Ratio (actual_load / avg_load) [e.g., 0-3]
        self.r_success = None          # Success Rate: Historical Comm. Success with CH (Normalized [0,1])
        self.e_send_total_ratio = None # Send Energy: Hist. Avg. Send Energy to CH Ratio (actual_send_e / avg_send_e_ref) [e.g., 0-3]

        # --- Consequents (Outputs) ---
        self.w_e_ch = None
        self.w_path = None
        self.w_load = None
        self.w_dist_bs = None

        self._define_antecedents_and_consequents_once()
        self.rules = self._define_rules_once() 
        
        if not self.rules:
            logger.error("NormalNodeCHSelectionFuzzySystem: No rules defined! System will not work.")
            self.control_system = None
        else:
            self.control_system = ctrl.ControlSystem(self.rules)
            logger.info("NormalNodeCHSelectionFuzzySystem: ControlSystem built successfully.")

    def _define_antecedents_and_consequents_once(self):
        logger.debug("Defining antecedents and consequents for NormalNodeCHSelectionFuzzySystem...")
        # 1. D_c_base
        max_dist_val = 250 * np.sqrt(2) 
        universe_dc_base = np.arange(0, max_dist_val + 1, 1)
        self.d_c_base = ctrl.Antecedent(universe_dc_base, 'd_c_base') # 标签与输入键匹配
        mid_dist_point = max_dist_val / 2
        self.d_c_base['Near'] = fuzz.zmf(self.d_c_base.universe, mid_dist_point * 0.6, mid_dist_point) 
        self.d_c_base['Medium'] = fuzz.trimf(self.d_c_base.universe, [mid_dist_point * 0.6, mid_dist_point, mid_dist_point * 1.4])
        self.d_c_base['Far'] = fuzz.smf(self.d_c_base.universe, mid_dist_point, mid_dist_point * 1.4)

        # 2. E_cluster
        universe_e_cluster = np.arange(0, 1.01, 0.01)
        self.e_cluster = ctrl.Antecedent(universe_e_cluster, 'e_cluster') # 标签与输入键匹配
        self.e_cluster['Low'] = fuzz.zmf(self.e_cluster.universe, 0.2, 0.4)
        self.e_cluster['Medium'] = fuzz.trimf(self.e_cluster.universe, [0.3, 0.5, 0.7])
        self.e_cluster['High'] = fuzz.smf(self.e_cluster.universe, 0.6, 0.8)

        # 3. P_cluster_Ratio (Input will be actual_load / avg_load_per_ch)
        universe_p_cluster_ratio = np.arange(0, 3.01, 0.01) 
        self.p_cluster_ratio = ctrl.Antecedent(universe_p_cluster_ratio, 'p_cluster_ratio') # 标签与输入键匹配
        self.p_cluster_ratio['Low'] = fuzz.zmf(self.p_cluster_ratio.universe, 0.5, 1.0)
        self.p_cluster_ratio['Medium'] = fuzz.trimf(self.p_cluster_ratio.universe, [0.75, 1.25, 1.75])
        self.p_cluster_ratio['High'] = fuzz.smf(self.p_cluster_ratio.universe, 1.5, 2.0)

        # 4. R_success
        universe_r_success = np.arange(0, 1.01, 0.01)
        self.r_success = ctrl.Antecedent(universe_r_success, 'r_success') # 标签与输入键匹配
        self.r_success['Low'] = fuzz.zmf(self.r_success.universe, 0.3, 0.6)
        self.r_success['Medium'] = fuzz.trimf(self.r_success.universe, [0.4, 0.7, 0.9])
        self.r_success['High'] = fuzz.smf(self.r_success.universe, 0.7, 0.95)

        # 5. E_send_total_Ratio (Input will be actual_e_send / avg_e_send_ref)
        universe_e_send_ratio = np.arange(0, 3.01, 0.01)
        self.e_send_total_ratio = ctrl.Antecedent(universe_e_send_ratio, 'e_send_total_ratio') # 标签与输入键匹配
        self.e_send_total_ratio['Low'] = fuzz.zmf(self.e_send_total_ratio.universe, 0.5, 1.0)
        self.e_send_total_ratio['Medium'] = fuzz.trimf(self.e_send_total_ratio.universe, [0.75, 1.25, 1.75])
        self.e_send_total_ratio['High'] = fuzz.smf(self.e_send_total_ratio.universe, 1.5, 2.0)
        
        # --- Consequents ---
        universe_weights = np.arange(0, 1.01, 0.01) # 输出权重范围 [0,1]
        self.w_e_ch = ctrl.Consequent(universe_weights, 'w_e_ch')
        self.w_path = ctrl.Consequent(universe_weights, 'w_path')
        self.w_load = ctrl.Consequent(universe_weights, 'w_load')
        self.w_dist_bs = ctrl.Consequent(universe_weights, 'w_dist_bs')

        for output_var in [self.w_e_ch, self.w_path, self.w_load, self.w_dist_bs]:
            output_var['Low'] = fuzz.zmf(output_var.universe, 0.2, 0.4)
            output_var['Medium'] = fuzz.trimf(output_var.universe, [0.3, 0.5, 0.7])
            output_var['High'] = fuzz.smf(output_var.universe, 0.6, 0.8)
        logger.debug("Antecedents and consequents for NormalNodeCHSelection defined.")


    def _define_rules_once(self):
        logger.debug("Defining fuzzy rules for NormalNodeCHSelection...")
        rules = []
        # Rules for w_e_ch
        # 修复覆盖漏洞并减少冲突
        rules.append(ctrl.Rule(self.e_cluster['Low'], self.w_e_ch['Low']))

        # Medium 集群规则优化
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & self.d_c_base['Far'] & self.p_cluster_ratio['High'], self.w_e_ch['Low']))
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & self.d_c_base['Far'] & (self.p_cluster_ratio['Medium'] | self.p_cluster_ratio['Low']), self.w_e_ch['Medium']))
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & self.d_c_base['Medium'], self.w_e_ch['Medium']))
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & self.d_c_base['Near'] & self.p_cluster_ratio['High'], self.w_e_ch['Medium']))
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & self.d_c_base['Near'] & (self.p_cluster_ratio['Medium'] | self.p_cluster_ratio['Low']), self.w_e_ch['High']))

        # High 集群规则优化
        rules.append(ctrl.Rule(self.e_cluster['High'] & self.d_c_base['Far'] & self.p_cluster_ratio['High'], self.w_e_ch['Medium']))
        rules.append(ctrl.Rule(self.e_cluster['High'] & self.d_c_base['Far'] & (self.p_cluster_ratio['Medium'] | self.p_cluster_ratio['Low']), self.w_e_ch['High']))
        rules.append(ctrl.Rule(self.e_cluster['High'] & self.d_c_base['Medium'], self.w_e_ch['High']))
        rules.append(ctrl.Rule(self.e_cluster['High'] & self.d_c_base['Near'], self.w_e_ch['High']))

        # Rules for w_path
        rules.append(ctrl.Rule(self.r_success['High'] & (self.e_send_total_ratio['Low'] | self.e_send_total_ratio['Medium']), self.w_path['Low']))
        rules.append(ctrl.Rule(self.r_success['High'] & self.e_send_total_ratio['High'], self.w_path['Medium']))
        rules.append(ctrl.Rule(self.r_success['Medium'] & (self.e_send_total_ratio['Low'] | self.e_send_total_ratio['Medium']), self.w_path['Medium']))
        rules.append(ctrl.Rule(self.r_success['Medium'] & self.e_send_total_ratio['High'], self.w_path['High']))
        rules.append(ctrl.Rule(self.r_success['Low'], self.w_path['High']))
        
        # --- [V2-健壮版] Rules for w_load (负载权重) ---
        # 核心思想：确保全覆盖，消除逻辑死区。
        # 规则主要由当前负载(p_cluster_ratio)和拥塞风险(d_c_base)决定。
        
        # 1. 高风险/高负载情况 -> 高权重
        rules.append(ctrl.Rule(self.p_cluster_ratio['High'] | self.d_c_base['Near'], self.w_load['High']))
        
        # 2. 中等风险/中等负载情况 -> 中权重
        rules.append(ctrl.Rule(self.p_cluster_ratio['Medium'] & self.d_c_base['Medium'], self.w_load['Medium']))
        
        # 3. 低风险/低负载情况 -> 低权重
        rules.append(ctrl.Rule(self.p_cluster_ratio['Low'] & self.d_c_base['Far'], self.w_load['Low']))
        
        # 4. 交叉情况，用于填补空白区域
        rules.append(ctrl.Rule(self.p_cluster_ratio['Medium'] & self.d_c_base['Far'], self.w_load['Medium'])) # 负载中等但位置安全，中等关注
        rules.append(ctrl.Rule(self.p_cluster_ratio['Low'] & self.d_c_base['Medium'], self.w_load['Low']))    # 负载低且位置中等，低关注

        # 5. 补充规则，确保即使能量很低，如果负载高或位置危险，依然要高度关注负载
        rules.append(ctrl.Rule(self.e_cluster['Low'] & (self.p_cluster_ratio['High'] | self.d_c_base['Near']), self.w_load['High']))

        # Rules for w_dist_bs
        # 优化后版本（5条规则）
        rules.append(ctrl.Rule(self.d_c_base['Near'], self.w_dist_bs['Low']))
        rules.append(ctrl.Rule(self.d_c_base['Medium'] & self.e_cluster['Low'], self.w_dist_bs['High']))
        rules.append(ctrl.Rule(self.d_c_base['Medium'] & self.e_cluster['Medium'], self.w_dist_bs['Medium']))
        rules.append(ctrl.Rule(self.d_c_base['Medium'] & self.e_cluster['High'], self.w_dist_bs['Low']))
        rules.append(ctrl.Rule(self.d_c_base['Far'] & self.e_cluster['Low'], self.w_dist_bs['High']))
        rules.append(ctrl.Rule(self.d_c_base['Far'] & (self.e_cluster['Medium'] | self.e_cluster['High']), self.w_dist_bs['Medium']))
        
        if not rules: # Default rules if none are defined (should not happen with above)
            logger.warning("NormalNodeCHSelectionFuzzySystem: No specific rules, adding default.")
            # ... (add some truly default rules if necessary) ...
        logger.debug(f"Defined {len(rules)} rules for NormalNodeCHSelectionFuzzySystem.")
        return rules

    def compute_weights(self, current_dc_base, current_e_cluster_normalized, 
                        current_p_cluster_ratio_val, # 接收已计算的比率
                        current_r_success_normalized, 
                        current_e_send_total_ratio_val):  # 接收已计算的比率
        
        if self.control_system is None:
            logger.error("NormalNodeCHSelectionFuzzySystem: ControlSystem is not built.")
            return {'w_e_ch': 0.5, 'w_path': 0.5, 'w_load': 0.5, 'w_dist_bs': 0.5} # Neutral weights

        simulation = ctrl.ControlSystemSimulation(self.control_system) # Create new simulation instance

        # --- 裁剪并设置输入值 ---
        simulation.input['d_c_base'] = np.clip(current_dc_base, self.d_c_base.universe.min(), self.d_c_base.universe.max())
        simulation.input['e_cluster'] = np.clip(current_e_cluster_normalized, self.e_cluster.universe.min(), self.e_cluster.universe.max())
        simulation.input['p_cluster_ratio'] = np.clip(current_p_cluster_ratio_val, self.p_cluster_ratio.universe.min(), self.p_cluster_ratio.universe.max())
        simulation.input['r_success'] = np.clip(current_r_success_normalized, self.r_success.universe.min(), self.r_success.universe.max())
        simulation.input['e_send_total_ratio'] = np.clip(current_e_send_total_ratio_val, self.e_send_total_ratio.universe.min(), self.e_send_total_ratio.universe.max())
        
        output_dict = {}
        try:
            simulation.compute()
            output_vars_to_get = {
                'w_e_ch': 0.5, 'w_path': 0.5, 'w_load': 0.5, 'w_dist_bs': 0.5 # Default to neutral if key missing
            }
            for var_name, default_val in output_vars_to_get.items():
                try:
                    output_dict[var_name] = simulation.output[var_name]
                except KeyError:
                    logger.warning(f"NormalNodeFuzzy: Output var '{var_name}' not found in simulation.output, using default {default_val}.")
                    output_dict[var_name] = default_val
            return output_dict
        except Exception as e:
            logger.error(f"Error in NormalNodeCHSelectionFuzzySystem compute: {e}", exc_info=True)
            return {'w_e_ch': 0.5, 'w_path': 0.5, 'w_load': 0.5, 'w_dist_bs': 0.5}

    def view_antecedent(self, name):
        if hasattr(self, name) and getattr(self, name) is not None:
            getattr(self, name).view()
            plt.show(block=True)
        else: print(f"Antecedent {name} not found or not initialized.")
            
    def view_consequent(self, name):
        if hasattr(self, name) and getattr(self, name) is not None:
            getattr(self, name).view()
            plt.show(block=True)
        else: print(f"Consequent {name} not found or not initialized.")


# =============================================================================
# Fuzzy System for Cluster Head (CH) Selecting Path to Base Station (BS)
# =============================================================================
class CHToBSPathSelectionFuzzySystem:
    def __init__(self, main_sim_config): # Add other necessary averages if needed
        """
        Initializes the fuzzy control system for CH selecting path to BS.
        Args:
            node_sum (int): Total number of nodes.
            cluster_sum (int): Current number of CHs.
        """
        self.main_config = main_sim_config
        self.p_opt_reference = float(self.main_config.get('deec', {}).get('p_opt', 0.1)) # 否则用deec的p_opt
        # --- Antecedents (Inputs) ---
        self.d_c_bs_neighbor = None     # Distance: Neighbor CH to BS
        self.e_c_neighbor = None        # Energy: Neighbor CH's Normalized Remaining Energy
        # self.lq_c_ratio = None          # Link Quality: Normalized or Ratio to Avg. (NEEDS avg_lq)
        self.load_c_ratio = None        # Load: Neighbor CH's Current Load Ratio
        self.r_c_success = None         # Success Rate: Historical Comm. Success with Neighbor (Normalized)
        # self.e_ctx_cost_ratio = None    # CTX Cost: Energy cost to send to this neighbor (NEEDS avg_ctx_cost)

        # For simplicity, LQ_c and E_ctx_cost are assumed to be directly normalized to [0,1] for now
        # Or you can pass their respective averages to __init__ like avg_load
        #self.lq_c_normalized = None     # Link Quality: Normalized [0,1]
        self.e_ctx_cost_normalized = None # CTX Cost: Normalized [0,1]


        # --- Consequents (Outputs) ---
        self.w_d_pro = None  # Path progress weight
        self.w_e_cost = None # Energy cost weight
        self.w_fur = None    # Sustainability/Future energy weight
        self.w_load_neighbor = None # Neighbor load weight (renamed from w_load to be specific)

        self._define_antecedents_and_consequents_once()
        self.rules = self._define_rules_once() 
        
        if not self.rules:
            logger.error("CHToBSPathSelectionFuzzySystem: No rules defined!")
            self.control_system = None
            # 确保即使control_system为None，后续也不会尝试创建simulation或引发错误
            # 或者在这里创建一个最小的、包含所有预期输入输出的空规则系统用于测试结构
            # 但更好的做法是确保规则被正确定义
        else:
            try:
                self.control_system = ctrl.ControlSystem(self.rules)
                logger.info("CHToBSPathSelectionFuzzySystem: ControlSystem built successfully.")
                
                # --- 调试打印 ControlSystem 的 Antecedents ---
                logger.debug("Antecedents registered in ControlSystem:")
                for antecedent in self.control_system.antecedents: # antecedents 是一个迭代器
                    logger.debug(f"  - Antecedent Label: '{antecedent.label}', Universe: {antecedent.universe.shape}")
                # --- 调试打印结束 ---

            except Exception as e:
                logger.error(f"Error creating ControlSystem: {e}", exc_info=True)
                self.control_system = None

    def _define_antecedents_and_consequents_once(self):
        # 1. D_c_bs_neighbor (Distance: Neighbor CH to BS)
        max_dist_val = 250 * np.sqrt(2)
        universe_dc_bs_neighbor = np.arange(0, max_dist_val + 1, 1)
        self.d_c_bs_neighbor = ctrl.Antecedent(universe_dc_bs_neighbor, 'd_c_bs_neighbor')
        mid_dist_point = max_dist_val / 2
        self.d_c_bs_neighbor['Near'] = fuzz.zmf(self.d_c_bs_neighbor.universe, mid_dist_point * 0.6, mid_dist_point)
        self.d_c_bs_neighbor['Medium'] = fuzz.trimf(self.d_c_bs_neighbor.universe, [mid_dist_point * 0.6, mid_dist_point, mid_dist_point * 1.4])
        self.d_c_bs_neighbor['Far'] = fuzz.smf(self.d_c_bs_neighbor.universe, mid_dist_point, mid_dist_point * 1.4)

        # 2. E_c_neighbor (Neighbor CH's Normalized Remaining Energy)
        universe_e_c_neighbor = np.arange(0, 1.01, 0.01)
        self.e_c_neighbor = ctrl.Antecedent(universe_e_c_neighbor, 'e_c_neighbor')
        self.e_c_neighbor['Low'] = fuzz.zmf(self.e_c_neighbor.universe, 0.25, 0.45)
        self.e_c_neighbor['Medium'] = fuzz.trimf(self.e_c_neighbor.universe, [0.35, 0.55, 0.75])
        self.e_c_neighbor['High'] = fuzz.smf(self.e_c_neighbor.universe, 0.65, 0.85)
        
        # 4. Load_c_ratio (Neighbor CH's Load Ratio - actual_load / avg_load_for_neighbor)
        universe_load_c_ratio = np.arange(0, 3.01, 0.01)
        self.load_c_ratio = ctrl.Antecedent(universe_load_c_ratio, 'load_c_ratio')
        self.load_c_ratio['Low'] = fuzz.zmf(self.load_c_ratio.universe, 0.5, 1.0)
        self.load_c_ratio['Medium'] = fuzz.trimf(self.load_c_ratio.universe, [0.75, 1.25, 1.75])
        self.load_c_ratio['High'] = fuzz.smf(self.load_c_ratio.universe, 1.5, 2.0)

        # 5. R_c_success (Historical Comm. Success Rate with Neighbor - Normalized [0,1])
        universe_r_c_success = np.arange(0, 1.01, 0.01)
        self.r_c_success = ctrl.Antecedent(universe_r_c_success, 'r_c_success')
        self.r_c_success['Low'] = fuzz.zmf(self.r_c_success.universe, 0.3, 0.6)
        self.r_c_success['Medium'] = fuzz.trimf(self.r_c_success.universe, [0.4, 0.7, 0.9])
        self.r_c_success['High'] = fuzz.smf(self.r_c_success.universe, 0.7, 0.95)

        # 6. E_ctx_cost_normalized (Energy cost to send to this neighbor - Normalized [0,1])
        universe_e_ctx_cost = np.arange(0, 1.01, 0.01)
        self.e_ctx_cost_normalized = ctrl.Antecedent(universe_e_ctx_cost, 'e_ctx_cost_normalized')
        self.e_ctx_cost_normalized['Low'] = fuzz.zmf(self.e_ctx_cost_normalized.universe, 0.2, 0.4)
        self.e_ctx_cost_normalized['Medium'] = fuzz.trimf(self.e_ctx_cost_normalized.universe, [0.3, 0.5, 0.7])
        self.e_ctx_cost_normalized['High'] = fuzz.smf(self.e_ctx_cost_normalized.universe, 0.6, 0.8)

        universe_weights = np.arange(0, 1.01, 0.01)
        self.w_d_pro = ctrl.Consequent(universe_weights, 'w_d_pro')
        self.w_e_cost = ctrl.Consequent(universe_weights, 'w_e_cost')
        self.w_fur = ctrl.Consequent(universe_weights, 'w_fur')
        self.w_load_neighbor = ctrl.Consequent(universe_weights, 'w_load_neighbor')

        for output_var in [self.w_d_pro, self.w_e_cost, self.w_fur, self.w_load_neighbor]:
            output_var['Low'] = fuzz.zmf(output_var.universe, 0.2, 0.4)
            output_var['Medium'] = fuzz.trimf(output_var.universe, [0.3, 0.5, 0.7])
            output_var['High'] = fuzz.smf(output_var.universe, 0.6, 0.8)
    
    def _define_rules_once(self):
        rules = []
        # Rules for w_d_pro
        rules.append(ctrl.Rule(self.r_c_success['High'], self.w_d_pro['High']))
        rules.append(ctrl.Rule(self.r_c_success['Medium'] & self.d_c_bs_neighbor['Near'], self.w_d_pro['High']))
        rules.append(ctrl.Rule(self.r_c_success['Medium'] & self.d_c_bs_neighbor['Medium'], self.w_d_pro['Medium']))
        rules.append(ctrl.Rule(self.r_c_success['Medium'] & self.d_c_bs_neighbor['Far'], self.w_d_pro['Low']))
        rules.append(ctrl.Rule(self.r_c_success['Low'], self.w_d_pro['Low']))

        # Rules for w_e_cost
        rules.append(ctrl.Rule(self.r_c_success['High'] & (self.e_ctx_cost_normalized['Low'] | self.e_ctx_cost_normalized['Medium']), self.w_e_cost['Low']))
        rules.append(ctrl.Rule(self.r_c_success['High'] & self.e_ctx_cost_normalized['High'], self.w_e_cost['Medium']))
        rules.append(ctrl.Rule(self.r_c_success['Medium'] & (self.e_ctx_cost_normalized['Low'] | self.e_ctx_cost_normalized['Medium']), self.w_e_cost['Medium']))
        rules.append(ctrl.Rule(self.r_c_success['Medium'] & self.e_ctx_cost_normalized['High'], self.w_e_cost['High']))
        rules.append(ctrl.Rule(self.r_c_success['Low'], self.w_e_cost['High']))

        # Rules for w_fur
        # 简化后的规则集

        # High输出规则
        rules.append(ctrl.Rule(
            self.e_c_neighbor['High'] & 
            (self.load_c_ratio['Low'] | self.load_c_ratio['Medium']) & 
            (self.d_c_bs_neighbor['Near'] | self.d_c_bs_neighbor['Medium']), 
            self.w_fur['High']
        ))

        # Medium输出规则
        rules.append(ctrl.Rule(
            self.e_c_neighbor['High'] & 
            (self.load_c_ratio['High'] | self.d_c_bs_neighbor['Far']), 
            self.w_fur['Medium']
        ))
        rules.append(ctrl.Rule(
            self.e_c_neighbor['Medium'] & 
            (self.load_c_ratio['Low'] | self.load_c_ratio['Medium']), 
            self.w_fur['Medium']
        ))
        rules.append(ctrl.Rule(
            self.e_c_neighbor['Medium'] & 
            self.load_c_ratio['High'] & 
            self.d_c_bs_neighbor['Near'], 
            self.w_fur['Medium']
        ))

        # Low输出规则
        rules.append(ctrl.Rule(
            self.e_c_neighbor['Medium'] & 
            self.load_c_ratio['High'] & 
            (self.d_c_bs_neighbor['Medium'] | self.d_c_bs_neighbor['Far']), 
            self.w_fur['Low']
        ))
        rules.append(ctrl.Rule(self.e_c_neighbor['Low'], self.w_fur['Low']))
        
        # Rules for w_load_neighbor
        rules.append(ctrl.Rule(self.load_c_ratio['High'], self.w_load_neighbor['High']))
        rules.append(ctrl.Rule(self.load_c_ratio['Medium'] & self.e_c_neighbor['Low'], self.w_load_neighbor['High']))
        rules.append(ctrl.Rule(self.load_c_ratio['Medium'] & self.e_c_neighbor['Medium'], self.w_load_neighbor['Medium']))
        rules.append(ctrl.Rule(self.load_c_ratio['Low'] & self.e_c_neighbor['Low'], self.w_load_neighbor['Medium']))
        rules.append(ctrl.Rule(self.load_c_ratio['Medium'] & self.e_c_neighbor['High'], self.w_load_neighbor['Low']))
        rules.append(ctrl.Rule(self.load_c_ratio['Low'] & (self.e_c_neighbor['Medium'] | self.e_c_neighbor['High']), self.w_load_neighbor['Low']))
        if not rules:
            logger.warning("CHToBSPathSelectionFuzzySystem: No specific rules, adding default.")
            # ... (添加默认规则的逻辑，确保所有输出都有一个默认规则)
            default_antecedent = self.e_c_neighbor['Medium'] # Pick one antecedent for default
            for out_var in [self.w_d_pro, self.w_e_cost, self.w_fur, self.w_load_neighbor]:
                if hasattr(out_var, 'terms') and 'Medium' in out_var.terms: # Assuming 'Medium' for outputs
                     rules.append(ctrl.Rule(default_antecedent, out_var['Medium']))
                elif hasattr(out_var, 'terms') and 'Neutral' in out_var.terms: # If you used Neutral for factors
                     rules.append(ctrl.Rule(default_antecedent, out_var['Neutral']))
                else:
                     logger.error(f"Cannot add default rule for {out_var.label} as 'Medium' or 'Neutral' MF is not defined.")

        logger.debug(f"Defined {len(rules)} rules for CHToBSPathSelectionFuzzySystem.")
        return rules

    def compute_weights(self, current_dc_bs_neighbor, current_e_c_neighbor, 
                        # current_lq_c_normalized, # 已移除
                        current_load_c_actual, current_r_c_success, 
                        current_e_ctx_cost_normalized,
                        avg_load_for_neighbor_ch):
        
        if self.control_system is None:
            logger.error("CHToBSPathSelectionFuzzySystem: ControlSystem is not built.")
            return {'w_d_pro': 0.5, 'w_e_cost': 0.5, 'w_fur': 0.5, 'w_load_neighbor': 0.5}

        simulation = ctrl.ControlSystemSimulation(self.control_system)
        
        # --- 准备并裁剪输入值 ---
        clipped_dc_bs_neighbor = np.clip(current_dc_bs_neighbor, self.d_c_bs_neighbor.universe.min(), self.d_c_bs_neighbor.universe.max())
        clipped_e_c_neighbor = np.clip(current_e_c_neighbor, self.e_c_neighbor.universe.min(), self.e_c_neighbor.universe.max())
        
        current_avg_load = avg_load_for_neighbor_ch if avg_load_for_neighbor_ch > 0 else \
                           (self.main_config.get('network',{}).get('node_count',100) / \
                            max(1, self.p_opt_reference * self.main_config.get('network',{}).get('node_count',100) ) )
        load_c_ratio_val = current_load_c_actual / current_avg_load if current_avg_load > 0 else 0
        clipped_load_c_ratio = np.clip(load_c_ratio_val, self.load_c_ratio.universe.min(), self.load_c_ratio.universe.max())
        
        clipped_r_c_success = np.clip(current_r_c_success, self.r_c_success.universe.min(), self.r_c_success.universe.max())
        clipped_e_ctx_cost_normalized = np.clip(current_e_ctx_cost_normalized, self.e_ctx_cost_normalized.universe.min(), self.e_ctx_cost_normalized.universe.max())

        # --- 设置输入值 ---
        simulation.input['d_c_bs_neighbor'] = clipped_dc_bs_neighbor
        simulation.input['e_c_neighbor'] = clipped_e_c_neighbor
        simulation.input['load_c_ratio'] = clipped_load_c_ratio
        simulation.input['r_c_success'] = clipped_r_c_success
        simulation.input['e_ctx_cost_normalized'] = clipped_e_ctx_cost_normalized
        
        output_dict = {}
        try:
            logger.debug(f"--- Fuzzy Inputs for CHToBS (before compute) ---")
            logger.debug(f"  d_c_bs_neighbor: {clipped_dc_bs_neighbor:.2f}")
            logger.debug(f"  e_c_neighbor: {clipped_e_c_neighbor:.2f}")
            logger.debug(f"  load_c_ratio: {clipped_load_c_ratio:.2f}")
            logger.debug(f"  r_c_success: {clipped_r_c_success:.2f}")
            logger.debug(f"  e_ctx_cost_normalized: {clipped_e_ctx_cost_normalized:.2f}")

            simulation.compute() 
            
            # logger.debug(f"--- Raw Simulation Output CHToBS (after compute) ---")
            # if hasattr(simulation, 'output') and isinstance(simulation.output, dict):
            #     for key, value in simulation.output.items():
            #         logger.debug(f"  Output '{key}': {value}")

            output_vars_to_get = {
                'w_d_pro': 0.5, 'w_e_cost': 0.5, 'w_fur': 0.5, 'w_load_neighbor': 0.5
            }
            
            for var_name, default_val in output_vars_to_get.items():
                try:
                    output_dict[var_name] = simulation.output[var_name]
                except KeyError:
                    logger.warning(f"CHToBSFuzzy: Output var '{var_name}' not found in simulation.output, using default {default_val}.")
                    # *** 修改这里的调试打印 ***
                    logger.warning(f"  Problematic Inputs that led to missing '{var_name}':")
                    logger.warning(f"    d_c_bs_neighbor (clipped): {clipped_dc_bs_neighbor:.3f}")
                    logger.warning(f"    e_c_neighbor (clipped): {clipped_e_c_neighbor:.3f}")
                    logger.warning(f"    load_c_ratio (clipped): {clipped_load_c_ratio:.3f}")
                    logger.warning(f"    r_c_success (clipped): {clipped_r_c_success:.3f}")
                    logger.warning(f"    e_ctx_cost_normalized (clipped): {clipped_e_ctx_cost_normalized:.3f}")
                    # *************************
                    output_dict[var_name] = default_val
                except TypeError: 
                    logger.warning(f"CHToBSFuzzy: simulation.output is not dict for '{var_name}'. Using default.")
                    output_dict[var_name] = default_val
            return output_dict

        except Exception as e: 
            logger.error(f"Error in CHToBSPathSelectionFuzzySystem compute: {e}", exc_info=True)
            # 打印原始传入的参数
            logger.error(f"  Original Inputs: dc_bs_neighbor={current_dc_bs_neighbor}, e_c_neighbor={current_e_c_neighbor}, "
                         f"load_c_actual={current_load_c_actual}, r_c_success={current_r_c_success}, "
                         f"e_ctx_cost_norm={current_e_ctx_cost_normalized}, avg_load_nh={avg_load_for_neighbor_ch}")
            return {'w_d_pro': 0.5, 'w_e_cost': 0.5, 'w_fur': 0.5, 'w_load_neighbor': 0.5}

    def view_antecedent(self, name): # Same as in NormalNodeCHSelectionFuzzySystem
        if hasattr(self, name) and getattr(self, name) is not None:
            getattr(self, name).view()
            plt.show(block=True)
        else: print(f"Antecedent {name} not found or not initialized.")
            
    def view_consequent(self, name): # Same as in NormalNodeCHSelectionFuzzySystem
        if hasattr(self, name) and getattr(self, name) is not None:
            getattr(self, name).view()
            plt.show(block=True)
        else: print(f"Consequent {name} not found or not initialized.")

class RewardWeightsFuzzySystemForCHCompetition:
    def __init__(self, main_sim_config): # 传入整个主仿真config
        """
        Initializes the fuzzy system for adjusting CH competition reward weights.
        """
        self.main_config = main_sim_config # 保存主配置的引用
        self.fuzzy_specific_config = self.main_config.get('fuzzy', {})
        
        # 从specific_config或main_config中获取p_opt_ref
        # 这个 p_opt_reference 用于定义 ch_density_global 的模糊集
        self.p_opt_reference = float(self.main_config.get('deec', {}).get('p_opt', 0.1)
                                          )
        logger.debug(f"RewardWeightsFuzzySystem: p_opt_reference set to {self.p_opt_reference}")

        # --- Antecedents (Inputs) ---
        self.network_energy_level = None
        self.node_self_energy = None
        self.ch_density_global = None
        self.ch_to_bs_dis = None

        # --- Consequents (Outputs) ---
        self.w_members_factor = None
        self.w_energy_self_factor = None
        self.w_cost_ch_factor = None
        self.w_rotation_factor = None
        self.w_dis = None

        # --- Define MFs and Rules once ---
        self._define_antecedents_and_consequents_once()
        self.rules = self._define_rules_once() # Store rules list as an attribute

        # ControlSystem will be created once, Simulation will be created per compute call
        if not self.rules:
            logger.error("RewardWeightsFuzzySystem: No rules defined! System will not work.")
            self.control_system = None # Mark as not built
        else:
            self.control_system = ctrl.ControlSystem(self.rules)
            logger.info("RewardWeightsFuzzySystem: ControlSystem built successfully.")


    def _define_antecedents_and_consequents_once(self):
        logger.debug("Defining antecedents and consequents for RewardWeightsFuzzySystem...")
        # 1. Network_Energy_Level (归一化 [0,1])
        universe_net_energy = np.arange(0, 1.01, 0.01)
        self.network_energy_level = ctrl.Antecedent(universe_net_energy, 'network_energy_level')
        self.network_energy_level['Low'] = fuzz.zmf(self.network_energy_level.universe, 0.2, 0.4)
        self.network_energy_level['Medium'] = fuzz.trimf(self.network_energy_level.universe, [0.3, 0.6, 0.8])
        self.network_energy_level['High'] = fuzz.smf(self.network_energy_level.universe, 0.7, 0.9)

        # 2. Node_Self_Energy (归一化 [0,1])
        universe_self_energy = np.arange(0, 1.01, 0.01)
        self.node_self_energy = ctrl.Antecedent(universe_self_energy, 'node_self_energy')
        self.node_self_energy['Low'] = fuzz.zmf(self.node_self_energy.universe, 0.2, 0.4)
        self.node_self_energy['Medium'] = fuzz.trimf(self.node_self_energy.universe, [0.3, 0.5, 0.7])
        self.node_self_energy['High'] = fuzz.smf(self.node_self_energy.universe, 0.6, 0.8)

        # 3. CH_Density_Global
        p_opt = self.p_opt_reference 
        universe_ch_density = np.arange(0, p_opt * 3 + 0.01, 0.01)
        self.ch_density_global = ctrl.Antecedent(universe_ch_density, 'ch_density_global')
        self.ch_density_global['Too_Low'] = fuzz.zmf(self.ch_density_global.universe, p_opt * 0.5, p_opt * 0.8)
        self.ch_density_global['Optimal'] = fuzz.trimf(self.ch_density_global.universe, [p_opt * 0.7, p_opt, p_opt * 1.3])
        self.ch_density_global['Too_High'] = fuzz.smf(self.ch_density_global.universe, p_opt * 1.2, p_opt * 1.5)

        # 4. CH_to_BS_Distance (节点到BS的归一化距离 [0,1])
        universe_distance = np.arange(0, 1.01, 0.01)
        self.ch_to_bs_dis = ctrl.Antecedent(universe_distance, 'ch_to_bs_dis')
        self.ch_to_bs_dis['Low'] = fuzz.zmf(self.ch_to_bs_dis.universe, 0.20, 0.30) 
        self.ch_to_bs_dis['Medium'] = fuzz.trimf(self.ch_to_bs_dis.universe, [0.25, 0.5, 0.75])
        self.ch_to_bs_dis['High'] = fuzz.smf(self.ch_to_bs_dis.universe, 0.70, 0.80)

        # --- Consequents ---
        universe_factors = np.arange(0.5, 1.51, 0.01) 
        self.w_members_factor = ctrl.Consequent(universe_factors, 'w_members_factor')
        self.w_energy_self_factor = ctrl.Consequent(universe_factors, 'w_energy_self_factor')
        self.w_cost_ch_factor = ctrl.Consequent(universe_factors, 'w_cost_ch_factor')
        self.w_rotation_factor = ctrl.Consequent(universe_factors, 'w_rotation_factor')
        self.w_dis = ctrl.Consequent(universe_factors, 'w_dis')

        for out_var in [self.w_members_factor, self.w_energy_self_factor, 
                        self.w_cost_ch_factor, self.w_rotation_factor, self.w_dis]:
            out_var['Decrease'] = fuzz.zmf(out_var.universe, 0.7, 0.9)
            out_var['Neutral'] = fuzz.trimf(out_var.universe, [0.85, 1.0, 1.15])
            out_var['Increase'] = fuzz.smf(out_var.universe, 1.1, 1.3)
        logger.debug("Antecedents and consequents defined.")

    # in fuzzy.py -> class RewardWeightsFuzzySystemForCHCompetition

    # in fuzzy.py -> class RewardWeightsFuzzySystemForCHCompetition

    def _define_rules_once(self):
        """
        定义一套修正后的、全覆盖的模糊规则。
        """
        logger.debug("Defining a new, robust, full-coverage set of fuzzy rules for RewardWeightsFuzzySystem...")
        rules = []

        # ======================================================================
        # 规则集 1: w_members_factor (成员收益权重)
        # 主要由全局CH密度决定。
        # ======================================================================
        rules.append(ctrl.Rule(self.ch_density_global['Too_Low'],  self.w_members_factor['Increase']))
        rules.append(ctrl.Rule(self.ch_density_global['Optimal'],  self.w_members_factor['Neutral']))
        rules.append(ctrl.Rule(self.ch_density_global['Too_High'], self.w_members_factor['Decrease']))

        # ======================================================================
        # 规则集 2: w_cost_ch_factor (成为CH的成本权重)
        # 主要由全局CH密度决定。
        # ======================================================================
        rules.append(ctrl.Rule(self.ch_density_global['Too_Low'],  self.w_cost_ch_factor['Decrease']))
        rules.append(ctrl.Rule(self.ch_density_global['Optimal'],  self.w_cost_ch_factor['Neutral']))
        rules.append(ctrl.Rule(self.ch_density_global['Too_High'], self.w_cost_ch_factor['Increase']))
        # 增加一条：当网络整体能量很低时，成为CH的成本应该降低，以鼓励有能力的节点站出来。
        rules.append(ctrl.Rule(self.network_energy_level['Low'], self.w_cost_ch_factor['Decrease']))

        # ======================================================================
        # 规则集 3: w_energy_self_factor (自身能量贡献权重) - **修正版**
        # 主要由节点自身能量决定，并由网络整体能量进行微调。
        # 这样可以保证任何自身能量状态都有对应的规则。
        # ======================================================================
        rules.append(ctrl.Rule(self.node_self_energy['Low'], self.w_energy_self_factor['Decrease']))
        rules.append(ctrl.Rule(self.node_self_energy['Medium'], self.w_energy_self_factor['Neutral']))
        # 当节点自身能量高时，再考虑网络整体情况
        rules.append(ctrl.Rule(self.node_self_energy['High'] & self.network_energy_level['High'], self.w_energy_self_factor['Neutral']))
        rules.append(ctrl.Rule(self.node_self_energy['High'] & (self.network_energy_level['Medium'] | self.network_energy_level['Low']), self.w_energy_self_factor['Increase']))

        # ======================================================================
        # 规则集 4: w_rotation_factor (轮换收益权重)
        # 主要由CH密度决定，不应该让能量低的节点被过度鼓励。
        # ======================================================================
        rules.append(ctrl.Rule(self.ch_density_global['Too_Low'], self.w_rotation_factor['Increase']))
        rules.append(ctrl.Rule(self.ch_density_global['Optimal'], self.w_rotation_factor['Neutral']))
        rules.append(ctrl.Rule(self.ch_density_global['Too_High'], self.w_rotation_factor['Decrease']))
        # 增加一条：能量低的节点不应该强调轮换收益
        rules.append(ctrl.Rule(self.node_self_energy['Low'], self.w_rotation_factor['Decrease']))

        # ======================================================================
        # 规则集 5: w_dis (距离影响权重)
        # 主要由距离本身决定，并由CH密度微调。
        # ======================================================================
        # 距离太近或太远都是缺点，增加惩罚（w_dis -> Increase）
        rules.append(ctrl.Rule(self.ch_to_bs_dis['Low'],  self.w_dis['Increase']))
        rules.append(ctrl.Rule(self.ch_to_bs_dis['High'], self.w_dis['Increase']))
        # 距离适中是优点，降低惩罚（w_dis -> Decrease）
        rules.append(ctrl.Rule(self.ch_to_bs_dis['Medium'], self.w_dis['Decrease']))
        # 微调：如果CH极度稀缺，可以稍微容忍位置不佳的CH
        rules.append(ctrl.Rule((self.ch_to_bs_dis['Low'] | self.ch_to_bs_dis['High']) & self.ch_density_global['Too_Low'], self.w_dis['Neutral']))


        if not rules:
            logger.warning("RewardWeightsFuzzySystem: No rules were defined, this should not happen.")
            # 添加一个绝对的后备规则
            rules.append(ctrl.Rule(self.network_energy_level['Medium'], (self.w_members_factor['Neutral'], self.w_cost_ch_factor['Neutral'])))

        logger.debug(f"Defined {len(rules)} new, robust, full-coverage rules for RewardWeightsFuzzySystem.")
        return rules

    def compute_reward_weights(self, current_net_energy_level, current_node_self_energy, 
                               current_ch_density_global, current_ch_to_bs_dis_normalized):
        
        if self.control_system is None:
            logger.error("RewardWeightsFuzzySystem: ControlSystem is not built. Cannot compute.")
            # Return neutral weights if system isn't ready
            return {'w_members_factor': 1.0, 'w_energy_self_factor': 1.0, 
                    'w_cost_ch_factor': 1.0, 'w_rotation_factor': 1.0, 'w_dis': 1.0}

        # Create a new simulation instance for each computation to ensure fresh state
        simulation = ctrl.ControlSystemSimulation(self.control_system)
        # logger.debug("Created new ControlSystemSimulation instance for compute_reward_weights.")

        # --- 准备并裁剪输入值 ---
        clipped_net_energy = np.clip(current_net_energy_level, self.network_energy_level.universe.min(), self.network_energy_level.universe.max())
        clipped_self_energy = np.clip(current_node_self_energy, self.node_self_energy.universe.min(), self.node_self_energy.universe.max())
        clipped_ch_density = np.clip(current_ch_density_global, self.ch_density_global.universe.min(), self.ch_density_global.universe.max())
        clipped_ch_to_bs_dis = np.clip(current_ch_to_bs_dis_normalized, self.ch_to_bs_dis.universe.min(), self.ch_to_bs_dis.universe.max())

        # --- 设置输入值 ---
        simulation.input['network_energy_level'] = clipped_net_energy
        simulation.input['node_self_energy'] = clipped_self_energy
        simulation.input['ch_density_global'] = clipped_ch_density
        simulation.input['ch_to_bs_dis'] = clipped_ch_to_bs_dis
        
        output_dict = {}
        try:
            # logger.debug(f"--- Fuzzy Inputs for compute_reward_weights (before compute) ---")
            # logger.debug(f"  network_energy_level (clipped): {clipped_net_energy:.3f}")
            # logger.debug(f"  node_self_energy (clipped): {clipped_self_energy:.3f}")
            # logger.debug(f"  ch_density_global (clipped): {clipped_ch_density:.3f}")
            # logger.debug(f"  ch_to_bs_dis (clipped): {clipped_ch_to_bs_dis:.3f}")

            simulation.compute() 
            
            # logger.debug(f"--- Raw Simulation Output (all available keys after compute) ---")
            # if hasattr(simulation, 'output') and isinstance(simulation.output, dict):
            #     for key, value in simulation.output.items():
            #         logger.debug(f"  Output '{key}': {value}")
            # else:
            #     logger.warning("  self.simulation.output is not a directly iterable dictionary after compute.")

            output_vars_to_get = {
                'w_members_factor': 1.0, 
                'w_energy_self_factor': 1.0, 
                'w_cost_ch_factor': 1.0, 
                'w_rotation_factor': 1.0,
                'w_dis': 1.0 
            }
            
            for var_name, default_val in output_vars_to_get.items():
                try:
                    output_dict[var_name] = simulation.output[var_name]
                except KeyError:
                    logger.warning(f"输出变量 '{var_name}' 在 simulation.output 中未找到，使用默认值 {default_val}.")
                    output_dict[var_name] = default_val
                except TypeError: 
                    logger.warning(f"simulation.output 不是字典类型，无法获取 '{var_name}'。使用默认值。")
                    output_dict[var_name] = default_val
            
            return output_dict

        except Exception as e: 
            logger.error(f"错误：在 RewardWeightsFuzzySystem 计算或输出获取中: {e}", exc_info=True) # Add exc_info for full traceback
            logger.error(f"  Input net_energy: {current_net_energy_level}, self_energy: {current_node_self_energy}, density: {current_ch_density_global}, bs_dist: {current_ch_to_bs_dis_normalized}")
            return {'w_members_factor': 1.0, 'w_energy_self_factor': 1.0, 
                    'w_cost_ch_factor': 1.0, 'w_rotation_factor': 1.0, 'w_dis': 1.0}

# in fuzzy.py
# (确保文件顶部有 import numpy as np, skfuzzy as fuzz, from skfuzzy import control as ctrl)

# =============================================================================
# Meta-Fuzzy Controller for CH Selection Strategy
# =============================================================================
class CHSelectionStrategyFuzzySystem:
    def __init__(self, main_sim_config):
        """
        初始化高阶模糊逻辑控制器，用于动态调整CH选举策略的权重。
        """
        self.main_config = main_sim_config
        logger.info("Initializing Meta-Fuzzy Controller for CH Selection Strategy...")

         # --- Antecedents (Inputs) ---
        self.net_health_pdr = None      # 网络PDR (0-1)
        self.net_energy_reserve = None  # 网络平均能量 (0-1)
        self.isolated_node_rate = None  # 孤立节点率 (0-1)
        self.net_congestion_level = None# 网络拥塞水平 (0-1)
        
        # --- Consequents (Outputs) ---
        self.p_opt_adjustment_factor = None

        self._define_antecedents_and_consequents()
        self.rules = self._define_rules()
        
        if not self.rules:
            logger.error("CHSelectionStrategyFuzzySystem: No rules defined!")
            self.control_system = None
        else:
            self.control_system = ctrl.ControlSystem(self.rules)
            logger.info("CHSelectionStrategyFuzzySystem: ControlSystem built successfully.")

    # in fuzzy.py -> class CHSelectionStrategyFuzzySystem

    def _define_antecedents_and_consequents(self):
        """定义输入和输出变量的模糊集。"""
        # --- 输入1: 网络健康度 (PDR) ---
        universe_pdr = np.arange(0, 1.01, 0.01)
        self.net_health_pdr = ctrl.Antecedent(universe_pdr, 'net_health_pdr')
        self.net_health_pdr['Poor'] = fuzz.zmf(self.net_health_pdr.universe, 0.2, 0.4)
        self.net_health_pdr['Acceptable'] = fuzz.trimf(self.net_health_pdr.universe, [0.3, 0.5, 0.7])
        self.net_health_pdr['Good'] = fuzz.smf(self.net_health_pdr.universe, 0.6, 0.8)

        # --- 输入2: 网络能量储备 ---
        universe_energy = np.arange(0, 1.01, 0.01)
        self.net_energy_reserve = ctrl.Antecedent(universe_energy, 'net_energy_reserve')
        self.net_energy_reserve['Critical'] = fuzz.zmf(self.net_energy_reserve.universe, 0.15, 0.35)
        self.net_energy_reserve['Medium'] = fuzz.trimf(self.net_energy_reserve.universe, [0.3, 0.55, 0.8])
        self.net_energy_reserve['High'] = fuzz.smf(self.net_energy_reserve.universe, 0.7, 0.9)

        # --- [新增] 输入3: 孤立节点率 ---
        # 反映网络覆盖的完整性
        universe_isolated_rate = np.arange(0, 1.01, 0.01)
        self.isolated_node_rate = ctrl.Antecedent(universe_isolated_rate, 'isolated_node_rate')
        self.isolated_node_rate['Low'] = fuzz.zmf(self.isolated_node_rate.universe, 0.05, 0.15)
        self.isolated_node_rate['Medium'] = fuzz.trimf(self.isolated_node_rate.universe, [0.1, 0.2, 0.3])
        self.isolated_node_rate['High'] = fuzz.smf(self.isolated_node_rate.universe, 0.25, 0.4)

        # --- [新增] 输入4: 网络拥塞水平 ---
        # 反映CH转发缓存的平均占用率
        universe_congestion = np.arange(0, 1.01, 0.01)
        self.net_congestion_level = ctrl.Antecedent(universe_congestion, 'net_congestion_level')
        self.net_congestion_level['Low'] = fuzz.zmf(self.net_congestion_level.universe, 0.2, 0.4)
        self.net_congestion_level['Medium'] = fuzz.trimf(self.net_congestion_level.universe, [0.3, 0.5, 0.7])
        self.net_congestion_level['High'] = fuzz.smf(self.net_congestion_level.universe, 0.6, 0.8)

        # [核心修改] 定义新的输出变量
        universe_factor = np.arange(0.5, 2.01, 0.01) # 调整因子范围：[0.5, 2.0]
        self.p_opt_adjustment_factor = ctrl.Consequent(universe_factor, 'p_opt_adjustment_factor')
        self.p_opt_adjustment_factor['Decrease_Significantly'] = fuzz.trimf(self.p_opt_adjustment_factor.universe, [0.5, 0.5, 0.8])
        self.p_opt_adjustment_factor['Decrease_Slightly'] = fuzz.trimf(self.p_opt_adjustment_factor.universe, [0.7, 0.9, 1.0])
        self.p_opt_adjustment_factor['Neutral'] = fuzz.trimf(self.p_opt_adjustment_factor.universe, [0.95, 1.0, 1.05])
        self.p_opt_adjustment_factor['Increase_Slightly'] = fuzz.trimf(self.p_opt_adjustment_factor.universe, [1.0, 1.1, 1.3])
        self.p_opt_adjustment_factor['Increase_Significantly'] = fuzz.trimf(self.p_opt_adjustment_factor.universe, [1.2, 1.5, 2.0])

    # in fuzzy.py -> class CHSelectionStrategyFuzzySystem

    def _define_rules(self):
        """定义核心策略规则。"""
        rules = []
        
            # 规则组1：服务质量优先
        # PDR差或孤立节点多，必须显著增加CH数量来修复网络
        rule1 = ctrl.Rule(self.net_health_pdr['Poor'] | self.isolated_node_rate['High'], 
                        self.p_opt_adjustment_factor['Increase_Significantly'])
        # PDR尚可，轻微增加CH以提升服务
        rule2 = ctrl.Rule(self.net_health_pdr['Acceptable'],
                        self.p_opt_adjustment_factor['Increase_Slightly'])

        # 规则组2：拥塞处理
        # 拥塞严重，需要增加CH来分担流量
        rule3 = ctrl.Rule(self.net_congestion_level['High'],
                        self.p_opt_adjustment_factor['Increase_Slightly'])

        # 规则组3：能量权衡
        # 只有在服务质量良好(PDR Good)的情况下，才考虑根据能量进行调整
        # 能量高，服务好 -> 维持现状
        rule4 = ctrl.Rule(self.net_health_pdr['Good'] & self.net_energy_reserve['High'],
                        self.p_opt_adjustment_factor['Neutral'])
        # 能量中等，服务好 -> 轻微减少CH，开始节能
        rule5 = ctrl.Rule(self.net_health_pdr['Good'] & self.net_energy_reserve['Medium'],
                        self.p_opt_adjustment_factor['Decrease_Slightly'])
        # 能量危急，服务好 -> 显著减少CH，全力保生存
        rule6 = ctrl.Rule(self.net_health_pdr['Good'] & self.net_energy_reserve['Critical'],
                        self.p_opt_adjustment_factor['Decrease_Significantly'])

        rules.extend([rule1, rule2, rule3, rule4, rule5, rule6])
        return rules


    def compute_p_opt_factor(self, pdr, energy, isolated_rate, congestion):
        """
        [V8.0 健壮版] 计算p_opt调整因子。
        返回一个单一的浮点数。
        """
        # 1. 检查控制系统是否已构建
        if self.control_system is None:
            logger.error("CHSelectionStrategyFuzzySystem: ControlSystem is not built. Returning neutral factor 1.0.")
            return 1.0  # 返回一个中性的、不会造成破坏的默认值

        simulation = ctrl.ControlSystemSimulation(self.control_system)
        
        # 2. 裁剪并设置所有输入值
        simulation.input['net_health_pdr'] = np.clip(pdr, 0, 1)
        simulation.input['net_energy_reserve'] = np.clip(energy, 0, 1)
        simulation.input['isolated_node_rate'] = np.clip(isolated_rate, 0, 1)
        simulation.input['net_congestion_level'] = np.clip(congestion, 0, 1)
        
        # 3. 计算并安全地获取输出
        try:
            simulation.compute()
            
            # 使用 .get() 方法来安全地访问字典，避免KeyError
            # 如果'p_opt_adjustment_factor'不存在，则返回一个中性的默认值 1.0
            adjustment_factor = simulation.output.get('p_opt_adjustment_factor', 1.0)
            
            if adjustment_factor is None:
                logger.warning("Fuzzy computation resulted in None for p_opt_adjustment_factor. Returning 1.0.")
                return 1.0
                
            return adjustment_factor
                
        except Exception as e:
            logger.error(f"Error during CHSelectionStrategyFuzzySystem computation: {e}", exc_info=True)
            # 在发生未知错误时，也返回一个安全的中性值
            return 1.0
        
# in fuzzy.py, at the end of the file

# =============================================================================
# Fuzzy System for Guiding CH Declaration Propensity
# =============================================================================
class CHDeclarationFuzzySystem:
    def __init__(self, main_sim_config):
        self.main_config = main_sim_config
        logger.info("Initializing CH Declaration Propensity Fuzzy System...")

        # --- Antecedents (Inputs) ---
        self.node_energy = None             # 节点自身归一化能量 [0, 1]
        self.distance_to_bs = None          # 节点到BS的归一化距离 [0, 1]
        self.local_density = None           # 节点的局部密度（归一化邻居数）[0, 1]
        self.coverage_gini = None           # CH覆盖范围的基尼系数 [0, 1]，值越大越不均衡

        # --- Consequent (Output) ---
        self.declaration_propensity = None  # 宣告倾向性 [0, 1]

        self._define_antecedents_and_consequents()
        self.rules = self._define_rules()
        
        if not self.rules:
            logger.error("CHDeclarationFuzzySystem: No rules defined!")
            self.control_system = None
        else:
            self.control_system = ctrl.ControlSystem(self.rules)
            logger.info("CHDeclarationFuzzySystem: ControlSystem built successfully.")

    def _define_antecedents_and_consequents(self):
        """定义输入和输出变量的模糊集。"""
        # --- Inputs ---
        universe_norm = np.arange(0, 1.01, 0.01)

        self.node_energy = ctrl.Antecedent(universe_norm, 'node_energy')
        self.node_energy['Low'] = fuzz.zmf(self.node_energy.universe, 0.2, 0.4)
        self.node_energy['Medium'] = fuzz.trimf(self.node_energy.universe, [0.3, 0.5, 0.7])
        self.node_energy['High'] = fuzz.smf(self.node_energy.universe, 0.6, 0.8)

        self.distance_to_bs = ctrl.Antecedent(universe_norm, 'distance_to_bs')
        self.distance_to_bs['Near'] = fuzz.zmf(self.distance_to_bs.universe, 0.2, 0.4)
        self.distance_to_bs['Medium'] = fuzz.trimf(self.distance_to_bs.universe, [0.3, 0.5, 0.7])
        self.distance_to_bs['Far'] = fuzz.smf(self.distance_to_bs.universe, 0.6, 0.8)

        self.local_density = ctrl.Antecedent(universe_norm, 'local_density')
        self.local_density['Sparse'] = fuzz.zmf(self.local_density.universe, 0.1, 0.3)
        self.local_density['Medium'] = fuzz.trimf(self.local_density.universe, [0.2, 0.4, 0.6])
        self.local_density['Dense'] = fuzz.smf(self.local_density.universe, 0.5, 0.7)

        self.coverage_gini = ctrl.Antecedent(universe_norm, 'coverage_gini')
        self.coverage_gini['Balanced'] = fuzz.zmf(self.coverage_gini.universe, 0.2, 0.4) # 基尼系数小，均衡
        self.coverage_gini['Unbalanced'] = fuzz.smf(self.coverage_gini.universe, 0.3, 0.5) # 基尼系数大，不均衡

        # --- Output ---
        self.declaration_propensity = ctrl.Consequent(universe_norm, 'declaration_propensity')
        self.declaration_propensity['Very_Low'] = fuzz.zmf(self.declaration_propensity.universe, 0.1, 0.2)
        self.declaration_propensity['Low'] = fuzz.trimf(self.declaration_propensity.universe, [0.1, 0.25, 0.4])
        self.declaration_propensity['Medium'] = fuzz.trimf(self.declaration_propensity.universe, [0.3, 0.5, 0.7])
        self.declaration_propensity['High'] = fuzz.trimf(self.declaration_propensity.universe, [0.6, 0.75, 0.9])
        self.declaration_propensity['Very_High'] = fuzz.smf(self.declaration_propensity.universe, 0.8, 0.9)

    # in fuzzy.py -> class CHDeclarationFuzzySystem

    # in fuzzy.py -> class CHDeclarationFuzzySystem

    # in fuzzy.py -> class CHDeclarationFuzzySystem

    # in fuzzy.py -> class CHDeclarationFuzzySystem

    def _define_rules(self):
        """
        [V10-解耦版] 定义一个绝对无死角的、全覆盖的核心引导规则。
        核心思想：每个输入独立地对输出产生影响，由模糊系统自动聚合。
        """
        rules = []

        # 规则组1：能量的影响 (Energy Contribution)
        # 能量越高，倾向性越高
        rules.append(ctrl.Rule(self.node_energy['Low'], self.declaration_propensity['Very_Low']))
        rules.append(ctrl.Rule(self.node_energy['Medium'], self.declaration_propensity['Medium']))
        rules.append(ctrl.Rule(self.node_energy['High'], self.declaration_propensity['High']))

        # 规则组2：到BS距离的影响 (Distance Contribution)
        # 越远（通常意味着越在边缘），越需要被扶持，倾向性越高
        rules.append(ctrl.Rule(self.distance_to_bs['Near'], self.declaration_propensity['Low']))
        rules.append(ctrl.Rule(self.distance_to_bs['Medium'], self.declaration_propensity['Medium']))
        rules.append(ctrl.Rule(self.distance_to_bs['Far'], self.declaration_propensity['High']))

        # 规则组3：局部密度的影响 (Density Contribution)
        # 越稀疏，越需要成为CH来服务该区域，倾向性越高
        rules.append(ctrl.Rule(self.local_density['Sparse'], self.declaration_propensity['High']))
        rules.append(ctrl.Rule(self.local_density['Medium'], self.declaration_propensity['Medium']))
        rules.append(ctrl.Rule(self.local_density['Dense'], self.declaration_propensity['Low']))

        # 规则组4：网络覆盖均衡度的影响 (Gini Contribution)
        # 网络越不均衡，越需要一个“颠覆性”的高倾向性建议；网络均衡，则倾向性建议趋于中性
        rules.append(ctrl.Rule(self.coverage_gini['Balanced'], self.declaration_propensity['Medium']))
        rules.append(ctrl.Rule(self.coverage_gini['Unbalanced'], self.declaration_propensity['Very_High']))

        logger.debug(f"Defined {len(rules)} decoupled rules for CHDeclarationFuzzySystem.")
        return rules

    def compute_propensity(self, energy, distance, density, gini):
        if self.control_system is None:
            return 0.5 # Fallback

        simulation = ctrl.ControlSystemSimulation(self.control_system)
        simulation.input['node_energy'] = np.clip(energy, 0, 1)
        simulation.input['distance_to_bs'] = np.clip(distance, 0, 1)
        simulation.input['local_density'] = np.clip(density, 0, 1)
        simulation.input['coverage_gini'] = np.clip(gini, 0, 1)
        
        try:
            simulation.compute()
            return simulation.output['declaration_propensity']
        except Exception as e:
            logger.warning(f"CHDeclarationFuzzySystem compute failed: {e}. Returning fallback 0.5")
            return 0.5
# =============================================================================
# Example Usage
# =============================================================================
if __name__ == '__main__':
    # --- Example for Normal Node CH Selection ---
    print("--- Normal Node CH Selection Example ---")
    sim_node_sum = 100
    sim_cluster_sum = 10
    sim_avg_e_send_normal = 0.002 # Avg send energy for normal node to CH

    normal_node_fuzzy_selector = NormalNodeCHSelectionFuzzySystem(
        node_sum=sim_node_sum,
        cluster_sum=sim_cluster_sum
    )
    # Pass the specific average send energy for this context
    weights_normal = normal_node_fuzzy_selector.compute_weights(
        current_dc_base=80, current_e_cluster=0.7, current_p_cluster_actual=5,
        current_r_success=0.9, current_e_send_total_actual=0.0015,
        avg_e_send_total_for_normal_node=sim_avg_e_send_normal
    )
    print("Normal Node CH Selection Weights:")
    for wn_name, wn_value in weights_normal.items(): # Changed variable names for clarity
        print(f"  {wn_name}: {wn_value:.4f}") # Format to 4 decimal places
    #normal_node_fuzzy_selector.view_antecedent('e_cluster')


    # --- Example for CH to BS Path Selection ---
    print("\n--- CH to BS Path Selection Example ---")
    # These averages might be different or obtained differently for CH-to-CH communication
    sim_avg_load_for_neighbor = sim_node_sum / sim_cluster_sum if sim_cluster_sum > 0 else 10

    ch_path_fuzzy_selector = CHToBSPathSelectionFuzzySystem(
        node_sum=sim_node_sum, # Might be used to calculate avg_load_for_neighbor if not passed directly
        cluster_sum=sim_cluster_sum
    )
    weights_ch = ch_path_fuzzy_selector.compute_weights(
        current_dc_bs_neighbor=100, current_e_c_neighbor=0.8,
        current_load_c_actual=8,
        current_r_c_success=0.95, current_e_ctx_cost_normalized=0.1,
        avg_load_for_neighbor_ch=sim_avg_load_for_neighbor
    )
    print("CH to BS Path Selection Weights:")
    for wc_name, wc_value in weights_ch.items(): # Changed variable names for clarity
        print(f"  {wc_name}: {wc_value:.4f}") # Format to 4 decimal places
    # ch_path_fuzzy_selector.view_antecedent('d_c_bs_neighbor')
    # ch_path_fuzzy_selector.view_antecedent('e_c_neighbor')
    # ch_path_fuzzy_selector.view_antecedent('lq_c_normalized')

    print("\nScript finished. Close any open plots to exit.")