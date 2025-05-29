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
        rules.append(ctrl.Rule(self.e_cluster['Low'], self.w_e_ch['Low']))
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & (self.d_c_base['Far'] | self.p_cluster_ratio['High']), self.w_e_ch['Low']))
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & (self.d_c_base['Medium'] | self.p_cluster_ratio['Medium']), self.w_e_ch['Medium']))
        rules.append(ctrl.Rule(self.e_cluster['High'] & (self.d_c_base['Far'] | self.p_cluster_ratio['High']), self.w_e_ch['Medium']))
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & (self.d_c_base['Near'] | self.p_cluster_ratio['Low']), self.w_e_ch['High']))
        rules.append(ctrl.Rule(self.e_cluster['High'] & (self.d_c_base['Medium'] | self.p_cluster_ratio['Medium']), self.w_e_ch['High']))
        rules.append(ctrl.Rule(self.e_cluster['High'] & (self.d_c_base['Near'] | self.p_cluster_ratio['Low']), self.w_e_ch['High']))

        # Rules for w_path
        rules.append(ctrl.Rule(self.r_success['High'] & (self.e_send_total_ratio['Low'] | self.e_send_total_ratio['Medium']), self.w_path['Low']))
        rules.append(ctrl.Rule(self.r_success['High'] & self.e_send_total_ratio['High'], self.w_path['Medium']))
        rules.append(ctrl.Rule(self.r_success['Medium'] & (self.e_send_total_ratio['Low'] | self.e_send_total_ratio['Medium']), self.w_path['Medium']))
        rules.append(ctrl.Rule(self.r_success['Medium'] & self.e_send_total_ratio['High'], self.w_path['High']))
        rules.append(ctrl.Rule(self.r_success['Low'], self.w_path['High']))
        
        # Rules for w_load
        rules.append(ctrl.Rule(self.p_cluster_ratio['High'], self.w_load['High']))
        rules.append(ctrl.Rule(self.p_cluster_ratio['Medium'] & self.e_cluster['Low'], self.w_load['High']))
        rules.append(ctrl.Rule(self.p_cluster_ratio['Medium'] & self.e_cluster['Medium'], self.w_load['Medium']))
        rules.append(ctrl.Rule(self.p_cluster_ratio['Low'] & self.e_cluster['Low'], self.w_load['Medium']))
        rules.append(ctrl.Rule(self.p_cluster_ratio['Medium'] & self.e_cluster['High'], self.w_load['Low']))
        rules.append(ctrl.Rule(self.p_cluster_ratio['Low'] & (self.e_cluster['Medium'] | self.e_cluster['High']), self.w_load['Low']))

        # Rules for w_dist_bs
        rules.append(ctrl.Rule(self.d_c_base['Near'], self.w_dist_bs['Low']))
        rules.append(ctrl.Rule(self.d_c_base['Medium'] & self.e_cluster['High'], self.w_dist_bs['Low']))
        rules.append(ctrl.Rule(self.d_c_base['Far'] & self.e_cluster['Medium'], self.w_dist_bs['Medium']))
        rules.append(ctrl.Rule(self.d_c_base['Far'] & self.e_cluster['High'], self.w_dist_bs['Medium']))
        rules.append(ctrl.Rule(self.d_c_base['Medium'] & self.e_cluster['Medium'], self.w_dist_bs['Medium']))
        rules.append(ctrl.Rule(self.d_c_base['Far'] & self.e_cluster['Low'], self.w_dist_bs['High']))
        rules.append(ctrl.Rule(self.d_c_base['Medium'] & self.e_cluster['Low'], self.w_dist_bs['High']))
        
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
    def __init__(self, node_sum, cluster_sum): # Add other necessary averages if needed
        """
        Initializes the fuzzy control system for CH selecting path to BS.
        Args:
            node_sum (int): Total number of nodes.
            cluster_sum (int): Current number of CHs.
        """
        if cluster_sum <= 0:
            self.avg_load_per_ch_for_neighbor = node_sum / 1.0 if node_sum > 0 else 10
        else:
            self.avg_load_per_ch_for_neighbor = node_sum / cluster_sum
        
        # --- Antecedents (Inputs) ---
        self.d_c_bs_neighbor = None     # Distance: Neighbor CH to BS
        self.e_c_neighbor = None        # Energy: Neighbor CH's Normalized Remaining Energy
        # self.lq_c_ratio = None          # Link Quality: Normalized or Ratio to Avg. (NEEDS avg_lq)
        self.load_c_ratio = None        # Load: Neighbor CH's Current Load Ratio
        self.r_c_success = None         # Success Rate: Historical Comm. Success with Neighbor (Normalized)
        # self.e_ctx_cost_ratio = None    # CTX Cost: Energy cost to send to this neighbor (NEEDS avg_ctx_cost)

        # For simplicity, LQ_c and E_ctx_cost are assumed to be directly normalized to [0,1] for now
        # Or you can pass their respective averages to __init__ like avg_load
        self.lq_c_normalized = None     # Link Quality: Normalized [0,1]
        self.e_ctx_cost_normalized = None # CTX Cost: Normalized [0,1]


        # --- Consequents (Outputs) ---
        self.w_d_pro = None  # Path progress weight
        self.w_e_cost = None # Energy cost weight
        self.w_fur = None    # Sustainability/Future energy weight
        self.w_load_neighbor = None # Neighbor load weight (renamed from w_load to be specific)

        self.control_system = None
        self.simulation = None
        self._build_system()

    def _define_antecedents(self):
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

    def _define_consequents(self):
        universe_weights = np.arange(0, 1.01, 0.01)
        self.w_d_pro = ctrl.Consequent(universe_weights, 'w_d_pro')
        self.w_e_cost = ctrl.Consequent(universe_weights, 'w_e_cost')
        self.w_fur = ctrl.Consequent(universe_weights, 'w_fur')
        self.w_load_neighbor = ctrl.Consequent(universe_weights, 'w_load_neighbor')

        for output_var in [self.w_d_pro, self.w_e_cost, self.w_fur, self.w_load_neighbor]:
            output_var['Low'] = fuzz.zmf(output_var.universe, 0.2, 0.4)
            output_var['Medium'] = fuzz.trimf(output_var.universe, [0.3, 0.5, 0.7])
            output_var['High'] = fuzz.smf(output_var.universe, 0.6, 0.8)
    
    def _define_rules(self):
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
        rules.append(ctrl.Rule(self.e_c_neighbor['High'] & (self.load_c_ratio['Low'] | self.load_c_ratio['Medium']) & 
                               (self.d_c_bs_neighbor['Near'] | self.d_c_bs_neighbor['Medium']), self.w_fur['High']))
        rules.append(ctrl.Rule(self.e_c_neighbor['High'] & self.load_c_ratio['High'] & self.d_c_bs_neighbor['Far'], self.w_fur['Medium']))
        rules.append(ctrl.Rule(self.e_c_neighbor['Medium'] & (self.load_c_ratio['Low'] | self.load_c_ratio['Medium']) & 
                               (self.d_c_bs_neighbor['Near'] | self.d_c_bs_neighbor['Medium']), self.w_fur['Medium']))
        rules.append(ctrl.Rule(self.e_c_neighbor['Medium'] & self.load_c_ratio['High'] & self.d_c_bs_neighbor['Far'], self.w_fur['Low']))
        rules.append(ctrl.Rule(self.e_c_neighbor['Low'], self.w_fur['Low']))
        
        # Rules for w_load_neighbor
        rules.append(ctrl.Rule(self.load_c_ratio['High'], self.w_load_neighbor['High']))
        rules.append(ctrl.Rule(self.load_c_ratio['Medium'] & self.e_c_neighbor['Low'], self.w_load_neighbor['High']))
        rules.append(ctrl.Rule(self.load_c_ratio['Medium'] & self.e_c_neighbor['Medium'], self.w_load_neighbor['Medium']))
        rules.append(ctrl.Rule(self.load_c_ratio['Low'] & self.e_c_neighbor['Low'], self.w_load_neighbor['Medium']))
        rules.append(ctrl.Rule(self.load_c_ratio['Medium'] & self.e_c_neighbor['High'], self.w_load_neighbor['Low']))
        rules.append(ctrl.Rule(self.load_c_ratio['Low'] & (self.e_c_neighbor['Medium'] | self.e_c_neighbor['High']), self.w_load_neighbor['Low']))
        return rules

    def _build_system(self):
        self._define_antecedents()
        self._define_consequents()
        rules = self._define_rules()
        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def compute_weights(self, current_dc_bs_neighbor, current_e_c_neighbor, 
                         current_load_c_actual, current_r_c_success, current_e_ctx_cost_normalized,
                        avg_load_for_neighbor_ch): # Pass the relevant average load
        if self.simulation is None:
            raise Exception("Fuzzy system not built.")

        # Normalize dynamic inputs (only load_c needs it here based on current design)
        current_avg_load = avg_load_for_neighbor_ch if avg_load_for_neighbor_ch > 0 else self.avg_load_per_ch_for_neighbor # Fallback
        load_c_ratio_val = current_load_c_actual / current_avg_load if current_avg_load > 0 else 0
        load_c_ratio_val = np.clip(load_c_ratio_val, self.load_c_ratio.universe.min(), self.load_c_ratio.universe.max())
        
        self.simulation.input['d_c_bs_neighbor'] = np.clip(current_dc_bs_neighbor, self.d_c_bs_neighbor.universe.min(), self.d_c_bs_neighbor.universe.max())
        self.simulation.input['e_c_neighbor'] = np.clip(current_e_c_neighbor, self.e_c_neighbor.universe.min(), self.e_c_neighbor.universe.max())
        self.simulation.input['load_c_ratio'] = load_c_ratio_val
        self.simulation.input['r_c_success'] = np.clip(current_r_c_success, self.r_c_success.universe.min(), self.r_c_success.universe.max())
        self.simulation.input['e_ctx_cost_normalized'] = np.clip(current_e_ctx_cost_normalized, self.e_ctx_cost_normalized.universe.min(), self.e_ctx_cost_normalized.universe.max())

        try:
            self.simulation.compute()
            return {
                'w_d_pro': self.simulation.output['w_d_pro'],
                'w_e_cost': self.simulation.output['w_e_cost'],
                'w_fur': self.simulation.output['w_fur'],
                'w_load_neighbor': self.simulation.output['w_load_neighbor']
            }
        except Exception as e:
            print(f"Error in CHToBSPathSelectionFuzzySystem computation: {e}")
            print(f"Inputs: d_c_bs_neighbor={self.simulation.input.get('d_c_bs_neighbor', 'N/A')}, "
                  f"e_c_neighbor={self.simulation.input.get('e_c_neighbor', 'N/A')}, "
                  f"lq_c_normalized={self.simulation.input.get('lq_c_normalized', 'N/A')}, "
                  f"load_c_ratio={self.simulation.input.get('load_c_ratio', 'N/A')}, "
                  f"r_c_success={self.simulation.input.get('r_c_success', 'N/A')}, "
                  f"e_ctx_cost_normalized={self.simulation.input.get('e_ctx_cost_normalized', 'N/A')}")
            return {'w_d_pro': 0.5, 'w_e_cost': 0.5, 'w_fur': 0.5, 'w_load_neighbor': 0.5} # Default

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

    def _define_rules_once(self):
        logger.debug("Defining fuzzy rules for RewardWeightsFuzzySystem...")
        rules = []
        # 调整 w_members_factor
        rules.append(ctrl.Rule(self.ch_density_global['Too_Low'], self.w_members_factor['Increase']))
        rules.append(ctrl.Rule(self.ch_density_global['Optimal'], self.w_members_factor['Neutral']))
        rules.append(ctrl.Rule(self.ch_density_global['Too_High'], self.w_members_factor['Decrease']))

        # 调整 w_energy_self_factor
        rules.append(ctrl.Rule(self.network_energy_level['Low'] & self.node_self_energy['High'], self.w_energy_self_factor['Increase']))
        rules.append(ctrl.Rule(self.network_energy_level['Medium'] & self.node_self_energy['High'], self.w_energy_self_factor['Increase']))
        rules.append(ctrl.Rule(self.network_energy_level['Low'] & self.node_self_energy['Medium'], self.w_energy_self_factor['Increase']))
        rules.append(ctrl.Rule(self.network_energy_level['High'] & self.node_self_energy['High'], self.w_energy_self_factor['Neutral']))
        rules.append(ctrl.Rule(self.network_energy_level['Medium'] & self.node_self_energy['Medium'], self.w_energy_self_factor['Neutral']))
        rules.append(ctrl.Rule(self.network_energy_level['High'] & self.node_self_energy['Medium'], self.w_energy_self_factor['Decrease']))
        rules.append(ctrl.Rule(self.node_self_energy['Low'], self.w_energy_self_factor['Decrease']))
        
        # 调整 w_cost_ch_factor
        rules.append(ctrl.Rule(self.ch_density_global['Too_High'], self.w_cost_ch_factor['Increase']))
        rules.append(ctrl.Rule(self.ch_density_global['Optimal'], self.w_cost_ch_factor['Neutral']))
        rules.append(ctrl.Rule(self.ch_density_global['Too_Low'], self.w_cost_ch_factor['Decrease']))
        rules.append(ctrl.Rule(self.network_energy_level['Low'], self.w_cost_ch_factor['Decrease']))

        # 调整 w_rotation_factor
        rules.append(ctrl.Rule(self.ch_density_global['Too_Low'] & (self.node_self_energy['High'] | self.node_self_energy['Medium']), self.w_rotation_factor['Increase']))
        rules.append(ctrl.Rule((self.ch_density_global['Too_Low'] | self.ch_density_global['Optimal']) & self.node_self_energy['Low'], self.w_rotation_factor['Decrease']))
        rules.append(ctrl.Rule(self.ch_density_global['Optimal'] & self.node_self_energy['High'], self.w_rotation_factor['Neutral']))
        rules.append(ctrl.Rule(self.ch_density_global['Optimal'] & self.node_self_energy['Medium'], self.w_rotation_factor['Decrease']))
        rules.append(ctrl.Rule(self.ch_density_global['Too_High'], self.w_rotation_factor['Decrease']))

        # 调整 w_dis (确保前提中的模糊集名称与定义一致)
        rules.append(ctrl.Rule(self.ch_to_bs_dis['Low'] & (self.ch_density_global['Too_High'] | self.ch_density_global['Optimal']), self.w_dis['Increase']))
        rules.append(ctrl.Rule(self.ch_to_bs_dis['Low'] & self.ch_density_global['Too_Low'], self.w_dis['Neutral']))
        rules.append(ctrl.Rule(self.ch_to_bs_dis['Medium'] & self.ch_density_global['Too_Low'], self.w_dis['Decrease']))
        rules.append(ctrl.Rule(self.ch_to_bs_dis['Medium'] & self.ch_density_global['Optimal'], self.w_dis['Neutral'])) # Corrected from ['Medium']
        rules.append(ctrl.Rule(self.ch_to_bs_dis['Medium'] & self.ch_density_global['Too_High'], self.w_dis['Increase']))
        rules.append(ctrl.Rule(self.ch_to_bs_dis['High'] & self.ch_density_global['Too_Low'], self.w_dis['Neutral']))
        rules.append(ctrl.Rule(self.ch_to_bs_dis['High'] & (self.ch_density_global['Too_High'] | self.ch_density_global['Optimal']), self.w_dis['Increase']))

        if not rules:
            logger.warning("RewardWeightsFuzzySystem: No specific rules were defined, adding a default neutral rule for all outputs.")
            default_antecedent = self.network_energy_level['Medium'] # Pick one antecedent for default
            for out_var in [self.w_members_factor, self.w_energy_self_factor, 
                            self.w_cost_ch_factor, self.w_rotation_factor, self.w_dis]:
                if hasattr(out_var, 'terms') and 'Neutral' in out_var.terms: # Check if 'Neutral' is defined
                     rules.append(ctrl.Rule(default_antecedent, out_var['Neutral']))
                else: # Fallback if 'Neutral' MF is not defined for some reason
                     logger.error(f"Cannot add default rule for {out_var.label} as 'Neutral' MF is not defined.")
        
        logger.debug(f"Defined {len(rules)} rules for RewardWeightsFuzzySystem.")
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