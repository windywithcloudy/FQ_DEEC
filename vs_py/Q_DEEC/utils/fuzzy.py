import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt # Keep for viewing, can be commented out for production

# =============================================================================
# Fuzzy System for Normal Node Selecting a Cluster Head (CH)
# =============================================================================
class NormalNodeCHSelectionFuzzySystem:
    def __init__(self, node_sum, cluster_sum):
        """
        Initializes the fuzzy control system for CH selection by a normal node.
        Args:
            node_sum (int): Total number of nodes in the network.
            cluster_sum (int): Current number of clusters (CHs) in the network.
        """
        if cluster_sum <= 0:
            self.avg_load_per_ch = node_sum / 1.0 if node_sum > 0 else 10 
        else:
            self.avg_load_per_ch = node_sum / cluster_sum
        
        # --- Antecedents (Inputs) ---
        self.d_c_base = None           # Distance: CH to Base Station
        self.e_cluster = None          # Energy: CH's Normalized Remaining Energy
        self.p_cluster_ratio = None    # Load: CH's Current Load Ratio (actual_load / avg_load)
        self.r_success = None          # Success Rate: Historical Comm. Success with CH (Normalized)
        self.e_send_total_ratio = None # Send Energy: Hist. Avg. Send Energy to CH Ratio (actual_send_e / avg_send_e)

        # --- Consequents (Outputs) ---
        self.w_e_ch = None
        self.w_path = None
        self.w_load = None
        self.w_dist_bs = None

        self.control_system = None
        self.simulation = None
        self._build_system()

    def _define_antecedents(self):
        # 1. D_c_base
        max_dist_val = 250 * np.sqrt(2) 
        universe_dc_base = np.arange(0, max_dist_val + 1, 1)
        self.d_c_base = ctrl.Antecedent(universe_dc_base, 'd_c_base')
        mid_dist_point = max_dist_val / 2
        self.d_c_base['Near'] = fuzz.zmf(self.d_c_base.universe, mid_dist_point * 0.6, mid_dist_point) 
        self.d_c_base['Medium'] = fuzz.trimf(self.d_c_base.universe, [mid_dist_point * 0.6, mid_dist_point, mid_dist_point * 1.4])
        self.d_c_base['Far'] = fuzz.smf(self.d_c_base.universe, mid_dist_point, mid_dist_point * 1.4)

        # 2. E_cluster
        universe_e_cluster = np.arange(0, 1.01, 0.01)
        self.e_cluster = ctrl.Antecedent(universe_e_cluster, 'e_cluster')
        self.e_cluster['Low'] = fuzz.zmf(self.e_cluster.universe, 0.2, 0.4)
        self.e_cluster['Medium'] = fuzz.trimf(self.e_cluster.universe, [0.3, 0.5, 0.7])
        self.e_cluster['High'] = fuzz.smf(self.e_cluster.universe, 0.6, 0.8)

        # 3. P_cluster_Ratio (Input will be actual_load / avg_load_per_ch)
        universe_p_cluster_ratio = np.arange(0, 3.01, 0.01) 
        self.p_cluster_ratio = ctrl.Antecedent(universe_p_cluster_ratio, 'p_cluster_ratio')
        self.p_cluster_ratio['Low'] = fuzz.zmf(self.p_cluster_ratio.universe, 0.5, 1.0)
        self.p_cluster_ratio['Medium'] = fuzz.trimf(self.p_cluster_ratio.universe, [0.75, 1.25, 1.75])
        self.p_cluster_ratio['High'] = fuzz.smf(self.p_cluster_ratio.universe, 1.5, 2.0)

        # 4. R_success
        universe_r_success = np.arange(0, 1.01, 0.01)
        self.r_success = ctrl.Antecedent(universe_r_success, 'r_success')
        self.r_success['Low'] = fuzz.zmf(self.r_success.universe, 0.3, 0.6)
        self.r_success['Medium'] = fuzz.trimf(self.r_success.universe, [0.4, 0.7, 0.9])
        self.r_success['High'] = fuzz.smf(self.r_success.universe, 0.7, 0.95)

        # 5. E_send_total_Ratio (Input will be actual_e_send / avg_e_send_total_for_normal_node)
        # Assuming avg_e_send_total_for_normal_node will be passed or handled similarly to avg_load
        universe_e_send_ratio = np.arange(0, 3.01, 0.01)
        self.e_send_total_ratio = ctrl.Antecedent(universe_e_send_ratio, 'e_send_total_ratio')
        self.e_send_total_ratio['Low'] = fuzz.zmf(self.e_send_total_ratio.universe, 0.5, 1.0)
        self.e_send_total_ratio['Medium'] = fuzz.trimf(self.e_send_total_ratio.universe, [0.75, 1.25, 1.75])
        self.e_send_total_ratio['High'] = fuzz.smf(self.e_send_total_ratio.universe, 1.5, 2.0)

    def _define_consequents(self):
        universe_weights = np.arange(0, 1.01, 0.01)
        self.w_e_ch = ctrl.Consequent(universe_weights, 'w_e_ch')
        self.w_path = ctrl.Consequent(universe_weights, 'w_path')
        self.w_load = ctrl.Consequent(universe_weights, 'w_load')
        self.w_dist_bs = ctrl.Consequent(universe_weights, 'w_dist_bs')

        for output_var in [self.w_e_ch, self.w_path, self.w_load, self.w_dist_bs]:
            output_var['Low'] = fuzz.zmf(output_var.universe, 0.2, 0.4)
            output_var['Medium'] = fuzz.trimf(output_var.universe, [0.3, 0.5, 0.7])
            output_var['High'] = fuzz.smf(output_var.universe, 0.6, 0.8)

    def _define_rules(self):
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
        return rules

    def _build_system(self):
        self._define_antecedents()
        self._define_consequents()
        rules = self._define_rules()
        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)

    def compute_weights(self, current_dc_base, current_e_cluster, current_p_cluster_actual, 
                        current_r_success, current_e_send_total_actual, avg_e_send_total_for_normal_node):
        if self.simulation is None:
            raise Exception("Fuzzy system not built.")

        p_cluster_ratio_val = current_p_cluster_actual / self.avg_load_per_ch if self.avg_load_per_ch > 0 else 0
        p_cluster_ratio_val = np.clip(p_cluster_ratio_val, self.p_cluster_ratio.universe.min(), self.p_cluster_ratio.universe.max())

        # Use passed avg_e_send_total_for_normal_node for normalization
        current_avg_e_send = avg_e_send_total_for_normal_node if avg_e_send_total_for_normal_node > 0 else 0.01 
        e_send_total_ratio_val = current_e_send_total_actual / current_avg_e_send
        e_send_total_ratio_val = np.clip(e_send_total_ratio_val, self.e_send_total_ratio.universe.min(), self.e_send_total_ratio.universe.max())
        
        self.simulation.input['d_c_base'] = np.clip(current_dc_base, self.d_c_base.universe.min(), self.d_c_base.universe.max())
        self.simulation.input['e_cluster'] = np.clip(current_e_cluster, self.e_cluster.universe.min(), self.e_cluster.universe.max())
        self.simulation.input['p_cluster_ratio'] = p_cluster_ratio_val
        self.simulation.input['r_success'] = np.clip(current_r_success, self.r_success.universe.min(), self.r_success.universe.max())
        self.simulation.input['e_send_total_ratio'] = e_send_total_ratio_val
        
        try:
            self.simulation.compute()
            return {
                'w_e_ch': self.simulation.output['w_e_ch'],
                'w_path': self.simulation.output['w_path'],
                'w_load': self.simulation.output['w_load'],
                'w_dist_bs': self.simulation.output['w_dist_bs']
            }
        except Exception as e:
            print(f"Error in NormalNodeCHSelectionFuzzySystem computation: {e}")
            # Provide input values for debugging
            print(f"Inputs: dc_base={self.simulation.input.get('d_c_base', 'N/A')}, "
                  f"e_cluster={self.simulation.input.get('e_cluster', 'N/A')}, "
                  f"p_cluster_ratio={self.simulation.input.get('p_cluster_ratio', 'N/A')}, "
                  f"r_success={self.simulation.input.get('r_success', 'N/A')}, "
                  f"e_send_total_ratio={self.simulation.input.get('e_send_total_ratio', 'N/A')}")
            return {'w_e_ch': 0.5, 'w_path': 0.5, 'w_load': 0.5, 'w_dist_bs': 0.5} # Default/neutral

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