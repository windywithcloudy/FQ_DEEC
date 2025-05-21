import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class NormalNodeCHSelectionFuzzySystem:
    def __init__(self, node_sum, cluster_sum, avg_e_send_total):
        """
        Initializes the fuzzy control system for CH selection by a normal node.

        Args:
            node_sum (int): Total number of nodes in the network.
            cluster_sum (int): Current number of clusters (CHs) in the network.
                            (Ensure cluster_sum > 0 to avoid division by zero)
            avg_e_send_total (float): Current average energy consumption for sending data.
        """
        if cluster_sum <= 0:
            # Handle this case, e.g., by using a default average load or raising an error
            # For simplicity here, we'll assume a default if cluster_sum is 0
            self.avg_load_per_ch = node_sum / 1.0 if node_sum > 0 else 10 # Default avg load
        else:
            self.avg_load_per_ch = node_sum / cluster_sum
        
        self.avg_e_send_total = avg_e_send_total if avg_e_send_total > 0 else 0.01 # Default avg send energy

        # --- Antecedents (Inputs) ---
        self.d_c_base = None
        self.e_cluster = None
        self.p_cluster = None
        self.r_success = None
        self.e_send_total_input = None # Renamed to avoid conflict with class member

        # --- Consequents (Outputs) ---
        self.w_e_ch = None
        self.w_path = None
        self.w_load = None
        self.w_dist_bs = None

        # --- Control System ---
        self.control_system = None
        self.simulation = None

        self._build_system()

    def _define_antecedents(self):
        # 1. D_c_base (Distance CH to Base Station)
        # Max distance for a 250x250 area
        max_dist_val = 250 * np.sqrt(2) 
        universe_dc_base = np.arange(0, max_dist_val + 1, 1) # Universe
        self.d_c_base = ctrl.Antecedent(universe_dc_base, 'D_c_base')
        mid_dist_point = max_dist_val / 2 # 125 * sqrt(2)

        # MF for D_c_base
        # Using zmf/smf for shoulders, trimf for middle for better coverage
        self.d_c_base['Near'] = fuzz.zmf(self.d_c_base.universe, mid_dist_point * 0.6, mid_dist_point) 
        self.d_c_base['Medium'] = fuzz.trimf(self.d_c_base.universe, [mid_dist_point * 0.6, mid_dist_point, mid_dist_point * 1.4])
        self.d_c_base['Far'] = fuzz.smf(self.d_c_base.universe, mid_dist_point, mid_dist_point * 1.4)

        # 2. E_cluster (CH's Normalized Remaining Energy)
        universe_e_cluster = np.arange(0, 1.01, 0.01) # Normalized [0, 1]
        self.e_cluster = ctrl.Antecedent(universe_e_cluster, 'E_cluster')
        self.e_cluster['Low'] = fuzz.zmf(self.e_cluster.universe, 0.2, 0.4)
        self.e_cluster['Medium'] = fuzz.trimf(self.e_cluster.universe, [0.3, 0.5, 0.7])
        self.e_cluster['High'] = fuzz.smf(self.e_cluster.universe, 0.6, 0.8)

        # 3. P_cluster (CH's Current Load - Normalized around average load)
        # Let's define universe relative to average load, e.g., 0 to 3 times avg_load
        # This makes MFs more stable if avg_load_per_ch changes significantly.
        # Or, normalize input P_cluster value before feeding to a fixed universe [0,X]
        # For this example, let's use a fixed universe [0,3] and assume input is normalized P_cluster / avg_load_per_ch
        universe_p_cluster_ratio = np.arange(0, 3.01, 0.01) # Ratio to average load
        self.p_cluster = ctrl.Antecedent(universe_p_cluster_ratio, 'P_cluster_Ratio') # Input will be actual_load / avg_load_per_ch
        self.p_cluster['Low'] = fuzz.zmf(self.p_cluster.universe, 0.5, 1.0)    # Load < 0.5*avg is definitely Low
        self.p_cluster['Medium'] = fuzz.trimf(self.p_cluster.universe, [0.75, 1.25, 1.75]) # Around avg to 1.75*avg
        self.p_cluster['High'] = fuzz.smf(self.p_cluster.universe, 1.5, 2.0)     # Load > 1.5*avg is High

        # 4. R_success (Historical Communication Success Rate with CH)
        universe_r_success = np.arange(0, 1.01, 0.01) # Normalized [0, 1]
        self.r_success = ctrl.Antecedent(universe_r_success, 'R_success')
        self.r_success['Low'] = fuzz.zmf(self.r_success.universe, 0.3, 0.6)
        self.r_success['Medium'] = fuzz.trimf(self.r_success.universe, [0.4, 0.7, 0.9])
        self.r_success['High'] = fuzz.smf(self.r_success.universe, 0.7, 0.95)

        # 5. E_send_total (Historical Avg. Send Energy to CH - Normalized)
        # Similar to P_cluster, let's assume input is normalized: actual_e_send / avg_e_send_total
        # Universe [0,3] for ratio
        universe_e_send_ratio = np.arange(0, 3.01, 0.01)
        self.e_send_total_input = ctrl.Antecedent(universe_e_send_ratio, 'E_send_total_Ratio')
        self.e_send_total_input['Low'] = fuzz.zmf(self.e_send_total_input.universe, 0.5, 1.0)
        self.e_send_total_input['Medium'] = fuzz.trimf(self.e_send_total_input.universe, [0.75, 1.25, 1.75])
        self.e_send_total_input['High'] = fuzz.smf(self.e_send_total_input.universe, 1.5, 2.0)

    def _define_consequents(self):
        universe_weights = np.arange(0, 1.01, 0.01) # Output weights [0, 1]

        self.w_e_ch = ctrl.Consequent(universe_weights, 'w_e_ch')
        self.w_path = ctrl.Consequent(universe_weights, 'w_path')
        self.w_load = ctrl.Consequent(universe_weights, 'w_load')
        self.w_dist_bs = ctrl.Consequent(universe_weights, 'w_dist_bs')

        # Common MFs for all output weights
        for output_var in [self.w_e_ch, self.w_path, self.w_load, self.w_dist_bs]:
            output_var['Low'] = fuzz.zmf(output_var.universe, 0.2, 0.4)
            output_var['Medium'] = fuzz.trimf(output_var.universe, [0.3, 0.5, 0.7])
            output_var['High'] = fuzz.smf(output_var.universe, 0.6, 0.8)
            # Set default defuzzification method if not using system default
            # output_var.defuzzify_method = 'centroid'


    def _define_rules(self):
        rules = []
        # Rules for w_e_ch
        rules.append(ctrl.Rule(self.e_cluster['Low'], self.w_e_ch['Low']))
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & (self.d_c_base['Far'] | self.p_cluster['High']), self.w_e_ch['Low']))
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & (self.d_c_base['Medium'] | self.p_cluster['Medium']), self.w_e_ch['Medium']))
        rules.append(ctrl.Rule(self.e_cluster['High'] & (self.d_c_base['Far'] | self.p_cluster['High']), self.w_e_ch['Medium']))
        rules.append(ctrl.Rule(self.e_cluster['Medium'] & (self.d_c_base['Near'] | self.p_cluster['Low']), self.w_e_ch['High']))
        rules.append(ctrl.Rule(self.e_cluster['High'] & (self.d_c_base['Medium'] | self.p_cluster['Medium']), self.w_e_ch['High']))
        rules.append(ctrl.Rule(self.e_cluster['High'] & (self.d_c_base['Near'] | self.p_cluster['Low']), self.w_e_ch['High']))

        # Rules for w_path
        rules.append(ctrl.Rule(self.r_success['High'] & (self.e_send_total_input['Low'] | self.e_send_total_input['Medium']), self.w_path['Low']))
        rules.append(ctrl.Rule(self.r_success['High'] & self.e_send_total_input['High'], self.w_path['Medium']))
        rules.append(ctrl.Rule(self.r_success['Medium'] & (self.e_send_total_input['Low'] | self.e_send_total_input['Medium']), self.w_path['Medium']))
        rules.append(ctrl.Rule(self.r_success['Medium'] & self.e_send_total_input['High'], self.w_path['High']))
        rules.append(ctrl.Rule(self.r_success['Low'], self.w_path['High']))

        # Rules for w_load
        rules.append(ctrl.Rule(self.p_cluster['High'], self.w_load['High']))
        rules.append(ctrl.Rule(self.p_cluster['Medium'] & self.e_cluster['Low'], self.w_load['High']))
        rules.append(ctrl.Rule(self.p_cluster['Medium'] & self.e_cluster['Medium'], self.w_load['Medium']))
        rules.append(ctrl.Rule(self.p_cluster['Low'] & self.e_cluster['Low'], self.w_load['Medium']))
        rules.append(ctrl.Rule(self.p_cluster['Medium'] & self.e_cluster['High'], self.w_load['Low']))
        rules.append(ctrl.Rule(self.p_cluster['Low'] & (self.e_cluster['Medium'] | self.e_cluster['High']), self.w_load['Low']))

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
        # You can set default T-Norm, S-Norm, etc. here if needed, e.g.
        # self.control_system.conjunction = np.fmin # For 'min' T-Norm
        # self.control_system.disjunction = np.fmax # For 'max' S-Norm
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
        # Set default defuzzification for outputs if not specified in _define_consequents
        # skfuzzy's ControlSystemSimulation defaults to 'centroid' for consequents
        # if their .defuzzify_method is not set.

    def compute_weights(self, current_dc_base, current_e_cluster, current_p_cluster_actual, 
                        current_r_success, current_e_send_total_actual):
        """
        Computes the output weights based on the current input values.

        Args:
            current_dc_base (float): Current distance from CH to BS.
            current_e_cluster (float): Current normalized energy of CH [0,1].
            current_p_cluster_actual (float): Current actual load of CH.
            current_r_success (float): Current normalized success rate [0,1].
            current_e_send_total_actual (float): Current actual avg send energy.

        Returns:
            dict: A dictionary कंटेनिंग the computed crisp output weights.
                  e.g., {'w_e_ch': 0.65, 'w_path': 0.3, ...}
        """
        if self.simulation is None:
            raise Exception("Fuzzy system not built. Call _build_system() first.")

        # Normalize dynamic inputs
        p_cluster_ratio_val = current_p_cluster_actual / self.avg_load_per_ch if self.avg_load_per_ch > 0 else 0
        # Clip to universe to avoid errors if ratio is outside defined universe
        p_cluster_ratio_val = np.clip(p_cluster_ratio_val, 
                                      self.p_cluster.universe.min(), 
                                      self.p_cluster.universe.max())


        e_send_total_ratio_val = current_e_send_total_actual / self.avg_e_send_total if self.avg_e_send_total > 0 else 0
        e_send_total_ratio_val = np.clip(e_send_total_ratio_val,
                                         self.e_send_total_input.universe.min(),
                                         self.e_send_total_input.universe.max())


        self.simulation.input['D_c_base'] = np.clip(current_dc_base, self.d_c_base.universe.min(), self.d_c_base.universe.max())
        self.simulation.input['E_cluster'] = np.clip(current_e_cluster, self.e_cluster.universe.min(), self.e_cluster.universe.max())
        self.simulation.input['P_cluster_Ratio'] = p_cluster_ratio_val
        self.simulation.input['R_success'] = np.clip(current_r_success, self.r_success.universe.min(), self.r_success.universe.max())
        self.simulation.input['E_send_total_Ratio'] = e_send_total_ratio_val
        
        try:
            self.simulation.compute()
            return {
                'w_e_ch': self.simulation.output['w_e_ch'],
                'w_path': self.simulation.output['w_path'],
                'w_load': self.simulation.output['w_load'],
                'w_dist_bs': self.simulation.output['w_dist_bs']
            }
        except Exception as e:
            print(f"Error during fuzzy computation: {e}")
            print("Inputs provided:")
            print(f"  D_c_base: {self.simulation.input['D_c_base']}")
            print(f"  E_cluster: {self.simulation.input['E_cluster']}")
            print(f"  P_cluster_Ratio: {self.simulation.input['P_cluster_Ratio']}")
            print(f"  R_success: {self.simulation.input['R_success']}")
            print(f"  E_send_total_Ratio: {self.simulation.input['E_send_total_Ratio']}")
            # Return default or neutral weights in case of error
            return {'w_e_ch': 0.5, 'w_path': 0.5, 'w_load': 0.5, 'w_dist_bs': 0.5}


    def view_antecedent(self, name):
        """Helper to view a specific antecedent's MFs."""
        if hasattr(self, name):
            getattr(self, name).view()
        else:
            print(f"Antecedent {name} not found.")
            
    def view_consequent(self, name):
        """Helper to view a specific consequent's MFs."""
        if hasattr(self, name):
            getattr(self, name).view()
        else:
            print(f"Consequent {name} not found.")

    def view_simulation_step(self, current_dc_base, current_e_cluster, current_p_cluster_actual, 
                             current_r_success, current_e_send_total_actual):
        """Computes and shows the detailed reasoning for one step."""
        _ = self.compute_weights(current_dc_base, current_e_cluster, current_p_cluster_actual,
                                 current_r_success, current_e_send_total_actual)
        # View individual outputs
        self.w_e_ch.view(sim=self.simulation)
        self.w_path.view(sim=self.simulation)
        self.w_load.view(sim=self.simulation)
        self.w_dist_bs.view(sim=self.simulation)


# --- Example Usage (Illustrative) ---
if __name__ == '__main__':
    # These dynamic values would come from your simulation environment
    current_node_sum = 100
    current_cluster_sum = 10 # Example, ensure > 0
    current_avg_e_send = 0.005 # Example average send energy

    fuzzy_selector = NormalNodeCHSelectionFuzzySystem(
        node_sum=current_node_sum,
        cluster_sum=current_cluster_sum,
        avg_e_send_total=current_avg_e_send
    )

    # View MFs to check (optional)
    # fuzzy_selector.view_antecedent('d_c_base')
    # fuzzy_selector.view_antecedent('e_cluster')
    # fuzzy_selector.view_antecedent('p_cluster') # This is P_cluster_Ratio now
    # fuzzy_selector.view_antecedent('r_success')
    # fuzzy_selector.view_antecedent('e_send_total_input') # This is E_send_total_Ratio now
    # fuzzy_selector.view_consequent('w_e_ch')


    # Example input values for a candidate CH
    # (These would be actual values from a CH in your simulation)
    input_dc_base = 150    # Actual distance
    input_e_cluster = 0.6  # Normalized energy
    input_p_cluster_actual = 15 # Actual number of nodes CH is serving
    input_r_success = 0.85 # Normalized success rate
    input_e_send_actual = 0.004 # Actual avg send energy to this CH

    print(f"Avg load per CH for MFs: {fuzzy_selector.avg_load_per_ch}")
    print(f"Avg send energy for MFs: {fuzzy_selector.avg_e_send_total}")
    
    computed_weights = fuzzy_selector.compute_weights(
        current_dc_base=input_dc_base,
        current_e_cluster=input_e_cluster,
        current_p_cluster_actual=input_p_cluster_actual,
        current_r_success=input_r_success,
        current_e_send_total_actual=input_e_send_actual
    )

    print("\nComputed Weights:")
    for weight_name, value in computed_weights.items():
        print(f"  {weight_name}: {value:.4f}")

    # Example of viewing simulation step for these inputs (optional)
    # print("\nViewing simulation step details:")
    # fuzzy_selector.view_simulation_step(
    #     current_dc_base=input_dc_base,
    #     current_e_cluster=input_e_cluster,
    #     current_p_cluster_actual=input_p_cluster_actual,
    #     current_r_success=input_r_success,
    #     current_e_send_total_actual=input_e_send_actual
    # )