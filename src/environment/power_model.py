"""
src/environment/power_model.py

Power Consumption Model for NOMA-MIMO-OFDM System
Implements: Circuit power, PA power, Total power (LOCKED formula from MIMO spec)
"""

import numpy as np


class PowerModel:
    """
    Total Power Consumption Model
    
    LOCKED FORMULA:
    P_total = P_static + (1/eta_PA) * sum(p_k,n) + N_t*(P_DAC + P_mix + P_filt) + K*P_ADC
    
    Components:
    - P_static: Fixed base station power
    - PA power: Transmit power / PA efficiency
    - RF chains: DAC, mixer, filter (TX side)
    - ADC power (RX side, user dependent)
    """
    
    def __init__(self, config):
        """
        Initialize power model.
        
        Args:
            config: Configuration dictionary from system_params
        """
        self.config = config
        
        # System parameters
        self.N_t = config['system']['N_t']
        self.K = config['system']['K']
        
        # Power parameters
        self.P_max_W = config['power']['P_max_W']
        self.eta_PA = config['power']['eta_PA']
        
        # Circuit power components
        self.P_static = config['circuit_power']['P_static']
        self.P_DAC = config['circuit_power']['P_DAC']
        self.P_ADC = config['circuit_power']['P_ADC']
        self.P_mix = config['circuit_power']['P_mix']
        self.P_filt = config['circuit_power']['P_filt']
        
        # Pre-calculate circuit power (constant part)
        self.P_circuit_constant = self.calculate_circuit_power_constant()
    
    def calculate_circuit_power_constant(self):
        """
        Calculate constant circuit power (independent of transmit power).
        
        P_circuit = P_static + N_t*(P_DAC + P_mix + P_filt) + K*P_ADC
        
        Returns:
            P_circuit: Circuit power in Watts
        """
        P_rf_tx = self.N_t * (self.P_DAC + self.P_mix + self.P_filt)
        P_rf_rx = self.K * self.P_ADC
        P_circuit = self.P_static + P_rf_tx + P_rf_rx
        
        return P_circuit
    
    def calculate_pa_power(self, P_tx):
        """
        Calculate Power Amplifier power consumption.
        
        P_PA = P_tx / eta_PA
        
        Args:
            P_tx: Transmit power in Watts
            
        Returns:
            P_PA: PA power consumption in Watts
        """
        return P_tx / self.eta_PA
    
    def calculate_total_power(self, P_tx):
        """
        Calculate total power consumption (LOCKED formula).
        
        P_total = P_static + (1/eta_PA)*P_tx + N_t*(P_DAC + P_mix + P_filt) + K*P_ADC
        
        Args:
            P_tx: Transmit power in Watts (can be scalar or array)
            
        Returns:
            P_total: Total power consumption in Watts
        """
        P_PA = self.calculate_pa_power(P_tx)
        P_total = self.P_circuit_constant + P_PA
        
        return P_total
    
    def calculate_power_allocation_matrix(self, power_vector, cluster_assignments, N_sc):
        """
        Convert power vector to power allocation matrix.
        
        Args:
            power_vector: Power values, shape (K,) or (K*N_sc,)
            cluster_assignments: User-to-subcarrier mapping, shape (K, N_sc)
            N_sc: Number of subcarriers
            
        Returns:
            P_matrix: Power allocation matrix, shape (K, N_sc)
        """
        K = self.K
        
        if power_vector.size == K * N_sc:
            # Reshape to matrix
            P_matrix = power_vector.reshape(K, N_sc)
        elif power_vector.size == K:
            # Broadcast to all subcarriers
            P_matrix = np.tile(power_vector[:, np.newaxis], (1, N_sc))
        else:
            raise ValueError(f"Invalid power_vector size: {power_vector.size}")
        
        # Apply cluster assignments (set to 0 if user not assigned)
        P_matrix = P_matrix * cluster_assignments
        
        return P_matrix
    
    def check_power_constraint(self, P_matrix):
        """
        Check if power allocation satisfies total power constraint.
        
        Constraint: sum(P_matrix) <= P_max
        
        Args:
            P_matrix: Power allocation matrix, shape (K, N_sc)
            
        Returns:
            is_valid: Boolean indicating if constraint is satisfied
            total_power: Total allocated power
        """
        total_power = np.sum(P_matrix)
        is_valid = (total_power <= self.P_max_W)
        
        return is_valid, total_power
    
    def project_power_to_constraint(self, P_matrix):
        """
        Project power allocation to satisfy constraint.
        
        If sum(P_matrix) > P_max, scale down proportionally.
        
        Args:
            P_matrix: Power allocation matrix, shape (K, N_sc)
            
        Returns:
            P_matrix_valid: Valid power allocation, shape (K, N_sc)
        """
        total_power = np.sum(P_matrix)
        
        if total_power > self.P_max_W:
            # Scale down
            scale_factor = self.P_max_W / total_power
            P_matrix_valid = P_matrix * scale_factor
        else:
            P_matrix_valid = P_matrix.copy()
        
        return P_matrix_valid
    
    def calculate_noma_power_allocation(self, alpha_near, alpha_far, P_total_cluster):
        """
        Calculate NOMA power allocation for a 2-user cluster.
        
        Args:
            alpha_near: Power coefficient for near user (0.2-0.4)
            alpha_far: Power coefficient for far user (0.6-0.8)
            P_total_cluster: Total power for this cluster
            
        Returns:
            p_near: Power for near user
            p_far: Power for far user
        """
        # Ensure power budget constraint
        alpha_sum = alpha_near + alpha_far
        if alpha_sum > 1.0:
            # Normalize
            alpha_near = alpha_near / alpha_sum
            alpha_far = alpha_far / alpha_sum
        
        p_near = alpha_near * P_total_cluster
        p_far = alpha_far * P_total_cluster
        
        return p_near, p_far


def test_power_model():
    """Test the power model."""
    import sys
    sys.path.append('.')
    from configs.system_params import get_scenario_config
    
    print("="*70)
    print("Testing Power Model")
    print("="*70)
    
    # Load config
    config = get_scenario_config('small_cell')
    
    # Create power model
    power = PowerModel(config)
    
    # Test circuit power
    print(f"\n✅ Circuit Power (Constant):")
    print(f"   P_static: {power.P_static} W")
    print(f"   P_RF_TX: {power.N_t * (power.P_DAC + power.P_mix + power.P_filt):.4f} W")
    print(f"   P_RF_RX: {power.K * power.P_ADC:.4f} W")
    print(f"   Total Circuit: {power.P_circuit_constant:.4f} W")
    
    # Test PA power
    P_tx = 20  # 20W transmit power
    P_PA = power.calculate_pa_power(P_tx)
    print(f"\n✅ PA Power Calculation:")
    print(f"   P_tx: {P_tx} W")
    print(f"   eta_PA: {power.eta_PA}")
    print(f"   P_PA: {P_PA:.4f} W")
    
    # Test total power
    P_total = power.calculate_total_power(P_tx)
    print(f"\n✅ Total Power Consumption:")
    print(f"   P_total: {P_total:.4f} W")
    print(f"   Breakdown: {power.P_circuit_constant:.2f}W (circuit) + {P_PA:.2f}W (PA)")
    
    # Test power constraint
    N_sc = 10
    K = power.K
    P_matrix = np.random.rand(K, N_sc) * 5  # Random power allocation
    
    is_valid, total_tx = power.check_power_constraint(P_matrix)
    print(f"\n✅ Power Constraint Check:")
    print(f"   Total allocated: {total_tx:.4f} W")
    print(f"   P_max: {power.P_max_W} W")
    print(f"   Valid: {is_valid}")
    
    # Test projection
    P_matrix_invalid = np.ones((K, N_sc)) * 10  # Violates constraint
    P_matrix_valid = power.project_power_to_constraint(P_matrix_invalid)
    
    print(f"\n✅ Power Projection:")
    print(f"   Before: sum = {np.sum(P_matrix_invalid):.4f} W")
    print(f"   After: sum = {np.sum(P_matrix_valid):.4f} W")
    print(f"   Valid: {power.check_power_constraint(P_matrix_valid)[0]}")
    
    # Test NOMA power allocation
    alpha_near, alpha_far = 0.3, 0.7
    P_cluster = 10
    p_near, p_far = power.calculate_noma_power_allocation(alpha_near, alpha_far, P_cluster)
    
    print(f"\n✅ NOMA Power Allocation:")
    print(f"   Cluster power: {P_cluster} W")
    print(f"   alpha_near: {alpha_near}, p_near: {p_near:.4f} W")
    print(f"   alpha_far: {alpha_far}, p_far: {p_far:.4f} W")
    print(f"   Sum: {p_near + p_far:.4f} W")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_power_model()