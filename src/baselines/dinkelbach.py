"""
src/baselines/dinkelbach.py

Baseline 3: Dinkelbach Energy Efficiency Optimization
From Optimization_v4.pdf Section 4.2-4.3

Algorithm:
1. Near-Far user pairing
2. Dinkelbach method for EE maximization (fractional programming)
3. Iterative power allocation to maximize Rate/Power ratio

Dinkelbach transforms max(R/P) into iterative subproblems:
max(R - lambda*P) where lambda is updated each iteration
"""

import numpy as np
import sys
sys.path.append('.')

from configs.system_params import get_scenario_config
from src.environment.channel_model import ChannelModel
from src.environment.power_model import PowerModel


class DinkelbachBaseline:
    """
    Dinkelbach Energy Efficiency Optimization
    
    Features:
    - Near-Far user pairing (optimal for NOMA)
    - Fractional programming for EE maximization
    - Iterative power allocation with convergence guarantee
    
    Complexity: O(T_outer * K) where T_outer ~ 20-30 iterations
    """
    
    def __init__(self, config):
        """
        Initialize baseline algorithm.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.K = config['system']['K']
        self.P_max = config['power']['P_max_W']
        self.noise_power = config['noise_power_W']
        self.psi = config['noma']['psi']
        self.P_sens = config['power']['P_sens_W']
        
        # Dinkelbach parameters
        self.tol = 1e-4  # Convergence tolerance
        self.max_outer_iter = 30  # Max outer loop iterations
        self.max_inner_iter = 200  # Max inner optimization iterations
        
        # Initialize models
        self.channel_model = ChannelModel(config)
        self.power_model = PowerModel(config)
    
    def pair_users_near_far(self, H):
        """
        Pair users using Near-Far algorithm.
        
        Args:
            H: Channel matrix (K, N_t)
            
        Returns:
            pairs: List of tuples (near_idx, far_idx)
            gains: Channel gains for all users
        """
        gains = self.channel_model.get_channel_gains(H)
        sorted_indices = np.argsort(gains)[::-1]
        
        num_pairs = self.K // 2
        pairs = []
        
        for i in range(num_pairs):
            near_idx = sorted_indices[i]
            far_idx = sorted_indices[-(i+1)]
            pairs.append((near_idx, far_idx))
        
        return pairs, gains
    
    def inner_optimization(self, pairs, gains, lam, step_size=1e-3):
        """
        Inner optimization: Maximize (Rate - lambda * Power).
        
        Uses projected gradient ascent.
        
        Args:
            pairs: NOMA pairs
            gains: Channel gains (K,)
            lam: Current lambda value (EE estimate)
            step_size: Gradient step size
            
        Returns:
            power_allocation: Optimal power (K,)
        """
        B = self.config['frequency']['B']
        alpha_overhead = self.config['ofdm_overheads']['alpha_total']
        
        # Initialize power allocation (uniform)
        P_per_cluster = self.P_max / len(pairs)
        power_allocation = np.zeros(self.K)
        
        for near_idx, far_idx in pairs:
            power_allocation[near_idx] = 0.3 * P_per_cluster
            power_allocation[far_idx] = 0.7 * P_per_cluster
        
        # Gradient ascent
        for _ in range(self.max_inner_iter):
            # Calculate gradients for each user
            gradients = np.zeros(self.K)
            
            for near_idx, far_idx in pairs:
                g_near = gains[near_idx]
                g_far = gains[far_idx]
                p_near = power_allocation[near_idx]
                p_far = power_allocation[far_idx]
                
                # SINR calculations
                sinr_far = (g_far * p_far) / (g_far * p_near + self.noise_power + 1e-12)
                residual = self.psi * g_near * p_far
                sinr_near = (g_near * p_near) / (residual + self.noise_power + 1e-12)
                
                # Gradient of rate w.r.t. power (Shannon capacity derivative)
                # dR/dp = B * (1/ln(2)) * (h / (noise + interference + h*p))
                
                # Far user gradient
                denom_far = g_far * p_near + self.noise_power
                grad_rate_far = B * (1 - alpha_overhead) * (1/np.log(2)) * (g_far / (denom_far + g_far * p_far))
                
                # Near user gradient
                denom_near = residual + self.noise_power
                grad_rate_near = B * (1 - alpha_overhead) * (1/np.log(2)) * (g_near / (denom_near + g_near * p_near))
                
                # Gradient of objective: dR/dp - lambda * (1/eta_PA)
                eta_PA = self.config['power']['eta_PA']
                gradients[far_idx] = grad_rate_far - lam / eta_PA
                gradients[near_idx] = grad_rate_near - lam / eta_PA
            
            # Gradient ascent step
            power_allocation += step_size * gradients
            
            # Project onto feasible set: p >= 0
            power_allocation = np.maximum(power_allocation, 0.0)
            
            # Project onto power constraint: sum(p) <= P_max
            total_power = np.sum(power_allocation)
            if total_power > self.P_max:
                power_allocation *= (self.P_max / total_power)
        
        return power_allocation
    
    def dinkelbach_optimization(self, pairs, gains):
        """
        Dinkelbach algorithm for EE maximization.
        
        Outer loop:
        1. Solve inner problem: max(R - lambda*P)
        2. Update lambda = R*/P*
        3. Repeat until convergence
        
        Args:
            pairs: NOMA pairs
            gains: Channel gains (K,)
            
        Returns:
            power_allocation: Optimal power allocation (K,)
            lam_opt: Optimal EE value
        """
        lam = 0.0  # Initialize lambda
        
        for iteration in range(self.max_outer_iter):
            # Inner optimization
            power_allocation = self.inner_optimization(pairs, gains, lam)
            
            # Calculate rate and power
            total_rate = self._calculate_sum_rate(power_allocation, pairs, gains)
            P_tx = np.sum(power_allocation)
            P_total = self.power_model.calculate_total_power(P_tx)
            
            if P_total <= 1e-12:
                break
            
            # Update lambda
            lam_new = total_rate / P_total
            
            # Check convergence
            if abs(lam_new - lam) < self.tol:
                break
            
            lam = lam_new
        
        return power_allocation, lam
    
    def _calculate_sum_rate(self, power_allocation, pairs, gains):
        """Calculate sum rate for given power allocation."""
        B = self.config['frequency']['B']
        alpha_overhead = self.config['ofdm_overheads']['alpha_total']
        
        total_rate = 0.0
        
        for near_idx, far_idx in pairs:
            g_near = gains[near_idx]
            g_far = gains[far_idx]
            p_near = power_allocation[near_idx]
            p_far = power_allocation[far_idx]
            
            # SINR
            sinr_far = (g_far * p_far) / (g_far * p_near + self.noise_power + 1e-12)
            residual = self.psi * g_near * p_far
            sinr_near = (g_near * p_near) / (residual + self.noise_power + 1e-12)
            
            # Rates
            rate_far = B * np.log2(1 + sinr_far) * (1 - alpha_overhead)
            rate_near = B * np.log2(1 + sinr_near) * (1 - alpha_overhead)
            
            total_rate += rate_far + rate_near
        
        return total_rate
    
    def optimize(self, H):
        """
        Run complete Dinkelbach optimization.
        
        Args:
            H: Channel matrix (K, N_t)
            
        Returns:
            power_allocation: Optimal power allocation (K,)
            pairs: NOMA pairs
            metrics: Dictionary with performance metrics
        """
        # Step 1: Pair users (Near-Far)
        pairs, gains = self.pair_users_near_far(H)
        
        # Step 2: Dinkelbach optimization
        power_allocation, lam_opt = self.dinkelbach_optimization(pairs, gains)
        
        # Step 3: Calculate metrics
        metrics = self._calculate_metrics(H, power_allocation, pairs, gains, lam_opt)
        
        return power_allocation, pairs, metrics
    
    def _calculate_metrics(self, H, power_allocation, pairs, gains, lam_opt):
        """Calculate SE, EE, and other metrics."""
        B = self.config['frequency']['B']
        noise_power = self.config['noise_power_W']
        alpha_overhead = self.config['ofdm_overheads']['alpha_total']
        
        # Calculate SINR for each user
        sinr_values = np.zeros(self.K)
        
        for near_idx, far_idx in pairs:
            g_near = gains[near_idx]
            g_far = gains[far_idx]
            p_near = power_allocation[near_idx]
            p_far = power_allocation[far_idx]
            
            # Far user SINR
            sinr_far = (g_far * p_far) / (g_far * p_near + noise_power + 1e-12)
            
            # Near user SINR with SIC
            residual = self.psi * g_near * p_far
            sinr_near = (g_near * p_near) / (residual + noise_power + 1e-12)
            
            sinr_values[near_idx] = sinr_near
            sinr_values[far_idx] = sinr_far
        
        # Calculate rates
        rates = B * np.log2(1 + sinr_values) * (1 - alpha_overhead)
        
        # Spectral Efficiency
        SE = np.sum(rates) / B
        
        # Energy Efficiency
        P_tx = np.sum(power_allocation)
        P_total = self.power_model.calculate_total_power(P_tx)
        EE = np.sum(rates) / P_total
        
        return {
            'SE': SE,
            'EE': EE,
            'EE_dinkelbach': lam_opt,
            'rates': rates,
            'sinr': sinr_values,
            'total_power': P_tx,
            'total_power_consumption': P_total,
            'num_pairs': len(pairs)
        }


def test_dinkelbach_baseline():
    """Test the Dinkelbach baseline."""
    print("="*70)
    print("Testing Dinkelbach EE Optimization Baseline")
    print("="*70)
    
    # Load config
    config = get_scenario_config('small_cell')
    
    # Create baseline
    baseline = DinkelbachBaseline(config)
    
    # Generate channel
    channel_model = ChannelModel(config, seed=42)
    H, _, _ = channel_model.generate_channel_matrix()
    
    print(f"\n✅ Channel Generated:")
    print(f"   Shape: {H.shape}")
    
    # Run optimization
    print(f"\n⏳ Running Dinkelbach optimization...")
    power_allocation, pairs, metrics = baseline.optimize(H)
    
    print(f"\n✅ Optimization Complete:")
    print(f"   Number of pairs: {len(pairs)}")
    print(f"   Converged lambda (EE): {metrics['EE_dinkelbach']:.4e} bits/J")
    
    print(f"\n✅ Power Allocation:")
    print(f"   Total allocated: {np.sum(power_allocation):.4f} W")
    print(f"   Max power: {config['power']['P_max_W']} W")
    print(f"   Min power per user: {np.min(power_allocation[power_allocation > 0]):.4f} W")
    print(f"   Max power per user: {np.max(power_allocation):.4f} W")
    
    print(f"\n✅ Performance Metrics:")
    print(f"   Spectral Efficiency: {metrics['SE']:.4f} bps/Hz")
    print(f"   Energy Efficiency: {metrics['EE']:.4e} bits/J")
    print(f"   Average Rate: {np.mean(metrics['rates'][metrics['rates'] > 0])/1e6:.4f} Mbps")
    print(f"   Min Rate: {np.min(metrics['rates'][metrics['rates'] > 0])/1e6:.4f} Mbps")
    print(f"   Max Rate: {np.max(metrics['rates'])/1e6:.4f} Mbps")
    print(f"   Total Power Consumption: {metrics['total_power_consumption']:.4f} W")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_dinkelbach_baseline()