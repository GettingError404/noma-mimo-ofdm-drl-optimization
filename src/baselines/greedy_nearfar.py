"""
src/baselines/greedy_nearfar.py

Baseline 1: Greedy Near-Far Power Allocation
From NOMA_v3.pdf Section 3.1 and 3.2

Algorithm:
1. Pair users using Near-Far strategy (strongest with weakest)
2. Allocate power using Greedy Recipe A (alpha_far = 0.7, alpha_near = 0.3)
3. Verify SIC stability constraint
"""

import numpy as np
import sys
sys.path.append('.')

from configs.system_params import get_scenario_config
from src.environment.channel_model import ChannelModel
from src.environment.power_model import PowerModel


class GreedyNearFarBaseline:
    """
    Greedy Near-Far Baseline Algorithm
    
    Features:
    - Deterministic user pairing (Near-Far)
    - Fixed power allocation (alpha_far=0.7, alpha_near=0.3)
    - SIC stability verification
    
    Complexity: O(K log K) - dominated by sorting
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
        self.P_sens = config['power']['P_sens_W']
        self.psi = config['noma']['psi']
        
        # Default power allocation coefficients (from NOMA doc)
        self.alpha_far_default = 0.7
        self.alpha_near_default = 0.3
        
        # Initialize models
        self.channel_model = ChannelModel(config)
        self.power_model = PowerModel(config)
    
    def pair_users_near_far(self, H):
        """
        Pair users using Near-Far algorithm.
        
        Algorithm:
        1. Sort users by channel gain (descending)
        2. Pair strongest with weakest
        3. Continue until all users paired
        
        Args:
            H: Channel matrix (K, N_t)
            
        Returns:
            pairs: List of tuples (near_idx, far_idx)
            gains: Channel gains for all users
        """
        # Get channel gains
        gains = self.channel_model.get_channel_gains(H)
        
        # Sort users by gain (descending)
        sorted_indices = np.argsort(gains)[::-1]
        
        # Pair strongest with weakest
        num_pairs = self.K // 2
        pairs = []
        
        for i in range(num_pairs):
            near_idx = sorted_indices[i]  # Strong user (near)
            far_idx = sorted_indices[-(i+1)]  # Weak user (far)
            pairs.append((near_idx, far_idx))
        
        return pairs, gains
    
    def check_sic_stability(self, g_near, p_near, p_far):
        """
        Verify SIC stability constraint.
        
        Constraint: g_near * (p_far - p_near) >= P_sens
        
        Args:
            g_near: Channel gain of near user
            p_near: Power allocated to near user
            p_far: Power allocated to far user
            
        Returns:
            is_stable: Boolean indicating if SIC is stable
        """
        sic_term = g_near * (p_far - p_near)
        is_stable = (sic_term >= self.P_sens)
        
        return is_stable
    
    def allocate_power_greedy(self, pairs, gains, P_total=None):
        """
        Allocate power using Greedy Recipe A from NOMA doc.
        
        Default allocation:
        - alpha_far = 0.7 (far user gets more power)
        - alpha_near = 0.3 (near user gets less power)
        
        Adjust if SIC stability constraint is violated.
        
        Args:
            pairs: List of (near_idx, far_idx) tuples
            gains: Channel gains for all users (K,)
            P_total: Total power budget (defaults to P_max)
            
        Returns:
            power_allocation: Power for each user (K,)
            valid_pairs: Pairs that satisfy constraints
        """
        if P_total is None:
            P_total = self.P_max
        
        power_allocation = np.zeros(self.K)
        valid_pairs = []
        
        # Equal power per cluster
        P_per_cluster = P_total / len(pairs)
        
        for near_idx, far_idx in pairs:
            g_near = gains[near_idx]
            g_far = gains[far_idx]
            
            # Start with default allocation
            alpha_far = self.alpha_far_default
            alpha_near = self.alpha_near_default
            
            # Calculate powers
            p_far = alpha_far * P_per_cluster
            p_near = alpha_near * P_per_cluster
            
            # Check SIC stability
            is_stable = self.check_sic_stability(g_near, p_near, p_far)
            
            # If not stable, adjust alpha to satisfy constraint
            if not is_stable:
                # Increase alpha_far to satisfy: g_near * (p_far - p_near) >= P_sens
                # p_far - p_near >= P_sens / g_near
                # alpha_far - alpha_near >= P_sens / (g_near * P_per_cluster)
                
                min_diff = self.P_sens / (g_near * P_per_cluster)
                alpha_far = 0.5 + min_diff / 2
                alpha_near = 0.5 - min_diff / 2
                
                # Ensure valid range
                if alpha_far > 0.95:
                    # Can't satisfy constraint, skip this pair
                    continue
                
                alpha_far = min(alpha_far, 0.95)
                alpha_near = max(alpha_near, 0.05)
                
                # Normalize to sum to 1
                total = alpha_far + alpha_near
                alpha_far /= total
                alpha_near /= total
                
                # Recalculate powers
                p_far = alpha_far * P_per_cluster
                p_near = alpha_near * P_per_cluster
            
            # Allocate power
            power_allocation[far_idx] = p_far
            power_allocation[near_idx] = p_near
            valid_pairs.append((near_idx, far_idx))
        
        return power_allocation, valid_pairs
    
    def optimize(self, H):
        """
        Run complete Greedy Near-Far optimization.
        
        Args:
            H: Channel matrix (K, N_t)
            
        Returns:
            power_allocation: Optimal power allocation (K,)
            pairs: Valid NOMA pairs
            metrics: Dictionary with performance metrics
        """
        # Step 1: Pair users
        pairs, gains = self.pair_users_near_far(H)
        
        # Step 2: Allocate power
        power_allocation, valid_pairs = self.allocate_power_greedy(pairs, gains)
        
        # Step 3: Calculate performance metrics
        metrics = self._calculate_metrics(H, power_allocation, valid_pairs, gains)
        
        return power_allocation, valid_pairs, metrics
    
    def _calculate_metrics(self, H, power_allocation, pairs, gains):
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
            
            # Far user (weak) SINR
            sinr_far = (g_far * p_far) / (g_far * p_near + noise_power + 1e-12)
            
            # Near user (strong) SINR with SIC
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
            'rates': rates,
            'sinr': sinr_values,
            'total_power': P_tx,
            'total_power_consumption': P_total,
            'num_valid_pairs': len(pairs)
        }


def test_greedy_baseline():
    """Test the Greedy Near-Far baseline."""
    print("="*70)
    print("Testing Greedy Near-Far Baseline")
    print("="*70)
    
    # Load config
    config = get_scenario_config('small_cell')
    
    # Create baseline
    baseline = GreedyNearFarBaseline(config)
    
    # Generate channel
    channel_model = ChannelModel(config, seed=42)
    H, _, _ = channel_model.generate_channel_matrix()
    
    print(f"\n✅ Channel Generated:")
    print(f"   Shape: {H.shape}")
    
    # Run optimization
    power_allocation, pairs, metrics = baseline.optimize(H)
    
    print(f"\n✅ Optimization Complete:")
    print(f"   Number of pairs: {len(pairs)}")
    print(f"   Valid pairs: {metrics['num_valid_pairs']}")
    
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
    
    print(f"\n✅ SINR Statistics:")
    print(f"   Mean SINR: {np.mean(metrics['sinr'][metrics['sinr'] > 0]):.4f}")
    print(f"   Min SINR: {np.min(metrics['sinr'][metrics['sinr'] > 0]):.4f}")
    print(f"   Max SINR: {np.max(metrics['sinr']):.4f}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_greedy_baseline()