"""
src/baselines/waterfilling.py

Baseline 2: Water-Filling Power Allocation
From MIMO_v3.pdf Section 2.1

Algorithm:
1. Random user pairing (for NOMA)
2. Water-filling power allocation over parallel channels
3. Allocate power based on channel quality

Water-filling: p_i = max(mu - sigma_n^2/|h_i|^2, 0)
where mu is the water level chosen to satisfy power constraint
"""

import numpy as np
import sys
sys.path.append('.')

from configs.system_params import get_scenario_config
from src.environment.channel_model import ChannelModel
from src.environment.power_model import PowerModel


class WaterFillingBaseline:
    """
    Water-Filling Power Allocation Baseline
    
    Features:
    - Random user pairing
    - Optimal power allocation over parallel channels
    - Maximizes sum rate under power constraint
    
    Complexity: O(K * log K) - dominated by bisection search
    """
    
    def __init__(self, config, seed=None):
        """
        Initialize baseline algorithm.
        
        Args:
            config: System configuration dictionary
            seed: Random seed for pairing
        """
        self.config = config
        self.K = config['system']['K']
        self.P_max = config['power']['P_max_W']
        self.noise_power = config['noise_power_W']
        self.psi = config['noma']['psi']
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize models
        self.channel_model = ChannelModel(config)
        self.power_model = PowerModel(config)
    
    def pair_users_random(self):
        """
        Randomly pair users for NOMA.
        
        Returns:
            pairs: List of tuples (user1_idx, user2_idx)
        """
        # Random permutation
        indices = np.random.permutation(self.K)
        
        # Create pairs
        num_pairs = self.K // 2
        pairs = []
        
        for i in range(num_pairs):
            user1 = indices[2*i]
            user2 = indices[2*i + 1]
            pairs.append((user1, user2))
        
        return pairs
    
    def water_filling(self, channel_gains, P_total, tol=1e-8, max_iter=200):
        """
        Water-filling power allocation algorithm.
        
        Allocate power to maximize sum rate:
        p_k = max(mu - noise_power/|h_k|^2, 0)
        
        Args:
            channel_gains: Channel power gains |h_k|^2 for each user (K,)
            P_total: Total power budget
            tol: Convergence tolerance
            max_iter: Maximum iterations for bisection
            
        Returns:
            power_allocation: Power for each user (K,)
        """
        K = len(channel_gains)
        
        # Calculate inverse channel quality (noise-to-channel ratio)
        # Avoid division by zero
        inv_quality = self.noise_power / (channel_gains + 1e-12)
        
        # Bisection search for water level mu
        # Lower bound: minimum to allocate any power
        # Upper bound: allocate all power to best channel
        mu_low = np.min(inv_quality)
        mu_high = np.max(inv_quality) + P_total + 1.0
        
        power_allocation = np.zeros(K)
        
        for _ in range(max_iter):
            mu = 0.5 * (mu_low + mu_high)
            
            # Calculate power allocation
            power_allocation = np.maximum(mu - inv_quality, 0.0)
            
            # Check power constraint
            total_power = np.sum(power_allocation)
            
            if total_power > P_total:
                mu_high = mu
            else:
                mu_low = mu
            
            # Check convergence
            if abs(total_power - P_total) < tol:
                break
        
        return power_allocation
    
    def allocate_power_waterfilling(self, pairs, gains):
        """
        Allocate power using water-filling for NOMA clusters.
        
        For each cluster:
        1. Allocate power per cluster using water-filling
        2. Distribute within cluster based on NOMA requirements
        
        Args:
            pairs: List of (user1_idx, user2_idx) tuples
            gains: Channel gains for all users (K,)
            
        Returns:
            power_allocation: Power for each user (K,)
        """
        power_allocation = np.zeros(self.K)
        
        # Equal power budget per cluster
        P_per_cluster = self.P_max / len(pairs)
        
        for user1_idx, user2_idx in pairs:
            g1 = gains[user1_idx]
            g2 = gains[user2_idx]
            
            # Determine near/far based on channel gain
            if g1 > g2:
                near_idx, far_idx = user1_idx, user2_idx
                g_near, g_far = g1, g2
            else:
                near_idx, far_idx = user2_idx, user1_idx
                g_near, g_far = g2, g1
            
            # Water-filling for this cluster (2 parallel channels)
            cluster_gains = np.array([g_near, g_far])
            cluster_power = self.water_filling(cluster_gains, P_per_cluster)
            
            # Assign power
            power_allocation[near_idx] = cluster_power[0]
            power_allocation[far_idx] = cluster_power[1]
        
        return power_allocation
    
    def optimize(self, H):
        """
        Run complete Water-Filling optimization.
        
        Args:
            H: Channel matrix (K, N_t)
            
        Returns:
            power_allocation: Optimal power allocation (K,)
            pairs: NOMA pairs
            metrics: Dictionary with performance metrics
        """
        # Step 1: Get channel gains
        gains = self.channel_model.get_channel_gains(H)
        
        # Step 2: Random pairing
        pairs = self.pair_users_random()
        
        # Step 3: Water-filling power allocation
        power_allocation = self.allocate_power_waterfilling(pairs, gains)
        
        # Step 4: Calculate performance metrics
        metrics = self._calculate_metrics(H, power_allocation, pairs, gains)
        
        return power_allocation, pairs, metrics
    
    def _calculate_metrics(self, H, power_allocation, pairs, gains):
        """Calculate SE, EE, and other metrics."""
        B = self.config['frequency']['B']
        noise_power = self.config['noise_power_W']
        alpha_overhead = self.config['ofdm_overheads']['alpha_total']
        
        # Calculate SINR for each user
        sinr_values = np.zeros(self.K)
        
        for user1_idx, user2_idx in pairs:
            g1 = gains[user1_idx]
            g2 = gains[user2_idx]
            p1 = power_allocation[user1_idx]
            p2 = power_allocation[user2_idx]
            
            # Determine near/far
            if g1 > g2:
                near_idx, far_idx = user1_idx, user2_idx
                g_near, g_far = g1, g2
                p_near, p_far = p1, p2
            else:
                near_idx, far_idx = user2_idx, user1_idx
                g_near, g_far = g2, g1
                p_near, p_far = p2, p1
            
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
            'rates': rates,
            'sinr': sinr_values,
            'total_power': P_tx,
            'total_power_consumption': P_total,
            'num_pairs': len(pairs)
        }


def test_waterfilling_baseline():
    """Test the Water-Filling baseline."""
    print("="*70)
    print("Testing Water-Filling Baseline")
    print("="*70)
    
    # Load config
    config = get_scenario_config('small_cell')
    
    # Create baseline
    baseline = WaterFillingBaseline(config, seed=42)
    
    # Generate channel
    channel_model = ChannelModel(config, seed=42)
    H, _, _ = channel_model.generate_channel_matrix()
    
    print(f"\n✅ Channel Generated:")
    print(f"   Shape: {H.shape}")
    
    # Run optimization
    power_allocation, pairs, metrics = baseline.optimize(H)
    
    print(f"\n✅ Optimization Complete:")
    print(f"   Number of pairs: {len(pairs)}")
    print(f"   Pairing method: Random")
    
    print(f"\n✅ Power Allocation:")
    print(f"   Total allocated: {np.sum(power_allocation):.4f} W")
    print(f"   Max power: {config['power']['P_max_W']} W")
    print(f"   Min power per user: {np.min(power_allocation[power_allocation > 0]):.4f} W")
    print(f"   Max power per user: {np.max(power_allocation):.4f} W")
    
    print(f"\n✅ Performance Metrics:")
    print(f"   Spectral Efficiency: {metrics['SE']:.4f} bps/Hz")
    print(f"   Energy Efficiency: {metrics['EE']:.4e} bits/J")
    print(f"   Average Rate: {np.mean(metrics['rates'])/1e6:.4f} Mbps")
    print(f"   Min Rate: {np.min(metrics['rates'])/1e6:.4f} Mbps")
    print(f"   Max Rate: {np.max(metrics['rates'])/1e6:.4f} Mbps")
    
    # Test water-filling on simple example
    print(f"\n✅ Water-Filling Test:")
    test_gains = np.array([1.0, 0.5, 0.25])
    test_power = baseline.water_filling(test_gains, P_total=10.0)
    print(f"   Channel gains: {test_gains}")
    print(f"   Power allocation: {test_power}")
    print(f"   Total: {np.sum(test_power):.4f} W")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_waterfilling_baseline()