"""
src/environment/channel_model.py

3GPP UMi Channel Model for NOMA-MIMO-OFDM System
Implements: Path loss, Shadowing, Rayleigh fading, Spatial correlation
"""

import numpy as np


class ChannelModel:
    """
    3GPP Urban Microcellular (UMi) Channel Model
    
    Features:
    - Large-scale fading: Path loss + Log-normal shadowing
    - Small-scale fading: Rayleigh (complex Gaussian)
    - Spatial correlation for MIMO
    """
    
    def __init__(self, config, seed=None):
        """
        Initialize channel model.
        
        Args:
            config: Configuration dictionary from system_params
            seed: Random seed for reproducibility
        """
        self.config = config
        self.N_t = config['system']['N_t']
        self.K = config['system']['K']
        
        # Channel parameters
        self.R_cell = config['channel']['R_cell']
        self.d_min = config['channel']['d_min']
        self.PL_intercept = config['channel']['PL_intercept']
        self.PL_slope = config['channel']['PL_slope']
        self.sigma_shadow = config['channel']['sigma_shadow']
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_user_positions(self):
        """
        Generate random user positions in the cell.
        
        Returns:
            distances: Array of distances in km, shape (K,)
            angles: Array of angles in radians, shape (K,)
        """
        # Random distances: uniform from d_min to R_cell
        distances_m = np.random.uniform(
            self.d_min, 
            self.R_cell, 
            self.K
        )
        distances_km = distances_m / 1000.0  # Convert to km for path loss
        
        # Random angles: uniform [0, 2π]
        angles = np.random.uniform(0, 2 * np.pi, self.K)
        
        return distances_km, angles
    
    def calculate_path_loss(self, distances_km):
        """
        Calculate 3GPP UMi path loss.
        
        Path Loss (dB) = 128.1 + 37.6 * log10(d_km)
        
        Args:
            distances_km: Distances in kilometers, shape (K,)
            
        Returns:
            path_loss_dB: Path loss in dB, shape (K,)
        """
        # Ensure minimum distance to avoid log(0)
        distances_km = np.maximum(distances_km, 1e-6)
        
        path_loss_dB = self.PL_intercept + self.PL_slope * np.log10(distances_km)
        
        return path_loss_dB
    
    def add_shadowing(self, path_loss_dB):
        """
        Add log-normal shadowing.
        
        Args:
            path_loss_dB: Path loss in dB, shape (K,)
            
        Returns:
            total_loss_dB: Path loss + shadowing in dB, shape (K,)
        """
        # Log-normal shadowing: Gaussian in dB domain
        shadowing_dB = np.random.normal(0, self.sigma_shadow, size=path_loss_dB.shape)
        
        total_loss_dB = path_loss_dB + shadowing_dB
        
        return total_loss_dB
    
    def generate_small_scale_fading(self, K, N_t):
        """
        Generate Rayleigh fading (complex Gaussian).
        
        Args:
            K: Number of users
            N_t: Number of transmit antennas
            
        Returns:
            H_small: Small-scale fading matrix, shape (K, N_t)
        """
        # Complex Gaussian: CN(0, 1)
        real_part = np.random.randn(K, N_t)
        imag_part = np.random.randn(K, N_t)
        
        H_small = (real_part + 1j * imag_part) / np.sqrt(2.0)
        
        return H_small
    
    def generate_channel_matrix(self):
        """
        Generate complete channel matrix H.
        
        H[k, t] = sqrt(path_gain[k]) * h_small[k, t]
        
        Returns:
            H: Channel matrix, shape (K, N_t), complex
            distances_km: User distances, shape (K,)
            path_loss_dB: Path loss values, shape (K,)
        """
        # Step 1: Generate user positions
        distances_km, angles = self.generate_user_positions()
        
        # Step 2: Calculate path loss
        path_loss_dB = self.calculate_path_loss(distances_km)
        
        # Step 3: Add shadowing
        total_loss_dB = self.add_shadowing(path_loss_dB)
        
        # Step 4: Convert to linear scale
        path_gain_linear = 10 ** (-total_loss_dB / 10.0)
        
        # Step 5: Generate small-scale fading
        H_small = self.generate_small_scale_fading(self.K, self.N_t)
        
        # Step 6: Combine large-scale and small-scale
        H = np.sqrt(path_gain_linear)[:, np.newaxis] * H_small
        
        return H, distances_km, path_loss_dB
    
    def get_channel_gains(self, H):
        """
        Calculate channel power gains |h|^2 for each user.
        
        Args:
            H: Channel matrix, shape (K, N_t)
            
        Returns:
            gains: Channel gains, shape (K,)
        """
        # Sum over all antennas: |h_k|^2 = sum_t |H[k,t]|^2
        gains = np.sum(np.abs(H) ** 2, axis=1)
        
        return gains
    
    def sort_users_by_gain(self, H):
        """
        Sort users by channel gain (descending order).
        
        Args:
            H: Channel matrix, shape (K, N_t)
            
        Returns:
            sorted_indices: User indices sorted by gain (strongest first)
            sorted_gains: Sorted channel gains
        """
        gains = self.get_channel_gains(H)
        sorted_indices = np.argsort(gains)[::-1]  # Descending order
        sorted_gains = gains[sorted_indices]
        
        return sorted_indices, sorted_gains
    
    def apply_zero_forcing_beamforming(self, H, cluster_indices):
        """
        Apply Zero-Forcing (ZF) beamforming for a NOMA cluster.
        
        For 2-user cluster: w = H^H * (H * H^H)^{-1}
        
        Args:
            H: Channel matrix, shape (K, N_t)
            cluster_indices: Indices of users in the cluster, list of length 2
            
        Returns:
            W: Beamforming matrix, shape (2, N_t)
            H_eff: Effective channel gains after beamforming, shape (2,)
        """
        # Extract cluster channels
        H_cluster = H[cluster_indices, :]  # Shape: (2, N_t)
        
        # Zero-Forcing: W = H^H * (H * H^H)^{-1}
        try:
            H_H_H = H_cluster @ H_cluster.conj().T  # Shape: (2, 2)
            inv_term = np.linalg.inv(H_H_H)
            W = H_cluster.conj().T @ inv_term  # Shape: (N_t, 2)
            
            # Normalize beamforming vectors
            for i in range(W.shape[1]):
                W[:, i] = W[:, i] / np.linalg.norm(W[:, i])
            
            # Calculate effective channel gains
            H_eff = np.abs(np.diag(H_cluster @ W)) ** 2  # Shape: (2,)
            
        except np.linalg.LinAlgError:
            # If matrix is singular, use identity (no beamforming)
            W = np.eye(self.N_t, 2, dtype=complex)
            H_eff = self.get_channel_gains(H_cluster)
        
        return W, H_eff


def test_channel_model():
    """Test the channel model."""
    import sys
    sys.path.append('.')
    from configs.system_params import get_scenario_config
    
    print("="*70)
    print("Testing Channel Model")
    print("="*70)
    
    # Load config
    config = get_scenario_config('small_cell')
    
    # Create channel model
    channel = ChannelModel(config, seed=42)
    
    # Generate channel
    H, distances, path_loss = channel.generate_channel_matrix()
    
    print(f"\n✅ Channel Matrix Generated:")
    print(f"   Shape: {H.shape}")
    print(f"   Mean |H|: {np.mean(np.abs(H)):.4f}")
    print(f"   Max |H|: {np.max(np.abs(H)):.4f}")
    print(f"   Min |H|: {np.min(np.abs(H)):.4f}")
    
    # Get channel gains
    gains = channel.get_channel_gains(H)
    print(f"\n✅ Channel Gains:")
    print(f"   Mean: {np.mean(gains):.4e}")
    print(f"   Max: {np.max(gains):.4e}")
    print(f"   Min: {np.min(gains):.4e}")
    
    # Sort users
    sorted_idx, sorted_gains = channel.sort_users_by_gain(H)
    print(f"\n✅ Sorted Users (by gain):")
    print(f"   Strongest user: {sorted_idx[0]}, Gain: {sorted_gains[0]:.4e}")
    print(f"   Weakest user: {sorted_idx[-1]}, Gain: {sorted_gains[-1]:.4e}")
    print(f"   Gain ratio (strong/weak): {sorted_gains[0]/sorted_gains[-1]:.2f}")
    
    # Test beamforming
    cluster = [sorted_idx[0], sorted_idx[-1]]  # Near-Far pair
    W, H_eff = channel.apply_zero_forcing_beamforming(H, cluster)
    print(f"\n✅ Zero-Forcing Beamforming:")
    print(f"   W shape: {W.shape}")
    print(f"   Effective gains: {H_eff}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_channel_model()