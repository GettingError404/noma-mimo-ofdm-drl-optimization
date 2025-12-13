"""
src/environment/noma_mimo_ofdm_env.py

Gymnasium Environment for NOMA-MIMO-OFDM System
Implements DRL-compatible interface for resource allocation optimization
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
sys.path.append('.')

from configs.system_params import get_scenario_config
from src.environment.channel_model import ChannelModel
from src.environment.power_model import PowerModel


class NomaMimoOfdmEnv(gym.Env):
    """
    Gymnasium Environment for Joint NOMA-MIMO-OFDM Optimization
    
    Objective: Maximize SE and EE through:
    - Power allocation
    - User pairing (NOMA clusters)
    - Resource block assignment
    
    State: Channel conditions, current allocations, QoS status
    Action: Power allocation + user pairing decisions
    Reward: Weighted sum of SE and EE with penalties
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, scenario='small_cell', seed=None, render_mode=None):
        """
        Initialize environment.
        
        Args:
            scenario: 'small_cell' or 'macro_cell'
            seed: Random seed
            render_mode: Rendering mode (not implemented)
        """
        super().__init__()
        
        # Load configuration
        self.config = get_scenario_config(scenario)
        self.scenario = scenario
        self.render_mode = render_mode
        
        # System parameters
        self.K = self.config['system']['K']  # Number of users
        self.N_t = self.config['system']['N_t']  # TX antennas
        self.N_sc = self.config['frequency']['N_sc']  # Subcarriers
        self.B = self.config['frequency']['B']  # Bandwidth
        self.cluster_size = self.config['system']['NOMA_CLUSTER_SIZE']  # 2
        
        # Power parameters
        self.P_max = self.config['power']['P_max_W']
        self.P_sens = self.config['power']['P_sens_W']
        
        # NOMA parameters
        self.psi = self.config['noma']['psi']  # SIC residual
        self.alpha_near_range = [
            self.config['noma']['alpha_near_min'],
            self.config['noma']['alpha_near_max']
        ]
        self.alpha_far_range = [
            self.config['noma']['alpha_far_min'],
            self.config['noma']['alpha_far_max']
        ]
        
        # QoS parameters
        self.R_min = self.config['qos']['R_min_bps']
        
        # OFDM overhead
        self.alpha_overhead = self.config['ofdm_overheads']['alpha_total']
        
        # Noise power
        self.noise_power = self.config['noise_power_W']
        
        # Reward weights
        self.w_SE = self.config['drl_reward']['w_SE']
        self.w_EE = self.config['drl_reward']['w_EE']
        self.w_QoS = self.config['drl_reward']['w_QoS_penalty']
        self.w_SIC = self.config['drl_reward']['w_SIC_penalty']
        
        # Initialize models
        self.channel_model = ChannelModel(self.config, seed=seed)
        self.power_model = PowerModel(self.config)
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.default_rng(seed)
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = self.config['simulation']['max_steps_per_episode']
        
        # State variables
        self.H = None  # Channel matrix
        self.distances = None
        self.channel_gains = None
        self.cluster_pairs = None
        self.current_power_allocation = None
        
    def _define_spaces(self):
        """Define action and observation spaces."""
        
        # === OBSERVATION SPACE ===
        # State includes:
        # 1. Effective channel gains: (K,)
        # 2. Current power allocation: (K,)
        # 3. Remaining power budget: (1,)
        # 4. QoS satisfaction indicators: (K,)
        # 5. Current rates: (K,)
        
        obs_dim = self.K + self.K + 1 + self.K + self.K
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # === ACTION SPACE ===
        # Continuous action:
        # 1. Normalized power allocation for each user: (K,) in [0, 1]
        # 2. User pairing scores: (K,) in [-1, 1]
        
        action_dim = self.K + self.K
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate new channel realization
        self.H, self.distances, _ = self.channel_model.generate_channel_matrix()
        self.channel_gains = self.channel_model.get_channel_gains(self.H)
        
        # Create initial NOMA clusters (Near-Far pairing)
        self.cluster_pairs = self._create_near_far_pairs()
        
        # Initialize power allocation (uniform)
        self.current_power_allocation = np.ones(self.K) * (self.P_max / self.K)
        
        # Reset step counter
        self.current_step = 0
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action from agent (power allocation + pairing)
            
        Returns:
            observation: New state
            reward: Reward signal
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Parse action
        power_actions = action[:self.K]  # Normalized power
        pairing_actions = action[self.K:]  # Pairing scores
        
        # Convert actions to actual power allocation
        power_allocation = self._process_power_action(power_actions)
        
        # Update user pairing based on pairing actions
        self._update_pairing(pairing_actions)
        
        # Calculate SINR for all users
        sinr_values = self._calculate_sinr(power_allocation)
        
        # Calculate rates
        rates = self._calculate_rates(sinr_values)
        
        # Calculate SE and EE
        SE = self._calculate_spectral_efficiency(rates)
        EE = self._calculate_energy_efficiency(rates, power_allocation)
        
        # Check constraints
        qos_violations = self._check_qos_constraints(rates)
        sic_violations = self._check_sic_stability(power_allocation)
        
        # Calculate reward
        reward = self._calculate_reward(SE, EE, qos_violations, sic_violations)
        
        # Update state
        self.current_power_allocation = power_allocation
        self.current_step += 1
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        info.update({
            'SE': SE,
            'EE': EE,
            'rates': rates,
            'qos_violations': qos_violations,
            'sic_violations': sic_violations
        })
        
        return observation, reward, terminated, truncated, info
    
    def _create_near_far_pairs(self):
        """
        Create NOMA clusters using Near-Far pairing.
        
        Returns:
            pairs: List of tuples (near_user_idx, far_user_idx)
        """
        # Sort users by channel gain
        sorted_indices, _ = self.channel_model.sort_users_by_gain(self.H)
        
        # Pair strongest with weakest
        num_pairs = self.K // 2
        pairs = []
        for i in range(num_pairs):
            near_idx = sorted_indices[i]  # Strong user
            far_idx = sorted_indices[-(i+1)]  # Weak user
            pairs.append((near_idx, far_idx))
        
        return pairs
    
    def _update_pairing(self, pairing_scores):
        """
        Update user pairing based on DRL agent's pairing scores.
        
        Args:
            pairing_scores: Scores for each user (K,)
        """
        # Sort users by pairing scores
        sorted_indices = np.argsort(pairing_scores)[::-1]
        
        # Create pairs
        num_pairs = self.K // 2
        pairs = []
        for i in range(num_pairs):
            user1 = sorted_indices[i]
            user2 = sorted_indices[-(i+1)]
            
            # Assign near/far based on channel gain
            if self.channel_gains[user1] > self.channel_gains[user2]:
                pairs.append((user1, user2))
            else:
                pairs.append((user2, user1))
        
        self.cluster_pairs = pairs
    
    def _process_power_action(self, power_actions):
        """
        Convert normalized power actions to actual power allocation.
        
        Args:
            power_actions: Normalized power (K,) in [-1, 1]
            
        Returns:
            power_allocation: Actual power in Watts (K,)
        """
        # Convert to [0, 1]
        power_normalized = (power_actions + 1) / 2
        power_normalized = np.clip(power_normalized, 0, 1)
        
        # Scale to total power budget
        power_allocation = power_normalized * self.P_max
        
        # Ensure constraint
        if np.sum(power_allocation) > self.P_max:
            power_allocation = power_allocation * (self.P_max / np.sum(power_allocation))
        
        return power_allocation
    
    def _calculate_sinr(self, power_allocation):
        """
        Calculate SINR for all users considering NOMA interference and SIC.
        
        Args:
            power_allocation: Power for each user (K,)
            
        Returns:
            sinr_values: SINR for each user (K,)
        """
        sinr_values = np.zeros(self.K)
        
        for near_idx, far_idx in self.cluster_pairs:
            # Get channel gains
            g_near = self.channel_gains[near_idx]
            g_far = self.channel_gains[far_idx]
            
            # Get allocated powers
            p_near = power_allocation[near_idx]
            p_far = power_allocation[far_idx]
            
            # Far user (weak) decodes directly
            # Interference from near user
            sinr_far = (g_far * p_far) / (g_far * p_near + self.noise_power + 1e-12)
            
            # Near user (strong) performs SIC
            # Residual interference after SIC
            residual_interference = self.psi * g_near * p_far
            sinr_near = (g_near * p_near) / (residual_interference + self.noise_power + 1e-12)
            
            sinr_values[near_idx] = sinr_near
            sinr_values[far_idx] = sinr_far
        
        return sinr_values
    
    def _calculate_rates(self, sinr_values):
        """
        Calculate achievable rates using Shannon capacity.
        
        Args:
            sinr_values: SINR for each user (K,)
            
        Returns:
            rates: Data rates in bps (K,)
        """
        # Shannon capacity: R = B * log2(1 + SINR) * (1 - overhead)
        rates = self.B * np.log2(1 + sinr_values) * (1 - self.alpha_overhead)
        
        return rates
    
    def _calculate_spectral_efficiency(self, rates):
        """
        Calculate Spectral Efficiency.
        
        Args:
            rates: Data rates (K,)
            
        Returns:
            SE: Spectral efficiency in bps/Hz
        """
        SE = np.sum(rates) / self.B
        return SE
    
    def _calculate_energy_efficiency(self, rates, power_allocation):
        """
        Calculate Energy Efficiency.
        
        Args:
            rates: Data rates (K,)
            power_allocation: Power allocation (K,)
            
        Returns:
            EE: Energy efficiency in bits/Joule
        """
        total_rate = np.sum(rates)
        P_tx = np.sum(power_allocation)
        P_total = self.power_model.calculate_total_power(P_tx)
        
        EE = total_rate / (P_total + 1e-12)  # bits/Joule
        
        return EE
    
    def _check_qos_constraints(self, rates):
        """
        Check QoS constraint violations.
        
        Args:
            rates: Data rates (K,)
            
        Returns:
            violations: Number of users below R_min
        """
        violations = np.sum(rates < self.R_min)
        return violations
    
    def _check_sic_stability(self, power_allocation):
        """
        Check SIC stability constraint.
        
        Constraint: |h_near|^2 * (p_far - p_near) >= P_sens
        
        Args:
            power_allocation: Power allocation (K,)
            
        Returns:
            violations: Number of cluster pairs violating SIC stability
        """
        violations = 0
        
        for near_idx, far_idx in self.cluster_pairs:
            g_near = self.channel_gains[near_idx]
            p_near = power_allocation[near_idx]
            p_far = power_allocation[far_idx]
            
            # SIC stability check
            sic_term = g_near * (p_far - p_near)
            if sic_term < self.P_sens:
                violations += 1
        
        return violations
    
    def _calculate_reward(self, SE, EE, qos_violations, sic_violations):
        """
        Calculate reward function.
        
        Reward = w_SE * SE + w_EE * EE - w_QoS * qos_violations - w_SIC * sic_violations
        
        Args:
            SE: Spectral efficiency
            EE: Energy efficiency
            qos_violations: Number of QoS violations
            sic_violations: Number of SIC violations
            
        Returns:
            reward: Scalar reward value
        """
        # Normalize SE and EE for better scaling
        SE_normalized = SE / 10.0  # Typical SE: 0-10 bps/Hz
        EE_normalized = EE / 1e6   # Typical EE: 0-10 Mbits/J
        
        reward = (
            self.w_SE * SE_normalized +
            self.w_EE * EE_normalized -
            self.w_QoS * qos_violations -
            self.w_SIC * sic_violations
        )
        
        return reward
    
    def _get_observation(self):
        """
        Construct observation vector.
        
        Returns:
            observation: State vector
        """
        # Calculate current rates
        sinr = self._calculate_sinr(self.current_power_allocation)
        current_rates = self._calculate_rates(sinr)
        
        # Construct state
        obs = np.concatenate([
            self.channel_gains / np.max(self.channel_gains),  # Normalized gains (K,)
            self.current_power_allocation / self.P_max,       # Normalized power (K,)
            [np.sum(self.current_power_allocation) / self.P_max],  # Power usage (1,)
            (current_rates >= self.R_min).astype(np.float32), # QoS indicators (K,)
            current_rates / 1e6                                # Rates in Mbps (K,)
        ])
        
        return obs.astype(np.float32)
    
    def _get_info(self):
        """Get additional info."""
        return {
            'step': self.current_step,
            'num_pairs': len(self.cluster_pairs),
            'total_power': np.sum(self.current_power_allocation)
        }
    
    def render(self):
        """Render the environment (not implemented)."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}, Pairs: {len(self.cluster_pairs)}")


def test_environment():
    """Test the environment."""
    print("="*70)
    print("Testing NOMA-MIMO-OFDM Environment")
    print("="*70)
    
    # Create environment
    env = NomaMimoOfdmEnv(scenario='small_cell', seed=42)
    
    print(f"\n✅ Environment Created:")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.shape}")
    print(f"   Number of users: {env.K}")
    print(f"   Number of subcarriers: {env.N_sc}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\n✅ Environment Reset:")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Info: {info}")
    
    # Take random actions
    print(f"\n✅ Testing Random Actions:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Step {i+1}: Reward={reward:.4f}, SE={info['SE']:.4f}, EE={info['EE']:.2e}")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    test_environment()