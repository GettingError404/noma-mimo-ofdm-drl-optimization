"""
Ablation Study: Test impact of each component
"""

# Test 1: Impact of reward weights
reward_configs = [
    {'w_SE': 1.0, 'w_EE': 0.0, 'w_QoS': 10, 'w_SIC': 5},  # SE only
    {'w_SE': 0.0, 'w_EE': 1.0, 'w_QoS': 10, 'w_SIC': 5},  # EE only
    {'w_SE': 1.0, 'w_EE': 1.0, 'w_QoS': 10, 'w_SIC': 5},  # Balanced (original)
    {'w_SE': 2.0, 'w_EE': 1.0, 'w_QoS': 10, 'w_SIC': 5},  # SE priority
    {'w_SE': 1.0, 'w_EE': 2.0, 'w_QoS': 10, 'w_SIC': 5},  # EE priority
]

# Test 2: Impact of QoS penalty
qos_penalties = [0, 5, 10, 20, 50]

# Test 3: Impact of SIC penalty
sic_penalties = [0, 2, 5, 10, 20]

# Test 4: Impact of network architecture
architectures = [
    [256, 128],           # Smaller
    [512, 256, 128],      # Original
    [1024, 512, 256],     # Larger
]

# Test 5: Without Twin Critics (DDPG vs TD3)
use_twin_critics = [True, False]

# Run each ablation, plot results