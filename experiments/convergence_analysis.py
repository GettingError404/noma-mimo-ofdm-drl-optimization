"""
Detailed convergence analysis
"""

import numpy as np
import matplotlib.pyplot as plt

# Load training logs
with open('results/training_logs/training_*.json', 'r') as f:
    logs = json.load(f)

# Metric 1: Sample Efficiency (how fast it learns)
episodes_to_converge = find_convergence_point(logs['episode_rewards'])
print(f"Converges after {episodes_to_converge} episodes")

# Metric 2: Final Performance Variance
final_100_rewards = logs['episode_rewards'][-100:]
variance = np.std(final_100_rewards)
print(f"Final variance: {variance:.4f}")

# Metric 3: Learning Curves
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Reward convergence with std bands
# Plot 2: SE convergence
# Plot 3: EE convergence  
# Plot 4: Constraint violations over time

plt.savefig('results/figures/convergence_detailed.png', dpi=300)