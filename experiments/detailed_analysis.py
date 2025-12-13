"""
experiments/detailed_analysis.py
Generate detailed analysis and additional figures for paper
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import sys
sys.path.append('.')

# Load results
with open('results/evaluation_results.json', 'r') as f:
    results = json.load(f)

# Load training logs
import glob
log_files = glob.glob('results/training_logs/*.json')
if log_files:
    with open(log_files[0], 'r') as f:
        training_logs = json.load(f)
else:
    training_logs = None

print("="*70)
print("DETAILED RESULTS ANALYSIS")
print("="*70)

# 1. Calculate percentage improvements
snr_20_idx = np.argmin(np.abs(np.array(results['SNR_dB']) - 20))

print("\n📊 Performance at SNR = 20 dB:")
print("-" * 70)

algos = ['DRL', 'Greedy', 'WaterFilling', 'Dinkelbach']
se_values = {algo: results[algo]['SE'][snr_20_idx] for algo in algos}
ee_values = {algo: results[algo]['EE'][snr_20_idx] for algo in algos}

# Find best baseline
baseline_se = max([se_values[a] for a in ['Greedy', 'WaterFilling', 'Dinkelbach']])
baseline_ee = max([ee_values[a] for a in ['Greedy', 'WaterFilling', 'Dinkelbach']])

print(f"\nSpectral Efficiency (bps/Hz):")
for algo in algos:
    improvement = ((se_values[algo] - baseline_se) / baseline_se * 100) if algo == 'DRL' else 0
    print(f"  {algo:15s}: {se_values[algo]:.4f}  ({improvement:+.2f}%)")

print(f"\nEnergy Efficiency (Mbits/J):")
for algo in algos:
    ee_mbits = ee_values[algo] / 1e6
    improvement = ((ee_values[algo] - baseline_ee) / baseline_ee * 100) if algo == 'DRL' else 0
    print(f"  {algo:15s}: {ee_mbits:.4f}  ({improvement:+.2f}%)")

# 2. Generate gain percentage table
print("\n" + "="*70)
print("PERFORMANCE GAIN OVER BEST BASELINE")
print("="*70)

df_gains = pd.DataFrame({
    'Metric': ['SE Gain (%)', 'EE Gain (%)'],
    'vs Greedy': [
        (se_values['DRL'] - se_values['Greedy']) / se_values['Greedy'] * 100,
        (ee_values['DRL'] - ee_values['Greedy']) / ee_values['Greedy'] * 100
    ],
    'vs Water-Filling': [
        (se_values['DRL'] - se_values['WaterFilling']) / se_values['WaterFilling'] * 100,
        (ee_values['DRL'] - ee_values['WaterFilling']) / ee_values['WaterFilling'] * 100
    ],
    'vs Dinkelbach': [
        (se_values['DRL'] - se_values['Dinkelbach']) / se_values['Dinkelbach'] * 100,
        (ee_values['DRL'] - ee_values['Dinkelbach']) / ee_values['Dinkelbach'] * 100
    ]
})

print("\n" + df_gains.to_string(index=False))
df_gains.to_csv('results/tables/performance_gains.csv', index=False)
print("\n✅ Saved: results/tables/performance_gains.csv")

# 3. Training convergence plot
if training_logs:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Reward
    rewards = training_logs['episode_rewards']
    window = 50
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed, linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Training Reward Convergence')
    axes[0].grid(True, alpha=0.3)
    
    # SE
    se = training_logs['episode_se']
    smoothed_se = np.convolve(se, np.ones(window)/window, mode='valid')
    axes[1].plot(smoothed_se, linewidth=2, color='green')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('SE (bps/Hz)')
    axes[1].set_title('Spectral Efficiency Convergence')
    axes[1].grid(True, alpha=0.3)
    
    # EE
    ee = training_logs['episode_ee']
    smoothed_ee = np.convolve(ee, np.ones(window)/window, mode='valid')
    axes[2].plot(smoothed_ee, linewidth=2, color='orange')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('EE (bits/J)')
    axes[2].set_title('Energy Efficiency Convergence')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/training_convergence.png', dpi=300, bbox_inches='tight')
    print("\n✅ Saved: results/figures/training_convergence.png")
    plt.close()

# 4. Summary statistics across all SNR
print("\n" + "="*70)
print("AVERAGE PERFORMANCE ACROSS ALL SNR")
print("="*70)

for algo in algos:
    avg_se = np.mean(results[algo]['SE'])
    avg_ee = np.mean(results[algo]['EE']) / 1e6
    print(f"\n{algo}:")
    print(f"  Avg SE: {avg_se:.4f} bps/Hz")
    print(f"  Avg EE: {avg_ee:.4f} Mbits/J")

print("\n" + "="*70)
print("✅ Analysis Complete!")
print("="*70)