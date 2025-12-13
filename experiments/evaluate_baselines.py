"""
experiments/evaluate_baselines.py

Evaluate and Compare All Algorithms:
- DRL (TD3)
- Baseline 1: Greedy Near-Far
- Baseline 2: Water-Filling
- Baseline 3: Dinkelbach
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
from tqdm import tqdm
import json

sys.path.append('.')

from configs.system_params import get_scenario_config
from src.environment.noma_mimo_ofdm_env import NomaMimoOfdmEnv
from src.environment.channel_model import ChannelModel
from src.drl.td3_agent import TD3Agent
from src.baselines.greedy_nearfar import GreedyNearFarBaseline
from src.baselines.waterfilling import WaterFillingBaseline
from src.baselines.dinkelbach import DinkelbachBaseline

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def evaluate_drl_agent(env, agent, num_episodes=100):
    """
    Evaluate trained DRL agent.
    
    Returns:
        metrics: Dict with SE, EE, rates, etc.
    """
    se_list = []
    ee_list = []
    rates_list = []
    rewards_list = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_se = 0
        episode_ee = 0
        episode_reward = 0
        steps = 0
        
        done = False
        while not done:
            # Select action (no exploration noise)
            action = agent.select_action(state, noise_scale=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            
            episode_se += info['SE']
            episode_ee += info['EE']
            episode_reward += reward
            steps += 1
            
            done = terminated or truncated
        
        se_list.append(episode_se / steps)
        ee_list.append(episode_ee / steps)
        rewards_list.append(episode_reward)
    
    return {
        'SE_mean': np.mean(se_list),
        'SE_std': np.std(se_list),
        'EE_mean': np.mean(ee_list),
        'EE_std': np.std(ee_list),
        'reward_mean': np.mean(rewards_list),
        'reward_std': np.std(rewards_list)
    }


def evaluate_baseline(baseline, channel_model, num_drops=100):
    """
    Evaluate a baseline algorithm.
    
    Returns:
        metrics: Dict with SE, EE, etc.
    """
    se_list = []
    ee_list = []
    
    for _ in range(num_drops):
        # Generate channel
        H, _, _ = channel_model.generate_channel_matrix()
        
        # Run baseline optimization
        power_allocation, pairs, metrics = baseline.optimize(H)
        
        se_list.append(metrics['SE'])
        ee_list.append(metrics['EE'])
    
    return {
        'SE_mean': np.mean(se_list),
        'SE_std': np.std(se_list),
        'EE_mean': np.mean(ee_list),
        'EE_std': np.std(ee_list)
    }


def evaluate_all_algorithms_vs_snr(
    scenario='small_cell',
    snr_range_db=np.arange(-10, 31, 2),
    num_drops=100,
    drl_model_path='results/models/td3_best.pth'
):
    """
    Evaluate all algorithms across SNR range.
    
    Returns:
        results: Dict with results for each algorithm
    """
    print("="*70)
    print("Evaluating All Algorithms vs SNR")
    print("="*70)
    
    # Load configuration
    config = get_scenario_config(scenario)
    
    # Initialize environment and agents
    env = NomaMimoOfdmEnv(scenario=scenario, seed=42)
    channel_model = ChannelModel(config, seed=42)
    
    # Create baselines
    baseline_greedy = GreedyNearFarBaseline(config)
    baseline_wf = WaterFillingBaseline(config, seed=42)
    baseline_dinkel = DinkelbachBaseline(config)
    
    # Load DRL agent
    agent = TD3Agent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0]
    )
    
    if os.path.exists(drl_model_path):
        agent.load(drl_model_path)
        print(f"✅ Loaded DRL model from {drl_model_path}")
    else:
        print(f"⚠️  DRL model not found at {drl_model_path}, skipping DRL evaluation")
        agent = None
    
    # Storage for results
    results = {
        'DRL': {'SE': [], 'EE': []},
        'Greedy': {'SE': [], 'EE': []},
        'WaterFilling': {'SE': [], 'EE': []},
        'Dinkelbach': {'SE': [], 'EE': []}
    }
    
    # Evaluate across SNR range
    print(f"\nEvaluating across SNR range: {snr_range_db[0]} to {snr_range_db[-1]} dB")
    
    for snr_db in tqdm(snr_range_db, desc="SNR sweep"):
        # Convert SNR to power
        # SNR = P_tx / noise_power
        noise_power = config['noise_power_W']
        P_tx = noise_power * (10 ** (snr_db / 10))
        
        # Update power budget in config
        config['power']['P_max_W'] = P_tx
        
        # Evaluate DRL
        if agent is not None:
            drl_metrics = evaluate_drl_agent(env, agent, num_episodes=num_drops)
            results['DRL']['SE'].append(drl_metrics['SE_mean'])
            results['DRL']['EE'].append(drl_metrics['EE_mean'])
        
        # Evaluate Greedy
        greedy_metrics = evaluate_baseline(baseline_greedy, channel_model, num_drops)
        results['Greedy']['SE'].append(greedy_metrics['SE_mean'])
        results['Greedy']['EE'].append(greedy_metrics['EE_mean'])
        
        # Evaluate Water-Filling
        wf_metrics = evaluate_baseline(baseline_wf, channel_model, num_drops)
        results['WaterFilling']['SE'].append(wf_metrics['SE_mean'])
        results['WaterFilling']['EE'].append(wf_metrics['EE_mean'])
        
        # Evaluate Dinkelbach
        dinkel_metrics = evaluate_baseline(baseline_dinkel, channel_model, num_drops)
        results['Dinkelbach']['SE'].append(dinkel_metrics['SE_mean'])
        results['Dinkelbach']['EE'].append(dinkel_metrics['EE_mean'])
    
    # Convert to arrays
    for algo in results:
        results[algo]['SE'] = np.array(results[algo]['SE'])
        results[algo]['EE'] = np.array(results[algo]['EE'])
    
    results['SNR_dB'] = snr_range_db
    
    return results


def plot_results(results, save_dir='results/figures'):
    """
    Plot SE and EE vs SNR.
    
    Args:
        results: Results dictionary from evaluation
        save_dir: Directory to save figures
    """
    os.makedirs(save_dir, exist_ok=True)
    
    snr_db = results['SNR_dB']
    
    # Figure 1: Spectral Efficiency vs SNR
    plt.figure(figsize=(10, 6))
    
    for algo, style in [
        ('DRL', {'marker': 'o', 'linewidth': 2.5, 'markersize': 8}),
        ('Greedy', {'marker': 's', 'linewidth': 2, 'markersize': 6}),
        ('WaterFilling', {'marker': '^', 'linewidth': 2, 'markersize': 6}),
        ('Dinkelbach', {'marker': 'd', 'linewidth': 2, 'markersize': 6})
    ]:
        if len(results[algo]['SE']) > 0:
            plt.plot(snr_db, results[algo]['SE'], label=algo, **style)
    
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Spectral Efficiency (bps/Hz)', fontsize=14)
    plt.title('Spectral Efficiency vs SNR', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    se_path = os.path.join(save_dir, 'SE_vs_SNR.png')
    plt.savefig(se_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {se_path}")
    plt.close()
    
    # Figure 2: Energy Efficiency vs SNR
    plt.figure(figsize=(10, 6))
    
    for algo, style in [
        ('DRL', {'marker': 'o', 'linewidth': 2.5, 'markersize': 8}),
        ('Greedy', {'marker': 's', 'linewidth': 2, 'markersize': 6}),
        ('WaterFilling', {'marker': '^', 'linewidth': 2, 'markersize': 6}),
        ('Dinkelbach', {'marker': 'd', 'linewidth': 2, 'markersize': 6})
    ]:
        if len(results[algo]['EE']) > 0:
            plt.plot(snr_db, results[algo]['EE'] / 1e6, label=algo, **style)  # Convert to Mbits/J
    
    plt.xlabel('SNR (dB)', fontsize=14)
    plt.ylabel('Energy Efficiency (Mbits/Joule)', fontsize=14)
    plt.title('Energy Efficiency vs SNR', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    ee_path = os.path.join(save_dir, 'EE_vs_SNR.png')
    plt.savefig(ee_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {ee_path}")
    plt.close()
    
    # Figure 3: SE-EE Pareto Frontier
    plt.figure(figsize=(10, 6))
    
    for algo, style in [
        ('DRL', {'marker': 'o', 's': 100, 'alpha': 0.7}),
        ('Greedy', {'marker': 's', 's': 80, 'alpha': 0.6}),
        ('WaterFilling', {'marker': '^', 's': 80, 'alpha': 0.6}),
        ('Dinkelbach', {'marker': 'd', 's': 80, 'alpha': 0.6})
    ]:
        if len(results[algo]['SE']) > 0:
            plt.scatter(results[algo]['SE'], results[algo]['EE'] / 1e6, 
                       label=algo, **style)
    
    plt.xlabel('Spectral Efficiency (bps/Hz)', fontsize=14)
    plt.ylabel('Energy Efficiency (Mbits/Joule)', fontsize=14)
    plt.title('SE-EE Pareto Frontier', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    pareto_path = os.path.join(save_dir, 'Pareto_Frontier.png')
    plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {pareto_path}")
    plt.close()


def generate_comparison_table(results, snr_target=20, save_dir='results/tables'):
    """
    Generate comparison table at specific SNR.
    
    Args:
        results: Results dictionary
        snr_target: SNR value to compare at (dB)
        save_dir: Directory to save table
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Find closest SNR index
    snr_db = results['SNR_dB']
    idx = np.argmin(np.abs(snr_db - snr_target))
    actual_snr = snr_db[idx]
    
    # Create comparison table
    table_data = []
    
    for algo in ['DRL', 'Greedy', 'WaterFilling', 'Dinkelbach']:
        if len(results[algo]['SE']) > 0:
            table_data.append({
                'Algorithm': algo,
                'SE (bps/Hz)': f"{results[algo]['SE'][idx]:.4f}",
                'EE (Mbits/J)': f"{results[algo]['EE'][idx]/1e6:.4f}",
            })
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_path = os.path.join(save_dir, f'comparison_SNR{actual_snr}dB.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Saved comparison table: {csv_path}")
    
    # Print table
    print(f"\n{'='*70}")
    print(f"Performance Comparison at SNR = {actual_snr} dB")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}")
    
    return df


def main():
    """Main evaluation function."""
    # Configuration
    SCENARIO = 'small_cell'
    SNR_RANGE = np.arange(-10, 31, 2)
    NUM_DROPS = 100
    DRL_MODEL_PATH = 'results/models/td3_best.pth'
    
    # Run evaluation
    results = evaluate_all_algorithms_vs_snr(
        scenario=SCENARIO,
        snr_range_db=SNR_RANGE,
        num_drops=NUM_DROPS,
        drl_model_path=DRL_MODEL_PATH
    )
    
    # Plot results
    print("\n📊 Generating plots...")
    plot_results(results)
    
    # Generate comparison table
    print("\n📋 Generating comparison table...")
    generate_comparison_table(results, snr_target=20)
    
    # Save results
    results_path = 'results/evaluation_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    results_json = {}
    for algo in results:
        if isinstance(results[algo], dict):
            results_json[algo] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in results[algo].items()}
        else:
            results_json[algo] = results[algo].tolist() if isinstance(results[algo], np.ndarray) else results[algo]
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✅ Results saved to {results_path}")
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()