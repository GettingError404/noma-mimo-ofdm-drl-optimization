"""
configs/system_params.py

Locked Global Constants for NOMA-MIMO-OFDM System
All parameters are FIXED and MUST NOT be changed to ensure reproducibility.

Source: NOMA_v3.pdf, OFDM_v3.pdf, MIMO_v3.pdf, Optimization_v4.pdf
"""

import numpy as np

# ==============================================================================
# SCENARIO CONFIGURATION
# ==============================================================================
SCENARIO = "small_cell"  # Options: "small_cell" or "macro_cell"

# ==============================================================================
# SYSTEM CONFIGURATION
# ==============================================================================
SYSTEM_CONFIG = {
    # Antenna Configuration
    'N_t': 64,                  # Number of transmit antennas at BS
    'N_r': 1,                   # Number of receive antennas per UE (single antenna)
    
    # User Configuration
    'K': 10,                    # Total number of users in the system
    'NOMA_CLUSTER_SIZE': 2,     # LOCKED: Exactly 2 users per NOMA cluster
    
    # Simulation Parameters
    'N_drops': 10000,           # Monte Carlo drops for averaging
    'seed': 42,                 # Random seed for reproducibility
}

# ==============================================================================
# FREQUENCY SPECIFICATIONS
# ==============================================================================
FREQUENCY_CONFIG = {
    'small_cell': {
        'B': 20e6,              # System bandwidth (Hz) - 20 MHz
        'N_sc': 600,            # Number of subcarriers
        'N_RB': 50,             # Number of Resource Blocks
        'N_FFT': 1024,          # FFT size
    },
    'macro_cell': {
        'B': 100e6,             # System bandwidth (Hz) - 100 MHz
        'N_sc': 3000,           # Number of subcarriers
        'N_RB': 250,            # Number of Resource Blocks
        'N_FFT': 2048,          # FFT size
    }
}

# Common frequency parameters
FREQUENCY_CONFIG['common'] = {
    'f_c': 2.0e9,               # Carrier frequency (Hz) - 2 GHz
    'delta_f': 15e3,            # Subcarrier spacing (Hz) - 15 kHz
}

# ==============================================================================
# CHANNEL MODEL (3GPP UMi)
# ==============================================================================
CHANNEL_CONFIG = {
    'small_cell': {
        'R_cell': 250,          # Cell radius (meters)
        'path_loss_model': '3GPP_UMi',  # Path loss model type
        'PL_intercept': 128.1,  # Path loss intercept (dB)
        'PL_slope': 37.6,       # Path loss slope
    },
    'macro_cell': {
        'R_cell': 500,          # Cell radius (meters)
        'path_loss_model': '3GPP_UMi',
        'PL_intercept': 128.1,
        'PL_slope': 37.6,
    }
}

# Common channel parameters
CHANNEL_CONFIG['common'] = {
    'd_min': 35,                # Minimum UE distance from BS (meters)
    'sigma_shadow': 8,          # Log-normal shadowing std dev (dB)
    'fading_model': 'Rayleigh', # Small-scale fading model
}

# ==============================================================================
# NOISE AND INTERFERENCE
# ==============================================================================
NOISE_CONFIG = {
    'N_0_dBm_Hz': -174,         # Noise power spectral density (dBm/Hz)
    'NF_dB': 9,                 # Noise figure (dB)
}

# Calculate thermal noise power (will be computed based on bandwidth)
def calculate_noise_power(bandwidth, N_0_dBm_Hz=-174, NF_dB=9):
    """Calculate thermal noise power in Watts."""
    N_0_W_Hz = 10**((N_0_dBm_Hz - 30) / 10)  # Convert dBm/Hz to W/Hz
    noise_power_W = N_0_W_Hz * bandwidth * 10**(NF_dB / 10)
    return noise_power_W

# ==============================================================================
# POWER SPECIFICATIONS
# ==============================================================================
POWER_CONFIG = {
    # Transmit Power
    'P_max_dBm': 46,            # Maximum BS transmit power (dBm) = 40W
    'P_max_W': 40,              # Maximum BS transmit power (Watts)
    'P_sens_dBm': -100,         # Receiver sensitivity (dBm)
    'P_sens_W': 1e-13,          # Receiver sensitivity (Watts)
    
    # Power Amplifier
    'eta_PA': 0.38,             # PA efficiency (small cell: 0.38, macro: 0.35)
}

# ==============================================================================
# CIRCUIT POWER MODEL (LOCKED - from MIMO spec)
# ==============================================================================
CIRCUIT_POWER_CONFIG = {
    'small_cell': {
        'P_static': 10,         # Static base station power (W)
        'P_DAC': 0.015,         # DAC power per TX chain (W)
        'P_ADC': 0.015,         # ADC power per RX chain (W)
        'P_mix': 0.020,         # Mixer power per chain (W)
        'P_filt': 0.020,        # Filter power per chain (W)
    },
    'macro_cell': {
        'P_static': 130,        # Static base station power (W)
        'P_DAC': 0.015,
        'P_ADC': 0.015,
        'P_mix': 0.020,
        'P_filt': 0.020,
    }
}

def calculate_circuit_power(N_t, K, scenario='small_cell'):
    """
    Calculate total circuit power consumption.
    LOCKED FORMULA: P_circuit = P_static + N_t*(P_DAC + P_mix + P_filt) + K*P_ADC
    """
    config = CIRCUIT_POWER_CONFIG[scenario]
    P_rf_tx = N_t * (config['P_DAC'] + config['P_mix'] + config['P_filt'])
    P_rf_rx = K * config['P_ADC']
    return config['P_static'] + P_rf_tx + P_rf_rx

# ==============================================================================
# NOMA PARAMETERS (LOCKED)
# ==============================================================================
NOMA_CONFIG = {
    'psi': 0.01,                # SIC residual interference factor (1%)
    'alpha_near_min': 0.2,      # Minimum power allocation for near user
    'alpha_near_max': 0.4,      # Maximum power allocation for near user
    'alpha_far_min': 0.6,       # Minimum power allocation for far user
    'alpha_far_max': 0.8,       # Maximum power allocation for far user
}

# Power allocation constraint: alpha_far + alpha_near <= 1
# SIC constraint: alpha_far > alpha_near (far user gets more power)

# ==============================================================================
# OFDM OVERHEADS (LOCKED)
# ==============================================================================
OFDM_OVERHEADS = {
    'alpha_CP': 0.0667,         # Cyclic Prefix overhead (6.67%)
    'alpha_pilot': 0.0476,      # Pilot/Reference signal overhead (4.76%)
    'alpha_GB': 0.10,           # Guard band overhead (10%)
    'alpha_ctrl': 0.214,        # Control channel overhead (21.4%)
}

# Total overhead
OFDM_OVERHEADS['alpha_total'] = sum([
    OFDM_OVERHEADS['alpha_CP'],
    OFDM_OVERHEADS['alpha_pilot'],
    OFDM_OVERHEADS['alpha_GB'],
    OFDM_OVERHEADS['alpha_ctrl']
])  # = 0.4283 (42.83%)

# ==============================================================================
# QoS REQUIREMENTS
# ==============================================================================
QOS_CONFIG = {
    'R_min_bps': 1e6,           # Minimum rate per user (1 Mbps)
    'BLER_target': 0.1,         # Target Block Error Rate (10%)
}

# ==============================================================================
# SIMULATION PARAMETERS
# ==============================================================================
SIMULATION_CONFIG = {
    'SNR_range_dB': np.arange(-10, 31, 2),  # SNR from -10 to 30 dB, step 2 dB
    'num_episodes': 1000,       # Number of training episodes for DRL
    'max_steps_per_episode': 100,  # Max time steps per episode
}

# ==============================================================================
# DRL REWARD WEIGHTS
# ==============================================================================
DRL_REWARD_CONFIG = {
    'w_SE': 1.0,                # Weight for Spectral Efficiency
    'w_EE': 1.0,                # Weight for Energy Efficiency
    'w_QoS_penalty': 10.0,      # Penalty weight for QoS violations
    'w_SIC_penalty': 5.0,       # Penalty weight for SIC instability
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_scenario_config(scenario='small_cell'):
    """Get all configuration parameters for a specific scenario."""
    assert scenario in ['small_cell', 'macro_cell'], "Invalid scenario"
    
    config = {
        'scenario': scenario,
        'system': SYSTEM_CONFIG.copy(),
        'frequency': FREQUENCY_CONFIG[scenario].copy(),
        'channel': CHANNEL_CONFIG[scenario].copy(),
        'power': POWER_CONFIG.copy(),
        'circuit_power': CIRCUIT_POWER_CONFIG[scenario].copy(),
        'noma': NOMA_CONFIG.copy(),
        'ofdm_overheads': OFDM_OVERHEADS.copy(),
        'qos': QOS_CONFIG.copy(),
        'noise': NOISE_CONFIG.copy(),
        'simulation': SIMULATION_CONFIG.copy(),
        'drl_reward': DRL_REWARD_CONFIG.copy(),
    }
    
    # Add common parameters
    config['frequency'].update(FREQUENCY_CONFIG['common'])
    config['channel'].update(CHANNEL_CONFIG['common'])
    
    # Calculate derived parameters
    config['noise_power_W'] = calculate_noise_power(
        config['frequency']['B'],
        config['noise']['N_0_dBm_Hz'],
        config['noise']['NF_dB']
    )
    
    config['circuit_power_total_W'] = calculate_circuit_power(
        config['system']['N_t'],
        config['system']['K'],
        scenario
    )
    
    return config

def print_config_summary(scenario='small_cell'):
    """Print a summary of the configuration."""
    config = get_scenario_config(scenario)
    
    print("="*70)
    print(f"NOMA-MIMO-OFDM System Configuration - {scenario.upper()}")
    print("="*70)
    print(f"\n📡 System Configuration:")
    print(f"   Transmit Antennas (N_t): {config['system']['N_t']}")
    print(f"   Number of Users (K): {config['system']['K']}")
    print(f"   NOMA Cluster Size: {config['system']['NOMA_CLUSTER_SIZE']}")
    
    print(f"\n📶 Frequency Configuration:")
    print(f"   Bandwidth: {config['frequency']['B']/1e6:.0f} MHz")
    print(f"   Subcarriers: {config['frequency']['N_sc']}")
    print(f"   Carrier Frequency: {config['frequency']['f_c']/1e9:.1f} GHz")
    
    print(f"\n🌐 Channel Configuration:")
    print(f"   Cell Radius: {config['channel']['R_cell']} m")
    print(f"   Path Loss Model: {config['channel']['path_loss_model']}")
    print(f"   Min Distance: {config['channel']['d_min']} m")
    
    print(f"\n⚡ Power Configuration:")
    print(f"   Max TX Power: {config['power']['P_max_dBm']} dBm ({config['power']['P_max_W']} W)")
    print(f"   PA Efficiency: {config['power']['eta_PA']*100:.0f}%")
    print(f"   Circuit Power: {config['circuit_power_total_W']:.2f} W")
    
    print(f"\n🔧 NOMA Configuration:")
    print(f"   SIC Residual (ψ): {config['noma']['psi']}")
    print(f"   Power Allocation Range: [{config['noma']['alpha_near_min']}, {config['noma']['alpha_far_max']}]")
    
    print(f"\n📊 OFDM Overheads:")
    print(f"   Total Overhead: {config['ofdm_overheads']['alpha_total']*100:.2f}%")
    
    print(f"\n🎯 QoS Requirements:")
    print(f"   Min Rate per User: {config['qos']['R_min_bps']/1e6:.1f} Mbps")
    
    print(f"\n🔬 Simulation:")
    print(f"   Monte Carlo Drops: {config['system']['N_drops']}")
    print(f"   SNR Range: {config['simulation']['SNR_range_dB'][0]} to {config['simulation']['SNR_range_dB'][-1]} dB")
    print(f"   Random Seed: {config['system']['seed']}")
    print("="*70)

# ==============================================================================
# MAIN - FOR TESTING
# ==============================================================================
if __name__ == "__main__":
    # Test configuration loading
    print_config_summary('small_cell')
    print("\n")
    print_config_summary('macro_cell')
    
    # Test helper functions
    config = get_scenario_config('small_cell')
    print(f"\n✅ Configuration loaded successfully!")
    print(f"✅ Noise Power: {config['noise_power_W']:.2e} W")
    print(f"✅ Circuit Power: {config['circuit_power_total_W']:.2f} W")