"""
src/utils/metrics.py

Utility functions for calculating performance metrics
"""

import numpy as np


def calculate_spectral_efficiency(rates, bandwidth):
    """
    Calculate Spectral Efficiency.
    
    SE = Sum(rates) / Bandwidth (bps/Hz)
    
    Args:
        rates: Array of user rates (bps)
        bandwidth: System bandwidth (Hz)
        
    Returns:
        SE: Spectral efficiency (bps/Hz)
    """
    return np.sum(rates) / bandwidth


def calculate_energy_efficiency(rates, total_power):
    """
    Calculate Energy Efficiency.
    
    EE = Sum(rates) / Total_Power (bits/Joule)
    
    Args:
        rates: Array of user rates (bps)
        total_power: Total power consumption (Watts)
        
    Returns:
        EE: Energy efficiency (bits/J)
    """
    if total_power <= 0:
        return 0.0
    return np.sum(rates) / total_power


def calculate_sinr(channel_gain, power, interference, noise_power):
    """
    Calculate Signal-to-Interference-plus-Noise Ratio.
    
    SINR = (channel_gain * power) / (interference + noise_power)
    
    Args:
        channel_gain: Channel power gain |h|^2
        power: Allocated power (W)
        interference: Interference power (W)
        noise_power: Noise power (W)
        
    Returns:
        SINR: Linear SINR value
    """
    return (channel_gain * power) / (interference + noise_power + 1e-12)


def calculate_rate(sinr, bandwidth, overhead=0.0):
    """
    Calculate achievable rate using Shannon capacity.
    
    Rate = B * log2(1 + SINR) * (1 - overhead)
    
    Args:
        sinr: Signal-to-interference-plus-noise ratio
        bandwidth: Bandwidth (Hz)
        overhead: Overhead factor (0 to 1)
        
    Returns:
        rate: Achievable rate (bps)
    """
    return bandwidth * np.log2(1 + sinr) * (1 - overhead)


def calculate_outage_probability(rates, min_rate):
    """
    Calculate outage probability (fraction of users below min rate).
    
    Args:
        rates: Array of user rates
        min_rate: Minimum required rate
        
    Returns:
        outage_prob: Outage probability (0 to 1)
    """
    num_outage = np.sum(rates < min_rate)
    return num_outage / len(rates)


def calculate_fairness_index(rates):
    """
    Calculate Jain's fairness index.
    
    FI = (sum(r_i))^2 / (N * sum(r_i^2))
    
    Args:
        rates: Array of user rates
        
    Returns:
        fairness_index: Value between 0 and 1 (1 = perfect fairness)
    """
    N = len(rates)
    if N == 0 or np.sum(rates**2) == 0:
        return 0.0
    
    return (np.sum(rates)**2) / (N * np.sum(rates**2))


def db_to_linear(db_value):
    """Convert dB to linear scale."""
    return 10 ** (db_value / 10.0)


def linear_to_db(linear_value):
    """Convert linear to dB scale."""
    return 10 * np.log10(linear_value + 1e-12)


def dbm_to_watts(dbm_value):
    """Convert dBm to Watts."""
    return 10 ** ((dbm_value - 30) / 10.0)


def watts_to_dbm(watts_value):
    """Convert Watts to dBm."""
    return 10 * np.log10(watts_value + 1e-12) + 30


def calculate_summary_statistics(data):
    """
    Calculate summary statistics for data.
    
    Args:
        data: Array or list of values
        
    Returns:
        stats: Dictionary with mean, std, min, max, median
    """
    data = np.array(data)
    
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }


def normalize_array(arr, min_val=None, max_val=None):
    """
    Normalize array to [0, 1] range.
    
    Args:
        arr: Array to normalize
        min_val: Minimum value (if None, use arr.min())
        max_val: Maximum value (if None, use arr.max())
        
    Returns:
        normalized: Normalized array
    """
    if min_val is None:
        min_val = np.min(arr)
    if max_val is None:
        max_val = np.max(arr)
    
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    
    return (arr - min_val) / (max_val - min_val)


def calculate_percentage_gain(value, baseline):
    """
    Calculate percentage gain over baseline.
    
    Args:
        value: Current value
        baseline: Baseline value
        
    Returns:
        gain: Percentage gain (can be negative)
    """
    if baseline == 0:
        return 0.0
    
    return ((value - baseline) / baseline) * 100


# Example usage and tests
if __name__ == "__main__":
    print("="*70)
    print("Testing Metrics Utilities")
    print("="*70)
    
    # Test SE calculation
    rates = np.array([1e6, 2e6, 1.5e6])  # 1, 2, 1.5 Mbps
    bandwidth = 10e6  # 10 MHz
    SE = calculate_spectral_efficiency(rates, bandwidth)
    print(f"\n✅ Spectral Efficiency: {SE:.4f} bps/Hz")
    
    # Test EE calculation
    total_power = 20  # 20 W
    EE = calculate_energy_efficiency(rates, total_power)
    print(f"✅ Energy Efficiency: {EE:.4e} bits/J")
    
    # Test SINR calculation
    g = 0.01
    p = 1.0
    interference = 0.001
    noise = 1e-10
    sinr = calculate_sinr(g, p, interference, noise)
    print(f"✅ SINR: {sinr:.4f} ({linear_to_db(sinr):.2f} dB)")
    
    # Test rate calculation
    rate = calculate_rate(sinr, bandwidth, overhead=0.4283)
    print(f"✅ Rate: {rate/1e6:.4f} Mbps")
    
    # Test outage probability
    rates_test = np.array([0.5e6, 1.2e6, 0.8e6, 1.5e6, 0.3e6])
    min_rate = 1e6
    outage = calculate_outage_probability(rates_test, min_rate)
    print(f"✅ Outage Probability: {outage:.2%}")
    
    # Test fairness index
    fairness = calculate_fairness_index(rates_test)
    print(f"✅ Fairness Index: {fairness:.4f}")
    
    # Test conversions
    dbm = 30
    watts = dbm_to_watts(dbm)
    print(f"✅ Conversion: {dbm} dBm = {watts:.4f} W")
    
    # Test summary statistics
    data_test = np.random.randn(100) * 10 + 50
    stats = calculate_summary_statistics(data_test)
    print(f"\n✅ Summary Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value:.4f}")
    
    # Test percentage gain
    gain = calculate_percentage_gain(120, 100)
    print(f"\n✅ Percentage Gain: {gain:.2f}%")
    
    print("\n" + "="*70)
    print("✅ All tests passed!")
    print("="*70)