"""
Statistical significance testing
"""

from scipy import stats

# Load results for DRL and best baseline
drl_se = results['DRL']['SE']  # Array of SE values
baseline_se = results['Dinkelbach']['SE']

# T-test
t_stat, p_value = stats.ttest_ind(drl_se, baseline_se)

print(f"T-test: t={t_stat:.4f}, p={p_value:.4e}")

if p_value < 0.01:
    print("✅ DRL is SIGNIFICANTLY better (p < 0.01)")
elif p_value < 0.05:
    print("✅ DRL is significantly better (p < 0.05)")
else:
    print("⚠️ Not statistically significant")

# Cohen's d (effect size)
d = (np.mean(drl_se) - np.mean(baseline_se)) / np.std(baseline_se)
print(f"Effect size (Cohen's d): {d:.4f}")