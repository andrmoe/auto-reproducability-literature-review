#!/usr/bin/env python3
"""
analysis.py
Simple analysis: read data.csv (sepal_length, petal_length),
compute Pearson correlation and plot scatter + regression line.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

DATA_PATH = Path("data.csv")
if not DATA_PATH.exists():
    print("Error: data.csv not found in current directory.", file=sys.stderr)
    sys.exit(1)

# Read CSV
df = pd.read_csv(DATA_PATH)

# Basic checks
if not {"sepal_length", "petal_length"}.issubset(df.columns):
    print("CSV must contain 'sepal_length' and 'petal_length' columns.",
          file=sys.stderr)
    sys.exit(1)

x = df["sepal_length"].astype(float)
y = df["petal_length"].astype(float)

# Pearson correlation
r, p_value = stats.pearsonr(x, y)

# Fisher z-based approximate 95% CI for r
def pearson_confidence_interval(r_val, n, alpha=0.05):
    if n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(r_val)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = z - z_crit * se, z + z_crit * se
    return (np.tanh(lo_z), np.tanh(hi_z))

ci_low, ci_high = pearson_confidence_interval(r, len(df))

# Print results
print(f"n = {len(df)}")
print(f"Pearson r = {r:.6f}")
print(f"p-value = {p_value:.3e}")
print(f"95% CI (approx) = [{ci_low:.6f}, {ci_high:.6f}]")

# Plot scatter with regression line
plt.figure(figsize=(6, 4.5))
sns.regplot(x=x, y=y, ci=95, scatter_kws={"s": 30})
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
plt.title("Sepal length vs Petal length (Iris subset)")
plt.tight_layout()
out_plot = "scatter_regression.png"
plt.savefig(out_plot, dpi=150)
print(f"Scatter plot saved to {out_plot}")