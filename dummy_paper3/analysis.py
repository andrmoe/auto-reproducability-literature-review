#!/usr/bin/env python3
"""
analysis.py

Combined analysis of existing Iris data and a computational experiment.

- Reads data.csv (must contain columns: sepal_length, petal_length, species)
- Computes observed Pearson correlation between sepal_length and petal_length
- Runs a permutation-based null model: for nsim repetitions, permute petal
  lengths within each species (preserving marginals and counts) and compute
  the overall Pearson correlation. This removes within-species association
  while keeping between-species structure.
- Prints summary and saves sim_plot.png (histogram of simulated correlations
  with observed correlation marked).

Usage:
    python3 analysis.py [--nsim N] [--seed S] [--out FILE]

Dependencies: numpy, pandas, scipy, matplotlib
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

DEFAULT_NSIM = 5000
DEFAULT_SEED = 12345
DATA_FILE = "data.csv"
OUT_PLOT = "simplot.png"


def run_permutation_test(sepal, petal, species, nsim=5000, seed=12345):
    rng = np.random.default_rng(seed)
    unique_species = np.unique(species)
    # Precompute indices for each species
    indices = {s: np.where(species == s)[0] for s in unique_species}
    r_sims = np.empty(nsim, dtype=float)

    for i in range(nsim):
        permuted_petal = np.empty_like(petal)
        for s, idx in indices.items():
            vals = petal[idx]
            permuted = vals[rng.permutation(len(vals))]
            permuted_petal[idx] = permuted
        r_sims[i], _ = pearsonr(sepal, permuted_petal)
    return r_sims


def main():
    p = argparse.ArgumentParser(
        description="Permutation test: are between-species differences sufficient?"
    )
    p.add_argument("--nsim", type=int, default=DEFAULT_NSIM,
                   help=f"Number of permutations (default: {DEFAULT_NSIM})")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"RNG seed (default: {DEFAULT_SEED})")
    p.add_argument("--out", default=OUT_PLOT,
                   help=f"Output plot filename (default: {OUT_PLOT})")
    args = p.parse_args()

    # Read data
    df = pd.read_csv(DATA_FILE)
    for col in ("sepal_length", "petal_length", "species"):
        if col not in df.columns:
            raise SystemExit(f"data.csv must contain column '{col}'")

    sepal = df["sepal_length"].to_numpy(dtype=float)
    petal = df["petal_length"].to_numpy(dtype=float)
    species = df["species"].to_numpy()

    # Observed correlation
    r_obs, p_obs = pearsonr(sepal, petal)

    # Permutation experiment
    r_sims = run_permutation_test(sepal, petal, species,
                                  nsim=args.nsim, seed=args.seed)

    # Empirical one-sided p-value: proportion of simulated correlations >= r_obs
    p_empirical = (np.sum(r_sims >= r_obs) + 1) / (len(r_sims) + 1)

    # Print summary
    print("Summary")
    print("-------")
    print(f"N = {len(df)}")
    print(f"Observed Pearson r = {r_obs:.6f} (scipy two-sided p = {p_obs:.3e})")
    print(f"Simulated null: mean r = {r_sims.mean():.6f}, sd = {r_sims.std(ddof=0):.6f}")
    print(f"Empirical (one-sided) p_perm = {p_empirical:.6f} "
          f"({np.sum(r_sims >= r_obs)} / {len(r_sims)})")
    print()
    print("Species summary (count, mean sepal, mean petal):")
    print(df.groupby("species").agg(
        count=("sepal_length", "size"),
        mean_sepal=("sepal_length", "mean"),
        mean_petal=("petal_length", "mean")
    ))

    # Plot histogram of simulated correlations
    plt.figure(figsize=(6, 4))
    plt.hist(r_sims, bins=50, color="#ccccff", edgecolor="k", alpha=0.9)
    plt.axvline(r_obs, color="C1", linestyle="--",
                label=f"Observed r = {r_obs:.3f}")
    plt.xlabel("Pearson correlation (simulated under null)")
    plt.ylabel("Frequency")
    plt.title("Permutation null: correlations with within-species association removed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved histogram to {args.out}")


if __name__ == "__main__":
    main()