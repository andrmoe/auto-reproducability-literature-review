#!/usr/bin/env python3
"""
analysis.py

Simple computational experiment to test whether
P(gcd(a,b)=1) -> 6/pi^2 by simulation.

Generates samples for several M values, computes the sample
proportion of coprime pairs, a 95% Wilson CI, and a binomial test
against p0 = 6/pi^2. Saves a plot (convergence.png) and prints a table.

The plot y-limits are chosen to not start at zero: they are tight around
the observed CIs with a small padding so convergence is more visible.

Usage:
    python3 analysis.py

Options:
    --Ms   Comma-separated M values (default: 1000,10000,100000,1000000)
    --n    Samples per M (default: 20000)
    --seed RNG seed (default: 12345)
    --out  Output plot filename (default: convergence.png)

Requires: numpy, scipy, matplotlib
"""
from __future__ import annotations

import argparse
import math
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Return (low, high) Wilson score interval for k/n."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (
        z
        * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
        / denom
    )
    return center - half, center + half


def parse_Ms(s: str):
    return [int(x) for x in s.split(',') if x.strip()]


def label_M(M: int):
    """Return a nice tick label: use 10^{k} if M is an exact power of 10."""
    try:
        k = int(round(math.log10(M)))
    except ValueError:
        return str(M)
    if 10 ** k == M:
        return f'$10^{{{k}}}$'
    return str(M)


def main():
    p = argparse.ArgumentParser(
        description='Test coprime probability by simulation.'
    )
    p.add_argument(
        '--Ms',
        default='1000,10000,100000,1000000',
        help='Comma-separated M values.',
    )
    p.add_argument(
        '--n',
        type=int,
        default=20000,
        help='Samples per M.',
    )
    p.add_argument(
        '--seed',
        type=int,
        default=12345,
        help='RNG seed.',
    )
    p.add_argument(
        '--out',
        default='convergence.png',
        help='Output plot filename.',
    )
    args = p.parse_args()

    M_values = parse_Ms(args.Ms)
    rng = np.random.default_rng(args.seed)
    p0 = 6 / (pi ** 2)

    results = []
    for M in M_values:
        a = rng.integers(1, M + 1, size=args.n, dtype=np.int64)
        b = rng.integers(1, M + 1, size=args.n, dtype=np.int64)

        # Elementwise gcd
        if hasattr(np, 'gcd'):
            g = np.gcd(a, b)
        else:
            import math as _math

            gcd_vec = np.vectorize(_math.gcd)
            g = gcd_vec(a, b).astype(np.int64)

        k = int(np.count_nonzero(g == 1))
        phat = k / args.n
        ci_low, ci_high = wilson_ci(k, args.n)

        # binomial test (use binomtest when available)
        try:
            pval = stats.binomtest(k, args.n, p=p0).pvalue
        except Exception:
            pval = stats.binom_test(k, args.n, p=p0)

        results.append((M, args.n, k, phat, ci_low, ci_high, pval))

    # Print table
    hdr = (
        '{:>10s} {:>8s} {:>8s} {:>8s} '
        '{:>10s} {:>10s} {:>10s}'
    ).format('M', 'n', 'k', 'p_hat', 'ci_low', 'ci_high', 'p_value')
    print(hdr)
    print('-' * len(hdr))
    for (M, n, k, phat, cli, chi, pv) in results:
        print(
            '{:10d} {:8d} {:8d} {:8.4f} {:10.4f} {:10.4f} {:10.3e}'
            .format(M, n, k, phat, cli, chi, pv)
        )

    # Plot proportions vs log10(M)
    xs = [math.log10(r[0]) for r in results]
    ys = [r[3] for r in results]
    yerr_low = [r[3] - r[4] for r in results]
    yerr_high = [r[5] - r[3] for r in results]

    # Determine y-limits that don't start at zero:
    ci_lows = [r[4] for r in results]
    ci_highs = [r[5] for r in results]
    y_min = min(ci_lows)
    y_max = max(ci_highs)
    # padding: at least 0.01, or 20% of range
    pad = max(0.01, (y_max - y_min) * 0.2)
    y_low = max(0.0, y_min - pad)
    y_high = min(1.0, y_max + pad)
    # ensure p0 is visible
    if p0 < y_low:
        y_low = max(0.0, p0 - pad)
    if p0 > y_high:
        y_high = min(1.0, p0 + pad)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, ys, yerr=[yerr_low, yerr_high], fmt='o-',
                capsize=4)
    ax.axhline(p0, color='C1', linestyle='--',
               label=f'$6/\\pi^2$ = {p0:.6f}')
    ax.set_xticks(xs)
    ax.set_xticklabels([label_M(M) for M in M_values])
    ax.set_xlabel('M (log$_{10}$)')
    ax.set_ylabel('Proportion coprime')
    ax.set_ylim(y_low, y_high)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f'Saved plot to {args.out}')


if __name__ == '__main__':
    main()