"""Generate the canonical analyses used by the README and the docs/ folder.

Outputs (under results/):
- convergence_binomial.png — log-log error plot for binomial vs BS
- convergence_mc.png — log-log error plot for MC vs BS
- hedge_sim_distribution.png — gamma-scalp P&L histogram across regimes
- summary_numbers.json — the headline numbers we cite in the README
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from options.hedging import simulate_distribution
from options.pricing import binomial as bn
from options.pricing import black_scholes as bs
from options.pricing import monte_carlo as mc

RESULTS = _REPO_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def convergence_studies():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    truth = float(bs.price(S, K, T, r, sigma, q, "call"))

    steps = [10, 25, 50, 100, 250, 500, 1000, 2000, 5000]
    tree_errs = [
        abs(bn.price(S, K, T, r, sigma, q, "call", "european", n) - truth) for n in steps
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(steps, tree_errs, marker="o", label="|binomial - BS|")
    # Reference line for O(1/N).
    ref = [tree_errs[0] * (steps[0] / n) for n in steps]
    ax.loglog(steps, ref, "--", color="grey", label="reference: 1/N")
    ax.set_xlabel("Number of steps N")
    ax.set_ylabel("Absolute error vs BS")
    ax.set_title("CRR binomial → BS convergence (S=K=100, T=1y, r=5%, σ=20%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "convergence_binomial.png", dpi=150)
    plt.close(fig)

    paths = [1_000, 5_000, 25_000, 100_000, 400_000, 1_000_000]
    mc_errs = []
    mc_ses = []
    for n in paths:
        out = mc.price_european(S, K, T, r, sigma, q, "call", n_paths=n, seed=42)
        mc_errs.append(abs(out["price"] - truth))
        mc_ses.append(out["std_error"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(paths, mc_errs, marker="o", label="|MC - BS|")
    ax.loglog(paths, mc_ses, marker="x", linestyle=":", label="MC standard error")
    ref = [mc_errs[0] * np.sqrt(paths[0] / n) for n in paths]
    ax.loglog(paths, ref, "--", color="grey", label="reference: 1/√N")
    ax.set_xlabel("Number of paths N")
    ax.set_ylabel("Absolute error vs BS")
    ax.set_title("Monte Carlo → BS convergence (antithetic, S=K=100, T=1y, σ=20%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "convergence_mc.png", dpi=150)
    plt.close(fig)

    return {
        "binomial": {"steps": steps, "abs_errors": [float(e) for e in tree_errs]},
        "monte_carlo": {"paths": paths, "abs_errors": [float(e) for e in mc_errs], "std_errors": [float(e) for e in mc_ses]},
        "bs_truth": truth,
    }


def hedge_sim_distribution():
    """Run the gamma-scalp sim across three regimes and plot all three on one chart."""
    base = dict(S0=100.0, K=100.0, T=30/365, r=0.04, q=0.0, n_steps=30, n_paths=400, seed=11)
    df_neutral = simulate_distribution(sigma_implied=0.20, sigma_realised=0.20, **base)
    df_win = simulate_distribution(sigma_implied=0.30, sigma_realised=0.15, **base)
    df_loss = simulate_distribution(sigma_implied=0.15, sigma_realised=0.30, **base)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_neutral["pnl"], bins=30, alpha=0.5, label="σ_imp=σ_real=20%")
    ax.hist(df_win["pnl"], bins=30, alpha=0.5, label="σ_imp=30%, σ_real=15% (short γ wins)")
    ax.hist(df_loss["pnl"], bins=30, alpha=0.5, label="σ_imp=15%, σ_real=30% (short γ loses)")
    ax.axvline(0, color="black", linestyle=":")
    ax.set_xlabel("P&L per straddle ($)")
    ax.set_ylabel("Number of paths")
    ax.set_title("Short ATM straddle, daily-rebalanced, 30-day option")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "hedge_sim_distribution.png", dpi=150)
    plt.close(fig)

    return {
        "regimes": {
            "neutral_20_20": {"mean": float(df_neutral["pnl"].mean()), "std": float(df_neutral["pnl"].std()), "win_rate": float((df_neutral["pnl"] > 0).mean())},
            "rich_30_15":   {"mean": float(df_win["pnl"].mean()),     "std": float(df_win["pnl"].std()),     "win_rate": float((df_win["pnl"] > 0).mean())},
            "cheap_15_30":  {"mean": float(df_loss["pnl"].mean()),    "std": float(df_loss["pnl"].std()),    "win_rate": float((df_loss["pnl"] > 0).mean())},
        },
    }


def main():
    print("Running convergence studies…")
    conv = convergence_studies()
    print("Running hedge-sim regimes…")
    hedge = hedge_sim_distribution()
    summary = {"convergence": conv, "hedging": hedge}
    (RESULTS / "summary_numbers.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {RESULTS}/summary_numbers.json plus 3 PNGs.")


if __name__ == "__main__":
    main()
