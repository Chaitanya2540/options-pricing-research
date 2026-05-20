"""Tests for the delta-hedging simulator.

Strategy:
- When sigma_realised == sigma_implied, the expected P&L over many paths
  should be near zero (within hedging error / sample noise).
- When sigma_realised < sigma_implied, short straddles should make money
  on average (short gamma wins when realised vol comes in below implied).
- When sigma_realised > sigma_implied, the short straddle should lose money.
- Theta + gamma attribution sums should approximate the realised hedge P&L
  per path within discretisation error.
"""
from __future__ import annotations

import numpy as np
import pytest

from options.hedging import (
    expected_pnl_textbook,
    simulate_distribution,
    simulate_short_straddle_path,
)


_N_STEPS = 30   # daily-ish rebalance over a 30-day option
_N_PATHS = 200  # enough for 3-sigma asserts; keeps tests under a few seconds


def test_zero_vol_gap_pnl_is_near_zero_on_average():
    """If implied == realised, expected P&L is ~0 by construction."""
    df = simulate_distribution(
        S0=100.0, K=100.0, T=30/365, r=0.04, q=0.0,
        sigma_implied=0.20, sigma_realised=0.20,
        n_steps=_N_STEPS, n_paths=_N_PATHS, seed=11,
    )
    se = df["pnl"].std(ddof=1) / np.sqrt(len(df))
    assert abs(df["pnl"].mean()) < 3 * se


def test_short_gamma_wins_when_realised_below_implied():
    df = simulate_distribution(
        S0=100.0, K=100.0, T=30/365, r=0.04, q=0.0,
        sigma_implied=0.30, sigma_realised=0.15,
        n_steps=_N_STEPS, n_paths=_N_PATHS, seed=23,
    )
    se = df["pnl"].std(ddof=1) / np.sqrt(len(df))
    assert df["pnl"].mean() > 3 * se


def test_short_gamma_loses_when_realised_above_implied():
    df = simulate_distribution(
        S0=100.0, K=100.0, T=30/365, r=0.04, q=0.0,
        sigma_implied=0.15, sigma_realised=0.30,
        n_steps=_N_STEPS, n_paths=_N_PATHS, seed=37,
    )
    se = df["pnl"].std(ddof=1) / np.sqrt(len(df))
    assert df["pnl"].mean() < -3 * se


def test_attribution_tracks_realised_pnl_in_direction_and_magnitude():
    """Per path, theta + gamma attribution should match the realised hedge
    P&L in direction and order of magnitude.

    Exact equality holds only in continuous time (Itô). With daily rebalancing
    over a 30-day option there's residual discretisation error from
    higher-order Itô-Taylor terms — typically a few dollars on a $100
    underlying. The attribution remains the right narrative tool for
    decomposing P&L into theta-collected vs gamma-paid.
    """
    out = simulate_short_straddle_path(
        S0=100.0, K=100.0, T=30/365, r=0.04, q=0.0,
        sigma_implied=0.20, sigma_realised=0.20,
        n_steps=_N_STEPS, seed=99,
    )
    attribution = out.theta_attribution + out.gamma_attribution
    # Same sign or both very small.
    if abs(out.hedge_pnl) > 0.5:
        assert np.sign(attribution) == np.sign(out.hedge_pnl)
    # Residual modest relative to underlying scale (5% of $100 spot).
    assert abs(out.residual) < 5.0


def test_textbook_formula_directionally_matches_simulation():
    """The Vega * (sigma_impl - sigma_real) closed form should agree in
    direction with the simulator and be in the right order of magnitude."""
    df = simulate_distribution(
        S0=100.0, K=100.0, T=30/365, r=0.04, q=0.0,
        sigma_implied=0.30, sigma_realised=0.15,
        n_steps=_N_STEPS, n_paths=_N_PATHS, seed=5,
    )
    expected = expected_pnl_textbook(
        S0=100.0, K=100.0, T=30/365, r=0.04, q=0.0,
        sigma_implied=0.30, sigma_realised=0.15,
    )
    sim_mean = df["pnl"].mean()
    assert expected > 0
    assert sim_mean > 0
    # Order of magnitude check (broad — discretisation + sample noise).
    assert 0.3 * expected < sim_mean < 2.5 * expected
