"""Tests for the implied-volatility solver and chain helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from options import iv as iv_mod
from options.pricing import black_scholes as bs


def test_iv_roundtrips_through_bs():
    """If we generate a price from BS at sigma=0.25 and then invert, we
    should recover sigma to floating-point precision."""
    sigma_true = 0.25
    S, K, T, r, q = 100.0, 100.0, 0.5, 0.04, 0.0
    p = bs.price(S, K, T, r, sigma_true, q, "call")
    sigma_recovered = iv_mod.implied_vol(p, S, K, T, r, q, "call")
    assert sigma_recovered == pytest.approx(sigma_true, abs=1e-6)


def test_iv_recovers_under_dividends():
    sigma_true = 0.30
    S, K, T, r, q = 100.0, 110.0, 0.75, 0.05, 0.02
    for opt in ("call", "put"):
        p = bs.price(S, K, T, r, sigma_true, q, opt)
        recovered = iv_mod.implied_vol(p, S, K, T, r, q, opt)
        assert recovered == pytest.approx(sigma_true, abs=1e-6)


def test_iv_returns_nan_for_below_intrinsic():
    """A price below intrinsic violates no-arbitrage. The solver should
    return NaN rather than crash."""
    S, K, T, r = 100.0, 90.0, 0.5, 0.05
    intrinsic = S - K * np.exp(-r * T)
    bad_price = intrinsic * 0.5  # too low
    result = iv_mod.implied_vol(bad_price, S, K, T, r, 0.0, "call")
    assert np.isnan(result)


def test_iv_chain_handles_mixed_calls_puts():
    S, r, q = 100.0, 0.04, 0.0
    rows = []
    for K in (90.0, 100.0, 110.0):
        for T in (0.25, 0.5):
            for opt in ("call", "put"):
                sigma = 0.25
                p = bs.price(S, K, T, r, sigma, q, opt)
                rows.append({
                    "strike": K, "T": T,
                    "market_price": float(p), "option_type": opt,
                })
    chain = pd.DataFrame(rows)
    out = iv_mod.implied_vol_chain(chain, S, r, q)
    assert "iv" in out.columns
    assert np.allclose(out["iv"], 0.25, atol=1e-6)


def test_smile_at_expiry_returns_sorted_strikes():
    chain = pd.DataFrame({
        "strike": [110.0, 90.0, 100.0],
        "T": [0.5, 0.5, 0.5],
        "market_price": [5.0, 12.0, 8.0],
        "option_type": ["call"] * 3,
        "iv": [0.22, 0.30, 0.25],
    })
    out = iv_mod.smile_at_expiry(chain, 0.5)
    assert list(out["strike"]) == [90.0, 100.0, 110.0]


def test_term_structure_atm_picks_closest_strike():
    chain = pd.DataFrame({
        "strike": [90, 100, 110, 90, 100, 110],
        "T": [0.25, 0.25, 0.25, 0.5, 0.5, 0.5],
        "market_price": [12, 8, 5, 14, 10, 7],
        "option_type": ["call"] * 6,
        "iv": [0.30, 0.25, 0.22, 0.32, 0.27, 0.24],
    })
    out = iv_mod.term_structure_atm(chain, 100.0)
    assert list(out["T"]) == [0.25, 0.5]
    assert list(out["strike"]) == [100.0, 100.0]
    assert list(out["iv"]) == [0.25, 0.27]
