"""Tests for the data-layer cleaning logic.

We do NOT hit the network in the unit tests — yfinance is exercised live in
the Streamlit app and via an opt-in integration script. Here we test the pure
DataFrame transforms with synthetic chains.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from options.data import clean_chain, realised_volatility


def _fake_chain(spot_date="2026-05-06"):
    return pd.DataFrame({
        "strike": [90, 95, 100, 105, 110, 100, 100],
        "expiry": [
            "2026-06-19", "2026-06-19", "2026-06-19",
            "2026-06-19", "2026-06-19",
            "2026-04-01",  # already expired vs spot_date — should be dropped
            "2026-06-19",  # duplicate, illiquid (zero bid) — should be dropped
        ],
        "option_type": ["call"] * 7,
        "bid": [10.5, 6.0, 2.5, 0.8, 0.2, 5.0, 0.0],
        "ask": [10.6, 6.1, 2.6, 0.85, 0.22, 5.1, 0.05],
    })


def test_clean_chain_drops_zero_bid_and_expired():
    raw = _fake_chain()
    out = clean_chain(raw, spot_date="2026-05-06")
    # Expect to keep the first 5 rows; drop the expired and the zero-bid ones.
    assert len(out) == 5
    assert (out["T"] > 0).all()
    assert (out["bid"] >= 0.05).all()


def test_clean_chain_preserves_required_columns():
    raw = _fake_chain()
    out = clean_chain(raw, spot_date="2026-05-06")
    for col in ("strike", "T", "market_price", "option_type", "expiry"):
        assert col in out.columns


def test_clean_chain_filters_toxic_spreads():
    df = pd.DataFrame({
        "strike": [100, 105, 110],
        "expiry": ["2026-06-19", "2026-06-19", "2026-06-19"],
        "option_type": ["call"] * 3,
        "bid": [2.5, 0.5, 1.0],
        "ask": [2.6, 1.5, 1.05],   # 105 strike has 100% relative spread
    })
    out = clean_chain(df, spot_date="2026-05-06", max_relative_spread=0.5)
    # 105 strike should be dropped — its spread is too wide.
    assert 105 not in out["strike"].values
    assert 100 in out["strike"].values
    assert 110 in out["strike"].values


def test_clean_chain_computes_T_in_years():
    raw = _fake_chain()
    out = clean_chain(raw, spot_date="2026-05-06")
    # 2026-06-19 - 2026-05-06 = 44 days = 44/365 ≈ 0.1205 years
    assert (out["T"] > 0.10).all()
    assert (out["T"] < 0.15).all()


def test_realised_volatility_basic():
    """Synthetic random walk: realised vol over a long window should be
    roughly sigma sqrt(252) of the per-step log-return std."""
    rng = np.random.default_rng(0)
    daily_vol = 0.01  # 1% daily
    n = 1000
    log_returns = rng.normal(0.0, daily_vol, n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    history = pd.DataFrame({"Close": close, "Adj Close": close})
    rv = realised_volatility(history, window=252)
    expected_annual = daily_vol * np.sqrt(252.0)
    assert rv.dropna().mean() == \
        __import__("pytest").approx(expected_annual, rel=0.10)
