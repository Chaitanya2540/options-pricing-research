"""Tests for the CRR binomial tree pricer."""
from __future__ import annotations

import numpy as np
import pytest

from options.pricing import binomial as bn
from options.pricing import black_scholes as bs


def test_european_call_converges_to_bs():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_price = bs.price(S, K, T, r, sigma, 0.0, "call")
    tree_price = bn.price(S, K, T, r, sigma, 0.0, "call", "european", n_steps=2000)
    assert tree_price == pytest.approx(bs_price, abs=0.01)


def test_european_put_converges_to_bs():
    S, K, T, r, sigma = 100.0, 110.0, 0.75, 0.04, 0.30
    bs_price = bs.price(S, K, T, r, sigma, 0.0, "put")
    tree_price = bn.price(S, K, T, r, sigma, 0.0, "put", "european", n_steps=2000)
    assert tree_price == pytest.approx(bs_price, abs=0.01)


def test_convergence_rate_decreases_with_steps():
    """As n_steps grows, the gap between tree and BS should shrink."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_price = bs.price(S, K, T, r, sigma, 0.0, "call")
    err_50 = abs(bn.price(S, K, T, r, sigma, 0.0, "call", "european", 50) - bs_price)
    err_500 = abs(bn.price(S, K, T, r, sigma, 0.0, "call", "european", 500) - bs_price)
    err_2000 = abs(bn.price(S, K, T, r, sigma, 0.0, "call", "european", 2000) - bs_price)
    assert err_500 < err_50
    assert err_2000 < err_500


def test_american_put_at_least_as_valuable_as_european():
    """The right to early-exercise is non-negative, so American >= European."""
    S, K, T, r, sigma = 100.0, 110.0, 1.0, 0.08, 0.30
    eur = bn.price(S, K, T, r, sigma, 0.0, "put", "european", 1000)
    amer = bn.price(S, K, T, r, sigma, 0.0, "put", "american", 1000)
    assert amer >= eur - 1e-8
    # For deep-ITM put with no dividends and positive r, early-exercise has
    # real value, so we expect a strict premium.
    assert amer - eur > 0.05


def test_american_call_no_dividends_equals_european():
    """Classic Merton result: with no dividends, never optimal to exercise an
    American call early, so the American call price equals the European one.
    """
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    eur = bn.price(S, K, T, r, sigma, 0.0, "call", "european", 1000)
    amer = bn.price(S, K, T, r, sigma, 0.0, "call", "american", 1000)
    assert amer == pytest.approx(eur, abs=1e-3)


def test_put_call_parity_european_tree():
    """Tree-priced European C - P should match S e^{-qT} - K e^{-rT}."""
    S, K, T, r, q, sigma = 100.0, 95.0, 0.5, 0.04, 0.02, 0.25
    n = 1000
    c = bn.price(S, K, T, r, sigma, q, "call", "european", n)
    p = bn.price(S, K, T, r, sigma, q, "put", "european", n)
    parity_lhs = c - p
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert parity_lhs == pytest.approx(parity_rhs, abs=0.01)


def test_tree_delta_matches_bs_delta():
    """The tree-based delta should agree with BS delta for European options."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_delta = bs.delta(S, K, T, r, sigma, 0.0, "call")
    out = bn.price_with_greeks(S, K, T, r, sigma, 0.0, "call", "european", n_steps=1000)
    assert out["delta"] == pytest.approx(bs_delta, abs=0.005)


def test_tree_gamma_matches_bs_gamma():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_gamma = bs.gamma(S, K, T, r, sigma, 0.0)
    out = bn.price_with_greeks(S, K, T, r, sigma, 0.0, "call", "european", n_steps=1000)
    assert out["gamma"] == pytest.approx(bs_gamma, abs=0.005)


def test_intrinsic_at_expiry():
    assert bn.price(110, 100, 0.0, 0.05, 0.2, 0.0, "call") == 10.0
    assert bn.price(90, 100, 0.0, 0.05, 0.2, 0.0, "put") == 10.0


def test_invalid_n_steps_raises():
    with pytest.raises(ValueError):
        bn.price(100, 100, 1.0, 0.05, 0.2, 0.0, "call", "european", n_steps=0)
