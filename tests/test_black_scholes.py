"""Tests for the closed-form Black-Scholes pricer.

Strategy:
- Pin against textbook values (Hull) so a refactor that breaks the formula
  fails loudly.
- Verify put-call parity, the most fundamental cross-check on European pricers.
- Verify Greek consistency: numerical bump-and-revalue must agree with the
  analytical formulas on the same parameters.
- Spot-check edge cases: T=0 returns intrinsic, sigma=0 returns discounted
  intrinsic.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from options.pricing import black_scholes as bs


def test_european_call_matches_hull(reference_european_call):
    p = bs.price(**{k: v for k, v in reference_european_call.items() if k != "expected_price"})
    assert p == pytest.approx(reference_european_call["expected_price"], abs=1e-3)


def test_european_put_matches_parity(reference_european_put):
    p = bs.price(**{k: v for k, v in reference_european_put.items() if k != "expected_price"})
    assert p == pytest.approx(reference_european_put["expected_price"], abs=1e-3)


def test_put_call_parity():
    """C - P = S e^{-qT} - K e^{-rT}."""
    S, K, T, r, q, sigma = 100.0, 95.0, 0.5, 0.04, 0.02, 0.25
    c = bs.price(S, K, T, r, sigma, q, "call")
    p = bs.price(S, K, T, r, sigma, q, "put")
    parity_lhs = c - p
    parity_rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert parity_lhs == pytest.approx(parity_rhs, abs=1e-10)


def test_intrinsic_at_expiry_call():
    assert bs.price(110, 100, 0.0, 0.05, 0.2, 0.0, "call") == 10.0
    assert bs.price(90, 100, 0.0, 0.05, 0.2, 0.0, "call") == 0.0


def test_intrinsic_at_expiry_put():
    assert bs.price(90, 100, 0.0, 0.05, 0.2, 0.0, "put") == 10.0
    assert bs.price(110, 100, 0.0, 0.05, 0.2, 0.0, "put") == 0.0


def test_zero_vol_collapses_to_discounted_forward():
    S, K, T, r, q = 100.0, 95.0, 0.5, 0.04, 0.0
    c = bs.price(S, K, T, r, sigma=0.0, q=q, option_type="call")
    expected = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    assert c == pytest.approx(expected, abs=1e-12)


def test_call_delta_in_range():
    deltas = bs.delta(np.linspace(50, 200, 50), 100.0, 0.5, 0.04, 0.25, 0.0, "call")
    assert np.all((deltas >= 0) & (deltas <= 1))


def test_put_delta_in_range():
    deltas = bs.delta(np.linspace(50, 200, 50), 100.0, 0.5, 0.04, 0.25, 0.0, "put")
    assert np.all((deltas >= -1) & (deltas <= 0))


def test_gamma_identical_for_call_and_put():
    """Call-put parity differentiates twice in S to gamma_C - gamma_P = 0."""
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.04, 0.25
    g = bs.gamma(S, K, T, r, sigma)
    # Confirm finite-difference call and put gammas match the analytical value.
    h = 0.01
    g_call_fd = (
        bs.price(S + h, K, T, r, sigma, 0.0, "call")
        - 2 * bs.price(S, K, T, r, sigma, 0.0, "call")
        + bs.price(S - h, K, T, r, sigma, 0.0, "call")
    ) / h**2
    g_put_fd = (
        bs.price(S + h, K, T, r, sigma, 0.0, "put")
        - 2 * bs.price(S, K, T, r, sigma, 0.0, "put")
        + bs.price(S - h, K, T, r, sigma, 0.0, "put")
    ) / h**2
    assert g == pytest.approx(g_call_fd, abs=1e-3)
    assert g == pytest.approx(g_put_fd, abs=1e-3)


def test_delta_finite_difference_agrees_with_analytical():
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.04, 0.25
    h = 0.01
    for option in ("call", "put"):
        analytical = bs.delta(S, K, T, r, sigma, 0.0, option)
        fd = (
            bs.price(S + h, K, T, r, sigma, 0.0, option)
            - bs.price(S - h, K, T, r, sigma, 0.0, option)
        ) / (2 * h)
        assert analytical == pytest.approx(fd, abs=1e-4)


def test_vega_finite_difference_agrees_with_analytical():
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.04, 0.25
    h = 1e-4
    analytical = bs.vega(S, K, T, r, sigma, 0.0)
    fd = (
        bs.price(S, K, T, r, sigma + h, 0.0, "call")
        - bs.price(S, K, T, r, sigma - h, 0.0, "call")
    ) / (2 * h)
    assert analytical == pytest.approx(fd, abs=1e-3)


def test_vectorised_pricing_over_strikes():
    S, T, r, sigma = 100.0, 0.5, 0.04, 0.25
    strikes = np.array([80, 90, 100, 110, 120], dtype=float)
    prices = bs.price(S, strikes, T, r, sigma, 0.0, "call")
    assert prices.shape == strikes.shape
    # Calls are monotone decreasing in strike, holding everything else fixed.
    assert np.all(np.diff(prices) < 0)


def test_input_validation():
    with pytest.raises(ValueError, match="Spot"):
        bs.price(-100, 100, 0.5, 0.04, 0.25)
    with pytest.raises(ValueError, match="Strike"):
        bs.price(100, -100, 0.5, 0.04, 0.25)
    with pytest.raises(ValueError, match="Volatility"):
        bs.price(100, 100, 0.5, 0.04, -0.25)
    with pytest.raises(ValueError, match="expiry"):
        bs.price(100, 100, -0.5, 0.04, 0.25)
