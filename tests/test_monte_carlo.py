"""Tests for the Monte Carlo pricer.

MC tests need to be tolerant: by definition the price has random error. We
verify that the BS analytical value lies within a 99% confidence interval
of the MC estimate, which gives a tiny false-positive rate per test.
"""
from __future__ import annotations

import numpy as np
import pytest

from options.pricing import black_scholes as bs
from options.pricing import monte_carlo as mc


def test_european_call_within_ci_of_bs():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_price = bs.price(S, K, T, r, sigma, 0.0, "call")
    out = mc.price_european(S, K, T, r, sigma, 0.0, "call", n_paths=200_000, seed=7)
    # BS price should be within ~3 standard errors of the MC estimate (99.7%).
    assert abs(out["price"] - bs_price) < 3 * out["std_error"]


def test_european_put_within_ci_of_bs():
    S, K, T, r, sigma = 100.0, 110.0, 0.5, 0.04, 0.30
    bs_price = bs.price(S, K, T, r, sigma, 0.0, "put")
    out = mc.price_european(S, K, T, r, sigma, 0.0, "put", n_paths=200_000, seed=11)
    assert abs(out["price"] - bs_price) < 3 * out["std_error"]


def test_antithetic_reduces_se_vs_no_antithetic():
    """Antithetic variates should produce lower SE than independent draws at
    the same path count."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    se_with = mc.price_european(S, K, T, r, sigma, 0.0, "call", n_paths=50_000, antithetic=True, seed=42)["std_error"]
    se_without = mc.price_european(S, K, T, r, sigma, 0.0, "call", n_paths=50_000, antithetic=False, seed=42)["std_error"]
    # Antithetic typically cuts SE by 30-50% for at-the-money calls.
    assert se_with < se_without * 0.9


def test_geometric_asian_closed_form_sanity():
    """Geometric Asian is bounded above by the European call (geometric
    average is bounded above by the terminal in the limit) and bounded
    below by zero."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    eur = bs.price(S, K, T, r, sigma, 0.0, "call")
    geo = mc.geometric_asian_closed_form(S, K, T, r, sigma, 0.0, "call")
    assert 0.0 <= geo <= eur


def test_arithmetic_asian_within_ci_of_geometric_closed_form():
    """Arithmetic Asian price should be within a few SE of, and very close to,
    the geometric Asian closed-form (they are highly correlated)."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    geo = mc.geometric_asian_closed_form(S, K, T, r, sigma, 0.0, "call")
    out = mc.price_arithmetic_asian(
        S, K, T, r, sigma, 0.0, "call",
        n_paths=20_000, n_steps=252, control_variates=True, seed=23,
    )
    # Arithmetic >= geometric by Jensen's inequality; the gap is small.
    assert out["price"] >= geo - 5 * out["std_error"]
    # Arithmetic shouldn't be drastically larger than geometric either.
    assert out["price"] <= geo + 0.5


def test_control_variates_reduce_variance():
    """The control-variates SE should be much smaller than naive SE for the
    arithmetic Asian, given the high correlation with the geometric average."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    out = mc.price_arithmetic_asian(
        S, K, T, r, sigma, 0.0, "call",
        n_paths=20_000, n_steps=100, control_variates=True, seed=5,
    )
    assert out["method"] == "control_variates"
    # Variance reduction ratio should be at least 5x in this regime.
    assert out["variance_reduction_ratio"] >= 5.0


def test_se_decreases_with_more_paths():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    out_small = mc.price_european(S, K, T, r, sigma, 0.0, "call", n_paths=10_000, seed=3)
    out_large = mc.price_european(S, K, T, r, sigma, 0.0, "call", n_paths=160_000, seed=3)
    # SE shrinks as 1/sqrt(N); 16x paths => ~4x smaller SE.
    assert out_large["std_error"] < out_small["std_error"] / 3.0


def test_seed_reproducibility():
    """Same seed -> identical results."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    a = mc.price_european(S, K, T, r, sigma, 0.0, "call", n_paths=10_000, seed=99)
    b = mc.price_european(S, K, T, r, sigma, 0.0, "call", n_paths=10_000, seed=99)
    assert a["price"] == b["price"]
    assert a["std_error"] == b["std_error"]
