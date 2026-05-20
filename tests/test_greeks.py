"""Cross-validation tests for Greeks across all three pricing methods.

The big idea: any new pricer we add to this repo must agree with the existing
ones on price *and* on first/second-order Greeks. The tests below run all
three engines on the same parameters and assert they agree within a
tolerance that reflects the precision each method can deliver.
"""
from __future__ import annotations

import pytest

from options import greeks
from options.pricing import black_scholes as bs


@pytest.fixture(scope="module")
def cv_european_call():
    """A single European-call cross-validation result, reused across tests."""
    return greeks.cross_validate_european(
        S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0,
        option_type="call", n_paths=300_000, n_steps_tree=1000, seed=11,
    )


def test_three_method_prices_agree_call(cv_european_call):
    a = cv_european_call["analytical"]["price"]
    t = cv_european_call["tree"]["price"]
    m = cv_european_call["mc"]["price"]
    assert t == pytest.approx(a, abs=0.01)
    # MC should be within 3 SE of the analytical truth.
    assert abs(m - a) < 3 * cv_european_call["mc"]["price_se"]


def test_three_method_deltas_agree_call(cv_european_call):
    a = cv_european_call["analytical"]["delta"]
    t = cv_european_call["tree"]["delta"]
    m = cv_european_call["mc"]["delta"]
    assert t == pytest.approx(a, abs=0.005)
    assert abs(m - a) < 3 * cv_european_call["mc"]["delta_se"]


def test_three_method_gammas_agree_call(cv_european_call):
    a = cv_european_call["analytical"]["gamma"]
    t = cv_european_call["tree"]["gamma"]
    m = cv_european_call["mc"]["gamma"]
    assert t == pytest.approx(a, abs=0.0005)
    # CRN gamma is noisy — allow 5 SE.
    assert abs(m - a) < 5 * cv_european_call["mc"]["gamma_se"]


def test_pathwise_vega_agrees_with_analytical(cv_european_call):
    a = cv_european_call["analytical"]["vega"]
    m = cv_european_call["mc"]["vega"]
    assert abs(m - a) < 3 * cv_european_call["mc"]["vega_se"]


def test_cross_validation_works_for_put():
    """Put delta is in [-1, 0]; check three methods agree."""
    out = greeks.cross_validate_european(
        S=100.0, K=110.0, T=0.75, r=0.04, sigma=0.30, q=0.0,
        option_type="put", n_paths=200_000, n_steps_tree=1000, seed=23,
    )
    a = out["analytical"]["delta"]
    t = out["tree"]["delta"]
    m = out["mc"]["delta"]
    assert -1.0 <= a <= 0.0
    assert t == pytest.approx(a, abs=0.005)
    assert abs(m - a) < 3 * out["mc"]["delta_se"]


def test_pathwise_delta_call_matches_bs():
    """Spot-check the pathwise delta in isolation (small param sweep)."""
    for K in (80.0, 100.0, 120.0):
        a = bs.delta(100.0, K, 1.0, 0.05, 0.20, 0.0, "call")
        m = greeks.mc_pathwise_delta(100.0, K, 1.0, 0.05, 0.20, 0.0, "call", n_paths=200_000, seed=3)
        assert abs(m["delta"] - a) < 3 * m["std_error"]


def test_pathwise_vega_call_matches_bs():
    for K in (80.0, 100.0, 120.0):
        a = bs.vega(100.0, K, 1.0, 0.05, 0.20, 0.0)
        m = greeks.mc_pathwise_vega(100.0, K, 1.0, 0.05, 0.20, 0.0, "call", n_paths=200_000, seed=3)
        assert abs(m["vega"] - a) < 3 * m["std_error"]
