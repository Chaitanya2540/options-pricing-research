"""Shared pytest fixtures.

A small, well-known reference parameter set is reused across pricers so that
convergence tests can compare any new method against the canonical
Black-Scholes value.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def reference_european_call():
    """Hull, 'Options, Futures, and Other Derivatives', canonical worked example.

    A 6-month European call on a non-dividend-paying stock with:
      S = 42, K = 40, T = 0.5, r = 0.10, sigma = 0.20
    has BS price ~ 4.7594 (Hull's worked example, reproducible from the formula
    to four decimal places).
    """
    return {
        "S": 42.0,
        "K": 40.0,
        "T": 0.5,
        "r": 0.10,
        "q": 0.0,
        "sigma": 0.20,
        "option_type": "call",
        "expected_price": 4.7594,
    }


@pytest.fixture(scope="session")
def reference_european_put():
    """Same parameters as the call fixture, paired by put-call parity:
        P = C - S + K * exp(-rT) = 4.7594 - 42 + 40 * exp(-0.05) = 0.8086.
    """
    return {
        "S": 42.0,
        "K": 40.0,
        "T": 0.5,
        "r": 0.10,
        "q": 0.0,
        "sigma": 0.20,
        "option_type": "put",
        "expected_price": 0.8086,
    }
