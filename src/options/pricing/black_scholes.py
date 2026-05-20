"""Closed-form Black-Scholes-Merton pricing and Greeks for European options.

The pricing formula assumes:
- Geometric Brownian motion for the underlying under the risk-neutral measure:
    dS = (r - q) S dt + sigma S dW
- Constant risk-free rate r, constant continuous dividend yield q, constant
  volatility sigma over the life of the option.
- No transaction costs, infinite divisibility, frictionless trading.
- European-style exercise (cannot be exercised early).

The implementation is fully vectorised over (S, K, T, sigma): you can price
an entire chain by passing arrays of strikes. The risk-free rate and dividend
yield are kept scalar because that's how they show up in practice (one yield
curve, one borrow assumption per pricing run).

References:
- Hull, "Options, Futures, and Other Derivatives", 10e, Chapter 15.
- Wilmott, "Paul Wilmott on Quantitative Finance", 2e, Chapter 8.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..utils import std_normal_cdf, std_normal_pdf, validate_inputs

OptionType = Literal["call", "put"]


def _d1_d2(S, K, T, r, sigma, q=0.0):
    """Compute the Black-Scholes auxiliary variables d1 and d2.

    These show up in every formula in this module. Splitting them out keeps
    the price/Greeks code short and matches the textbook notation.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma_sqrt_T = sigma * np.sqrt(T)
    # Guard against sigma_sqrt_T == 0 — at T=0 the option is at expiry and
    # has only intrinsic value; we handle that case in price() directly.
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


def price(
    S, K, T, r, sigma, q: float = 0.0, option_type: OptionType = "call"
) -> np.ndarray:
    """Black-Scholes price for a European call or put.

    At T = 0 returns intrinsic value. At sigma = 0 returns the discounted
    intrinsic value, which is the limit of the BS formula as sigma -> 0+.
    """
    validate_inputs(S, K, T, r, sigma, q)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)

    # Edge case: at expiry, payoff is purely intrinsic.
    if T == 0:
        if option_type == "call":
            return np.maximum(S - K, 0.0)
        return np.maximum(K - S, 0.0)

    # Edge case: zero volatility -> deterministic forward, discounted intrinsic.
    sigma_arr = np.asarray(sigma, dtype=float)
    if np.all(sigma_arr == 0):
        forward = S * np.exp(-q * T)
        kd = K * np.exp(-r * T)
        if option_type == "call":
            return np.maximum(forward - kd, 0.0)
        return np.maximum(kd - forward, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * np.exp(-q * T) * std_normal_cdf(d1) - K * np.exp(-r * T) * std_normal_cdf(d2)
    if option_type == "put":
        return K * np.exp(-r * T) * std_normal_cdf(-d2) - S * np.exp(-q * T) * std_normal_cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def delta(S, K, T, r, sigma, q: float = 0.0, option_type: OptionType = "call") -> np.ndarray:
    """Sensitivity of price to spot. Range: [0, 1] for calls, [-1, 0] for puts.

    Practical interpretation: a delta of 0.6 means a $1 move in the underlying
    moves the option price by $0.60, *to first order*. Market makers hedge
    delta to stay first-order neutral, then are exposed to the higher-order
    terms (gamma, vega, theta).
    """
    validate_inputs(S, K, T, r, sigma, q)
    if T == 0:
        # Heaviside step at expiry; tie at the strike resolves to 0.5 by convention.
        if option_type == "call":
            return np.where(S > K, 1.0, np.where(S < K, 0.0, 0.5))
        return np.where(S < K, -1.0, np.where(S > K, 0.0, -0.5))
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    sign = 1.0 if option_type == "call" else -1.0
    return sign * np.exp(-q * T) * std_normal_cdf(sign * d1)


def gamma(S, K, T, r, sigma, q: float = 0.0) -> np.ndarray:
    """Second derivative of price w.r.t. spot. Identical for calls and puts.

    Gamma is highest for ATM options near expiry — that's where small spot
    moves cause the largest changes in delta. A short-gamma position is a
    short-convexity position: you collect theta when the market is quiet
    and pay when it moves a lot.
    """
    validate_inputs(S, K, T, r, sigma, q)
    if T == 0 or np.all(np.asarray(sigma, dtype=float) == 0):
        # Limit is a delta function at the strike; not useful numerically. Return 0
        # off-the-strike so downstream code doesn't break.
        return np.zeros_like(np.asarray(S, dtype=float))
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * std_normal_pdf(d1) / (np.asarray(S, dtype=float) * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q: float = 0.0) -> np.ndarray:
    """Sensitivity to volatility. Reported per unit sigma (NOT per 1% move).

    To get 'vega per 1 vol point', divide by 100. Identical for calls and puts.
    """
    validate_inputs(S, K, T, r, sigma, q)
    if T == 0:
        return np.zeros_like(np.asarray(S, dtype=float))
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    return np.asarray(S, dtype=float) * np.exp(-q * T) * std_normal_pdf(d1) * np.sqrt(T)


def theta(
    S, K, T, r, sigma, q: float = 0.0, option_type: OptionType = "call"
) -> np.ndarray:
    """Time decay (per year). Most pricers and trading systems quote theta
    per calendar day — divide by 365 for that. We keep the 'per year' unit
    here so all Greeks are in the same time scale; the streamlit / report
    layer handles the daily conversion.
    """
    validate_inputs(S, K, T, r, sigma, q)
    if T == 0:
        return np.zeros_like(np.asarray(S, dtype=float))
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S_arr = np.asarray(S, dtype=float)
    K_arr = np.asarray(K, dtype=float)
    common = -(S_arr * np.exp(-q * T) * std_normal_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    if option_type == "call":
        return (
            common
            - r * K_arr * np.exp(-r * T) * std_normal_cdf(d2)
            + q * S_arr * np.exp(-q * T) * std_normal_cdf(d1)
        )
    if option_type == "put":
        return (
            common
            + r * K_arr * np.exp(-r * T) * std_normal_cdf(-d2)
            - q * S_arr * np.exp(-q * T) * std_normal_cdf(-d1)
        )
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def rho(
    S, K, T, r, sigma, q: float = 0.0, option_type: OptionType = "call"
) -> np.ndarray:
    """Sensitivity to risk-free rate. Reported per unit r (NOT per bp)."""
    validate_inputs(S, K, T, r, sigma, q)
    if T == 0:
        return np.zeros_like(np.asarray(S, dtype=float))
    _, d2 = _d1_d2(S, K, T, r, sigma, q)
    K_arr = np.asarray(K, dtype=float)
    if option_type == "call":
        return K_arr * T * np.exp(-r * T) * std_normal_cdf(d2)
    if option_type == "put":
        return -K_arr * T * np.exp(-r * T) * std_normal_cdf(-d2)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def greeks(
    S, K, T, r, sigma, q: float = 0.0, option_type: OptionType = "call"
) -> dict:
    """Convenience wrapper returning all Greeks in a dict.

    Used by the Streamlit calculator and by the cross-validation module.
    """
    return {
        "price": price(S, K, T, r, sigma, q, option_type),
        "delta": delta(S, K, T, r, sigma, q, option_type),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, q, option_type),
        "rho": rho(S, K, T, r, sigma, q, option_type),
    }
