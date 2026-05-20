"""Implied-volatility solver and IV-surface fitting.

Given a market option price, we invert the Black-Scholes formula for sigma.
The function p(sigma) is monotone increasing in sigma (vega > 0), so any
bracketed root finder works. We use Brent's method (scipy.optimize.brentq):
super-linear convergence, no derivatives needed, robust on bracketed monotonic
roots.

The surface fit is intentionally non-parametric — we just compute IV per
strike per expiry and plot. Parametric smile models (SVI, SABR, etc.) are
deeper than this project intends to go; we treat them as a clearly-scoped
'next step'.

Failure modes documented in the docstring of `implied_vol`: arbitrage-violating
quotes (price < intrinsic) and effectively-zero-vega tails return NaN rather
than crashing the surface fit.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .pricing import black_scholes as bs

OptionType = Literal["call", "put"]

_SIGMA_LO = 1e-6
_SIGMA_HI = 5.0  # 500% IV upper bracket — covers everything except complete data garbage


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Invert BS for sigma given a market price.

    Returns NaN if:
    - market_price is below intrinsic (arbitrage)
    - the bracketed root finder cannot find a sign change
    - inputs are invalid (T <= 0, etc.)

    Returning NaN rather than raising lets surface fits fail gracefully on a
    handful of dirty quotes without abandoning the entire chain.
    """
    if T <= 0 or market_price < 0 or S <= 0 or K <= 0:
        return float("nan")

    intrinsic = (
        max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        if option_type == "call"
        else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    )
    if market_price < intrinsic - 1e-8:
        return float("nan")

    def f(sigma):
        return float(bs.price(S, K, T, r, sigma, q, option_type)) - market_price

    f_lo = f(_SIGMA_LO)
    f_hi = f(_SIGMA_HI)
    if f_lo * f_hi > 0:
        return float("nan")

    try:
        return float(brentq(f, _SIGMA_LO, _SIGMA_HI, xtol=1e-8, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


def implied_vol_chain(
    chain: pd.DataFrame,
    S: float,
    r: float,
    q: float = 0.0,
) -> pd.DataFrame:
    """Compute IV for every row of a chain DataFrame.

    Expected columns:
        strike, T (years), market_price, option_type ('call' | 'put')

    Returns the same DataFrame with an added 'iv' column.
    """
    required = {"strike", "T", "market_price", "option_type"}
    missing = required - set(chain.columns)
    if missing:
        raise ValueError(f"chain DataFrame is missing columns: {missing}")

    out = chain.copy()
    out["iv"] = [
        implied_vol(row.market_price, S, row.strike, row.T, r, q, row.option_type)
        for row in out.itertuples()
    ]
    return out


def smile_at_expiry(iv_chain: pd.DataFrame, T: float, tol: float = 1e-6) -> pd.DataFrame:
    """Slice the IV chain at a single expiry and return a sorted-by-strike
    DataFrame. Useful for plotting the smile."""
    mask = np.isclose(iv_chain["T"], T, atol=tol)
    sub = iv_chain.loc[mask, ["strike", "iv", "option_type", "market_price"]].copy()
    return sub.sort_values("strike").reset_index(drop=True)


def term_structure_atm(iv_chain: pd.DataFrame, S: float) -> pd.DataFrame:
    """Pick the strike closest to spot (ATM) for each expiry and return
    (T, iv) for plotting the term structure."""
    rows = []
    for T_val, group in iv_chain.groupby("T"):
        idx = (group["strike"] - S).abs().idxmin()
        rows.append({"T": T_val, "strike": group.loc[idx, "strike"], "iv": group.loc[idx, "iv"]})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
