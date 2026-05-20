"""Shared utilities used across pricing modules.

Kept deliberately small — only things that genuinely don't belong in a single
pricer module live here.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Union

import numpy as np
from scipy.stats import norm

DateLike = Union[str, date, datetime, np.datetime64]

# 252 trading days per year is the convention used in the equity-options
# literature; 365.25 calendar days appears in some quant texts. We use 365
# (calendar days) for time-to-expiry because option expiries are wall-clock
# dates and decay accrues over weekends. Volatilities are still typically
# annualised from daily log-returns × sqrt(252) — that's a separate convention
# applied only when annualising historical realised vol.
DAYS_PER_YEAR = 365.0


def year_fraction(t0: DateLike, t1: DateLike) -> float:
    """Calendar-day year fraction between two dates."""
    a = np.datetime64(_to_date(t0))
    b = np.datetime64(_to_date(t1))
    days = (b - a) / np.timedelta64(1, "D")
    return float(days) / DAYS_PER_YEAR


def _to_date(d: DateLike) -> date:
    if isinstance(d, str):
        return datetime.fromisoformat(d).date()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, np.datetime64):
        return datetime.utcfromtimestamp(
            (d - np.datetime64("1970-01-01")) / np.timedelta64(1, "s")
        ).date()
    return d


def std_normal_cdf(x):
    return norm.cdf(x)


def std_normal_pdf(x):
    return norm.pdf(x)


def validate_inputs(S, K, T, r, sigma, q=0.0):
    """Centralised input validation. Pricers call this so that error messages
    are uniform across modules.
    """
    if np.any(np.asarray(S) <= 0):
        raise ValueError("Spot S must be positive.")
    if np.any(np.asarray(K) <= 0):
        raise ValueError("Strike K must be positive.")
    if T < 0:
        raise ValueError("Time to expiry T must be non-negative.")
    if np.any(np.asarray(sigma) < 0):
        raise ValueError("Volatility sigma must be non-negative.")
    # r and q can be negative (negative rates are real, dividends can be modelled negative for funding adjustments)
