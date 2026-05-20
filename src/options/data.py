"""Market-data ingestion for SPY/SPX options and underlying spot.

We use yfinance as the primary source because it's free, reasonably reliable,
and ships with a dead-simple API. Production-grade option-pricing work uses
OPRA-feed vendors (CBOE DataShop, Polygon, IB), but for a portfolio project
yfinance is the right scope: zero-cost, reproducible, and the data quality
is good enough that the IV smile is recognisable.

Module split:
- `fetch_*` functions touch the network. Cached to disk under data/raw/.
- `clean_chain` is a pure DataFrame transform. This is the function we test.

The fetch path is intentionally tolerant of yfinance's schema drift: we
collect the columns we need and silently drop the rest.
"""
from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import year_fraction

# Where on-disk caches live.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = _REPO_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def fetch_spy_history(
    start: str, end: Optional[str] = None, ticker: str = "SPY",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily OHLCV history for SPY (or any other yfinance ticker)."""
    cache_key = f"history_{ticker}_{start}_{end or 'today'}"
    cache = _cache_path(cache_key)
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} between {start} and {end}.")
    df.index.name = "date"
    df = df.reset_index()
    # Newer yfinance versions return MultiIndex columns; flatten.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == "" else c[0] for c in df.columns]
    df.to_parquet(cache)
    return df


def fetch_spy_chain(
    ticker: str = "SPY",
    expiry: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch the option chain for one or all available expiries.

    Returns a long-format DataFrame with columns:
        symbol, strike, expiry, option_type, bid, ask, last, mid, volume, openInterest

    If `expiry` is None, fetches every expiry yfinance lists for the ticker
    and concatenates them. Heavy network call — use `use_cache=True` in interactive
    sessions.
    """
    cache_key = f"chain_{ticker}_{expiry or 'all'}_{date.today().isoformat()}"
    cache = _cache_path(cache_key)
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    import yfinance as yf
    tk = yf.Ticker(ticker)
    expiries = [expiry] if expiry else list(tk.options)
    if not expiries:
        raise RuntimeError(f"No option expiries returned for {ticker}.")

    rows: list[pd.DataFrame] = []
    for exp in expiries:
        try:
            chain = tk.option_chain(exp)
        except Exception:
            # yfinance can throw on individual expiries; skip and continue.
            continue
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty:
                continue
            sub = df.copy()
            sub["expiry"] = exp
            sub["option_type"] = kind
            rows.append(sub)
    if not rows:
        raise RuntimeError(f"All expiries failed for {ticker}.")
    out = pd.concat(rows, ignore_index=True)
    # Standard column names
    keep = {
        "contractSymbol": "symbol", "strike": "strike", "lastPrice": "last",
        "bid": "bid", "ask": "ask", "volume": "volume",
        "openInterest": "openInterest", "expiry": "expiry", "option_type": "option_type",
    }
    out = out[[c for c in keep if c in out.columns]].rename(columns=keep)
    if "bid" in out.columns and "ask" in out.columns:
        out["mid"] = 0.5 * (out["bid"].astype(float) + out["ask"].astype(float))
    out.to_parquet(cache)
    return out


def clean_chain(
    chain: pd.DataFrame,
    spot_date: str | date | datetime,
    min_bid: float = 0.05,
    max_relative_spread: float = 0.5,
) -> pd.DataFrame:
    """Filter and standardise a raw chain for IV fitting.

    Steps:
    1. Drop rows with bid < min_bid (illiquid / stale).
    2. Drop rows with (ask - bid) / mid > max_relative_spread (toxic spreads).
    3. Compute time-to-expiry T (years) from spot_date and expiry.
    4. Drop rows with T <= 0.

    Returns a fresh DataFrame with the columns that downstream pricers expect:
    `strike, T, market_price, option_type, expiry, bid, ask, mid, volume, openInterest`.
    """
    df = chain.copy()
    if "bid" in df.columns and "ask" in df.columns:
        df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
        df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
    elif "mid" not in df.columns:
        raise ValueError("clean_chain requires 'bid'+'ask' or a 'mid' column.")
    df = df.dropna(subset=["mid"])
    df = df[df["bid"] >= min_bid]
    rel_spread = (df["ask"] - df["bid"]) / df["mid"]
    df = df[rel_spread <= max_relative_spread]

    df["T"] = df["expiry"].apply(lambda e: year_fraction(spot_date, e))
    df = df[df["T"] > 0]
    df["market_price"] = df["mid"]

    cols = ["strike", "T", "market_price", "option_type", "expiry"]
    extra = [c for c in ("bid", "ask", "mid", "volume", "openInterest", "symbol") if c in df.columns]
    return df[cols + extra].reset_index(drop=True)


def estimate_dividend_yield(
    spot: float, history: pd.DataFrame, lookback_days: int = 365 * 3,
) -> float:
    """Crude dividend-yield estimate from price history.

    Uses the gap between the (Close) and (Adj Close) implied dividend rate
    if available; otherwise returns 0.013 — SPY's ~1.3% historical yield —
    as a sensible default. This is intentionally simple; the IV smile fit is
    insensitive to small q errors.
    """
    if "Adj Close" in history.columns and "Close" in history.columns:
        recent = history.tail(lookback_days)
        # Annualised geometric divergence between adj close and close
        ratio_end = recent["Adj Close"].iloc[-1] / recent["Close"].iloc[-1]
        ratio_start = recent["Adj Close"].iloc[0] / recent["Close"].iloc[0]
        if ratio_start > 0 and ratio_end > 0:
            years = lookback_days / 365.0
            yld = -np.log(ratio_end / ratio_start) / years
            if 0.0 <= yld <= 0.05:
                return float(yld)
    return 0.013  # SPY's long-run trailing yield, a defensible fallback.


def realised_volatility(history: pd.DataFrame, window: int = 21) -> pd.Series:
    """Trailing realised vol from daily log-returns, annualised by sqrt(252)."""
    close = history["Adj Close"] if "Adj Close" in history.columns else history["Close"]
    returns = np.log(close / close.shift(1))
    return returns.rolling(window).std() * np.sqrt(252.0)
