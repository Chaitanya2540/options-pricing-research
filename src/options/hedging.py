"""Discrete delta-hedging simulator for a short ATM straddle.

The textbook result we are demonstrating: for a delta-hedged short option
position, instantaneous P&L is approximately

    dP&L ≈ -½ · Γ · S² · (σ_realised² − σ_implied²) · dt

So a short straddle (which is short gamma) makes money when realised
volatility comes in below the implied volatility we sold at, and loses
money in the opposite direction. The simulator confirms this empirically
across many paths, accumulating both the realised P&L and a step-by-step
gamma-scalp decomposition that we can compare to the theoretical mean.

Mechanics:
1. At t = 0, sell one ATM call + one ATM put (a straddle). Premium received
   is computed at σ_implied.
2. Establish the initial delta hedge against the straddle's delta (close to
   zero for ATM but not exactly zero).
3. At each rebalancing time t_i, recompute the straddle delta at (S_i, K,
   T - t_i, σ_implied) and trade the underlying to bring net delta to zero.
   Cash is accrued at the risk-free rate r.
4. At expiry, pay out |S_T − K| (the straddle's payoff) and close the hedge.
5. Net P&L = premium - sum(hedge cash flows) - final payout, with
   appropriate carry on the cash position.

The simulator separately tracks an *attribution* P&L computed step-by-step
from theta and the local gamma-scalp term — so the user can see the
accounting tie out (within hedging error and discretisation noise).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .pricing import black_scholes as bs


@dataclass
class HedgeResult:
    """Per-path simulation result for a short straddle."""
    pnl: float                  # net P&L at expiry
    premium: float              # straddle premium received at t=0
    final_payout: float         # |S_T - K|, paid at expiry
    hedge_pnl: float            # cumulative hedge P&L (delta * dS net of carry)
    theta_attribution: float    # sum over steps of theta * dt
    gamma_attribution: float    # sum over steps of -0.5 * gamma * (dS)^2
    residual: float             # pnl - (premium - final_payout - hedge_pnl) [should be ~0]
    S_path: np.ndarray
    delta_path: np.ndarray


def simulate_short_straddle_path(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma_implied: float,
    sigma_realised: float,
    n_steps: int,
    seed: Optional[int] = None,
) -> HedgeResult:
    """Run one path. Path is generated under sigma_realised; deltas are
    computed under sigma_implied (the IV we sold at)."""
    if T <= 0 or n_steps < 1:
        raise ValueError("Need T > 0 and n_steps >= 1.")
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    # Generate path under realised vol (risk-neutral drift; could equally
    # use real-world drift — for the gamma-scalp decomposition the drift
    # only contributes a small bias to the residual).
    z = rng.standard_normal(n_steps)
    increments = (r - q - 0.5 * sigma_realised**2) * dt + sigma_realised * np.sqrt(dt) * z
    log_path = np.cumsum(increments)
    S_path = np.empty(n_steps + 1)
    S_path[0] = S0
    S_path[1:] = S0 * np.exp(log_path)

    # Initial straddle premium received (we are short).
    premium = float(
        bs.price(S0, K, T, r, sigma_implied, q, "call")
        + bs.price(S0, K, T, r, sigma_implied, q, "put")
    )

    # Track positions and cash.
    cash = premium
    delta_path = np.zeros(n_steps + 1)
    theta_sum = 0.0
    gamma_sum = 0.0

    for i in range(n_steps + 1):
        t_remaining = T - i * dt
        if t_remaining > 1e-12:
            d_call = float(bs.delta(S_path[i], K, t_remaining, r, sigma_implied, q, "call"))
            d_put = float(bs.delta(S_path[i], K, t_remaining, r, sigma_implied, q, "put"))
            straddle_delta = d_call + d_put
        else:
            straddle_delta = 0.0
        # We are short the straddle, so we want to be long `straddle_delta` shares
        # of underlying as a hedge (i.e. holdings = +straddle_delta).
        hedge_target = straddle_delta
        delta_change = hedge_target - delta_path[i - 1] if i > 0 else hedge_target
        # Cost of trade at S_path[i]; cash decreases when we buy, increases when we sell.
        cash -= delta_change * S_path[i]
        delta_path[i] = hedge_target

        if i < n_steps:
            # Accrue interest on the cash balance (cash already reflects the
            # cost of any stock we bought, so no separate financing cost).
            cash *= np.exp(r * dt)
            # Dividends earned on stock holdings, paid into cash.
            cash += hedge_target * S_path[i] * (np.exp(q * dt) - 1.0)

            # Step attribution from Itô: dP&L_short = -theta dt - 0.5 gamma dS^2
            theta_call = float(bs.theta(S_path[i], K, t_remaining, r, sigma_implied, q, "call"))
            theta_put = float(bs.theta(S_path[i], K, t_remaining, r, sigma_implied, q, "put"))
            gamma_val = float(bs.gamma(S_path[i], K, t_remaining, r, sigma_implied, q))
            theta_sum += -(theta_call + theta_put) * dt
            dS = S_path[i + 1] - S_path[i]
            gamma_sum += -0.5 * gamma_val * dS**2
            # In expectation, this decomposition reproduces the textbook:
            #   E[dP&L] ≈ 0.5 * gamma * S^2 * (sigma_impl^2 - sigma_real^2) * dt
            # because E[-theta dt] is approximately +0.5 gamma sigma_impl^2 S^2 dt
            # via the BS PDE, and E[-0.5 gamma dS^2] is -0.5 gamma sigma_real^2 S^2 dt.

    # At expiry, settle the straddle: we pay |S_T - K|.
    final_payout = float(abs(S_path[-1] - K))
    # Close out the hedge.
    cash += delta_path[-1] * S_path[-1]
    delta_path[-1] = 0.0
    # We pay the payout from cash.
    cash -= final_payout
    pnl = cash

    hedge_pnl = pnl - (premium - final_payout)
    # Attribution should sum to roughly the same hedge_pnl up to discretisation.
    attribution = theta_sum + gamma_sum
    residual = hedge_pnl - attribution

    return HedgeResult(
        pnl=pnl,
        premium=premium,
        final_payout=final_payout,
        hedge_pnl=hedge_pnl,
        theta_attribution=theta_sum,
        gamma_attribution=gamma_sum,
        residual=residual,
        S_path=S_path,
        delta_path=delta_path,
    )


def simulate_distribution(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma_implied: float,
    sigma_realised: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Run many independent paths and return a tidy DataFrame of P&L.

    Columns: pnl, premium, final_payout, hedge_pnl, theta_attribution,
    gamma_attribution, residual.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=n_paths)
    rows = []
    for s in seeds:
        out = simulate_short_straddle_path(
            S0, K, T, r, q, sigma_implied, sigma_realised, n_steps, seed=int(s),
        )
        rows.append({
            "pnl": out.pnl,
            "premium": out.premium,
            "final_payout": out.final_payout,
            "hedge_pnl": out.hedge_pnl,
            "theta_attribution": out.theta_attribution,
            "gamma_attribution": out.gamma_attribution,
            "residual": out.residual,
        })
    return pd.DataFrame(rows)


def expected_pnl_textbook(
    S0: float, K: float, T: float, r: float, q: float,
    sigma_implied: float, sigma_realised: float,
) -> float:
    """Textbook closed-form expected P&L of a delta-hedged short straddle.

    Using the integral E[ ∫ 0.5 Γ S² (σ_imp² − σ_real²) dt ] over a BS path.
    For an ATM short straddle with σ_imp held constant, this is approximately
    Vega * (σ_imp − σ_real) at first order. We return the exact form we use
    in the gamma-attribution series for an apples-to-apples comparison with
    the simulator.
    """
    # Vega-of-straddle approximation: 2 * vega(call ATM)
    v = 2.0 * float(bs.vega(S0, K, T, r, sigma_implied, q))
    return v * (sigma_implied - sigma_realised)
