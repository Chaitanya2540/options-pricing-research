"""Greeks computation across all three pricing methods, with cross-validation.

This module ties the three pricers together for the single question every
options interview eventually asks: "How do you know your Greeks are right?"
The answer is "I compute them three ways and check they agree."

For European options:
- BS analytical Greeks: closed form, exact.
- Tree-based delta and gamma: from layers 1 and 2 of the CRR tree.
- MC pathwise delta and vega: differentiate the payoff w.r.t. the parameter
  before taking the expectation. Works whenever the payoff and the SDE are
  smooth enough that you can swap derivative and integral.
- MC bump-and-revalue with common random numbers (CRN): use *the same* random
  draws for the bumped and unbumped runs so the noise cancels in the
  difference. This is how we get gamma cleanly from MC.

Pathwise vs bump-and-revalue. Pathwise estimators typically have lower variance
when applicable (smooth payoffs). Bump-and-revalue with CRN works for any
payoff but has slightly higher variance. We use pathwise for delta/vega and
CRN for gamma — gamma's pathwise estimator runs into the indicator's
non-differentiability at the strike.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .pricing import binomial as bn
from .pricing import black_scholes as bs

OptionType = Literal["call", "put"]


def _mc_paths_terminal(S, T, r, q, sigma, n_paths, antithetic, seed):
    """Sample terminal prices and the associated standard normals (we need
    the Z's themselves for the pathwise vega estimator)."""
    if antithetic and n_paths % 2:
        n_paths += 1
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(half)
    if antithetic:
        z = np.concatenate([z, -z])
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    S_T = S * np.exp(drift + diffusion)
    return S_T, z


def _antithetic_pair_units(arr, antithetic: bool):
    if antithetic:
        half = len(arr) // 2
        return 0.5 * (arr[:half] + arr[half:])
    return arr


def mc_pathwise_delta(
    S, K, T, r, sigma, q=0.0, option_type: OptionType = "call",
    n_paths: int = 200_000, antithetic: bool = True, seed: int | None = None,
) -> dict:
    """MC delta via the pathwise method.

    For a European call:
        Delta = e^{-rT} E[ 1_{S_T > K} * S_T / S_0 ]
    For a put:
        Delta = -e^{-rT} E[ 1_{S_T < K} * S_T / S_0 ]
    """
    S_T, _ = _mc_paths_terminal(S, T, r, q, sigma, n_paths, antithetic, seed)
    if option_type == "call":
        contributions = (S_T > K).astype(float) * S_T / S
    elif option_type == "put":
        contributions = -(S_T < K).astype(float) * S_T / S
    else:
        raise ValueError(option_type)
    discounted = np.exp(-r * T) * contributions
    units = _antithetic_pair_units(discounted, antithetic)
    delta_est = float(np.mean(units))
    se = float(np.std(units, ddof=1) / np.sqrt(len(units)))
    return {"delta": delta_est, "std_error": se}


def mc_pathwise_vega(
    S, K, T, r, sigma, q=0.0, option_type: OptionType = "call",
    n_paths: int = 200_000, antithetic: bool = True, seed: int | None = None,
) -> dict:
    """MC vega via the pathwise method.

    Differentiating S_T = S_0 exp((r-q-0.5 sigma^2)T + sigma sqrt(T) Z) w.r.t. sigma:
        dS_T/dsigma = S_T * (-sigma T + sqrt(T) Z)
    For a call this gives:
        Vega = e^{-rT} E[ 1_{S_T > K} * S_T * (-sigma T + sqrt(T) Z) ]
    """
    S_T, z = _mc_paths_terminal(S, T, r, q, sigma, n_paths, antithetic, seed)
    dS_dsigma = S_T * (-sigma * T + np.sqrt(T) * z)
    if option_type == "call":
        contributions = (S_T > K).astype(float) * dS_dsigma
    elif option_type == "put":
        contributions = -(S_T < K).astype(float) * dS_dsigma
    else:
        raise ValueError(option_type)
    discounted = np.exp(-r * T) * contributions
    units = _antithetic_pair_units(discounted, antithetic)
    vega_est = float(np.mean(units))
    se = float(np.std(units, ddof=1) / np.sqrt(len(units)))
    return {"vega": vega_est, "std_error": se}


def mc_gamma_crn_bump(
    S, K, T, r, sigma, q=0.0, option_type: OptionType = "call",
    n_paths: int = 200_000, antithetic: bool = True, seed: int | None = None,
    bump_pct: float = 0.01,
) -> dict:
    """MC gamma via bump-and-revalue with common random numbers.

    We re-use the same Z draws for S, S+h, S-h so most of the noise cancels in
    the second-difference. The seed must be fixed for this to work.
    """
    h = S * bump_pct

    def _price_at(spot):
        S_T, _ = _mc_paths_terminal(spot, T, r, q, sigma, n_paths, antithetic, seed)
        if option_type == "call":
            payoff = np.maximum(S_T - K, 0.0)
        else:
            payoff = np.maximum(K - S_T, 0.0)
        discounted = np.exp(-r * T) * payoff
        units = _antithetic_pair_units(discounted, antithetic)
        return float(np.mean(units)), units

    p_up, u_up = _price_at(S + h)
    p_mid, u_mid = _price_at(S)
    p_down, u_down = _price_at(S - h)
    gamma_est = (p_up - 2.0 * p_mid + p_down) / (h * h)
    # SE from the per-sample second difference (CRN keeps these correlated).
    second_diff_units = (u_up - 2.0 * u_mid + u_down) / (h * h)
    se = float(np.std(second_diff_units, ddof=1) / np.sqrt(len(second_diff_units)))
    return {"gamma": gamma_est, "std_error": se}


def cross_validate_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 200_000,
    n_steps_tree: int = 1000,
    seed: int | None = 7,
) -> dict:
    """Run all three pricers and return a side-by-side Greeks comparison.

    The shape of the returned dict is:
        {
          'analytical': {'price', 'delta', 'gamma', 'vega', 'theta', 'rho'},
          'tree':       {'price', 'delta', 'gamma'},
          'mc':         {'price', 'std_error', 'delta', 'delta_se', 'vega',
                         'vega_se', 'gamma', 'gamma_se'},
        }
    """
    analytical = bs.greeks(S, K, T, r, sigma, q, option_type)
    tree = bn.price_with_greeks(S, K, T, r, sigma, q, option_type, "european", n_steps_tree)
    delta_pw = mc_pathwise_delta(S, K, T, r, sigma, q, option_type, n_paths, True, seed)
    vega_pw = mc_pathwise_vega(S, K, T, r, sigma, q, option_type, n_paths, True, seed)
    gamma_crn = mc_gamma_crn_bump(S, K, T, r, sigma, q, option_type, n_paths, True, seed)

    # Re-use the European MC pricer for price+SE; import lazily to avoid cycle.
    from .pricing.monte_carlo import price_european

    mc_price = price_european(S, K, T, r, sigma, q, option_type, n_paths, True, seed)
    return {
        "analytical": {k: float(v) for k, v in analytical.items()},
        "tree": tree,
        "mc": {
            "price": mc_price["price"],
            "price_se": mc_price["std_error"],
            "delta": delta_pw["delta"],
            "delta_se": delta_pw["std_error"],
            "vega": vega_pw["vega"],
            "vega_se": vega_pw["std_error"],
            "gamma": gamma_crn["gamma"],
            "gamma_se": gamma_crn["std_error"],
        },
    }
