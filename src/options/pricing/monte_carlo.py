"""Monte Carlo pricers for European and arithmetic Asian options.

The MC engine uses geometric Brownian motion under the risk-neutral measure
with two variance-reduction techniques:

1. **Antithetic variates.** For every standard-normal draw Z we also use -Z.
   The pair has the same expected payoff but lower variance for any monotonic
   payoff function. Effectively halves the standard error for free.

2. **Control variates** (Asian only). We use the discounted geometric-Asian
   payoff as a control: its true mean has a closed form (Kemna-Vorst, continuous
   limit) and it is ~99% correlated with the arithmetic Asian payoff. The
   variance-reduced estimator is

       V_arith_CV = V_arith_MC - beta * (V_geom_MC - E[V_geom_closed])

   with beta = Cov(arith, geom) / Var(geom), which is the regression slope of
   arithmetic on geometric. Empirically this knocks 1-2 orders of magnitude
   off the standard error vs naive MC at the same path count.

Limitations to declare in interviews:
- The Kemna-Vorst closed form assumes *continuous* geometric averaging; our MC
  samples discretely on n_steps points. For large n_steps the discrete-vs-
  continuous bias in the control variate is small but non-zero. We document
  the bias in the convergence notebook.
- We use NumPy's PCG64 generator (default_rng); for production-grade MC you'd
  want stratified sampling or quasi-random sequences (Sobol) for further
  variance reduction. That's a clearly-scoped next-step.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..utils import validate_inputs
from . import black_scholes as bs

OptionType = Literal["call", "put"]


def gbm_terminal(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    antithetic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Sample terminal prices S_T under risk-neutral GBM in one step."""
    if antithetic and n_paths % 2:
        n_paths += 1  # round up to even for antithetic pairing
    rng = np.random.default_rng(seed)
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal(half)
    if antithetic:
        z = np.concatenate([z, -z])
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * z
    return S0 * np.exp(drift + diffusion)


def gbm_paths(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_steps: int,
    n_paths: int,
    antithetic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Sample full paths on a uniform time grid. Returns (n_paths, n_steps+1).

    Path[i, 0] = S0, Path[i, n_steps] = S_T. Used for path-dependent payoffs.
    """
    if antithetic and n_paths % 2:
        n_paths += 1
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    half = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((half, n_steps))
    if antithetic:
        z = np.concatenate([z, -z], axis=0)
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = S0 * np.exp(log_paths)
    initial = np.full((paths.shape[0], 1), S0)
    return np.concatenate([initial, paths], axis=1)


def price_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: int | None = None,
) -> dict:
    """MC price for a European call/put with antithetic variance reduction.

    Returns a dict with the point estimate, the standard error, and a 95%
    confidence interval. Always quote the SE alongside the estimate.

    Important — antithetic SE bookkeeping. With antithetic variates, the
    samples are NOT independent: each pair (Z_i, -Z_i) is negatively
    correlated by construction. The correct SE is computed from the
    *pair-means*, not from the raw payoff array. Otherwise you get the
    naive SE and the variance reduction is hidden by your own bookkeeping.
    """
    validate_inputs(S, K, T, r, sigma, q)
    S_T = gbm_terminal(S, T, r, q, sigma, n_paths, antithetic, seed)
    if option_type == "call":
        payoff = np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    discounted = np.exp(-r * T) * payoff

    if antithetic:
        half = len(discounted) // 2
        sample_units = 0.5 * (discounted[:half] + discounted[half:])
    else:
        sample_units = discounted

    price_est = float(np.mean(sample_units))
    se = float(np.std(sample_units, ddof=1) / np.sqrt(len(sample_units)))
    return {
        "price": price_est,
        "std_error": se,
        "ci_95": (price_est - 1.96 * se, price_est + 1.96 * se),
        "n_paths": int(len(discounted)),
        "n_effective": int(len(sample_units)),
    }


def geometric_asian_closed_form(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> float:
    """Kemna-Vorst (1990) closed-form price of a *continuous* geometric Asian.

    The geometric average of GBM is itself log-normal, so a BS-like formula
    applies with adjusted volatility and drift:
        sigma_G = sigma / sqrt(3)
        b_G    = 0.5 * (r - q - sigma^2 / 6)

    The price is then BS with an effective dividend yield q_eff = r - b_G.
    """
    if T == 0:
        if option_type == "call":
            return float(max(S - K, 0.0))
        return float(max(K - S, 0.0))
    sigma_G = sigma / np.sqrt(3.0)
    b_G = 0.5 * (r - q - sigma**2 / 6.0)
    q_eff = r - b_G
    return float(bs.price(S, K, T, r, sigma_G, q_eff, option_type))


def price_arithmetic_asian(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 50_000,
    n_steps: int = 252,
    antithetic: bool = True,
    control_variates: bool = True,
    seed: int | None = None,
) -> dict:
    """MC price for an arithmetic-average Asian option.

    The average is taken over the n_steps post-spot observations (i.e.
    excluding S_0). Control variates use the geometric Asian as the control.
    """
    validate_inputs(S, K, T, r, sigma, q)
    paths = gbm_paths(S, T, r, q, sigma, n_steps, n_paths, antithetic, seed)
    n_actual = paths.shape[0]
    half = n_actual // 2 if antithetic else None
    sample = paths[:, 1:]  # exclude S_0
    arith_avg = sample.mean(axis=1)
    if option_type == "call":
        payoff_arith = np.maximum(arith_avg - K, 0.0)
    else:
        payoff_arith = np.maximum(K - arith_avg, 0.0)
    disc_arith = np.exp(-r * T) * payoff_arith

    def _pair_or_pass(arr):
        """Apply antithetic pairing if enabled, else return the raw samples."""
        if antithetic:
            return 0.5 * (arr[:half] + arr[half:])
        return arr

    if not control_variates:
        units = _pair_or_pass(disc_arith)
        price_est = float(np.mean(units))
        se = float(np.std(units, ddof=1) / np.sqrt(len(units)))
        return {
            "price": price_est,
            "std_error": se,
            "ci_95": (price_est - 1.96 * se, price_est + 1.96 * se),
            "n_paths": int(n_actual),
            "n_effective": int(len(units)),
            "method": "naive",
        }

    geom_avg = np.exp(np.mean(np.log(sample), axis=1))
    if option_type == "call":
        payoff_geom = np.maximum(geom_avg - K, 0.0)
    else:
        payoff_geom = np.maximum(K - geom_avg, 0.0)
    disc_geom = np.exp(-r * T) * payoff_geom

    # Closed-form mean of the geometric Asian (continuous approximation).
    E_geom = geometric_asian_closed_form(S, K, T, r, sigma, q, option_type)

    # Optimal beta = Cov(arith, geom) / Var(geom). Computed on the raw (unpaired)
    # samples — the regression slope is the same regardless of pairing.
    cov = np.cov(disc_arith, disc_geom, ddof=1)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else 1.0
    adjusted = disc_arith - beta * (disc_geom - E_geom)

    units_cv = _pair_or_pass(adjusted)
    units_naive = _pair_or_pass(disc_arith)

    price_est = float(np.mean(units_cv))
    se = float(np.std(units_cv, ddof=1) / np.sqrt(len(units_cv)))
    se_naive = float(np.std(units_naive, ddof=1) / np.sqrt(len(units_naive)))
    return {
        "price": price_est,
        "std_error": se,
        "ci_95": (price_est - 1.96 * se, price_est + 1.96 * se),
        "n_paths": int(n_actual),
        "n_effective": int(len(units_cv)),
        "method": "control_variates",
        "beta": beta,
        "se_naive": se_naive,
        "variance_reduction_ratio": (se_naive / se) ** 2 if se > 0 else float("nan"),
    }
