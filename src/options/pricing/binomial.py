"""Cox-Ross-Rubinstein (CRR) binomial tree pricer for European and American options.

Why this module exists in the project: Black-Scholes has no closed form for
American options because the holder can choose to exercise at any time before
expiry. The tree handles this naturally — at every node we compare
continuation value to immediate exercise and take the maximum. That single
modification is the entire reason the tree earns its place alongside BS.

Construction (per Hull, Chapter 21):
    Time step: dt = T / N
    Up factor: u  = exp(sigma * sqrt(dt))
    Down factor: d  = 1 / u                (recombining tree)
    Risk-neutral probability: p = (exp((r - q) * dt) - d) / (u - d)

The tree at step n has n+1 nodes. For an American put we backstep through the
tree comparing exp(-r dt) * (p * V_up + (1-p) * V_down) to (K - S_n,i)+ at
each node and keeping the larger.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ..utils import validate_inputs

OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    exercise_style: ExerciseStyle = "european",
    n_steps: int = 500,
) -> float:
    """Price a vanilla call or put using a CRR tree.

    Parameters
    ----------
    n_steps : int
        Number of time steps. Convergence to Black-Scholes is roughly O(1/N)
        for European options with the CRR scheme. n_steps=500 typically gives
        4-5 decimal places of accuracy on near-the-money options. The
        convergence study in docs/ uses n_steps in {10, 50, 100, 500, 1000,
        5000} to show the rate.
    """
    validate_inputs(S, K, T, r, sigma, q)
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")

    if T == 0:
        if option_type == "call":
            return float(max(S - K, 0.0))
        return float(max(K - S, 0.0))

    dt = T / n_steps
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    discount = float(np.exp(-r * dt))
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        # Risk-neutral probability falls outside [0,1] only when dt is too
        # large relative to volatility — extremely small N or extreme params.
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} is outside (0,1). "
            "Increase n_steps or check your inputs."
        )

    # Spot prices at expiry, one entry per terminal node.
    # Node i corresponds to i up-moves and (n_steps - i) down-moves.
    i = np.arange(n_steps + 1)
    S_terminal = S * (u ** i) * (d ** (n_steps - i))

    # Terminal payoff.
    if option_type == "call":
        V = np.maximum(S_terminal - K, 0.0)
    else:
        V = np.maximum(K - S_terminal, 0.0)

    # Backward induction.
    is_american = exercise_style == "american"
    for step in range(n_steps - 1, -1, -1):
        # Risk-neutral discounted expectation: V_step,i = e^{-r dt} (p V_step+1,i+1 + (1-p) V_step+1,i)
        V = discount * (p * V[1:] + (1.0 - p) * V[:-1])
        if is_american:
            # Spot tree at this step.
            i_step = np.arange(step + 1)
            S_step = S * (u ** i_step) * (d ** (step - i_step))
            if option_type == "call":
                exercise = np.maximum(S_step - K, 0.0)
            else:
                exercise = np.maximum(K - S_step, 0.0)
            V = np.maximum(V, exercise)

    return float(V[0])


def price_with_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    exercise_style: ExerciseStyle = "european",
    n_steps: int = 500,
) -> dict:
    """Price + delta and gamma from the tree itself (not bump-and-revalue).

    The standard tree-based Greeks come from the first two layers:
        delta = (V_1,1 - V_1,0) / (S_1,1 - S_1,0)
        gamma is taken from layer 2 with a centred second-difference.

    Theta and vega are not exposed here because the tree only gives you a
    natural finite-difference at the *initial* node — for theta we'd need to
    compare V_0 with V_{2,1} (centre node at step 2, same spot), and for vega
    we'd need to bump sigma and rebuild the tree. We expose those as
    bump-and-revalue helpers in `greeks.py` to keep this module focused.
    """
    validate_inputs(S, K, T, r, sigma, q)
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2 to extract gamma from the tree.")

    dt = T / n_steps
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    discount = float(np.exp(-r * dt))
    p = (np.exp((r - q) * dt) - d) / (u - d)

    i = np.arange(n_steps + 1)
    S_terminal = S * (u ** i) * (d ** (n_steps - i))
    if option_type == "call":
        V = np.maximum(S_terminal - K, 0.0)
    else:
        V = np.maximum(K - S_terminal, 0.0)

    is_american = exercise_style == "american"
    layers: dict[int, np.ndarray] = {}
    for step in range(n_steps - 1, -1, -1):
        V = discount * (p * V[1:] + (1.0 - p) * V[:-1])
        if is_american:
            i_step = np.arange(step + 1)
            S_step = S * (u ** i_step) * (d ** (step - i_step))
            if option_type == "call":
                exercise = np.maximum(S_step - K, 0.0)
            else:
                exercise = np.maximum(K - S_step, 0.0)
            V = np.maximum(V, exercise)
        if step in (1, 2):
            layers[step] = V.copy()

    price_at_root = float(V[0])

    # Tree-based Greeks.
    V1 = layers[1]  # V at step 1: [V_down, V_up]
    S1_down = S * d
    S1_up = S * u
    delta_tree = (V1[1] - V1[0]) / (S1_up - S1_down)

    V2 = layers[2]  # V at step 2: [V_dd, V_du, V_uu]
    S2_dd = S * d * d
    S2_du = S
    S2_uu = S * u * u
    delta_up = (V2[2] - V2[1]) / (S2_uu - S2_du)
    delta_down = (V2[1] - V2[0]) / (S2_du - S2_dd)
    gamma_tree = (delta_up - delta_down) / ((S2_uu - S2_dd) / 2.0)

    return {
        "price": price_at_root,
        "delta": float(delta_tree),
        "gamma": float(gamma_tree),
    }
