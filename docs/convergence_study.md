# Convergence study: binomial → BS, Monte Carlo → BS

A pricing engine that doesn't show its convergence is one you shouldn't trust.
The numbers below come from `scripts/run_analyses.py` on a vanilla European
call with `S = K = 100, T = 1y, r = 5%, σ = 20%`. Black-Scholes truth:
**$10.4506**.

## CRR binomial tree

The Cox-Ross-Rubinstein tree converges to Black-Scholes at order **O(1/N)**.
The error roughly halves every time `N` doubles.

| `N` (steps) | `|tree - BS|` |
|--:|--:|
| 10 | 0.1972 |
| 25 | 0.0704 |
| 50 | 0.0399 |
| 100 | 0.0200 |
| 250 | 0.0080 |
| 500 | 0.0040 |
| 1000 | 0.0020 |
| 2000 | 0.0010 |
| 5000 | 0.00040 |

That's a ~500× error reduction from N=10 to N=5000, exactly as the theoretical
rate predicts. The straight-line slope on a log-log plot is the visual
signature of O(1/N).

![binomial convergence](../results/convergence_binomial.png)

## Monte Carlo (antithetic, no control variates)

MC converges at the standard `O(1/√N)` rate. To halve the standard error you
need to quadruple the path count.

| Paths | `|MC - BS|` | MC SE |
|--:|--:|--:|
| 1k | 0.4502 | 0.314 |
| 5k | 0.0004 | 0.150 |
| 25k | 0.0029 | 0.067 |
| 100k | 0.0167 | 0.033 |
| 400k | 0.0214 | 0.017 |
| 1M | 0.0060 | 0.010 |

The absolute error is noisy at small `N` because we are looking at one
realisation of the random estimator. The **standard error** is the right
metric, and it decreases by exactly √4 ≈ 2× whenever the path count
quadruples (1k→5k drops SE from 0.31 to 0.15; 5k→25k from 0.15 to 0.067).

![mc convergence](../results/convergence_mc.png)

## Why these rates hold (and where they break)

**Why these are the right rates:**
- Binomial: each node introduces a `dt` discretisation. The local error is
  `O(dt²)` per step, summed across `N = T/dt` steps → global `O(dt) = O(1/N)`.
- MC: variance of the sample mean is `σ²/N`, so SE is `σ/√N`.

**Where the rates break:**
- Binomial: discontinuous payoffs (digitals, barriers near a barrier) hit
  CRR with oscillation — error doesn't shrink monotonically. Mitigations:
  finer tree near the discontinuity, or a smoothing scheme (Heston-Lewis).
- MC: pathwise non-smoothness (deep OTM with finite path count) produces
  high variance and slow convergence. Mitigations: stratified sampling,
  importance sampling, low-discrepancy (Sobol) sequences.
