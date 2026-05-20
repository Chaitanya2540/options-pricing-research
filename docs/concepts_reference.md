# Concepts reference

Deep-dive reference for every concept this project implicates, plus adjacent
material interviewers commonly probe. Read alongside the mastery guide; this
is the searchable encyclopaedia, that is the linear story.

Conventions:
- Formulas use `S` for spot, `K` strike, `T` time to expiry (years), `r`
  risk-free rate (continuously compounded), `q` dividend yield, `σ`
  volatility (annualised), `Z ~ N(0,1)`, `W_t` standard Brownian motion.

---

## 1. Stochastic calculus foundations

### 1.1 Brownian motion / Wiener process

A Wiener process `W_t` satisfies: (1) `W_0 = 0`, (2) increments `W_t − W_s`
are normally distributed with mean 0 and variance `t − s`, (3) increments
are independent across non-overlapping intervals, (4) sample paths are
continuous but nowhere differentiable.

Key facts you should be able to recite:
- `E[W_t] = 0`, `Var[W_t] = t`, `Cov(W_s, W_t) = min(s, t)`.
- `(dW)² = dt` (the heuristic that drives Itô calculus).
- `dW · dt = 0`, `(dt)² = 0` (in the Itô-Doeblin product table).

Why it's in our project: GBM uses `dW` directly. Every MC path is a discrete
Brownian-motion sampling.

### 1.2 Stochastic differential equations (SDEs)

An SDE has the form `dX = μ(X,t) dt + σ(X,t) dW`. The first term is the
**drift**; the second is the **diffusion**. For GBM:

> `dS = μ S dt + σ S dW` (real-world)
> `dS = (r − q) S dt + σ S dW` (risk-neutral)

The transformation between the two measures is Girsanov's theorem (§2.3).

### 1.3 Itô's lemma — the chain rule for stochastic processes

If `dX = μ dt + σ dW` and `f = f(X, t)`, then:

> `df = (∂f/∂t) dt + (∂f/∂X) dX + ½ (∂²f/∂X²) (dX)²`

The third term is the surprise — ordinary calculus drops it. Because
`(dW)² = dt`, `(dX)² = σ² dt`, so:

> `df = [∂f/∂t + μ ∂f/∂X + ½ σ² ∂²f/∂X²] dt + σ ∂f/∂X dW`

Why it's in our project: Itô underpins both the Black-Scholes PDE
derivation (§3.2) and the per-step P&L decomposition for delta-hedged
positions (§9.1). If an interviewer asks you to derive *anything* in
options pricing, Itô is your starting hammer.

### 1.4 Quadratic variation

For a Brownian motion, `[W, W]_T = T`. For GBM `dS = μS dt + σS dW`, the
quadratic variation of `ln S` is `σ² T` — i.e., **realised variance** is
literally the quadratic variation of log-prices. This is why σ enters
pricing as `σ² T`, not `σ T`.

### 1.5 Martingales

A process `M_t` is a martingale if `E[M_t | F_s] = M_s` for `s ≤ t` (under
some measure and filtration). Discounted prices of tradeable assets are
martingales under the risk-neutral measure — that's the no-arbitrage
condition. You may hear "the discounted call price is a Q-martingale"; that
is the formal statement of "options are priced fairly under Q."

---

## 2. Probability measures: real-world vs risk-neutral

### 2.1 Real-world (P)

The actual probability the market would assign to outcomes. Drift `μ` here
is whatever the asset actually expects to earn — depends on risk
preferences, beta, etc.

### 2.2 Risk-neutral (Q)

A constructed measure under which **every tradeable asset (after dividend
yield) drifts at the risk-free rate**. Why useful: under Q, the price of a
contingent claim equals the discounted expected payoff
`V_0 = E^Q[e^{-rT} payoff(S_T)]`. No need to know real-world drift.

### 2.3 Girsanov's theorem

Tells you how to switch from P to Q. Formally: if `dW^P = dW^Q + λ dt`,
where `λ = (μ − r) / σ` is the market price of risk, then `W^Q` is a
Brownian motion under Q. Practical takeaway: **changing measure shifts the
drift, not the volatility**. That's why σ is the same in both measures —
and why we can estimate σ from real-world prices and then plug it into a
Q-pricing formula.

### 2.4 Feynman-Kac formula

A bridge between PDEs and expectations. If `V` solves a parabolic PDE of
the BS form with terminal condition `V(S, T) = payoff(S)`, then
`V(S, t) = E^Q[e^{-r(T-t)} payoff(S_T) | S_t = S]`. So pricing by PDE and
pricing by MC expectation give the same answer — they are two computational
windows on the same mathematical object.

---

## 3. Black-Scholes deep dive

### 3.1 Assumptions

- Constant `r`, `q`, `σ` over the life of the option.
- GBM dynamics for `S` (continuous, no jumps).
- Frictionless market: no transaction costs, infinite divisibility,
  continuous trading, no taxes.
- European exercise.
- No arbitrage.

Most-violated in practice: **constant σ**. The IV smile is the market's
patch.

### 3.2 The Black-Scholes PDE

Construct a self-financing replicating portfolio: hold one option, short Δ
shares of underlying. By Itô, the portfolio's instantaneous return is
deterministic. No-arbitrage forces it to grow at `r`. Setting up and solving:

> `∂V/∂t + (r − q) S ∂V/∂S + ½ σ² S² ∂²V/∂S² − r V = 0`

The boundary condition is the payoff at expiry. For a European call:
`V(S, T) = max(S − K, 0)`.

### 3.3 The closed-form solution

Solving the PDE (or applying Feynman-Kac with risk-neutral GBM):

> `C = S e^{-qT} N(d₁) − K e^{-rT} N(d₂)`
> `d₁ = [ln(S/K) + (r − q + ½σ²) T] / (σ √T)`
> `d₂ = d₁ − σ √T`

For puts: `P = K e^{-rT} N(-d₂) − S e^{-qT} N(-d₁)`.

### 3.4 Put-call parity

Model-free identity:
> `C − P = S e^{-qT} − K e^{-rT}`

Holds regardless of the pricing model (BS, Heston, anything). Two
applications: (a) check your pricer for sign errors; (b) given a call
price, you know the put price exactly without re-pricing.

### 3.5 Self-financing replication

The hedge in BS is a **self-financing** portfolio: rebalancing requires no
cash injections beyond the initial premium. The replication is exact in
continuous time under BS assumptions. In discrete time you get hedging
error, which is the topic of §9.

---

## 4. Numerical methods overview

### 4.1 Three families

- **Closed form / quasi-closed form** (BS, Black, Merton). Fastest, exact
  for the model. Limited to simple payoffs and dynamics.
- **Lattice / tree methods** (CRR, JR, Tian, trinomial). Discrete time,
  discrete state. Handle American exercise naturally.
- **Monte Carlo** (forward simulation). Discrete time, continuous state.
  Handles path-dependence, high dimensions, complex payoffs.
- **PDE / finite-difference** (explicit Euler, implicit Euler, Crank-Nicolson).
  Discrete time and state. Excellent for low-dimensional problems with
  early-exercise; mature stability theory.

### 4.2 Trade-offs at a glance

| Method | European vanilla | American | Path-dependent | High-dim | American + path |
|---|---|---|---|---|---|
| Closed form | ✓ exact | rare | rare | ✗ | ✗ |
| Tree | ✓ O(1/N) | ✓ | ✗ | ✗ | ✗ |
| MC | ✓ O(1/√N) | (Longstaff-Schwartz) | ✓ | ✓ | ✓ (LSM) |
| FDM | ✓ | ✓ | (limited) | up to 3-4D | (limited) |

We ship closed-form, tree, and MC. FDM is a credible v2 — the BS PDE solved
on a fine grid is what production trading systems often use.

---

## 5. Binomial trees in depth

### 5.1 CRR construction

Per step `dt = T / N`:
- `u = exp(σ √dt)`, `d = 1/u` (recombining)
- `p = (e^{(r−q)dt} − d) / (u − d)` is the risk-neutral up-probability.

Tree at step `n` has `n + 1` nodes (because `ud = du`). Total nodes
`O(N²)`, which is manageable.

### 5.2 Backward induction

Compute terminal payoffs at step `N`. Step backwards:

> `V_{n,i} = e^{-r dt} [p · V_{n+1, i+1} + (1−p) · V_{n+1, i}]`

For American exercise, replace the right-hand side with `max(continuation,
exercise_value)`. That single change is the entire reason the tree exists
in our project.

### 5.3 Why CRR converges at O(1/N)

Local truncation error is `O(dt²) = O(1/N²)` per step. There are `N`
steps, so global error is `O(1/N)`. Numerically observed in
docs/convergence_study.md: error halves when `N` doubles.

### 5.4 Tree variants

- **Cox-Ross-Rubinstein (CRR)**: `u·d = 1`. Simple, recombining, slight
  variance mismatch.
- **Jarrow-Rudd (JR)**: choose `u`, `d` so the per-step variance matches
  σ²·dt exactly; sets `p = 0.5`. Recombining.
- **Tian**: matches first three moments. Slightly better for skewed
  problems.
- **Trinomial**: three branches per node. Faster convergence near
  discontinuities (barriers, digitals) and easier to extend with
  state-dependent vol.

### 5.5 Pitfalls

- **Discontinuous payoffs** (digitals, barriers) cause oscillation: error
  doesn't shrink monotonically. Use a finer tree near the discontinuity, a
  smoothing scheme, or trinomial.
- **Risk-neutral probability outside (0,1)** when `dt` is too large
  relative to `σ`. We guard against this in `binomial.py`.

---

## 6. Monte Carlo in depth

### 6.1 Path simulation under GBM

Exact one-step (since GBM is closed-form):

> `S_T = S_0 · exp[(r − q − ½σ²) T + σ √T · Z]`

For path-dependent payoffs, sample on a time grid `t_1, ..., t_N`:

> `S_{t_{k+1}} = S_{t_k} · exp[(r − q − ½σ²) Δt + σ √Δt · Z_k]`

This is **exact** for GBM; no discretisation error in the path itself
(only in the discretisation of the *payoff observation grid* for path-
dependent payoffs).

For SDEs without closed form (Heston, SABR), use Euler-Maruyama or
Milstein discretisation, which introduce path-discretisation error.

### 6.2 Naive Monte Carlo and its convergence

Estimator: `V̂ = (1/N) Σ payoff_i · e^{-rT}`. Variance: `σ²_payoff / N`.
Standard error: `σ_payoff / √N`. **You quadruple paths to halve the SE.**

### 6.3 Variance reduction techniques

#### Antithetic variates
For each `Z`, also use `−Z`. Pair-mean has same expectation but lower
variance whenever the payoff is monotonic in `Z`. **Crucial bookkeeping**:
SE must be computed from pair-means, not from the 2N raw samples — else
the variance reduction is hidden by your own accounting (we caught and
fixed this exact bug in our test suite).

#### Control variates
Pair the unknown payoff with a correlated payoff whose true mean is known:
> `V̂_CV = V̂_target − β · (V̂_control − E[V_control])`
with `β = Cov(target, control) / Var(control)`. We use the **geometric
Asian** as control for the **arithmetic Asian** because (a) the geometric
Asian has a Kemna-Vorst closed form, and (b) the two are ~99% correlated.
Empirical variance reduction: 5-50× at the same path count.

#### Importance sampling
Sample under a different measure that puts more weight where the payoff is
non-zero, then re-weight using the Radon-Nikodym derivative. Useful for
deep-OTM options where naive MC produces mostly zero payoffs.

#### Stratified sampling
Partition the input space (e.g., on `Z`) and sample a fixed number from
each stratum. Eliminates within-stratum variance.

#### Quasi-random sequences (low-discrepancy)
**Sobol** and **Halton** sequences fill the input space more uniformly
than IID uniforms. Convergence rate improves to `O((log N)^d / N)`.
Pseudo-Monte Carlo (QMC) can give 10-100× speedup for low- to
moderate-dimensional integrals.

### 6.4 MC Greeks

Three estimator families:

1. **Pathwise**: differentiate the payoff w.r.t. the parameter before
   averaging. For European call delta:
   > `Δ_call = e^{-rT} E[ 1{S_T > K} · S_T / S_0 ]`
   Lower variance when applicable. Requires payoff smoothness. We use this
   for delta and vega.

2. **Likelihood ratio**: differentiate the density rather than the payoff.
   Works for non-smooth payoffs (digitals, barriers). Higher variance.

3. **Bump-and-revalue with common random numbers (CRN)**: same `Z`'s for
   bumped and unbumped runs so noise cancels. General-purpose, slightly
   higher variance than pathwise. We use this for gamma (gamma's pathwise
   estimator hits the indicator's non-differentiability).

### 6.5 American Monte Carlo: Longstaff-Schwartz (LSM)

Backward induction on simulated paths. At each exercise date, regress the
*continuation value* (next-step discounted price) on a basis of polynomials
in `S_t`. Compare the regression's prediction to the immediate exercise
value at each node; exercise where exercise > continuation. The genius is
turning a forward-simulated MC into a backward-induction procedure for
early-exercise decisions. We don't ship LSM (binomial handles American
fine for vanillas), but it's the right tool for American + path-dependent.

### 6.6 Kemna-Vorst formula

Closed form for the **geometric Asian** call (continuous averaging):
> `σ_G = σ / √3`,  `b_G = ½ (r − q − σ²/6)`

Then price as BS with adjusted vol `σ_G` and effective dividend yield
`q_eff = r − b_G`. Because the geometric average of GBM is itself
log-normal, you get a BS-like closed form. We use this as the control
variate's known mean.

---

## 7. Greeks — first to third order

### 7.1 First order

| Greek | Symbol | Definition | Long call sign |
|---|---|---|---|
| Delta | Δ | `∂V/∂S` | `+`, in [0, 1] |
| Vega | ν | `∂V/∂σ` | `+` |
| Theta | Θ | `∂V/∂t` | `−` |
| Rho | ρ | `∂V/∂r` | `+` |

### 7.2 Second order

| Greek | Definition | What it measures |
|---|---|---|
| Gamma (Γ) | `∂²V/∂S²` | Curvature in spot |
| Vomma / Volga | `∂²V/∂σ²` | Curvature in vol — vol-of-vol exposure |
| Vanna | `∂²V/∂S ∂σ` | Skew sensitivity (delta wrt vol, vega wrt spot) |
| Charm | `∂²V/∂S ∂t` (delta decay) | How delta changes as time passes |
| Veta | `∂²V/∂σ ∂t` | Vega's time decay |

### 7.3 Third order (the more exotic ones)

- **Speed**: `∂³V/∂S³` — how gamma changes with spot.
- **Color**: `∂³V/∂S² ∂t` — how gamma decays.
- **Ultima**: `∂³V/∂σ³` — third-order vol sensitivity.
- **Zomma**: `∂Γ/∂σ` — gamma's vol sensitivity.

You won't be asked to compute these by hand in interviews, but you should
recognise the names. They matter when you're hedging a complex book.

### 7.4 Practical interpretation

- **Gamma is highest ATM near expiry**. That's where a small spot move
  causes the largest delta change → most rebalance risk.
- **Vega is highest ATM with longer T**. Vol sensitivity grows with time
  to expiry.
- **Theta is most negative for short-dated ATM options** — they decay fast.
- **Vanna lives in skew**: a non-zero vanna means delta changes when vol
  moves, which matters whenever the IV smile is sloping (always, in
  equities).

### 7.5 Our cross-validation (real numbers)

For `S = K = 100, T = 1y, r = 5%, σ = 20%, q = 0`:

| Greek | BS (analytical) | Tree (1000 steps) | MC (200k paths) |
|---|--:|--:|--:|
| Delta | 0.6368 | 0.6367 | 0.6364 ± 0.0005 |
| Gamma | 0.0188 | 0.0188 | 0.0187 ± 0.0008 (CRN bump) |
| Vega | 37.52 | — | 37.50 ± 0.07 (pathwise) |

All three within statistical tolerance — confirms no method has a
sign-flip or scaling bug.

---

## 8. Implied volatility, smile, and vol models

### 8.1 IV definition and inversion

The σ that, plugged into BS, recovers the market price. We invert via
Brent's method (root-finding on a bracketed monotonic function — vega ≥ 0
means BS price is monotone increasing in σ).

### 8.2 Why the smile/skew exists

1. **Fat-tailed real returns**: empirical returns have higher kurtosis
   than log-normal. Markets pay up for tail protection → higher IV in the
   wings.
2. **Stochastic volatility**: σ itself moves, which adds value to options
   beyond constant-vol BS.
3. **Demand for puts**: portfolio insurance buyers structurally demand
   downside puts → put IVs lifted → skew (more pronounced in equity
   indices than single stocks).
4. **Jumps**: discrete price moves (earnings, news) add tail risk that
   BS's continuous diffusion misses.

### 8.3 Smile shapes

- **Equity index** (SPX, SPY, NIFTY): heavy *skew* — strong negative slope
  (lower strikes → higher IV), little upside smile.
- **Equity single-stock**: skew + some upside smile around earnings.
- **FX**: roughly symmetric smile (because FX risk is two-sided).
- **Commodities**: wide smile, often *forward* skew (calls richer than
  puts in some markets — supply shocks).

### 8.4 Vol models beyond BS

#### Local volatility (Dupire)
The unique deterministic σ(S, t) that makes the model reproduce the entire
observed IV surface today. Formula:
> `σ²_loc(K, T) = ∂C/∂T / (½ K² ∂²C/∂K²)`
Pros: matches today's surface exactly. Cons: predicts the smile flattens
unrealistically with time (which empirically it doesn't).

#### Heston (stochastic volatility)
Two-factor model: `S` follows GBM with stochastic σ that itself follows
mean-reverting CIR. Closed form via characteristic function + Fourier
inversion. Captures vol clustering and the "smile dynamics" local vol
misses. Five parameters: long-run vol, mean reversion, vol-of-vol,
correlation, initial vol.

#### SABR
Stochastic-α-β-ρ model. Common in interest-rate / FX markets. Approximate
closed form for IV (Hagan formula). β controls the "vol regime"
(log-normal at β=1, normal at β=0).

#### Jump diffusion (Merton)
Add a Poisson jump component to BS. Tractable in some forms; captures
fat-tail behaviour BS misses.

### 8.5 Smile dynamics (sticky)

When the underlying moves, what happens to the smile? Three idealised
regimes:
- **Sticky strike**: IV at each strike is constant; the smile shape moves
  with spot. Implies large vanna effects.
- **Sticky delta**: IV at each delta is constant; the smile is anchored to
  moneyness, not strike.
- **Sticky moneyness**: IV at each `K/S` is constant.
Real markets are somewhere between sticky strike and sticky delta. This
matters because it determines how to attribute hedging P&L to spot vs vol.

### 8.6 Variance swap pricing

A variance swap pays `realised_variance − strike_variance` at expiry. Its
fair strike has a closed form: `K_var = (2/T) ∫₀^∞ p(K)/K² dK` where
`p(K)` is the OTM option price at strike `K`. So **the entire IV smile
prices a variance swap.** Powerful identity that connects vanilla pricing
to volatility products.

---

## 9. Hedging — the trader's view

### 9.1 The Itô decomposition you must memorise

For a delta-hedged short option:
> `dP&L_short = −θ dt − ½ Γ (dS)²`

Take expectations under realised vol:
> `E[dP&L] ≈ ½ Γ S² (σ²_imp − σ²_real) dt`

Short gamma wins when realised vol comes in below implied. This is the
entire short-gamma trade in one line.

### 9.2 Discrete hedging error

Continuous hedging gives perfect replication (in BS world). Discrete
hedging introduces a residual proportional to `√(rebalance_interval)`. The
trade-off:
- More frequent rebalance → smaller hedging error but higher
  transaction costs.
- The optimal rebalance frequency depends on `σ`, `Γ`, and bid-ask spread.

### 9.3 Higher-order hedging

- **Gamma hedging**: trade other options to flatten Γ. Common for desks
  with large books.
- **Vega hedging**: offset vega risk with longer-dated options or variance
  swaps.
- **Vanna / volga hedging**: relevant when hedging skew exposure
  (FX desks).

### 9.4 Transaction costs

Adds a quadratic-variation-style penalty to hedging P&L. Leland (1985)
showed that with proportional transaction costs, the right hedging volatility
to use is *higher* than the actual volatility — the cost of rebalancing
gets baked in.

### 9.5 Volatility arbitrage

The strategic version of gamma scalping: systematically sell options when
implied is rich relative to forecast realised. Two-sigma simple version:
delta-hedge a short straddle held to expiry, P&L equals
`½ Γ S² (σ²_imp − σ²_real) dt` integrated. Refinements: variance swaps
(linear in σ², no Γ-S² weighting), VIX futures (forward variance).

---

## 10. Option types — the menagerie

| Type | Payoff | Where it shows up |
|---|---|---|
| European call/put | `max(S_T − K, 0)`, `max(K − S_T, 0)` | Index options (SPX, NIFTY) |
| American call/put | Early exercise allowed any time `t ≤ T` | Single-stock options (US) |
| Bermudan | Early exercise on specific dates | Callable bonds, swaptions |
| Asian (arithmetic) | `max(avg(S) − K, 0)` | Commodities, energy |
| Asian (geometric) | `max((∏ S)^(1/n) − K, 0)` | Theory; control variate |
| Barrier (knock-out) | Pays vanilla unless `S` hits barrier | Structured products |
| Barrier (knock-in) | Pays vanilla only if `S` hits barrier | Same |
| Lookback (fixed) | `max(max(S) − K, 0)` | Exotic; few liquid |
| Lookback (floating) | `max(S_T − min(S), 0)` | Same |
| Digital / binary | Pays $1 if `S_T > K`, else 0 | Building block; FX OTC |
| Compound | Option on an option | Real options, exec comp |
| Cliquet / ratchet | Series of forward-starting options | Insurance, structured |
| Forward-start | Strike set at a future date | Vol-targeting products |

Pricing tool of choice (rule of thumb):
- European vanilla → BS closed form.
- American vanilla → binomial tree or PDE.
- Path-dependent (Asian, barrier, lookback) → MC.
- American + path-dependent → Longstaff-Schwartz MC or PDE.

---

## 11. Numerical PDE methods

Used by sell-side trading systems for vanilla and many exotics. Four
flavours:

- **Explicit Euler**: simple, fast per step, conditionally stable
  (Courant condition `dt ≤ ½ dS² / σ² S²` or you blow up).
- **Implicit Euler**: solve a tridiagonal system per step, unconditionally
  stable, first-order in time.
- **Crank-Nicolson**: average of explicit and implicit. Second-order in
  time, unconditionally stable. Industry standard for BS PDE.
- **Operator splitting** (ADI): handle multi-dimensional problems by
  alternating direction implicit. Used for stochastic-vol PDEs.

We don't ship a PDE solver; v2 worth considering if you want to hit
American puts with smoothness around the exercise boundary.

---

## 12. Risk management primer

### 12.1 VaR (Value at Risk)
The α-quantile of the loss distribution over a horizon. Doesn't capture
tail-shape — two portfolios with the same VaR can have very different
expected shortfalls.

### 12.2 CVaR / Expected Shortfall
Expected loss conditional on exceeding VaR. Coherent risk measure (VaR is
not). Now the regulatory standard (FRTB).

### 12.3 Stress testing
Run the book through historical shocks (1987, 2008, 2020) and hypothetical
scenarios. Useful for fat-tail risks the parametric VaR misses.

### 12.4 Greek aggregation
Total portfolio Δ = Σ position_i · Δ_i. Same for Γ, ν, Θ, ρ. Trading
desks live or die by their portfolio Greek dashboard.

---

## 13. Microstructure (light touch)

- **Bid-ask spread**: cost of round-trip immediacy. Typically 1-3 ticks
  for liquid options, much wider for illiquid wings.
- **Volume / Open Interest**: volume = today's trades; OI = open contracts.
  OI spikes near earnings or expiry.
- **Pin risk**: at expiry, options near the strike can pin → unhedgeable
  delta jump for short positions. Real risk on monthly expiries.
- **Settlement**: cash-settled (SPX) vs physical (single-stock). Matters
  for early exercise considerations.
- **Liquidity tiers**: nearer expiries and ATM strikes are more liquid;
  far OTM and longer-dated less so.

---

## 14. Real-world adjustments

### 14.1 Dividends
- **Continuous yield** (`q`): SPX, indices. Plug into BS directly.
- **Discrete cash**: single stocks. Requires modelling each ex-dividend
  date as a deterministic price drop.
- **Forecast vs known**: known dividends are deterministic; forecasts have
  uncertainty.

### 14.2 Funding rates
Post-2008, the "risk-free rate" is OIS (USD: SOFR; previously LIBOR/Fed
funds). For collateralised trades, OIS-discount; for uncollateralised, use
the funding curve (FVA = funding valuation adjustment).

### 14.3 Borrow / repo
Short-selling a stock requires borrowing it; the borrow rate (negative for
hard-to-borrow names) affects forward construction. For an index it's
small; for a single stock at the wrong time, it's huge.

### 14.4 Forward construction
`F = S e^{(r − q − borrow)T}`. The forward, not spot, is the right
underlying for option pricing (the "Black formula" prices options on
forwards directly).

### 14.5 Curve building
Bootstrapping a yield curve from money-market deposits, FRA, swaps, and
basis swaps. Required for any non-trivial multi-currency or multi-maturity
pricing.

---

## 15. Singapore-relevant context

- **Index options**: STI (Straits Times Index) options on SGX; relatively
  thin compared to SPX.
- **Cross-listed activity**: SGX historically traded NIFTY futures
  (since relisted), Nikkei 225 futures, MSCI Asia. Asian-hours liquidity
  for global products is the SGX selling point.
- **Major MM firms in SG**: Optiver, IMC, Tower Research, Jump Trading,
  Jane Street (Hong Kong office services SG). All are options
  market-makers and all interview for entry-level QT and QD seats.
- **Buy-side**: Squarepoint, Two Sigma, Citadel, Millennium, Dymon, Tudor.
  More QR / systematic-strategy roles.
- **Banks**: DBS, OCBC, UOB (rates / risk), GS / MS / JPM (Asia derivatives).

For your interviews specifically: the MM seats will probe options
pricing + Greeks + hedging math hard. The systematic seats will probe
factor-style alpha and statistical methods. This project leans MM — the
gamma-scalp story is exactly the right signal there.

---

## 16. Where every concept above lives in our code

| Concept | Module | Test |
|---|---|---|
| GBM path simulation | `pricing/monte_carlo.py:gbm_paths` | `test_monte_carlo.py` |
| BS PDE solution (closed form) | `pricing/black_scholes.py:price` | `test_black_scholes.py` |
| All BS Greeks | `pricing/black_scholes.py:{delta,gamma,vega,theta,rho}` | `test_black_scholes.py` |
| Put-call parity check | `test_black_scholes.py:test_put_call_parity` | — |
| CRR tree, American + European | `pricing/binomial.py:price` | `test_binomial.py` |
| Tree-based Greeks | `pricing/binomial.py:price_with_greeks` | `test_binomial.py` |
| Antithetic MC | `pricing/monte_carlo.py:price_european` | `test_monte_carlo.py` |
| Control variates (geometric Asian) | `pricing/monte_carlo.py:price_arithmetic_asian` | `test_monte_carlo.py` |
| Kemna-Vorst closed form | `pricing/monte_carlo.py:geometric_asian_closed_form` | `test_monte_carlo.py` |
| Pathwise MC Greeks | `greeks.py:mc_pathwise_delta`, `mc_pathwise_vega` | `test_greeks.py` |
| CRN bump-and-revalue | `greeks.py:mc_gamma_crn_bump` | `test_greeks.py` |
| Three-method cross-validation | `greeks.py:cross_validate_european` | `test_greeks.py` |
| Brent IV inversion | `iv.py:implied_vol` | `test_iv.py` |
| Smile and term-structure helpers | `iv.py:smile_at_expiry`, `term_structure_atm` | `test_iv.py` |
| Itô-based hedging P&L decomposition | `hedging.py:simulate_short_straddle_path` | `test_hedging.py` |
| Distribution across regimes | `hedging.py:simulate_distribution` | `test_hedging.py` |
| Real chain ingestion + cleaning | `data.py:fetch_spy_chain`, `clean_chain` | `test_data.py` |
| Realised vol estimator | `data.py:realised_volatility` | `test_data.py` |

If you understand each row of this table, you can defend every line in the
codebase to an interviewer.
