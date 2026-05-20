# Study questions and concise answers

A categorised Q&A reference covering every concept this framework
implicates, plus adjacent material commonly probed in derivatives
interviews. Answers are short and dense; deeper exposition for any
concept lives in `concepts_reference.md`.

Categories:
- A. Stochastic calculus and probability measures
- B. Black-Scholes and the pricing PDE
- C. Numerical methods (trees, Monte Carlo, PDE)
- D. Greeks (first, second, higher order)
- E. Implied volatility and vol models
- F. Hedging
- G. Option types — vanilla and exotic
- H. Risk management and microstructure
- I. Project-specific

---

## A. Stochastic calculus and probability measures

**Q: What's a Brownian motion / Wiener process?**
A: A continuous stochastic process with `W_0 = 0`, normal increments
(`W_t − W_s ~ N(0, t−s)`), independent across non-overlapping intervals,
and continuous-but-nowhere-differentiable paths. Its variance grows
linearly with time, which is why option vol scales as `σ√T`.

**Q: State Itô's lemma and explain why it has an extra term vs ordinary
calculus.**
A: For `dX = μ dt + σ dW` and `f = f(X, t)`:
`df = (∂f/∂t) dt + (∂f/∂X) dX + ½ (∂²f/∂X²) (dX)²`. The third term is
new because `(dW)² = dt` rather than zero — Brownian paths have non-zero
quadratic variation. Ordinary calculus drops `(dx)²` because for smooth
paths it's `O(dt²)`; for Brownian paths it's `O(dt)`.

**Q: What's the difference between the real-world and risk-neutral
measures?**
A: Real-world (P) has actual drift `μ` reflecting risk preferences.
Risk-neutral (Q) is a constructed measure under which every tradeable
asset (after dividend yield) drifts at `r`. Pricing happens under Q
because the discounted price of any tradeable is a Q-martingale —
equivalent to no-arbitrage.

**Q: What does Girsanov's theorem give you?**
A: A tool to switch between measures. Specifically:
`dW^P = dW^Q + λ dt` where `λ = (μ − r) / σ` is the market price of
risk. **Switching measure shifts drift, not volatility.** That's why σ
is the same under P and Q, and why we can estimate σ from real prices and
use it in Q-pricing.

**Q: What's the Feynman-Kac formula?**
A: A bridge between PDEs and expectations. The solution `V(S, t)` of a
parabolic PDE of BS form with terminal condition `payoff(S)` equals
`E^Q[e^{-r(T-t)} payoff(S_T) | S_t = S]`. So pricing by PDE and pricing
by MC give the same answer — they're two computational windows on the
same object.

**Q: Why is the discounted price of a tradeable asset a Q-martingale?**
A: It's the no-arbitrage condition expressed mathematically. If
`E^Q[e^{-rT} S_T] = S_0` (the Q-martingale property), then there's no
trading strategy that produces a riskless profit. Conversely, the
existence of an equivalent martingale measure is equivalent to absence
of arbitrage (Fundamental Theorem of Asset Pricing).

**Q: What's quadratic variation and why does it matter for options?**
A: For Brownian motion, `[W,W]_T = T`. For GBM, the quadratic variation
of `ln S_t` equals `σ²T` — i.e., realised variance is literally the
quadratic variation of log-prices. That's why σ enters pricing as `σ²T`,
and why variance swaps pay realised quadratic variation.

---

## B. Black-Scholes and the pricing PDE

**Q: Walk me through the Black-Scholes formula.**
A: Under Q, the underlying follows GBM `dS = (r-q)S dt + σS dW`. The
price of a European call is the discounted Q-expectation of its payoff,
which integrates to `C = S e^{-qT} N(d₁) − K e^{-rT} N(d₂)` with
`d₁ = [ln(S/K) + (r-q+σ²/2)T] / (σ√T)` and `d₂ = d₁ − σ√T`. `N(d₂)` is
the risk-neutral probability of exercise; `N(d₁)` is delta.

**Q: Derive the BS PDE.**
A: Set up a self-financing replicating portfolio: long one option, short
`Δ` shares. The portfolio's Itô expansion has a `dW` term equal to
`(∂V/∂S − Δ) dW`. Choosing `Δ = ∂V/∂S` kills the random component. The
remaining drift, by no-arbitrage, must equal `r × portfolio_value`.
Rearranging gives:
`∂V/∂t + (r-q)S ∂V/∂S + ½σ²S² ∂²V/∂S² − rV = 0`.

**Q: State put-call parity. Why does it always hold?**
A: `C − P = S e^{-qT} − K e^{-rT}`. It's model-free — derived from a
static replication argument: long call + short put + cash position
= forward = stock minus dividends. Independent of model assumptions.

**Q: What's the most violated assumption in BS, and how does the market
respond?**
A: Constant volatility. The market's response is the implied-vol smile:
prices of OTM options imply higher σ than ATM, especially on the put
side. Reflects fat tails, stochastic vol, jumps, and structural demand
for downside puts.

**Q: Why is N(d₁) delta and not N(d₂)?**
A: `N(d₂)` is the *probability* the call expires ITM. `N(d₁)` is the
*marginal contribution of S* to the price — exactly `∂C/∂S`, by direct
differentiation. Confusing them is the most common BS mistake.

**Q: What's a self-financing portfolio?**
A: A trading strategy where rebalancing requires no external cash
injections — gains from one leg fund the changes in the other. The BS
delta hedge is self-financing in continuous time.

**Q: How would you price an option on a forward (Black's formula)?**
A: Same form as BS but the underlying is the forward `F`, not spot.
`C = e^{-rT} [F · N(d₁) − K · N(d₂)]` with `d₁ = [ln(F/K) + ½σ²T] / (σ√T)`.
The `(r-q)` drift disappears because the forward is already adjusted.

---

## C. Numerical methods

**Q: Why three pricing methods, not just one?**
A: Each has a unique strength. BS is closed-form for European vanillas —
instant, exact. Binomial trees handle American options because we can
compare continuation vs immediate exercise at every node. Monte Carlo
handles path-dependent payoffs (Asians, lookbacks) where trees can't
recombine. Right tool for the right job.

**Q: What's the convergence rate of CRR? Where does it break?**
A: O(1/N) for European options — error halves when N doubles, confirmed
empirically (0.0200 at N=100, 0.0020 at N=1000). It breaks for
discontinuous payoffs (digitals, barriers near a barrier): you get
oscillation rather than smooth convergence. Mitigations: smoothing
schemes, finer trees near the discontinuity, or trinomial.

**Q: What's the convergence rate of Monte Carlo and why?**
A: O(1/√N). The variance of a sample mean is `σ²/N`, so SE = `σ/√N`.
You need 4× the paths to halve the SE.

**Q: Explain antithetic variates.**
A: For every random draw `Z`, also use `−Z`. The pair has the same
expected payoff but lower variance whenever the payoff is monotonic in
`Z` (true for vanilla calls and puts). **Crucial bookkeeping**: SE must
be computed from pair-means, not from raw 2N samples — otherwise the
variance reduction is hidden by your own accounting. I caught and fixed
that bug in my test suite.

**Q: What's a control variate?**
A: A correlated random variable whose true mean we know in closed form.
Estimator: `V̂_target − β · (V̂_control − E[V_control])`, with optimal
`β = Cov(target, control) / Var(control)`. For arithmetic Asian options I
use the geometric Asian as control — it has a Kemna-Vorst closed form
and ~99% correlation with the arithmetic. Empirical variance reduction
5-50× at the same path count.

**Q: What about other variance reduction techniques?**
A: (1) Importance sampling — sample under a measure that puts more
weight where the payoff is non-zero. (2) Stratified sampling — partition
input space, sample fixed counts per stratum. (3) Quasi-Monte Carlo
(Sobol, Halton sequences) — convergence improves to `O((log N)^d / N)`.
(4) Control variates as above.

**Q: How do you compute Greeks via Monte Carlo?**
A: Three estimators. (1) **Pathwise**: differentiate the payoff w.r.t.
the parameter before averaging. Lower variance, requires payoff
smoothness. Works for delta and vega on smooth payoffs. (2)
**Likelihood ratio**: differentiate the density. Works for non-smooth
payoffs (digitals). (3) **Common-random-number bump-and-revalue**: same
Z's for bumped and unbumped runs so noise cancels in the difference.
General-purpose. I use pathwise for delta/vega and CRN for gamma.

**Q: How do you price an American option in Monte Carlo (Longstaff-
Schwartz)?**
A: Forward-simulate paths, then backward-induct. At each exercise date,
regress the continuation value (next-step discounted) on a polynomial
basis of `S_t`. Compare regression's prediction to immediate exercise
value at each node; exercise where exercise > continuation. The clever
bit is turning forward MC into backward induction.

**Q: What about PDE methods (finite difference)?**
A: Discretise the BS PDE on a grid in (S, t) and step backwards from
expiry. Three flavours: explicit (simple, conditionally stable),
implicit (unconditionally stable, first-order), Crank-Nicolson
(second-order, unconditionally stable, industry standard for BS).
Useful for American options because you can apply the early-exercise
constraint at each grid node naturally. Industry trading systems
typically use Crank-Nicolson + projected SOR for American options.

**Q: When would you choose a tree vs PDE vs MC?**
A: Vanilla European → closed form (no question). Vanilla American →
binomial or PDE (PDE is smoother near the early-exercise boundary).
Path-dependent → MC. American + path-dependent → Longstaff-Schwartz MC
or 2D PDE if the path-dependence is summary-state (e.g., running
average). High-dimensional basket options → MC always.

**Q: What's the Kemna-Vorst formula?**
A: Closed form for the geometric-Asian option. Because the geometric
mean of GBM is itself log-normal, the price is BS-like with
`σ_G = σ/√3` and an effective dividend yield. Useful as a control
variate for MC pricing of arithmetic Asians.

---

## D. Greeks

**Q: Define the first-order Greeks and give a one-line interpretation.**
A: **Delta** `∂V/∂S`: first-order spot sensitivity, range [0,1] calls,
[-1,0] puts. **Vega** `∂V/∂σ`: vol sensitivity. **Theta** `∂V/∂t`: time
decay (negative for long, positive for short). **Rho** `∂V/∂r`: rate
sensitivity, usually small for short-dated.

**Q: Define gamma and explain when it's largest.**
A: `Γ = ∂²V/∂S²`. Largest for ATM options near expiry — that's where
small spot moves cause the largest changes in delta. A short-gamma
position is a short-convexity position: collect when markets are quiet,
pay when they move.

**Q: Name three second-order Greeks and what each measures.**
A: **Vomma / volga** = `∂²V/∂σ²`: vol-of-vol exposure (curvature in σ).
**Vanna** = `∂²V/∂S ∂σ`: how delta changes with vol — relevant to skew
hedging. **Charm** = delta decay = `∂²V/∂S ∂t`: how delta moves as time
passes, important for overnight risk.

**Q: Have you heard of higher-order Greeks like Speed or Color?**
A: Yes — Speed is `∂Γ/∂S` (third-order in spot), Color is `∂Γ/∂t`
(gamma's time decay). They matter on large books or when hedging
complex exotic positions; for vanilla projects they're more vocabulary
than computation.

**Q: How do you validate Greek computations?**
A: Independent confirmation across methods. In my project I compute
delta, gamma, vega three ways (BS analytical, binomial tree, MC
pathwise/CRN) and assert they agree within statistical tolerance. If
two of three methods agree and the third disagrees, the third has the
bug. If all three disagree, your test fixture is wrong.

**Q: Why is gamma identical for European calls and puts?**
A: Put-call parity: `C − P = S e^{-qT} − K e^{-rT}` is linear in `S`,
so the second derivative of both sides w.r.t. `S` gives `Γ_C = Γ_P`.
Same reasoning gives equal vega.

**Q: What's a delta-neutral portfolio?**
A: One whose total delta is zero. First-order insensitive to spot moves.
But it's not risk-free — it's still exposed to gamma (curvature),
vega (vol), and theta (time).

**Q: How does delta change as the option becomes deep-ITM or deep-OTM?**
A: Deep-ITM call delta → 1 (option behaves like the stock). Deep-OTM
call delta → 0 (no exposure to the underlying). The transition through
the strike is where gamma lives — that's the curvature region.

---

## E. Implied volatility and vol models

**Q: What is implied volatility?**
A: The σ that, plugged into BS, recovers the market price. Markets quote
options in IV rather than dollars because IV is the only parameter not
directly observable (S, K, T, r are given). It captures the market's
view on future volatility.

**Q: Why does the IV smile/skew exist?**
A: Four reasons. (1) Fat-tailed real returns — actual returns have more
extreme moves than log-normal predicts. (2) Stochastic volatility — σ
itself moves; vol-of-vol commands a premium. (3) Jumps — discrete moves
that BS's continuous diffusion misses. (4) Demand for puts — portfolio
insurance buyers structurally demand downside puts, lifting put IVs.

**Q: How do you compute IV from a market price?**
A: Brent's method (root-finding) on `f(σ) = BS_price(σ) − market_price`.
Brent converges super-linearly without derivatives, and BS price is
monotone increasing in σ (vega ≥ 0), so the bracket `[ε, 5]` is
guaranteed to contain the root. Failure modes: market price below
intrinsic (return NaN — arb violation), or near-zero vega tails
(numerical instability).

**Q: What's the difference between local vol and stochastic vol?**
A: **Local vol** (Dupire): σ is a deterministic function of `S` and
`t`. The unique σ(S,t) that reproduces today's IV surface exactly. Pros:
matches today's market. Cons: predicts the smile flattens unrealistically
with time. **Stochastic vol** (Heston, SABR): σ itself is a random
process. Captures vol-clustering and smile dynamics local vol misses.
Cons: more parameters, no closed form for arbitrary payoffs.

**Q: What is the Heston model?**
A: A two-factor stoch-vol model. `dS = (r-q)S dt + √v · S dW^S` and
`dv = κ(θ − v) dt + ξ √v dW^v` with `Corr(dW^S, dW^v) = ρ`. Five
parameters: long-run vol `θ`, mean reversion `κ`, vol-of-vol `ξ`,
correlation `ρ`, initial vol `v_0`. Has a closed form via Fourier
inversion of the characteristic function. Industry-standard for FX,
equity index. Captures negative correlation (`ρ < 0` matches equity
skew).

**Q: What is SABR?**
A: Stochastic-α-β-ρ. Common in interest-rate and FX markets. Has an
approximate closed form for IV (Hagan formula). β controls the regime:
β=1 is log-normal (BS-like), β=0 is normal (Bachelier-like). Fits a
smile with four parameters per slice.

**Q: What's the relationship between IV smile and variance swap
strikes?**
A: `K_var = (2/T) ∫₀^∞ p(K)/K² dK` where `p(K)` is the OTM option price.
The entire smile prices a variance swap. Practical implication: you can
construct a variance swap by holding a portfolio of OTM options with
weights inversely proportional to `K²`.

**Q: Sticky strike vs sticky delta — why does it matter?**
A: When spot moves, what happens to the smile? Sticky strike: IV at each
strike is constant; the smile shape moves with spot. Sticky delta: IV at
each delta is constant; the smile is anchored to moneyness. Sticky
moneyness: IV at each `K/S` is constant. Real markets are between
sticky strike and sticky delta. This determines vanna effects and
attribution of P&L to spot vs vol.

---

## F. Hedging

**Q: Derive the per-step P&L of a delta-hedged short option.**
A: By Itô, `dV = θ dt + δ dS + ½γ (dS)²`. For a short position with
hedge `+δ` shares, `dP&L = -dV + δ dS = -θ dt − ½γ (dS)²`. So short
positions collect theta and pay realised gamma. Expected daily P&L
under realised vol: `E[dP&L] ≈ ½γ S² (σ²ᵢₘₚ − σ²ᵣₑₐₗ) dt`. Short
gamma wins when realised vol comes in below implied.

**Q: What's the difference between continuous and discrete hedging?**
A: Continuous gives perfect replication in the BS world (zero hedging
error). Discrete introduces a residual proportional to `√(rebalance_dt)`
from higher-order Itô-Taylor terms. Real desks balance hedging error
against transaction costs.

**Q: How would you hedge gamma?**
A: Trade other options. To make a portfolio gamma-neutral, add `−Γ_p / Γ_h`
units of a hedging option (typically the same expiry, ATM for max
gamma). This keeps you delta-and-gamma neutral but transfers the
exposure to vega and higher Greeks.

**Q: How do you hedge vega?**
A: With other options at different expiries (vega is not a property of
the underlying). For a calendar-vega hedge, you'd offset front-month
vega against back-month vega. Variance swaps are also a clean vega
hedge — they pay realised variance directly.

**Q: How do transaction costs change optimal hedging?**
A: Leland (1985): with proportional bid-ask costs, the right
volatility to use in your hedging delta is *higher* than actual σ —
this bakes the cost of rebalancing into the price. Practically: hedge
less frequently, or only when delta moves outside a band.

**Q: What's whipsaw / gap risk?**
A: A jump in the underlying that's larger than your rebalance interval
can absorb. Realised gamma cost spikes; your discrete hedge is
mispositioned at the gap. Mitigations: jump-aware hedging (hedge tail
risk separately), volatility floors, position size limits.

**Q: Explain the gamma-scalp trade.**
A: Sell options, delta-hedge daily. Each day collect theta. Each day
pay realised gamma against the underlying's realised quadratic
variation. P&L per day ≈ `½γ S² (σ²ᵢₘₚ − σ²ᵣₑₐₗ) dt`. Win when
realised vol comes in below implied; lose when it doesn't. My
simulator confirms this: at σ_imp=30% / σ_real=15%, every single one
of 400 paths makes money.

**Q: What's volatility arbitrage?**
A: Strategically expressing a forecast that realised vol will diverge
from implied. Two-sigma simple form: short straddle, delta-hedge to
expiry. More refined: variance swaps (linear in σ², cleaner exposure),
VIX futures (forward variance).

---

## G. Option types

**Q: What's an Asian option and why is it harder to price than a
European?**
A: Asian option's payoff depends on the average price over a window,
not just the terminal price. Path-dependent → trees can't recombine →
Monte Carlo is the natural method. Two flavours: arithmetic average
(no closed form) and geometric average (closed form via Kemna-Vorst,
because geom of GBM is log-normal). I price arithmetic Asians with MC
+ control variates.

**Q: What's the difference between a European, American, and Bermudan
option?**
A: European: exercise only at expiry. American: exercise any time `t ≤ T`.
Bermudan: exercise on a finite set of dates. Pricing complexity:
European cheapest, American most complex, Bermudan in between.

**Q: When is it ever optimal to early-exercise an American call?**
A: On a non-dividend-paying stock, *never* (Merton's theorem) — better
to sell the call than exercise. With dividends or financing carry, it
can be optimal just before an ex-dividend date if the dividend exceeds
the time value lost. American puts can always be optimally exercised
early when deep-ITM (interest on the strike makes immediate exercise
attractive).

**Q: What's a barrier option?**
A: An option that activates (knock-in) or extinguishes (knock-out)
when the underlying touches a barrier level. Cheaper than a vanilla
because the optionality is conditional. Pricing: closed-form under BS,
otherwise PDE or MC. Hedging is unstable near the barrier (gamma
spikes).

**Q: What's a digital option?**
A: Pays `$1` if `S_T > K`, else 0 (cash-or-nothing) or pays `S_T` if
`S_T > K` (asset-or-nothing). Discontinuous payoff makes pathwise
Greek estimators fail; use likelihood-ratio for MC.

**Q: What's a lookback option?**
A: Payoff depends on the max or min of the underlying over the option
life. Fixed-strike: `max(max_S − K, 0)`. Floating-strike:
`max(S_T − min_S, 0)`. Path-dependent → MC.

---

## H. Risk management and microstructure

**Q: What's VaR and what's its main limitation?**
A: Value at Risk: the α-quantile of the loss distribution over a
horizon. Limitation: doesn't capture tail shape — two portfolios with
same VaR can have very different expected shortfalls. Also fails to be
a coherent risk measure (sub-additivity can fail).

**Q: What's CVaR / Expected Shortfall?**
A: Expected loss conditional on exceeding VaR. Coherent risk measure.
Now the regulatory standard under FRTB.

**Q: What's pin risk?**
A: At expiry, options very near the strike can pin → unhedgeable delta
jump for short positions (delta is 0 if just OTM, 1 if just ITM, with
no time to rebalance through the discontinuity). Real risk on monthly
expiries.

**Q: What's the difference between cash and physical settlement?**
A: Physical: at exercise, you actually deliver/receive the underlying.
Cash: you pay/receive the difference. Affects early-exercise calculations
(physical American calls on dividend-payers might be exercised just before
ex-div) and pin risk near expiry.

**Q: What does open interest tell you that volume doesn't?**
A: Volume = today's trades. Open interest = total open contracts. OI
spikes near earnings or expiry. A high-volume / low-OI day suggests
day-trading; high volume + rising OI suggests new positioning.

---

## I. Project-specific

**Q: Walk me through your project.**
A: I built an options pricing and hedging research framework on
SPY/SPX. Three pricers — Black-Scholes analytical, CRR binomial tree,
Monte Carlo with antithetic + control variates — each applied to the
option type where it earns its place: BS for European, binomial for
American (early exercise), MC for Asian (path-dependent). I compute
Greeks three ways and cross-validate. I fit the IV surface from real
SPY chains via Brent inversion. And I built a daily-rebalanced
short-straddle gamma-scalp simulator that empirically reproduces the
textbook short-gamma result. 54-test pytest suite, Streamlit dashboard,
deployed end-to-end.

**Q: Why SPY rather than another underlying?**
A: SPY has the cleanest IV smile of any liquid equity-index option and
the deepest free historical chain data, so the convergence and
smile-fitting stories are sharper. The methodology generalises to other
indices (NIFTY, Nikkei, HSI) without modification.

**Q: Why a short straddle for the hedging sim?**
A: A short straddle is the canonical gamma-scalp setup: ATM call + ATM
put, delta near zero at inception, maximum gamma. P&L decomposes most
cleanly into theta vs gamma. Other setups (covered call, condor) add
either underlying exposure or wing exposure that muddy the gamma
story.

**Q: What was the most surprising thing you learned?**
A: The antithetic-variates standard-error bookkeeping bug. The price
estimate was correct but the SE was wrong because I treated 50k
antithetic-paired samples as if they were 50k independent draws. The
variance reduction was hidden by my own accounting. Fixing the SE to
compute from pair-means restored the textbook 30-50% SE reduction at
the same path count. Subtle and instructive.

**Q: What's the biggest limitation of your framework?**
A: Two: (1) GBM as the data-generating process — no jumps, no
stochastic vol, no microstructure. The gamma-scalp story is
approximately right under GBM but real markets jump. (2) Daily
rebalance — real desks hedge faster and trade off transaction costs
explicitly. v2: Heston-vol the underlying and add an intraday rebalance
comparison.

**Q: What would you do next?**
A: Three concrete extensions, ranked. (1) Heston stochastic-vol
underlying — does the gamma-scalp story survive when σ is itself
moving? (2) Compare daily / 4h / hourly rebalance frequencies and back
out the hedging-error vs transaction-cost trade-off (Leland-style). (3)
Real-data backtest: roll a monthly short ATM SPY straddle 2018-2024,
daily-hedge, measure realised IV vs RV and compare to simulator.

**Q: How would you extend this to American options pricing?**
A: I already do — the binomial tree handles American natively. To go
deeper: PDE solver with projected SOR, or Longstaff-Schwartz MC if I
also need path-dependence. The framework's pricing/binomial.py is the
hook; adding PDE is a clean module addition.

**Q: How did you verify your code is correct?**
A: 54-test pytest suite. Pin against textbook BS values (Hull worked
example). Verify convergence rates: binomial O(1/N), MC O(1/√N).
Cross-validate Greeks across BS, binomial, MC — three independent
methods agreeing is evidence of correctness. Check put-call parity. For
the hedging sim: verify that with σ_imp = σ_real, mean P&L is
statistically zero across 200 paths (3-sigma test); with σ_imp ≠ σ_real,
P&L direction matches the textbook formula.

---

## How to study from these notes

1. Read `concepts_reference.md` cover-to-cover for the deeper exposition
   behind each answer here.
2. For each question above, attempt the answer without looking back at
   the source. Categories A (stochastic calculus) and E (vol models) are
   the ones most worth drilling for first-time derivatives interviews.
3. Project-specific answers (section I) are the layer where personal
   narrative meets the codebase — write your own version of those for
   the project you are presenting.
