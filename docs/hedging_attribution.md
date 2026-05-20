# Hedging attribution: short straddle gamma-scalp

For a delta-hedged short option position, instantaneous P&L decomposes as:

> dP&L ≈ −θ · dt − ½ · Γ · (dS)²

In expectation under the realised vol process:

> E[dP&L] ≈ ½ · Γ · S² · (σ²ᵢₘₚ − σ²ᵣₑₐₗ) · dt

So a short ATM straddle (which is short Γ) **wins when realised vol comes in
below implied** and loses in the opposite regime. The simulator confirms
this empirically.

## Setup

- 30-day ATM short straddle on a $100 underlying.
- Daily rebalance (30 hedges over 30 days).
- 400 simulated paths per regime.
- `σ_implied` is the vol we sold the straddle at (drives the initial premium and
  the delta we use for hedging).
- `σ_realised` is the vol that drives the actual price path.

## Three regimes

| Regime | σ_imp | σ_real | Mean P&L | Win rate | Std P&L |
|---|--:|--:|--:|--:|--:|
| Neutral | 20% | 20% | **$0.02** | 51% | $0.73 |
| Sold rich vol | 30% | 15% | **+$3.39** | 100% | $1.05 |
| Sold cheap vol | 15% | 30% | **−$3.34** | 0% | $2.04 |

The neutral regime returns near-zero on average, with all the variability
coming from discretisation noise (we hedge daily, not continuously). When we
sell at 30% IV and realised vol prints at 15%, **every single path makes
money** — the textbook short-gamma trade. When we sell cheap and realised
vol turns up, every single path loses.

![hedge sim](../results/hedge_sim_distribution.png)

## Attribution per regime (representative path)

The simulator decomposes per-step P&L into theta and gamma terms following the
Itô identity above. In aggregate over the 30-day path:

- **Theta term** (`−θ · dt` summed) — the time decay we collect for being short.
- **Gamma term** (`−½Γ(dS)²` summed) — the realised-vol cost we pay.
- **Residual** — discretisation error from rebalancing on a daily grid rather
  than continuously. Typically a few dollars on a $100 underlying.

The textbook closed form `Vega · (σ_imp − σ_real)` (using ATM straddle vega ≈
2× ATM call vega) gives $5.55 for the rich-vol regime, vs the simulator's
$3.39. The simulator's number is *lower* because (a) we rebalance discretely,
not continuously, and (b) gamma decays slightly as the option moves OTM/ITM
during the path. Both numbers tell the same story: short gamma + falling
realised vol = positive expected P&L.

## What this proves on a resume

1. Understanding of **why** market-makers sell options when implied vol looks
   rich relative to realised — and why they hedge.
2. The empirical link between `σ_imp − σ_real` and short-gamma P&L, with
   supporting attribution.
3. A simulator that lets you stress-test the regime — how does the P&L
   distribution shrink/widen with rebalancing frequency, with vol-of-vol,
   with gap risk?

## What we would extend in v2

- Stop-loss / trailing P&L to model real-world straddle management.
- Stochastic volatility (Heston) underlying — does the gamma-scalp story
  survive when realised vol itself is moving?
- Intraday rebalancing comparison — daily vs 4h vs hourly hedge errors.
- Real-data backtest: roll a monthly short ATM SPY straddle, hedge daily,
  measure realised P&L attribution against historical IV vs RV.
