"""Options pricing and delta-hedging research framework.

Top-level package. Subpackages:
- options.pricing : Black-Scholes, binomial tree, Monte Carlo pricers.
- options.greeks  : Greeks computation + cross-validation.
- options.iv      : implied-volatility solver and surface fitting.
- options.hedging : delta-hedging simulator (short straddle).
- options.data    : market data ingestion (SPY chains + spot).
- options.utils   : shared utilities.
"""

__version__ = "0.1.0"
