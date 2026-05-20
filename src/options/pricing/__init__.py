"""Pricing engines.

Each module exposes a small functional API rather than a class hierarchy:
the goal is composability and easy vectorisation, not OOP for its own sake.

Conventions used across pricers:
- S : spot price (float or array)
- K : strike (float or array)
- T : time to expiry in years (float)
- r : continuously-compounded risk-free rate (float)
- q : continuous dividend yield (float, default 0.0)
- sigma : annualised volatility (float)
- option_type : 'call' or 'put'
"""
