"""Streamlit dashboard for the options-pricing-research framework.

Five tabs:
1. Pricing calculator — price + Greeks across BS / binomial / MC, with a
   convergence plot showing tree → BS and MC → BS.
2. IV surface viewer — pulls a live SPY chain via yfinance, shows the smile
   at a chosen expiry and the ATM term structure.
3. Hedge sim — runs a short ATM straddle gamma-scalp simulation across many
   paths and shows the P&L distribution + theta/gamma attribution.
4. Payoff diagrams — long call, long put, short straddle profit profiles
   at expiry across spot.
5. P&L heatmap — option P&L across a 2D grid of spot × volatility for a
   chosen long position.

Notes for self / for whoever maintains this:
- Do NOT use @st.cache_data with dict arguments — silent blank-page bug on
  Streamlit 1.57. Cache on primitive args only.
- Use width="stretch" (Streamlit 1.57+); use_container_width=True is deprecated.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make the src/ package importable when run via `streamlit run app/streamlit_app.py`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from options import iv as iv_mod
from options.greeks import cross_validate_european
from options.hedging import simulate_distribution
from options.pricing import binomial as bn
from options.pricing import black_scholes as bs
from options.pricing import monte_carlo as mc

st.set_page_config(
    page_title="Options Pricing & Hedging",
    page_icon="📈",
    layout="wide",
)

st.title("Options Pricing & Delta-Hedging Research")
st.caption(
    "Three pricers (Black-Scholes, CRR binomial, Monte Carlo) with cross-validated "
    "Greeks, IV surface fitting on real SPY data, and a short-straddle gamma-scalp "
    "simulator."
)

tab_pricing, tab_iv, tab_hedge, tab_payoff, tab_heatmap = st.tabs([
    "💰 Pricing & Greeks",
    "🌋 IV Surface (SPY)",
    "🎯 Hedge Sim",
    "📉 Payoff Diagrams",
    "🔥 P&L Heatmap",
])


# ---------------------------------------------------------------------------
# Tab 1 — Pricing & Greeks
# ---------------------------------------------------------------------------
with tab_pricing:
    st.subheader("Side-by-side: Black-Scholes vs Binomial Tree vs Monte Carlo")
    col1, col2, col3 = st.columns(3)
    with col1:
        S = st.number_input("Spot S", value=100.0, step=1.0)
        K = st.number_input("Strike K", value=100.0, step=1.0)
    with col2:
        T = st.number_input("Time to expiry T (years)", value=0.5, step=0.05, min_value=0.0)
        r = st.number_input("Risk-free rate r", value=0.04, step=0.005)
    with col3:
        sigma = st.number_input("Volatility σ", value=0.20, step=0.01, min_value=0.0)
        q = st.number_input("Dividend yield q", value=0.0, step=0.005)
    option_type = st.radio("Option type", ("call", "put"), horizontal=True)

    if T > 0:
        out = cross_validate_european(
            S=S, K=K, T=T, r=r, sigma=sigma, q=q,
            option_type=option_type,
            n_paths=80_000, n_steps_tree=500, seed=7,
        )

        rows = [
            {
                "Method": "Black-Scholes (analytical)",
                "Price": out["analytical"]["price"],
                "Δ Delta": out["analytical"]["delta"],
                "Γ Gamma": out["analytical"]["gamma"],
                "ν Vega": out["analytical"]["vega"],
                "Θ Theta": out["analytical"]["theta"],
                "ρ Rho": out["analytical"]["rho"],
            },
            {
                "Method": "Binomial tree (CRR, 500 steps)",
                "Price": out["tree"]["price"],
                "Δ Delta": out["tree"]["delta"],
                "Γ Gamma": out["tree"]["gamma"],
                "ν Vega": np.nan,
                "Θ Theta": np.nan,
                "ρ Rho": np.nan,
            },
            {
                "Method": f"Monte Carlo (80k paths, ±{1.96*out['mc']['price_se']:.3f} 95% CI)",
                "Price": out["mc"]["price"],
                "Δ Delta": out["mc"]["delta"],
                "Γ Gamma": out["mc"]["gamma"],
                "ν Vega": out["mc"]["vega"],
                "Θ Theta": np.nan,
                "ρ Rho": np.nan,
            },
        ]
        st.dataframe(pd.DataFrame(rows).set_index("Method").round(4), width="stretch")

        st.markdown("#### Convergence: binomial steps → BS, MC paths → BS")
        bs_truth = float(bs.price(S, K, T, r, sigma, q, option_type))
        steps_grid = [10, 25, 50, 100, 250, 500, 1000, 2000]
        tree_errs = [
            abs(bn.price(S, K, T, r, sigma, q, option_type, "european", n) - bs_truth)
            for n in steps_grid
        ]
        paths_grid = [1_000, 5_000, 25_000, 100_000, 400_000]
        mc_errs = [
            abs(mc.price_european(S, K, T, r, sigma, q, option_type, n_paths=n, seed=42)["price"] - bs_truth)
            for n in paths_grid
        ]

        cfig = go.Figure()
        cfig.add_trace(go.Scatter(x=steps_grid, y=tree_errs, mode="lines+markers", name="Binomial vs BS"))
        cfig.add_trace(go.Scatter(x=paths_grid, y=mc_errs, mode="lines+markers", name="MC vs BS"))
        cfig.update_layout(
            xaxis=dict(type="log", title="Steps / paths (log)"),
            yaxis=dict(type="log", title="|method - BS| (log)"),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(cfig, width="stretch")
        st.caption(
            "Log-log slope ≈ -1 for the binomial tree (CRR convergence is O(1/N)) and "
            "≈ -0.5 for Monte Carlo (standard √N convergence). The straight lines on a "
            "log-log axis are the visual signature of these rates."
        )


# ---------------------------------------------------------------------------
# Tab 2 — IV Surface
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _fetch_spy_chain_cached(spot_date_str: str) -> pd.DataFrame:
    """Cached on the date string only (not on the full DataFrame, which would
    blow up Streamlit's cache hashing)."""
    from options.data import fetch_spy_chain
    return fetch_spy_chain(ticker="SPY", expiry=None, use_cache=True)


@st.cache_data(show_spinner=False)
def _fetch_spy_spot_cached(spot_date_str: str) -> float:
    import yfinance as yf
    return float(yf.Ticker("SPY").history(period="5d")["Close"].iloc[-1])


with tab_iv:
    st.subheader("SPY implied volatility surface")
    st.caption(
        "Live chain pulled from yfinance. Mid prices are inverted to IV via "
        "Brent's method. Wide spreads and zero-bid quotes are filtered out."
    )

    fetch = st.button("Fetch SPY chain")
    if fetch:
        with st.spinner("Pulling SPY chain (this can take 5-15s)…"):
            today = date.today().isoformat()
            try:
                chain_raw = _fetch_spy_chain_cached(today)
                spot = _fetch_spy_spot_cached(today)
            except Exception as exc:
                st.error(f"Could not fetch live data: {exc}. Try again or use a smaller window.")
                chain_raw = None
                spot = None

        if chain_raw is not None:
            from options.data import clean_chain
            cleaned = clean_chain(chain_raw, spot_date=date.today())
            r_proxy = 0.045
            q_proxy = 0.013
            with st.spinner("Solving for implied vols…"):
                iv_chain = iv_mod.implied_vol_chain(cleaned, S=spot, r=r_proxy, q=q_proxy)
            iv_chain = iv_chain.dropna(subset=["iv"])
            iv_chain = iv_chain[(iv_chain["iv"] > 0.05) & (iv_chain["iv"] < 2.0)]
            st.write(f"Spot: **${spot:.2f}** · {len(iv_chain)} clean quotes across {iv_chain['expiry'].nunique()} expiries.")
            expiries = sorted(iv_chain["expiry"].unique())
            chosen = st.selectbox("Expiry to view", expiries)
            T_chosen = float(iv_chain.loc[iv_chain["expiry"] == chosen, "T"].iloc[0])

            smile = iv_mod.smile_at_expiry(iv_chain, T_chosen)
            sm_fig = go.Figure()
            for opt, marker in (("call", "circle"), ("put", "x")):
                sub = smile[smile["option_type"] == opt]
                sm_fig.add_trace(go.Scatter(
                    x=sub["strike"], y=sub["iv"] * 100,
                    mode="markers", name=opt, marker=dict(symbol=marker, size=8),
                ))
            sm_fig.add_vline(x=spot, line_dash="dash", annotation_text=f"spot {spot:.0f}")
            sm_fig.update_layout(
                xaxis_title="Strike", yaxis_title="Implied vol (%)",
                height=400, title=f"Smile @ {chosen} (T = {T_chosen:.2f}y)",
            )
            st.plotly_chart(sm_fig, width="stretch")

            term = iv_mod.term_structure_atm(iv_chain, S=spot)
            tm_fig = go.Figure()
            tm_fig.add_trace(go.Scatter(x=term["T"], y=term["iv"] * 100, mode="lines+markers"))
            tm_fig.update_layout(
                xaxis_title="Time to expiry (years)", yaxis_title="ATM IV (%)",
                height=300, title="ATM term structure",
            )
            st.plotly_chart(tm_fig, width="stretch")
    else:
        st.info("Click **Fetch SPY chain** to pull the live chain.")


# ---------------------------------------------------------------------------
# Tab 3 — Hedge Simulator
# ---------------------------------------------------------------------------
with tab_hedge:
    st.subheader("Short ATM straddle: gamma-scalp P&L")
    st.markdown(
        "Sell an ATM call + put at σ_implied. Drive the underlying with σ_realised "
        "and rebalance delta on a daily grid. Short straddles **win** when realised "
        "vol comes in below implied (`σ_real < σ_imp`) and **lose** in the opposite "
        "regime — this is the textbook short-gamma trade."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        S0_h = st.number_input("Spot S₀", value=100.0, step=1.0, key="h_s")
        K_h = st.number_input("Strike K", value=100.0, step=1.0, key="h_k")
    with col2:
        days_to_expiry = st.slider("Days to expiry", min_value=7, max_value=90, value=30)
        n_paths_h = st.slider("Paths", min_value=50, max_value=500, value=200, step=50)
    with col3:
        sigma_implied = st.slider("σ_implied (sold at)", 0.05, 0.60, 0.20, step=0.01)
        sigma_realised = st.slider("σ_realised (actual)", 0.05, 0.60, 0.20, step=0.01)

    if st.button("Run simulation"):
        with st.spinner(f"Running {n_paths_h} paths × {days_to_expiry} daily rebalances…"):
            df = simulate_distribution(
                S0=S0_h, K=K_h, T=days_to_expiry / 365.0, r=0.04, q=0.0,
                sigma_implied=sigma_implied, sigma_realised=sigma_realised,
                n_steps=days_to_expiry, n_paths=n_paths_h, seed=11,
            )
        mean_pnl = df["pnl"].mean()
        median_pnl = df["pnl"].median()
        win_rate = float((df["pnl"] > 0).mean())
        sharpe_per_trade = mean_pnl / df["pnl"].std(ddof=1) if df["pnl"].std() > 0 else float("nan")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean P&L", f"${mean_pnl:.2f}")
        m2.metric("Median P&L", f"${median_pnl:.2f}")
        m3.metric("Win rate", f"{win_rate*100:.1f}%")
        m4.metric("P&L Sharpe", f"{sharpe_per_trade:.2f}")

        h_fig = go.Figure()
        h_fig.add_trace(go.Histogram(x=df["pnl"], nbinsx=40, name="P&L"))
        h_fig.add_vline(x=0, line_dash="dot")
        h_fig.add_vline(x=mean_pnl, line_dash="dash", line_color="red",
                        annotation_text=f"mean {mean_pnl:.2f}")
        h_fig.update_layout(
            xaxis_title="P&L per trade ($)", yaxis_title="Paths",
            height=400, title=f"P&L distribution: σ_imp={sigma_implied:.0%}, σ_real={sigma_realised:.0%}",
        )
        st.plotly_chart(h_fig, width="stretch")

        st.markdown("##### Attribution")
        attr = pd.DataFrame({
            "Component": ["Theta collected (mean)", "Gamma paid (mean)", "Residual (mean)"],
            "Value ($)": [
                df["theta_attribution"].mean(),
                df["gamma_attribution"].mean(),
                df["residual"].mean(),
            ],
        })
        st.dataframe(attr.set_index("Component").round(3), width="stretch")


# ---------------------------------------------------------------------------
# Tab 4 — Payoff Diagrams
# ---------------------------------------------------------------------------
with tab_payoff:
    st.subheader("Profit / loss at expiry")
    st.markdown(
        "The simplest visualisation in options. Each curve is the realised "
        "P&L *at expiry* as a function of the underlying's terminal price, "
        "net of the premium paid (or received). The kink at K is the "
        "non-linear payoff that makes options interesting."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        S0_p = st.number_input("Spot S₀ (today)", value=100.0, step=1.0, key="p_s")
        K_p = st.number_input("Strike K", value=100.0, step=1.0, key="p_k")
    with col2:
        T_p = st.number_input("T (years, for premium)", value=0.25, step=0.05, min_value=0.01, key="p_t")
        sigma_p = st.number_input("σ (for premium)", value=0.20, step=0.01, min_value=0.0, key="p_sigma")
    with col3:
        r_p = st.number_input("r", value=0.04, step=0.005, key="p_r")
        q_p = st.number_input("q", value=0.0, step=0.005, key="p_q")

    spot_grid = np.linspace(S0_p * 0.5, S0_p * 1.5, 200)
    call_premium = float(bs.price(S0_p, K_p, T_p, r_p, sigma_p, q_p, "call"))
    put_premium = float(bs.price(S0_p, K_p, T_p, r_p, sigma_p, q_p, "put"))
    straddle_premium = call_premium + put_premium

    long_call_pnl = np.maximum(spot_grid - K_p, 0.0) - call_premium
    long_put_pnl = np.maximum(K_p - spot_grid, 0.0) - put_premium
    short_straddle_pnl = -(np.maximum(spot_grid - K_p, 0.0) + np.maximum(K_p - spot_grid, 0.0)) + straddle_premium

    pf_fig = go.Figure()
    pf_fig.add_trace(go.Scatter(x=spot_grid, y=long_call_pnl, mode="lines",
                                name=f"Long call (premium ${call_premium:.2f})"))
    pf_fig.add_trace(go.Scatter(x=spot_grid, y=long_put_pnl, mode="lines",
                                name=f"Long put (premium ${put_premium:.2f})"))
    pf_fig.add_trace(go.Scatter(x=spot_grid, y=short_straddle_pnl, mode="lines",
                                name=f"Short straddle (premium ${straddle_premium:.2f})",
                                line=dict(dash="dash")))
    pf_fig.add_hline(y=0, line_dash="dot", line_color="grey")
    pf_fig.add_vline(x=K_p, line_dash="dot", line_color="grey",
                     annotation_text=f"K={K_p:.0f}")
    pf_fig.update_layout(
        xaxis_title="Spot at expiry ($)",
        yaxis_title="P&L per contract ($)",
        height=460,
        title="Payoff at expiry, net of today's premium",
    )
    st.plotly_chart(pf_fig, width="stretch")

    st.caption(
        "Reading these: a long call's max loss is the premium (paid up front); "
        "the upside is unbounded above K. A long put's max loss is the premium; "
        "the upside is K minus the premium (only realisable if spot → 0). The "
        "short straddle is the inverse — bounded profit equal to the premium "
        "received, unbounded loss either side. Hedging delta turns this into "
        "the gamma-scalp game in tab 3."
    )


# ---------------------------------------------------------------------------
# Tab 5 — P&L Heatmap (spot × volatility, before expiry)
# ---------------------------------------------------------------------------
with tab_heatmap:
    st.subheader("P&L heatmap across spot × volatility")
    st.markdown(
        "Mark-to-market P&L of a long option position *before expiry* as the "
        "underlying's spot and the implied volatility move. Shows how much of "
        "your P&L comes from delta (horizontal) vs vega (vertical). A long "
        "call gets greener up-and-right; a long put up-and-left."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        S0_hm = st.number_input("Spot S₀ (entry)", value=100.0, step=1.0, key="hm_s")
        K_hm = st.number_input("Strike K", value=100.0, step=1.0, key="hm_k")
    with col2:
        T_hm = st.number_input("T_remaining (years)", value=0.25, step=0.05, min_value=0.01, key="hm_t")
        sigma_entry = st.number_input("σ at entry (sets premium)", value=0.20, step=0.01, min_value=0.01, key="hm_sigma")
    with col3:
        r_hm = st.number_input("r", value=0.04, step=0.005, key="hm_r")
        position = st.radio("Position", ("long call", "long put"), horizontal=True)

    spot_lo = st.slider("Spot range — lower (% of S₀)", 50, 95, 80) / 100.0
    spot_hi = st.slider("Spot range — upper (% of S₀)", 105, 200, 120) / 100.0
    vol_lo = st.slider("σ range — lower", 0.05, 0.40, 0.10, step=0.01)
    vol_hi = st.slider("σ range — upper", 0.20, 1.00, 0.50, step=0.01)

    spot_grid_hm = np.linspace(S0_hm * spot_lo, S0_hm * spot_hi, 25)
    vol_grid_hm = np.linspace(vol_lo, vol_hi, 25)
    opt = "call" if position == "long call" else "put"
    entry_premium = float(bs.price(S0_hm, K_hm, T_hm, r_hm, sigma_entry, 0.0, opt))

    pnl_grid = np.zeros((len(vol_grid_hm), len(spot_grid_hm)))
    for i, v in enumerate(vol_grid_hm):
        # Vectorised over spot for speed.
        prices_at_v = bs.price(spot_grid_hm, K_hm, T_hm, r_hm, v, 0.0, opt)
        pnl_grid[i, :] = prices_at_v - entry_premium

    hm_fig = go.Figure(data=go.Heatmap(
        z=pnl_grid,
        x=spot_grid_hm,
        y=vol_grid_hm * 100,
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="P&L ($)"),
    ))
    hm_fig.update_layout(
        xaxis_title="Spot ($)",
        yaxis_title="Implied vol (%)",
        height=520,
        title=f"{position.title()} mark-to-market P&L (entry ${entry_premium:.2f}, K={K_hm:.0f}, T={T_hm:.2f}y)",
    )
    hm_fig.add_vline(x=K_hm, line_dash="dot", line_color="black", annotation_text=f"K={K_hm:.0f}")
    hm_fig.add_vline(x=S0_hm, line_dash="dash", line_color="blue", annotation_text=f"entry S={S0_hm:.0f}")
    st.plotly_chart(hm_fig, width="stretch")

    st.caption(
        "Read this as a stress-test grid. Each cell answers: 'if spot moves to "
        "X and IV moves to Y, what is my P&L?' Long calls love spot up + vol up "
        "(positive delta + positive vega). Long puts love spot down + vol up. "
        "The contour where P&L = 0 is your breakeven manifold — useful for "
        "stop-loss placement."
    )
