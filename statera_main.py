import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from statera_utils import (
    DEFAULT_TICKERS, fetch_price_data, compute_basic_metrics,
    fetch_news_sentiment, score_stocks, get_sectors, factor_contributions_row
)
from statera_optimizer import optimize_weights, backtest_portfolio, cap_weights
from statera_risk import max_drawdown, historical_var, historical_cvar, risk_decomposition, stress_scenarios

st.set_page_config(page_title="Statera Partners", layout="wide", page_icon="🏛️")

# ====== Header ======
st.markdown(
    """
    <h1 style="margin:0;">Statera Partners</h1>
    <p style="color:#5b5b5b;margin-top:-6px;">Balanced intelligence for modern portfolios.</p>
    """, unsafe_allow_html=True
)

# ====== Sidebar ======
with st.sidebar:
    st.header("1) Data & Universe")
    universe = st.selectbox("Universe", ["Default Large Caps", "Custom (enter tickers)"])
    if universe.startswith("Default"):
        tickers = DEFAULT_TICKERS
    else:
        t_input = st.text_area("Tickers (comma separated)", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA")
        tickers = [t.strip().upper() for t in t_input.split(",") if t.strip()]
    period = st.selectbox("History", ["1y","2y","3y"], index=2)
    fetch_btn = st.button("Fetch Market Data", type="primary")

    st.divider()
    st.header("2) Policy — ESG / Sector Caps")
    exclude_tickers = st.text_input("Exclude tickers (comma‑sep)", value="").upper()
    exclude_tickers = [t.strip() for t in exclude_tickers.split(",") if t.strip()]

    cap_on = st.checkbox("Enable sector caps", value=False)
    sector_caps = {}
    if cap_on:
        st.caption("Set max % per sector (applied after optimization).")
        for sec in ["Information Technology","Communication Services","Consumer Discretionary","Consumer Staples",
                    "Health Care","Financials","Industrials","Energy","Materials","Utilities","Real Estate","Unknown"]:
            sector_caps[sec] = st.slider(f"{sec} cap", 0, 100, 40, step=5)

    st.divider()
    st.header("3) Auto Builder & News")
    risk_tol = st.slider("Risk tolerance", 0.0, 1.0, 0.5)
    target_profile = st.selectbox("Target profile", ["balanced","growth"])
    build_btn = st.button("Build Suggested Portfolio", type="secondary")
    news_key = st.secrets.get("NEWSAPI_KEY") if "NEWSAPI_KEY" in st.secrets else os.environ.get("NEWSAPI_KEY","")
    st.caption("Optional: add `NEWSAPI_KEY` in Secrets for news-driven scoring.")

def alt_line(series: pd.Series, title: str):
    df = series.reset_index()
    df.columns = ["Date","Value"]
    c = alt.Chart(df).mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Value:Q", title=title, scale=alt.Scale(zero=False)),
        tooltip=["Date:T", alt.Tooltip("Value:Q", format=".2f")]
    ).properties(height=320, title=title)
    st.altair_chart(c, use_container_width=True)

def format_weights(weights: dict) -> pd.DataFrame:
    if not weights: 
        return pd.DataFrame({"Weight":[]})
    s = pd.Series(weights).sort_values(ascending=False)
    return (s*100).round(2).astype(str).add("%").to_frame("Weight")

def apply_sector_caps(weights: dict, sectors: dict, caps: dict) -> dict:
    if not caps:
        return weights
    w = pd.Series(weights, dtype=float)
    sec_series = pd.Series({t: sectors.get(t, "Unknown") for t in w.index})
    # iterate to enforce caps (simple proportional scaling)
    for _ in range(5):
        sec_w = w.groupby(sec_series).sum()
        over = [sec for sec,val in sec_w.items() if val > (caps.get(sec, 100)/100)]
        if not over:
            break
        for sec in over:
            mask = sec_series==sec
            cap = caps.get(sec,100)/100
            if w[mask].sum() > 0:
                w[mask] = w[mask] * (cap / w[mask].sum())
        # renormalize
        if w.sum() > 0:
            w = w / w.sum()
    return {k: float(v) for k,v in w.items()}

# ====== A) Market Data & Scores ======
st.subheader("A) Market Data & Stock Scores")
if fetch_btn:
    with st.spinner("Fetching market data (cached)..."):
        prices = fetch_price_data(tickers, period=period)
    if prices is None or prices.empty:
        st.error("No data. Try fewer tickers or a shorter period.")
    else:
        st.success(f"Downloaded {prices.shape[1]} tickers • {len(prices):,} rows")
        with st.expander("Recent price table", expanded=False):
            st.dataframe(prices.tail(5).style.format("{:.2f}"))

        # Exclusions
        if exclude_tickers:
            keep_cols = [c for c in prices.columns if c not in exclude_tickers]
            prices = prices[keep_cols]
            if not keep_cols:
                st.error("All tickers excluded. Remove exclusions to continue.")
        sectors = get_sectors(list(prices.columns))

        # News (optional)
        news_sent = fetch_news_sentiment(list(prices.columns), api_key=news_key, pause=0.4) if news_key else {}

        # Score
        metrics = compute_basic_metrics(prices)
        scored = score_stocks(metrics, news_sent)
        top = scored.sort_values("score_0_100", ascending=False).head(20)

        with st.expander("How we score stocks (0–100)", expanded=False):
            st.markdown(
                """
                **Components**  
                - **Return**: Annualized mean return (higher is better)  
                - **Momentum (3m)**: 3‑month performance (higher is better)  
                - **Volatility**: Annualized std. dev. (lower is better)  
                - **News Sentiment (optional)**: Headlines mapped to 0–1 (neutral if disabled)  
                
                **Composite Score** = 0.4×Return + 0.3×Momentum + 0.2×(1–Vol) + 0.1×News, rescaled to **0–100**.
                """
            )

        st.dataframe(
            top[["mean_return","momentum_3m","volatility","score_0_100"]]
                .rename(columns={"mean_return":"Ann. Return","momentum_3m":"3M Momentum","volatility":"Ann. Vol","score_0_100":"Score (0–100)"})
                .style.format({"Ann. Return":"{:.2%}","3M Momentum":"{:.2%}","Ann. Vol":"{:.2%}","Score (0–100)":"{:.1f}"})
        )

        with st.expander("Explain a stock's score", expanded=False):
            ticker_ex = st.selectbox("Pick a ticker to explain", options=list(top.index))
            row = scored.loc[ticker_ex]
            contrib = factor_contributions_row(row)
            st.write({k: f"{v*100:.1f}%" for k,v in contrib.items()})

        st.markdown("**Build / Edit Portfolio**")
        default = list(top.index[:10])
        selected = st.multiselect("Select holdings", options=list(scored.index), default=default)
        if selected:
            st.caption("Enter raw weights (we normalize to sum to 1).")
            weights = {}
            total = 0.0
            for s in selected:
                wv = st.number_input(f"Weight for {s}", min_value=0.0, max_value=1.0, value=round(1/len(selected),3), step=0.01)
                weights[s] = wv; total += wv
            if total > 0:
                weights = {k: v/total for k,v in weights.items()}

            daily = prices[selected].pct_change().dropna() @ pd.Series(weights)
            ann_vol = float(daily.std() * (252**0.5))
            port_score = float((scored.loc[selected, "score_0_100"] * pd.Series(weights)).sum())
            penalty = min(20.0, (ann_vol / 0.6) * 20.0)
            port_score = max(0.0, port_score - penalty)

            c1, c2 = st.columns(2)
            c1.metric("Portfolio Score (0–100)", f"{port_score:.2f}")
            c2.metric("Annualized Volatility", f"{ann_vol:.2%}")

            b1, b2, b3 = st.columns(3)
            if b1.button("Optimize (Max Sharpe)"):
                with st.spinner("Optimizing..."):
                    optw = optimize_weights(prices[selected], method="max_sharpe", max_pos=0.2)
                st.success("Optimized weights")
                st.dataframe(format_weights(optw).style.hide_index())

            if b2.button("Min Volatility"):
                with st.spinner("Optimizing..."):
                    optw = optimize_weights(prices[selected], method="min_vol", max_pos=0.2)
                st.success("Min‑Vol weights")
                st.dataframe(format_weights(optw).style.hide_index())

            if b3.button("Backtest & Risk Report"):
                cum = backtest_portfolio(prices[selected], weights)
                if cum.empty:
                    st.error("Backtest failed.")
                else:
                    # chart
                    df = cum.reset_index()
                    df.columns = ["Date","Cumulative"]
                    chart = alt.Chart(df).mark_line().encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y("Cumulative:Q", title="Growth of $1", scale=alt.Scale(zero=False)),
                        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Cumulative:Q", format=".2f")],
                    ).properties(height=320, title="Custom Portfolio — Growth of $1")
                    st.altair_chart(chart, use_container_width=True)

                    sharpe = float(daily.mean()*252 / (daily.std()*(252**0.5) + 1e-12))
                    mdd = max_drawdown(cum); var95 = historical_var(daily, 0.95); cvar95 = historical_cvar(daily, 0.95)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Sharpe (est.)", f"{sharpe:.2f}")
                    m2.metric("Max Drawdown", f"{mdd:.2%}")
                    m3.metric("VaR 95%", f"{var95:.2%}")
                    m4.metric("CVaR 95%", f"{cvar95:.2%}")
                    st.subheader("Risk Decomposition")
                    rd = risk_decomposition(prices[selected], weights)
                    if rd.empty:
                        st.info("Risk decomposition unavailable.")
                    else:
                        st.dataframe(rd.sort_values("pct_of_risk", ascending=False).style.format({"weight":"{:.2%}","pct_of_risk":"{:.2%}"}))
                    st.subheader("Stress Scenarios")
                    st.dataframe(stress_scenarios(prices[selected], weights).style.format("{:.2%}"))
                    # Exports
                    st.download_button("Download Weights CSV", data=pd.Series(weights).to_csv().encode("utf-8"),
                                       file_name="custom_portfolio_weights.csv", mime="text/csv")
                    st.download_button("Download Backtest CSV", data=cum.to_csv().encode("utf-8"),
                                       file_name="custom_portfolio_backtest.csv", mime="text/csv")

# ====== B) Auto Portfolio Builder ======
st.subheader("B) Auto Portfolio Builder (One‑Click)")
st.caption("Ranks by composite score, optimizes with caps, applies ESG & sector constraints, then scores the result.")

if build_btn:
    try:
        prices = fetch_price_data(tickers, period=period)
        if prices is None or prices.empty:
            st.error("No data for builder. Try fewer tickers or a shorter history.")
        else:
            # apply exclusions early
            if exclude_tickers:
                keep_cols = [c for c in prices.columns if c not in exclude_tickers]
                prices = prices[keep_cols]
            sectors = get_sectors(list(prices.columns))

            metrics = compute_basic_metrics(prices)
            news_sent = fetch_news_sentiment(list(prices.columns), api_key=news_key, pause=0.4) if news_key else {}
            scored = score_stocks(metrics, news_sent)
            ranked = scored.sort_values("score_0_100", ascending=False).head(16)

            # Optimize
            optw = optimize_weights(prices[ranked.index], method="max_sharpe", max_pos=0.2)
            optw = {k: v for k, v in optw.items() if v > 0}

            # Sector caps
            if cap_on and sector_caps:
                optw = apply_sector_caps(optw, sectors, sector_caps)

            # Final normalize
            if optw:
                s = pd.Series(optw, dtype=float)
                if s.sum() > 0:
                    s = s / s.sum()
                optw = s.to_dict()

        if optw:
            st.success("Suggested Portfolio (weights)")
            st.dataframe(format_weights(optw).style.hide_index())
            sel = list(optw.keys())
            daily = prices[sel].pct_change().dropna() @ pd.Series(optw)
            pf_score_raw = float((scored.loc[sel, "score_0_100"] * pd.Series(optw)).sum())
            ann_vol = float(daily.std() * (252**0.5))
            penalty = min(20.0, (ann_vol / 0.6) * 20.0) * (1.0 - risk_tol + 0.5)  # use slider value
            pf_score = max(0.0, pf_score_raw - penalty)

            c1, c2 = st.columns(2)
            c1.metric("Suggested Portfolio Score (0–100)", f"{pf_score:.2f}")
            c2.metric("Annualized Volatility", f"{ann_vol:.2%}")

            if st.button("Backtest Suggested Portfolio"):
                cum = backtest_portfolio(prices[sel], optw)
                df = cum.reset_index()
                df.columns = ["Date","Cumulative"]
                chart = alt.Chart(df).mark_line().encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Cumulative:Q", title="Growth of $1", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Cumulative:Q", format=".2f")],
                ).properties(height=320, title="Suggested Portfolio — Growth of $1")
                st.altair_chart(chart, use_container_width=True)
                st.download_button("Download Weights CSV", data=pd.Series(optw).to_csv().encode("utf-8"),
                                   file_name="suggested_portfolio_weights.csv", mime="text/csv")
                st.download_button("Download Backtest CSV", data=cum.to_csv().encode("utf-8"),
                                   file_name="suggested_portfolio_backtest.csv", mime="text/csv")
        else:
            st.warning("Builder produced empty weights. Try again or relax constraints.")
    except Exception as e:
        st.error("Auto Builder encountered an issue. Try fewer tickers or different settings.")
        st.caption(f"(Developer hint) {type(e).__name__}: {e}")

# ====== FAQ ======
with st.expander("FAQ — How to read the dashboard & why we score this way", expanded=False):
    st.markdown(
        """
        **Stock Score (0–100)** blends: **annualized return**, **3‑month momentum**, **volatility** (lower is better),
        and optional **news sentiment**. Scores are rescaled between 0 and 100.

        **Portfolio Score (0–100)** is a **weighted average of holdings' scores** minus a small **volatility penalty**.
        This balances quality and risk.

        **Risk Metrics**  
        - **Sharpe (est.)**: Return per unit of risk (higher is better)  
        - **VaR 95%**: Daily loss not exceeded 95% of the time (smaller magnitude is better)  
        - **CVaR 95%**: Average loss on the worst 5% of days (smaller is better)  
        - **Max Drawdown**: Worst peak‑to‑trough decline in backtest (closer to 0% is better)

        **Sector Caps** place a ceiling on exposure per sector to avoid concentration risk.
        **ESG Exclusions** let you remove tickers from consideration.

        **Data**: Free EOD via `yfinance`. For institutional use, consider paid vendors (Polygon, IEX).
        """
    )
