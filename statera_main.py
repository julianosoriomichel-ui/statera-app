import os
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

st.markdown("<h1 style='margin:0'>Statera Partners</h1><p style='color:#5b5b5b;margin-top:-6px;'>Balanced intelligence for modern portfolios.</p>", unsafe_allow_html=True)

# ====== Sidebar: Intake Wizard ======
with st.sidebar:
    st.header("Start — Investor Profile")
    invest_amount = st.number_input("Amount to invest (USD)", min_value=0.0, value=10000.0, step=100.0, help="We'll convert weights into dollar allocations.")
    horizon = st.selectbox("Time horizon", ["1 year","3 years","5 years"], index=1)
    objective = st.selectbox("Objective", ["Balanced","Growth","Income"], index=0)
    risk_tol = st.slider("Risk tolerance", 0.0, 1.0, 0.5, help="0=Conservative, 1=Aggressive")
    max_positions = st.slider("Max positions", 5, 25, 12)
    st.divider()

    st.header("Universe & Data")
    universe = st.selectbox("Universe", ["Default Large Caps", "Custom (enter tickers)"])
    if universe.startswith("Default"):
        tickers = DEFAULT_TICKERS
    else:
        t_input = st.text_area("Tickers (comma separated)", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA")
        tickers = [t.strip().upper() for t in t_input.split(",") if t.strip()]
    period = st.selectbox("History", ["1y","2y","3y"], index=2)
    fetch_btn = st.button("Fetch Market Data", type="primary")
    st.divider()

    st.header("Policy — ESG / Sector Caps")
    exclude_tickers = st.text_input("Exclude tickers (comma‑sep)").upper()
    exclude_tickers = [t.strip() for t in exclude_tickers.split(",") if t.strip()]
    cap_on = st.checkbox("Enable sector caps", value=False)
    sector_caps = {}
    if cap_on:
        st.caption("Max % per sector (applied after optimization).")
        for sec in ["Information Technology","Communication Services","Consumer Discretionary","Consumer Staples",
                    "Health Care","Financials","Industrials","Energy","Materials","Utilities","Real Estate","Unknown"]:
            sector_caps[sec] = st.slider(f"{sec} cap", 0, 100, 40, step=5)
    st.divider()

    st.header("News (optional)")
    news_key = st.secrets.get("NEWSAPI_KEY") if "NEWSAPI_KEY" in st.secrets else os.environ.get("NEWSAPI_KEY","")
    st.caption("Add NEWSAPI_KEY in Secrets to incorporate headlines into scoring.")

# ====== Helpers ======
def dollars_fmt(x):
    try: return f"${x:,.0f}"
    except: return "-"

def weights_table(weights: dict, invest_amount: float) -> pd.DataFrame:
    s = pd.Series(weights, dtype=float).sort_values(ascending=False)
    dollars = (s * invest_amount).round(2)
    return pd.DataFrame({
        "Ticker": s.index,
        "Weight %": (s*100).round(2),
        "Dollars": dollars
    })

def mini_price_panel(prices: pd.DataFrame, symbols):
    if prices.empty: return
    sub = prices[symbols].copy()
    sub = sub / sub.iloc[0]
    df = sub.reset_index().melt("Date", var_name="Ticker", value_name="Index")
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("Date:T", title=""),
        y=alt.Y("Index:Q", title="", scale=alt.Scale(zero=False)),
        facet=alt.Facet("Ticker:N", columns=4),
        tooltip=["Date:T", alt.Tooltip("Index:Q", format=".2f")]
    ).properties(height=120, title="Price index since window start")
    st.altair_chart(chart, use_container_width=True)

# ====== A) Market Data & Scores ======
st.markdown("## A) Market Data & Stock Scores")
if fetch_btn:
    with st.spinner("Fetching market data (cached & batched)..."):
        prices = fetch_price_data(tickers, period=period)
    if prices is None or prices.empty:
        st.error("No data. Try fewer tickers or a shorter period.")
    else:
        # Apply exclusions
        if exclude_tickers:
            keep = [c for c in prices.columns if c not in exclude_tickers]
            prices = prices[keep]
        st.success(f"Downloaded {prices.shape[1]} tickers • {len(prices):,} rows")
        with st.expander("Recent prices (last 5 rows)"):
            st.dataframe(prices.tail(5).style.format("{:.2f}"))

        sectors = get_sectors(list(prices.columns))
        mini_price_panel(prices, list(prices.columns[:min(12, prices.shape[1])]))

        news_sent = fetch_news_sentiment(list(prices.columns), api_key=news_key, pause=0.4) if news_key else {}
        metrics = compute_basic_metrics(prices)
        scored = score_stocks(metrics, news_sent)
        top = scored.sort_values("score_0_100", ascending=False).head(20)
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

        st.markdown("---")

        # ====== B) Auto Portfolio Builder ======
        st.markdown("## B) Auto Portfolio Builder (One‑Click)")
        st.caption("Ranks by composite score, optimizes with position caps, optional sector caps, then scores the result.")

        if st.button("Build Suggested Portfolio", type="secondary"):
            try:
                ranked = scored.sort_values("score_0_100", ascending=False).head(max_positions)
                optw = optimize_weights(prices[ranked.index], method="max_sharpe", max_pos=0.2)
                optw = {k: v for k, v in optw.items() if v > 0}

                # Sector caps (optional)
                if cap_on and sector_caps:
                    w = pd.Series(optw, dtype=float)
                    w = w / w.sum() if w.sum()>0 else w
                    sec = pd.Series({t: sectors.get(t,"Unknown") for t in w.index})
                    for _ in range(5):
                        sec_w = w.groupby(sec).sum()
                        over = [s for s,val in sec_w.items() if val > (sector_caps.get(s,100)/100)]
                        if not over: break
                        for s in over:
                            mask = (sec==s); cap = sector_caps.get(s,100)/100
                            w[mask] = w[mask] * (cap / w[mask].sum())
                        w = w / w.sum()
                    optw = w.to_dict()

                # normalize
                s = pd.Series(optw, dtype=float)
                if s.sum()>0: s = s/s.sum()
                optw = s.to_dict()

                st.success("Suggested Portfolio")
                wt = weights_table(optw, invest_amount)
                wt["Dollars"] = wt["Dollars"].apply(lambda v: f"${v:,.0f}")
                wt["Weight %"] = wt["Weight %"].map(lambda x: f"{x:.2f}")
                st.dataframe(wt, use_container_width=True)

                # Score & risk
                sel = list(optw.keys())
                daily = prices[sel].pct_change().dropna() @ pd.Series({k: optw.get(k,0.0) for k in sel})
                pf_score_raw = float((scored.loc[sel, "score_0_100"] * pd.Series(optw)).sum())
                ann_vol = float(daily.std() * (252**0.5))
                penalty = min(20.0, (ann_vol / 0.6) * 20.0) * (1.0 - risk_tol + 0.5)
                pf_score = max(0.0, pf_score_raw - penalty)

                c1, c2 = st.columns(2)
                c1.metric("Suggested Portfolio Score (0–100)", f"{pf_score:.2f}")
                c2.metric("Annualized Volatility", f"{ann_vol:.2%}")

                if st.button("Backtest Suggested Portfolio"):
                    cum = backtest_portfolio(prices[sel], optw)
                    df = cum.reset_index(); df.columns = ["Date","Cumulative"]
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
            except Exception as e:
                st.error("Auto Builder encountered an issue. Try fewer tickers or different settings.")
                st.caption(f"(Developer hint) {type(e).__name__}: {e}")

# ====== FAQ ======
st.markdown("## FAQ — How to read the dashboard & why we score this way")
with st.expander("Open FAQ"):
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
