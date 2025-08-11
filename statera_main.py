import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from statera_utils import DEFAULT_TICKERS, fetch_price_data, compute_basic_metrics, get_sectors
from statera_ml import train_and_predict
from statera_optimizer import optimize_with_custom_mu
from statera_risk import max_drawdown, historical_var, historical_cvar, backtest

st.set_page_config(page_title="Statera Partners — Pro v6", layout="wide", page_icon="🏛️")

st.markdown("<h1 style='margin:0'>Statera Partners</h1><p style='color:#5b5b5b;margin-top:-6px;'>JP‑level portfolio intelligence, consumer‑simple UX.</p>", unsafe_allow_html=True)

# ====== Sidebar: Investor Questionnaire ======
with st.sidebar:
    st.header("Investor Profile")
    invest_amount = st.number_input("Amount to invest (USD)", min_value=0.0, value=20000.0, step=100.0)
    horizon = st.selectbox("Time horizon", ["1 year","3 years","5+ years"], index=2)
    risk_profile = st.selectbox("Risk profile", ["Conservative","Balanced","Growth"], index=1)
    max_positions = st.slider("Max positions", 5, 30, 12)
    st.divider()

    st.header("Universe & Data")
    universe = st.selectbox("Universe", ["Default Large Caps","Custom (enter tickers)"])
    if universe.startswith("Default"):
        tickers = DEFAULT_TICKERS
    else:
        t_input = st.text_area("Tickers (comma separated)", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,BRK-B,JNJ,JPM,WMT")
        tickers = [t.strip().upper() for t in t_input.split(",") if t.strip()]
    period = st.selectbox("History window", ["2y","3y"], index=1)
    st.caption("Tip: 3y gives ML more training data, 2y is faster.")
    st.divider()

    st.header("Policies")
    sector_caps_on = st.checkbox("Enable sector caps", value=False)
    sector_caps = {}
    if sector_caps_on:
        for sec in ["Information Technology","Communication Services","Consumer Discretionary","Consumer Staples",
                    "Health Care","Financials","Industrials","Energy","Materials","Utilities","Real Estate","Unknown"]:
            sector_caps[sec] = st.slider(f"{sec} cap %", 0, 100, 40, step=5)
    exclude = st.text_input("ESG / Manual exclusions (comma-separated)").upper()
    exclude_list = [t.strip() for t in exclude.split(",") if t.strip()]

# ====== Helpers ======
def price_panel(prices: pd.DataFrame, per_page: int = 12):
    if prices.empty: return
    page = st.number_input("Mini‑chart page", min_value=1, max_value=max(1, int(np.ceil(prices.shape[1]/per_page))), value=1, step=1)
    cols = prices.columns[(page-1)*per_page: page*per_page]
    if len(cols)==0:
        return
    sub = prices[cols].copy()
    sub = sub / sub.iloc[0]
    df = sub.reset_index().melt("Date", var_name="Ticker", value_name="Index")
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("Date:T", title=""),
        y=alt.Y("Index:Q", title="", scale=alt.Scale(zero=False)),
        facet=alt.Facet("Ticker:N", columns=4),
        tooltip=["Date:T", alt.Tooltip("Index:Q", format=".2f")]
    ).properties(height=120, title="Price index since window start")
    st.altair_chart(chart, use_container_width=True)

def weights_table(weights: dict, invest_amount: float) -> pd.DataFrame:
    s = pd.Series(weights, dtype=float).sort_values(ascending=False)
    return pd.DataFrame({
        "Ticker": s.index,
        "Weight %": (s*100).round(2),
        "Dollars": (s*invest_amount).round(2)
    })

def apply_sector_caps(weights: dict, sectors: dict, caps: dict) -> dict:
    if not caps: return weights
    w = pd.Series(weights, dtype=float)
    sec = pd.Series({t: sectors.get(t, "Unknown") for t in w.index})
    for _ in range(5):
        sec_w = w.groupby(sec).sum()
        over = [s for s,val in sec_w.items() if val > (caps.get(s,100)/100)]
        if not over: break
        for s in over:
            mask = (sec==s); cap = caps.get(s,100)/100
            w[mask] = w[mask] * (cap / w[mask].sum())
        if w.sum()>0: w = w / w.sum()
    return {k: float(v) for k,v in w.items()}

# ====== Tabs ======
tab_plan, tab_build, tab_explore, tab_faq = st.tabs(["Plan", "Auto Portfolio", "Explore", "FAQ"])

with tab_plan:
    st.subheader("Your plan")
    st.write(f"**Goal:** {risk_profile} | **Horizon:** {horizon} | **Capital:** ${invest_amount:,.0f}")
    st.caption("When you proceed to Auto Portfolio, the builder will respect your risk profile and position caps.")

with tab_build:
    st.subheader("One‑Click ML Auto Portfolio")
    run = st.button("Generate Suggested Portfolio", type="primary")
    if run:
        with st.spinner("Fetching market data..."):
            prices = fetch_price_data(tickers, period=period)
        if prices is None or prices.empty:
            st.error("No data. Try a smaller universe or shorter period.")
        else:
            if exclude_list:
                keep = [c for c in prices.columns if c not in exclude_list]
                prices = prices[keep]
            sectors = get_sectors(list(prices.columns))

            # ML predictions
            with st.spinner("Training ML model & predicting next‑month returns..."):
                mu_ann, feat_imp = train_and_predict(prices)
            if mu_ann.empty:
                st.warning("ML needs more data; falling back to annualized mean returns.")
                mu_ann = compute_basic_metrics(prices)["mean_return"].sort_values(ascending=False)

            # Focus on top names then optimize
            top_mu = mu_ann.head(max_positions)
            ra = {"Conservative":4.0,"Balanced":2.5,"Growth":1.5}[risk_profile]
            with st.spinner("Optimizing portfolio..."):
                weights = optimize_with_custom_mu(prices[top_mu.index], top_mu, risk_aversion=ra, max_pos=0.25)

            if sector_caps_on and sector_caps:
                weights = apply_sector_caps(weights, sectors, sector_caps)
                s = pd.Series(weights, dtype=float)
                if s.sum()>0: s = s/s.sum()
                weights = s.to_dict()

            if not weights:
                st.error("Optimizer returned empty weights. Try increasing max positions or relaxing caps.")
            else:
                st.success("Suggested Portfolio")
                wt = weights_table(weights, invest_amount)
                wt["Weight %"] = wt["Weight %"].map(lambda x: f"{x:.2f}")
                wt["Dollars"] = wt["Dollars"].map(lambda v: f"${v:,.0f}")
                st.dataframe(wt, use_container_width=True)

                # Backtest
                cum = backtest(weights, prices[list(weights.keys())])
                df = cum.reset_index(); df.columns = ["Date","Cumulative"]
                chart = alt.Chart(df).mark_line().encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Cumulative:Q", title="Growth of $1", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Cumulative:Q", format=".2f")]
                ).properties(height=320, title="Suggested Portfolio — Growth of $1")
                st.altair_chart(chart, use_container_width=True)

                # Risk stats
                daily = prices[list(weights.keys())].pct_change().dropna() @ pd.Series(weights)
                sharpe = float(daily.mean()*252 / (daily.std()*(252**0.5) + 1e-12))
                mdd = max_drawdown(cum); var95 = historical_var(daily, 0.95); cvar95 = historical_cvar(daily, 0.95)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Sharpe (est.)", f"{sharpe:.2f}")
                m2.metric("Max Drawdown", f"{mdd:.2%}")
                m3.metric("VaR 95%", f"{var95:.2%}")
                m4.metric("CVaR 95%", f"{cvar95:.2%}")

                st.download_button("Download Weights CSV", data=pd.Series(weights).to_csv().encode("utf-8"),
                                   file_name="suggested_portfolio_weights.csv", mime="text/csv")

with tab_explore:
    st.subheader("Explore Market Data")
    if st.button("Fetch & score"):
        prices = fetch_price_data(tickers, period=period)
        if prices is None or prices.empty:
            st.error("No data.")
        else:
            st.success(f"Fetched {prices.shape[1]} tickers • {len(prices):,} rows")
            price_panel(prices, per_page=12)
            metrics = compute_basic_metrics(prices)
            st.dataframe(
                metrics.rename(columns={"mean_return":"Ann. Return","momentum_3m":"3M Momentum","volatility":"Ann. Vol","return_1y":"1Y Return"})
                .style.format({"Ann. Return":"{:.2%}","3M Momentum":"{:.2%}","Ann. Vol":"{:.2%}","1Y Return":"{:.2%}"})
            )

with tab_faq:
    st.subheader("FAQ — What am I seeing?")
    st.markdown('''
**How the ML works** — We train a Random Forest to predict next‑month returns from recent **momentum** (5/21/63/126‑day) and **volatility** (21/63‑day). Predictions are converted to annualized expected returns and sent to the optimizer.

**How the optimizer works** — We use your risk profile to set a **risk‑aversion** level and solve for weights given expected returns and the covariance matrix. Position caps and optional **sector caps** prevent over‑concentration.

**Risk metrics** —  
- **Sharpe (est.)**: return per unit of risk (higher is better)  
- **Max Drawdown**: worst peak‑to‑trough drop in backtest (closer to 0% is better)  
- **VaR / CVaR 95%**: typical and average loss in the worst 5% of days  

**Why allocations are in dollars** — You tell us how much you want to invest; we convert weights into exact dollar amounts for easy execution.

**Data** — Free end‑of‑day equities via `yfinance`. For production use, consider paid feeds (e.g., IEX Cloud, Polygon).
''')
