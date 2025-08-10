import os
import streamlit as st
import pandas as pd

from statera_utils import (
    DEFAULT_TICKERS, fetch_price_data, compute_basic_metrics,
    fetch_news_sentiment, score_stocks
)
from statera_optimizer import optimize_weights, backtest_portfolio, cap_weights
from statera_risk import max_drawdown, historical_var, historical_cvar, risk_decomposition, stress_scenarios

st.set_page_config(page_title="Statera Partners", layout="wide", page_icon="🏛️")
st.markdown("# Statera Partners\n**Balanced Intelligence for Modern Portfolios**")

with st.sidebar:
    st.header("Data & Universe")
    universe = st.selectbox("Universe", ["Default Large Caps", "Custom (enter tickers)"])
    if universe.startswith("Default"):
        tickers = DEFAULT_TICKERS
    else:
        t_input = st.text_area("Tickers (comma separated)", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA")
        tickers = [t.strip().upper() for t in t_input.split(",") if t.strip()]
    period = st.selectbox("History", ["1y", "2y", "3y"], index=2)
    fetch_btn = st.button("Fetch Market Data")

    st.markdown("---")
    st.header("Auto Builder & News")
    risk_tol = st.slider("Risk tolerance", 0.0, 1.0, 0.5)
    target_profile = st.selectbox("Target profile", ["balanced", "growth"])
    build_btn = st.button("Build Suggested Portfolio")

    st.caption("News (optional): add NEWSAPI_KEY in Secrets to enable news-driven signals.")
    news_key = st.secrets.get("NEWSAPI_KEY") if "NEWSAPI_KEY" in st.secrets else os.environ.get("NEWSAPI_KEY", "")

st.header("1) Market Data, Scores & Picks")
if fetch_btn:
    with st.spinner("Downloading prices..."):
        prices = fetch_price_data(tickers, period=period)
    if prices is None or prices.empty:
        st.error("No data retrieved. Try fewer tickers or a shorter history.")
    else:
        st.success(f"Downloaded {prices.shape[1]} tickers.")
        st.dataframe(prices.tail(3))

        news_sent = {}
        if news_key:
            with st.spinner("Fetching news sentiment..."):
                news_sent = fetch_news_sentiment(list(prices.columns), api_key=news_key, pause=0.4)

        metrics = compute_basic_metrics(prices)
        scored = score_stocks(metrics, news_sent)
        st.subheader("Top 20 by Composite Score (0–100)")
        st.dataframe(scored.sort_values("score_0_100", ascending=False).head(20))

        st.subheader("Build / Edit Portfolio")
        default = list(scored.sort_values("score_0_100", ascending=False).head(10).index)
        selected = st.multiselect("Select holdings", options=list(scored.index), default=default)
        if selected:
            weights = {}
            st.caption("Enter raw weights (we normalize to sum to 1)")
            total = 0.0
            for s in selected:
                w = st.number_input(f"Weight for {s}", min_value=0.0, max_value=1.0, value=round(1/len(selected),3), step=0.01, key=f"w_{s}")
                weights[s] = w
                total += w
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}

            # Portfolio score with risk penalty
            port_score = float((scored.loc[selected, "score_0_100"] * pd.Series(weights)).sum())
            daily = prices[selected].pct_change().dropna() @ pd.Series(weights)
            ann_vol = float(daily.std() * (252**0.5))
            penalty = min(20.0, (ann_vol / 0.6) * 20.0)
            port_score = max(0.0, port_score - penalty)
            st.metric("Portfolio Score (0–100)", round(port_score, 2))

            cols = st.columns(3)
            if cols[0].button("Optimize (Max Sharpe)"):
                with st.spinner("Optimizing..."):
                    optw = optimize_weights(prices[selected], method="max_sharpe", max_pos=0.2)
                st.write("Optimized weights (<=20% per name):", optw)
            if cols[1].button("Min Volatility"):
                with st.spinner("Optimizing..."):
                    optw = optimize_weights(prices[selected], method="min_vol", max_pos=0.2)
                st.write("Min-Vol weights:", optw)
            if cols[2].button("Backtest & Risk Report"):
                with st.spinner("Running backtest..."):
                    cum = backtest_portfolio(prices[selected], weights)
                if cum.empty:
                    st.error("Backtest failed.")
                else:
                    st.line_chart(cum)

                    sharpe = float(daily.mean()*252 / (daily.std()*(252**0.5) + 1e-12))
                    mdd = max_drawdown(cum)
                    var95 = historical_var(daily, 0.95)
                    cvar95 = historical_cvar(daily, 0.95)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Sharpe (est.)", f"{sharpe:.2f}")
                    c2.metric("Max Drawdown", f"{mdd:.2%}")
                    c3.metric("VaR 95%", f"{var95:.2%}")
                    c4.metric("CVaR 95%", f"{cvar95:.2%}")

                    st.subheader("Risk Decomposition")
                    rd = risk_decomposition(prices[selected], weights)
                    if rd.empty:
                        st.info("Risk decomposition unavailable.")
                    else:
                        st.dataframe(rd.style.format({"weight":"{:.2%}", "pct_of_risk":"{:.2%}"}))

                    st.subheader("Stress Scenarios (historical worst windows)")
                    st.dataframe(stress_scenarios(prices[selected], weights))

st.header("2) Auto Portfolio Builder (One-Click)")
if st.sidebar.button("Run Auto Builder Now"):
    with st.spinner("Building suggested portfolio..."):
        prices = fetch_price_data(tickers, period=period)
        metrics = compute_basic_metrics(prices)
        news_key = st.secrets.get("NEWSAPI_KEY") if "NEWSAPI_KEY" in st.secrets else os.environ.get("NEWSAPI_KEY", "")
        news_sent = fetch_news_sentiment(list(prices.columns), api_key=news_key, pause=0.4) if news_key else {}
        scored = score_stocks(metrics, news_sent)
        ranked = scored.sort_values("score_0_100", ascending=False).head(12)
        optw = optimize_weights(prices[ranked.index], method="max_sharpe", max_pos=0.2)
    optw = {k: v for k, v in optw.items() if v > 0}
    capped = cap_weights(optw, max_pos=0.2)
    st.subheader("Suggested Portfolio (weights)")
    st.write(capped)

    pf_score = float((ranked.loc[list(capped.keys()), "score_0_100"] * pd.Series(capped)).sum())
    daily = prices[list(capped.keys())].pct_change().dropna() @ pd.Series(capped)
    ann_vol = float(daily.std() * (252**0.5))
    penalty = min(20.0, (ann_vol / 0.6) * 20.0) * (1.0 - st.sidebar.session_state.get("risk_tol", 0.5) + 0.5)
    pf_score = max(0.0, pf_score - penalty)
    st.metric("Suggested Portfolio Score (0–100)", round(pf_score, 2))

    if st.button("Backtest Suggested Portfolio"):
        cum = backtest_portfolio(prices[list(capped.keys())], capped)
        if cum.empty:
            st.error("Backtest failed.")
        else:
            st.line_chart(cum)
