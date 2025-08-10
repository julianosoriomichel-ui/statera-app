# Statera Partners — Fixed Build

**Main file:** `statera_main.py` (set this in Streamlit Cloud)  
**Python version:** pinned via `runtime.txt` to **3.11** (ensures PyPortfolioOpt is available)

Includes:
- yfinance batching + retries + 15‑min caching
- Stock scoring (return, momentum, vol, + optional news sentiment)
- Portfolio score (0–100) with risk penalty
- Optimizer (Max Sharpe / Min Vol) with 20% cap
- Backtest + Sharpe, Max Drawdown, VaR(95%), CVaR(95%)
- Risk decomposition & stress scenarios
- Auto Portfolio Builder

**News:** add `NEWSAPI_KEY` in Secrets to enable news-driven signals.
