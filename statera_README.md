# Statera Partners (Renamed Files Build)

**Main entry:** `statera_main.py` (set this in Streamlit Cloud).  
`requirements.txt` is included for deployment; `statera_requirements.txt` is a duplicate for your local organization.

Features
- Stock suggestions with composite 0–100 score (return, momentum, volatility, optional news sentiment).
- Portfolio score (0–100) with risk penalty.
- Max Sharpe / Min Vol optimization with position caps (PyPortfolioOpt).
- Backtesting, Sharpe, Max Drawdown, Historical VaR & CVaR.
- Risk decomposition & simple historical stress tests.
- Auto Portfolio Builder (one click) using scores + optimizer.
- Optional news integration via `NEWSAPI_KEY` in Streamlit Secrets.
