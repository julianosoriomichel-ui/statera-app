# Statera Partners — Pro v4 (Ready)

**Main file:** `statera_main.py`  
**Python version:** pinned via `runtime.txt` to **3.11**

### What's new
- Auto Builder bug fixed (uses `risk_tol` correctly)
- Altair charts for clean, labeled visuals
- Formatted tables & readable weights
- Factor explanations per stock
- ESG exclusions & sector caps (optional)
- CSV downloads (weights, backtests)
- yfinance retries/batching/caching for stability

### Deploy
1) Upload all files to your GitHub repo.
2) Streamlit Cloud → New app  
   - Repo: `youruser/yourrepo`
   - Branch: `main`
   - Main file path: `statera_main.py`
3) (Optional) Add `NEWSAPI_KEY` in Secrets for headline sentiment.
