import time
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

DEFAULT_TICKERS = [
 "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK-B","JNJ","V","PG",
 "JPM","MA","UNH","HD","DIS","BAC","XOM","PFE","INTC","CSCO","ADBE","CRM",
 "NFLX","CMCSA","KO","PEP","ABT","T","ORCL","NKE","MCD","WMT","COST","AVGO",
 "TXN","LLY","MDT","SBUX","IBM","AMAT","NOW","AMD","QCOM","GILD","LOW","AXP",
 "MS","CAT","RTX","SCHW"
]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(tickers: List[str], period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame()
    frames = []
    last_err = None
    for i in range(0, len(tickers), 20):
        batch = tickers[i:i+20]
        for _ in range(3):
            try:
                data = yf.download(batch, period=period, interval=interval, progress=False, threads=True, auto_adjust=False)
                break
            except Exception as e:
                last_err = e
                time.sleep(0.7)
        else:
            raise RuntimeError(f"yfinance failed: {last_err}")
        if isinstance(data.columns, pd.MultiIndex):
            part = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data["Close"]
        else:
            part = data
        frames.append(part)
    prices = pd.concat(frames, axis=1)
    prices = prices.ffill().dropna(how="all")
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices

def compute_basic_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    if prices is None or prices.empty:
        return pd.DataFrame()
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252.0
    vol = returns.std() * (252.0**0.5)
    momentum_3m = (prices / prices.shift(63) - 1).iloc[-1]
    ret_1y = (prices.iloc[-1] / prices.iloc[max(0, len(prices)-252)] - 1)
    out = pd.DataFrame({
        "mean_return": mean_returns,
        "volatility": vol,
        "momentum_3m": momentum_3m,
        "return_1y": ret_1y
    }).replace([np.inf, -np.inf], np.nan).dropna()
    return out

@st.cache_data(ttl=3600, show_spinner=False)
def get_sectors(tickers: List[str]) -> Dict[str, str]:
    out: Dict[str,str] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            out[t] = info.get("sector") or "Unknown"
        except Exception:
            out[t] = "Unknown"
        time.sleep(0.05)
    return out
