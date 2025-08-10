import os
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import requests

DEFAULT_TICKERS = [
 "AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK-B","JNJ","V","PG",
 "JPM","MA","UNH","HD","DIS","BAC","XOM","PFE","INTC","CSCO","ADBE","CRM",
 "NFLX","CMCSA","KO","PEP","ABT","T","ORCL","NKE","MCD","WMT","COST","AVGO",
 "TXN","LLY","MDT","SBUX","IBM","AMAT","NOW","AMD","QCOM","GILD","LOW","AXP",
 "MS","CAT","RTX","SCHW"
]

POS_WORDS = {
 "good","great","positive","gain","gains","beat","beats","beats expectations","upgrade","upgraded",
 "strong","outperform","outperforming","growth","surge","record","beat estimates","raise","raised",
 "profit","profits","guidance raise","guidance increased","buyback","dividend increase"
}
NEG_WORDS = {
 "bad","poor","negative","miss","missed","downgrade","downgraded","weak","underperform","decline",
 "drop","falls","fall","loss","losses","profit warning","cut","cut guidance","lawsuit","investigation",
 "probe","recall","fraud","restatement","layoffs","ban"
}

def _safe_download(batch: List[str], period: str, interval: str, tries: int = 3, sleep: float = 1.0) -> pd.DataFrame:
    last_err = None
    for _ in range(tries):
        try:
            data = yf.download(
                tickers=batch, period=period, interval=interval,
                progress=False, threads=True, auto_adjust=False
            )
            if data is None or data.empty:
                last_err = RuntimeError("Empty download result")
            else:
                return data
        except Exception as e:
            last_err = e
        time.sleep(sleep)
    raise RuntimeError(f"yfinance failed for {batch}: {last_err}")

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(tickers: List[str], period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    if not tickers:
        return pd.DataFrame()
    batches, step = [], 20
    for i in range(0, len(tickers), step):
        batches.append(tickers[i:i+step])

    frames = []
    for b in batches:
        data = _safe_download(b, period, interval)
        if isinstance(data.columns, pd.MultiIndex):
            if "Adj Close" in data.columns.get_level_values(0):
                part = data["Adj Close"]
            else:
                lvl0 = data.columns.get_level_values(0)
                if "Close" in set(lvl0):
                    part = data["Close"]
                else:
                    # fallback to the first level
                    part = data.xs(data.columns[0], axis=1, level=0)
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
    vol = returns.std() * np.sqrt(252.0)
    momentum_3m = (prices / prices.shift(63) - 1).iloc[-1]
    ret_1y = (prices.iloc[-1] / prices.iloc[max(0, len(prices)-252)] - 1)
    m = pd.DataFrame({
        "mean_return": mean_returns,
        "volatility": vol,
        "momentum_3m": momentum_3m,
        "return_1y": ret_1y
    }).replace([np.inf, -np.inf], np.nan).dropna()
    return m

def lexicon_sentiment(text: str) -> float:
    t = (text or "").lower()
    if not t.strip():
        return 0.0
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)

def get_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker

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

@st.cache_data(ttl=600, show_spinner=False)
def fetch_news_sentiment(tickers: List[str], api_key: str | None = None, pause: float = 0.4) -> Dict[str, float]:
    if not api_key:
        return {t: 0.0 for t in tickers}
    sentiments: Dict[str,float] = {}
    base = "https://newsapi.org/v2/everything"
    for tk in tickers:
        q = f"\"{get_company_name(tk)}\" OR {tk}"
        params = {"q": q, "language": "en", "pageSize": 20, "sortBy": "publishedAt", "apiKey": api_key}
        try:
            r = requests.get(base, params=params, timeout=10)
            data = r.json()
            arts = data.get("articles", [])
        except Exception:
            arts = []
        scores = []
        for a in arts:
            text = f"{a.get('title','')} {a.get('description','')} {a.get('content','')}"
            pos = sum(1 for w in POS_WORDS if w in text.lower())
            neg = sum(1 for w in NEG_WORDS if w in text.lower())
            if pos + neg == 0:
                scores.append(0.0)
            else:
                scores.append((pos - neg) / (pos + neg))
        sentiments[tk] = float(np.mean(scores)) if scores else 0.0
        time.sleep(pause)
    return sentiments

def score_stocks(metrics: pd.DataFrame, news_sent: Dict[str,float] | None = None,
                 weights: Dict[str,float] | None = None) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    if weights is None:
        weights = {"return": 0.4, "mom": 0.3, "vol": 0.2, "news": 0.1}
    df = metrics.copy()
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-12)
    df["s_return"] = norm(df["mean_return"])
    df["s_mom"] = norm(df["momentum_3m"])
    df["s_vol"] = 1 - norm(df["volatility"])
    if news_sent is None:
        df["s_news"] = 0.5
    else:
        ns = pd.Series(news_sent, dtype=float).reindex(df.index).fillna(0.0)
        df["s_news"] = (ns + 1) / 2.0  # -1..1 -> 0..1
    df["raw_score"] = (weights["return"]*df["s_return"] +
                       weights["mom"]*df["s_mom"] +
                       weights["vol"]*df["s_vol"] +
                       weights["news"]*df["s_news"])
    df["score_0_100"] = (df["raw_score"] - df["raw_score"].min()) / (df["raw_score"].max() - df["raw_score"].min() + 1e-12) * 100.0
    return df

def factor_contributions_row(row: pd.Series, w: Dict[str, float] | None = None) -> Dict[str, float]:
    weights = w or {"return": 0.4, "mom": 0.3, "vol": 0.2, "news": 0.1}
    parts = {
        "Return": weights["return"]*row.get("s_return", 0),
        "Momentum": weights["mom"]*row.get("s_mom", 0),
        "Volatility (inverted)": weights["vol"]*row.get("s_vol", 0),
        "News": weights["news"]*row.get("s_news", 0),
    }
    total = sum(parts.values()) + 1e-12
    return {k: float(v/total) for k, v in parts.items()}
