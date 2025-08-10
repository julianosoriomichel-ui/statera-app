import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import os

DEFAULT_TICKERS = [
 'AAPL','MSFT','GOOGL','AMZN','NVDA','TSLA','META','BRK-B','JNJ','V','PG',
 'JPM','MA','UNH','HD','DIS','BAC','XOM','PFE','INTC','CSCO','ADBE','CRM',
 'NFLX','CMCSA','KO','PEP','ABT','T','ORCL','NKE','MCD','WMT','COST','AVGO',
 'TXN','LLY','MDT','SBUX','IBM','AMAT','NOW','AMD','QCOM','GILD','LOW','AXP',
 'MS','CAT','RTX','SCHW'
]

POS_WORDS = set([
 'good','great','positive','gain','gains','beat','beats','beats expectations','upgrade','upgraded',
 'strong','outperform','outperforming','growth','surge','record','beat estimates','raise','raised',
 'profit','profits','guidance raise','guidance increased','buyback','dividend increase'
])
NEG_WORDS = set([
 'bad','poor','negative','miss','missed','downgrade','downgraded','weak','underperform','decline',
 'drop','falls','fall','loss','losses','profit warning','cut','cut guidance','lawsuit','investigation',
 'probe','recall','fraud','restatement','layoffs','ban'
])

def fetch_price_data(tickers, period='3y', interval='1d'):
    data = None
    try:
        data = yf.download(tickers, period=period, interval=interval, progress=False, threads=True)
    except Exception as e:
        return pd.DataFrame()
    if data is None or data.empty:
        return pd.DataFrame()
    if 'Adj Close' in data:
        prices = data['Adj Close'].dropna(how='all')
    else:
        prices = data.dropna(how='all')
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.ffill().dropna(how='all')
    return prices

def compute_basic_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    vol = returns.std() * (252**0.5)
    momentum_3m = (prices / prices.shift(63) - 1).iloc[-1]
    ret_1y = (prices.iloc[-1] / prices.iloc[max(0, len(prices)-252)] - 1)
    m = pd.DataFrame({
        'mean_return': mean_returns,
        'volatility': vol,
        'momentum_3m': momentum_3m,
        'return_1y': ret_1y
    }).replace([np.inf, -np.inf], np.nan).dropna()
    return m

def lexicon_sentiment(text: str) -> float:
    t = (text or "").lower()
    if not t.strip():
        return 0.0
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos+neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)

def get_company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get('longName') or info.get('shortName') or ticker
    except Exception:
        return ticker

def fetch_news_sentiment(tickers, api_key=None, pause=0.4):
    # Return dict ticker -> sentiment in [-1,1] using NewsAPI if api_key provided, else zeros.
    if not api_key:
        return {t: 0.0 for t in tickers}
    sentiments = {}
    base = "https://newsapi.org/v2/everything"
    for tk in tickers:
        q = f"\"{get_company_name(tk)}\" OR {tk}"
        params = {
            "q": q, "language": "en", "pageSize": 20, "sortBy": "publishedAt", "apiKey": api_key
        }
        try:
            r = requests.get(base, params=params, timeout=10)
            data = r.json()
            arts = data.get("articles", [])
        except Exception:
            arts = []
        scores = []
        for a in arts:
            text = f"{a.get('title','')} {a.get('description','')} {a.get('content','')}"
            scores.append(lexicon_sentiment(text))
        sentiments[tk] = float(np.mean(scores)) if scores else 0.0
        time.sleep(pause)
    return sentiments

def score_stocks(metrics: pd.DataFrame, news_sent: dict | None = None,
                 weights: dict | None = None) -> pd.DataFrame:
    if weights is None:
        weights = {'return': 0.4, 'mom': 0.3, 'vol': 0.2, 'news': 0.1}
    df = metrics.copy()
    if df.empty:
        return df
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-12)
    df['s_return'] = norm(df['mean_return'])
    df['s_mom'] = norm(df['momentum_3m'])
    df['s_vol'] = 1 - norm(df['volatility'])
    if news_sent is None:
        df['s_news'] = 0.5
    else:
        ns = pd.Series(news_sent, dtype=float).reindex(df.index).fillna(0.0)
        df['s_news'] = (ns + 1) / 2
    df['raw_score'] = (weights['return']*df['s_return'] +
                       weights['mom']*df['s_mom'] +
                       weights['vol']*df['s_vol'] +
                       weights['news']*df['s_news'])
    df['score_0_100'] = (df['raw_score'] - df['raw_score'].min()) / (df['raw_score'].max() - df['raw_score'].min() + 1e-12) * 100
    return df
