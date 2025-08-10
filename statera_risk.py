import pandas as pd
import numpy as np
from typing import Dict

def max_drawdown(series: pd.Series) -> float:
    cummax = series.cummax()
    dd = (series / cummax - 1).min()
    return float(dd)

def historical_var(returns: pd.Series, alpha=0.95) -> float:
    return float(-np.quantile(returns, 1-alpha))

def historical_cvar(returns: pd.Series, alpha=0.95) -> float:
    var = np.quantile(returns, 1-alpha)
    tail = returns[returns <= var]
    return float(-tail.mean()) if len(tail)>0 else 0.0

def risk_decomposition(prices: pd.DataFrame, weights: Dict[str,float]) -> pd.DataFrame:
    rets = prices.pct_change().dropna()
    cov = rets.cov()
    w = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return pd.DataFrame()
    mrc = cov @ w
    rc = w * mrc
    pct = rc / port_var
    out = pd.DataFrame({'weight': w, 'risk_contrib': rc, 'pct_of_risk': pct})
    return out.sort_values('pct_of_risk', ascending=False)

def stress_scenarios(prices: pd.DataFrame, weights: Dict[str,float]) -> pd.DataFrame:
    rets = prices.pct_change().dropna()
    port_ret = (rets @ pd.Series(weights).reindex(prices.columns).fillna(0)).dropna()
    worst_day = port_ret.min()
    worst_5d = (1+port_ret).rolling(5).apply(lambda x: x.prod()-1).min()
    worst_20d = (1+port_ret).rolling(20).apply(lambda x: x.prod()-1).min()
    return pd.DataFrame({
        'Worst 1D': [float(worst_day)],
        'Worst 5D': [float(worst_5d) if pd.notna(worst_5d) else None],
        'Worst 20D': [float(worst_20d) if pd.notna(worst_20d) else None]
    })
