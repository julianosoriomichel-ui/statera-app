import numpy as np
import pandas as pd

def max_drawdown(series: pd.Series) -> float:
    cummax = series.cummax()
    dd = (series / cummax - 1).min()
    return float(dd)

def historical_var(returns: pd.Series, alpha: float = 0.95) -> float:
    return float(-np.quantile(returns, 1 - alpha))

def historical_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    var = np.quantile(returns, 1 - alpha)
    tail = returns[returns <= var]
    return float(-tail.mean()) if len(tail) > 0 else 0.0

def backtest(weights: dict, prices: pd.DataFrame) -> pd.Series:
    w = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    rets = prices.pct_change().dropna()
    port = (rets @ w).rename("Portfolio")
    return (1 + port).cumprod()
