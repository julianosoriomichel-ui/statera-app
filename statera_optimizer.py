import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

def optimize_weights(prices: pd.DataFrame, method: str = "max_sharpe", max_pos: float = 0.2, min_pos: float = 0.0):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=(min_pos, max_pos))
    if method == "max_sharpe":
        ef.add_objective(objective_functions.L2_reg, gamma=0.001)
        ef.max_sharpe()
    elif method == "min_vol":
        ef.min_volatility()
    elif method == "target_vol":
        try:
            ef.efficient_risk(0.12)
        except Exception:
            ef.min_volatility()
    else:
        ef.max_sharpe()
    return ef.clean_weights()

def backtest_portfolio(prices: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights).reindex(prices.columns).fillna(0.0)
    rets = prices.pct_change().dropna()
    port = (rets @ w).rename("Portfolio")
    return (1 + port).cumprod()

def cap_weights(weights: dict, max_pos: float = 0.2) -> dict:
    s = pd.Series(weights)
    s = s.clip(0, max_pos)
    s = s / s.sum() if s.sum() > 0 else s
    return s.to_dict()
