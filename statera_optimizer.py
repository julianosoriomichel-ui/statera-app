import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models

def optimize_with_custom_mu(prices: pd.DataFrame, mu: pd.Series, risk_aversion: float = 2.0, max_pos: float = 0.2):
    cols = [c for c in prices.columns if c in mu.index]
    prices = prices[cols]
    mu = mu.reindex(cols).fillna(0.0)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S, weight_bounds=(0.0, max_pos))
    try:
        ef.max_quadratic_utility(risk_aversion=risk_aversion)
    except Exception:
        ef.min_volatility()
    w = ef.clean_weights()
    return {k: float(v) for k,v in w.items() if v>0}
