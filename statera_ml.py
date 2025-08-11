from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def _features_from_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    rets = prices.pct_change().dropna()
    feats = []
    targets = []
    tickers = prices.columns
    lookbacks = [5, 21, 63, 126]
    fcols = [f"ret_{lb}" for lb in lookbacks] + [f"vol_{lb}" for lb in [21,63]]
    for t in range(126, len(prices)-21):
        row_X = []
        row_y = []
        for tk in tickers:
            p = prices[tk]
            if p.isna().iloc[t-126:t+1].any():
                row_X.append([np.nan]*len(fcols)); row_y.append(np.nan); continue
            vals = []
            for lb in lookbacks:
                vals.append(p.iloc[t] / p.iloc[t-lb] - 1.0)
            for lb in [21,63]:
                vals.append(rets[tk].iloc[t-lb+1:t+1].std() * (252.0**0.5))
            row_X.append(vals)
            y = p.iloc[t+21] / p.iloc[t] - 1.0
            row_y.append(y)
        X_block = pd.DataFrame(row_X, columns=fcols, index=tickers)
        X_block["date"] = rets.index[t]
        feats.append(X_block)
        targets.append(pd.Series(row_y, index=tickers, name=rets.index[t]))
    X = pd.concat(feats).dropna()
    y = pd.concat(targets).dropna()
    X = X.drop(columns=["date"])
    return X, y

def train_and_predict(prices: pd.DataFrame) -> tuple[pd.Series, dict]:
    if prices is None or prices.empty or prices.shape[0] < 220:
        return pd.Series(dtype=float), {}
    X, y = _features_from_prices(prices)
    if len(y) < 500:
        return pd.Series(dtype=float), {}
    rf = RandomForestRegressor(n_estimators=300, max_depth=7, min_samples_leaf=5, random_state=11, n_jobs=-1)
    rf.fit(X.values, y.values)
    t = len(prices) - 1
    rets = prices.pct_change().dropna()
    lookbacks = [5,21,63,126]
    fcols = [f"ret_{lb}" for lb in lookbacks] + [f"vol_{lb}" for lb in [21,63]]
    latest = []
    idx = []
    for tk in prices.columns:
        p = prices[tk]
        if p.isna().iloc[t-126:t+1].any():
            continue
        vals = []
        for lb in lookbacks:
            vals.append(p.iloc[t] / p.iloc[t-lb] - 1.0)
        for lb in [21,63]:
            vals.append(rets[tk].iloc[-lb:].std() * (252.0**0.5))
        latest.append(vals); idx.append(tk)
    if not latest:
        return pd.Series(dtype=float), {}
    X_last = pd.DataFrame(latest, columns=fcols, index=idx)
    preds_21d = pd.Series(rf.predict(X_last.values), index=X_last.index)
    ann = (1.0 + preds_21d).clip(-0.9, 5.0) ** (252.0/21.0) - 1.0
    fi = dict(zip(fcols, rf.feature_importances_))
    return ann.sort_values(ascending=False), fi
