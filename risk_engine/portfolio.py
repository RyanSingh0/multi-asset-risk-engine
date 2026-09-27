"""Weighting schemes and a simple backtest loop.

Long-only, fully invested unless the vol target levers up (capped).
Schemes: equal_weight, inverse_vol, min_variance, risk_parity (ERC), plus a vol-target overlay.
Anything estimated at date t only uses returns before t.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TD = 252


def equal_weight(cov):
    n = len(cov)
    return np.full(n, 1.0 / n)


def inverse_vol(cov):
    iv = 1 / np.sqrt(np.diag(cov))
    return iv / iv.sum()


def min_variance(cov):
    n = len(cov)
    res = minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), method='SLSQP',
                   bounds=[(0, 1)] * n, constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
                   options={'ftol': 1e-12, 'maxiter': 500})
    return res.x


def risk_parity(cov, tol=1e-10, max_iter=10000):
    """ERC by fixed-point iteration (Chaves et al. 2012): w_i ~ 1 / (cov @ w)_i.
    Converges when cov is positive definite."""
    n = len(cov)
    w = inverse_vol(cov)
    for _ in range(max_iter):
        m = cov @ w
        w_new = 1 / m
        w_new /= w_new.sum()
        w_new = 0.5 * w + 0.5 * w_new                     # damp it, otherwise it can oscillate
        if np.abs(w_new - w).max() < tol:
            return w_new
        w = w_new
    return w


SCHEMES = {'Equal weight': equal_weight, 'Inverse volatility': inverse_vol,
           'Minimum variance': min_variance, 'Risk parity (ERC)': risk_parity}


def backtest(returns, scheme, lookback=252, rebalance='M', cost_bp=2.0, vol_target=None,
             max_leverage=2.0, rf=None):
    """Rebalance at each period end on the trailing `lookback` sessions; weights drift in between.
    With vol_target, exposure = target / forecast vol, capped at max_leverage.
    Unused cash earns rf if I pass it."""
    R = returns.dropna()
    dates = R.index
    key = pd.Series(np.asarray(dates.to_period(rebalance)), index=np.arange(len(dates)))
    # last session of each period is the rebalance day
    reb = key.groupby(key).tail(1).index.values
    reb = reb[(reb >= lookback) & (reb < len(dates) - 1)]   # need a full window and at least one day after
    f = SCHEMES[scheme] if isinstance(scheme, str) else scheme
    n = R.shape[1]
    port = pd.Series(np.nan, index=dates)
    wts = []
    w_drift = np.zeros(n)
    lev_prev = 0.0
    X = R.values
    for k, s in enumerate(reb):
        window = X[s - lookback + 1:s + 1]
        cov = np.cov(window.T)
        w = f(cov)
        lev = 1.0
        if vol_target:
            pv = np.sqrt(w @ cov @ w * TD)
            lev = min(max_leverage, vol_target / pv)
        target = w * lev
        turnover = np.abs(target - w_drift).sum()
        e = reb[k + 1] if k + 1 < len(reb) else len(dates) - 1
        seg = X[s + 1:e + 1]
        growth = np.cumprod(1 + seg, axis=0)
        val = np.concatenate([[1.0], (growth @ target) + (1 - target.sum())])
        if rf is not None and vol_target:
            cash = (1 - target.sum()) * rf.reindex(dates[s + 1:e + 1]).fillna(0).values
            val[1:] += np.cumsum(cash)                    # simple (not compounded) interest on the cash
        pr = val[1:] / val[:-1] - 1
        pr[0] -= turnover * cost_bp / 1e4                 # cost charged on the first day after rebalance
        port.iloc[s + 1:e + 1] = pr
        drift = target * growth[-1]
        w_drift = drift / (drift.sum() + (1 - target.sum())) if drift.sum() > 0 else drift
        wts.append(pd.Series(target, index=R.columns, name=dates[s]))
    return port.dropna(), pd.DataFrame(wts)
