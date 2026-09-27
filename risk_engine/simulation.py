"""Scenario generators: Gaussian and Student-t Monte Carlo (sample or EWMA cov) and the
Politis-Romano (1994) stationary block bootstrap.

Every generator returns DAILY asset returns shaped (n_paths, horizon, n_assets), so the
path and risk functions below don't care which one made the scenarios.
"""
import numpy as np
import pandas as pd


def ewma_cov(returns, lam=0.94):
    # seed with the first 50 days' sample cov, then run the EWMA update
    x = np.asarray(returns) - np.asarray(returns).mean(0)
    S = np.cov(x[:50].T)
    for t in range(50, len(x)):
        S = lam * S + (1 - lam) * np.outer(x[t], x[t])
    return S


def mc_gaussian(returns, n_paths=10000, horizon=21, cov='sample', seed=0):
    rng = np.random.default_rng(seed)
    R = np.asarray(returns)
    mu = R.mean(0)
    S = np.cov(R.T) if cov == 'sample' else ewma_cov(R)
    L = np.linalg.cholesky(S + 1e-14 * np.eye(len(S)))   # tiny jitter so Cholesky doesn't fail
    z = rng.standard_normal((n_paths, horizon, R.shape[1]))
    return mu + z @ L.T


def mc_student_t(returns, n_paths=10000, horizon=21, df=None, cov='sample', seed=0):
    """Multivariate t with the SAME cov as the data. If df is None I fit it from median kurtosis."""
    rng = np.random.default_rng(seed)
    R = np.asarray(returns)
    mu = R.mean(0)
    S = np.cov(R.T) if cov == 'sample' else ewma_cov(R)
    if df is None:
        from scipy import stats
        k = np.median([stats.kurtosis(R[:, j]) for j in range(R.shape[1])])
        df = max(4.5, 4 + 6 / max(k, 1e-6))              # t excess kurtosis = 6/(df-4); floor at 4.5
    # scale by (df-2)/df so the t draws end up with cov S, not S * df/(df-2)
    L = np.linalg.cholesky(S * (df - 2) / df + 1e-14 * np.eye(len(S)))
    z = rng.standard_normal((n_paths, horizon, R.shape[1]))
    w = rng.chisquare(df, (n_paths, horizon, 1)) / df
    return mu + (z @ L.T) / np.sqrt(w)


def stationary_bootstrap_indices(T, n_paths, horizon, mean_block=21, seed=0):
    """Stationary bootstrap indices: geometric block lengths (mean `mean_block`), random starts,
    wrapping past the end of the sample."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    idx[:, 0] = rng.integers(T, size=n_paths)
    new_block = rng.random((n_paths, horizon)) < p
    starts = rng.integers(T, size=(n_paths, horizon))
    for h in range(1, horizon):
        idx[:, h] = np.where(new_block[:, h], starts[:, h], (idx[:, h - 1] + 1) % T)
    return idx


def block_bootstrap(returns, n_paths=10000, horizon=21, mean_block=21, seed=0):
    """Resample whole days (all assets together), so correlations, fat tails and vol clustering
    inside a block survive. No distribution assumed."""
    R = np.asarray(returns)
    idx = stationary_bootstrap_indices(len(R), n_paths, horizon, mean_block, seed)
    return R[idx]


def portfolio_paths(sim, weights, rebalance='daily'):
    """Book value paths starting at 1. 'daily' = constant weights, 'none' = buy and hold."""
    w = np.asarray(weights, float)
    if rebalance == 'daily':
        pr = sim @ w
        return np.cumprod(1 + pr, axis=1)
    growth = np.cumprod(1 + sim, axis=1)                 # (paths, horizon, assets)
    return growth @ w


def horizon_risk(paths, levels=(0.95, 0.99)):
    """VaR / ES of the end-of-horizon return, plus P(drawdown > x) along the path."""
    end = paths[:, -1] - 1
    out = {'Mean horizon return': float(end.mean()), 'Median horizon return': float(np.median(end)),
           'P(loss)': float((end < 0).mean())}
    for lv in levels:
        q = np.quantile(end, 1 - lv)
        out[f'VaR {int(lv * 100)}%'] = float(-q)
        out[f'ES {int(lv * 100)}%'] = float(-end[end <= q].mean())
    # prepend the starting value 1 so a path that falls from day 1 counts as a drawdown
    peak = np.maximum.accumulate(np.concatenate([np.ones((len(paths), 1)), paths], axis=1), axis=1)[:, 1:]
    mdd = (paths / peak - 1).min(axis=1)
    for d in (0.05, 0.10, 0.20):
        out[f'P(drawdown > {int(d * 100)}%)'] = float((mdd < -d).mean())
    return out


def bootstrap_ci(x, stat, n_boot=5000, mean_block=21, level=0.95, seed=0):
    """Block-bootstrap CI for any stat of a return series (I mostly use it for Sharpe)."""
    x = np.asarray(x)
    idx = stationary_bootstrap_indices(len(x), n_boot, len(x), mean_block, seed)
    vals = np.array([stat(x[i]) for i in idx])
    a = (1 - level) / 2
    return float(stat(x)), float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a)), vals


def sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(252))
