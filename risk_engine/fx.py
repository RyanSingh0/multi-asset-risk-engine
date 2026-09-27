"""FX checks for a USD book.

dollar_exposure       rolling beta of each asset to DX futures
min_variance_hedge    Ederington (1979) hedge ratio against a currency future
carry_attribution     split FX futures returns into carry (rate differential) and spot
correlation_regimes   avg cross-asset correlation, calm vs stressed
"""
import numpy as np
import pandas as pd

TD = 252


def rolling_beta(r, factor, window=126):
    j = pd.concat([r.rename('r'), factor.rename('f')], axis=1).dropna()
    c = j['r'].rolling(window).cov(j['f'])
    v = j['f'].rolling(window).var()
    return (c / v).rename(r.name)


def dollar_exposure(returns, dxy_col='DXY', window=126):
    f = returns[dxy_col]
    return pd.DataFrame({c: rolling_beta(returns[c], f, window) for c in returns.columns if c != dxy_col})


def min_variance_hedge(asset, fx_future):
    """h* = cov(asset, fx) / var(fx), the hedge that minimises var(asset - h * fx).
    Returns (h, variance reduction)."""
    j = pd.concat([asset, fx_future], axis=1).dropna()
    a, f = j.iloc[:, 0], j.iloc[:, 1]
    h = np.cov(a, f)[0, 1] / f.var()
    red = 1 - (a - h * f).var() / a.var()
    return float(h), float(red)


def carry_attribution(fx_returns, differentials):
    """Split FX futures excess returns into carry and spot.

    Quotes are USD per foreign unit. CIP gives ln F = ln S + (r_usd - r_for) * tau, so over time
    d ln F = d ln S + (r_for - r_usd) dt. A long future earns spot plus (foreign - US) as it converges.
    I take carry = daily differential accrual; whatever is left is spot plus basis noise."""
    out = {}
    for c in fx_returns.columns:
        if c not in differentials:
            continue
        carry = differentials[c].reindex(fx_returns.index).ffill() / TD
        tot = fx_returns[c]
        out[c] = pd.DataFrame({'total': tot, 'carry (rate differential)': carry, 'spot and other': tot - carry})
    summ = pd.DataFrame({c: {'Total return (ann.)': v['total'].mean() * TD,
                             'Carry component (ann.)': v['carry (rate differential)'].mean() * TD,
                             'Spot and other (ann.)': v['spot and other'].mean() * TD,
                             'Total vol (ann.)': v['total'].std() * np.sqrt(TD)} for c, v in out.items()}).T
    return summ, out


def correlation_regimes(returns, vol_col, window=63, q=0.8):
    """Avg pairwise correlation when trailing vol of `vol_col` is in its top (1-q) ('stressed')
    vs the rest ('calm'). If diversification breaks in stress, the stressed number is higher."""
    vol = returns[vol_col].rolling(window).std().shift(1)
    thr = vol.quantile(q)
    calm = returns[vol <= thr]
    stress = returns[vol > thr]
    def avg(c):
        m = c.corr().values
        return m[np.triu_indices_from(m, 1)].mean()
    return {'calm': float(avg(calm)), 'stressed': float(avg(stress)), 'threshold_vol_ann': float(thr * np.sqrt(TD)),
            'calm_corr': calm.corr(), 'stressed_corr': stress.corr()}
