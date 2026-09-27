"""VaR / ES (four ways), drawdowns, risk decomposition, stress tests.

Inputs are simple daily returns. VaR and ES come out as POSITIVE loss fractions of the book,
so 0.02 means a 2% loss.
"""
import numpy as np
import pandas as pd
from scipy import stats

TD = 252


# ---------------------------------
# 1) VALUE AT RISK
# ---------------------------------
def var_historical(r, level=0.99, horizon=1):
    x = _horizon_returns(r, horizon)
    return float(-np.quantile(x, 1 - level))


def es_historical(r, level=0.99, horizon=1):
    x = _horizon_returns(r, horizon)
    q = np.quantile(x, 1 - level)
    return float(-x[x <= q].mean())


def var_normal(r, level=0.99, horizon=1):
    mu, sd = np.mean(r) * horizon, np.std(r, ddof=1) * np.sqrt(horizon)
    return float(-(mu + sd * stats.norm.ppf(1 - level)))


def es_normal(r, level=0.99, horizon=1):
    mu, sd = np.mean(r) * horizon, np.std(r, ddof=1) * np.sqrt(horizon)
    a = 1 - level
    return float(-(mu - sd * stats.norm.pdf(stats.norm.ppf(a)) / a))


def var_cornish_fisher(r, level=0.99, horizon=1):
    """Modified VaR (Zangari 1996): normal quantile corrected for skew and excess kurtosis."""
    r = np.asarray(r)
    mu, sd = r.mean(), r.std(ddof=1)
    s, k = stats.skew(r), stats.kurtosis(r)
    z = stats.norm.ppf(1 - level)
    zc = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    return float(-(mu * horizon + zc * sd * np.sqrt(horizon)))


def ewma_vol(r, lam=0.94):
    """RiskMetrics EWMA vol. Forecast for day t only uses returns up to t-1."""
    r = pd.Series(r)
    v = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(v.shift(1))


def var_filtered_historical(r, level=0.99, lam=0.94):
    """Filtered HS: divide returns by EWMA vol, take the quantile of that, scale back up by today's vol."""
    r = pd.Series(r).dropna()
    s = ewma_vol(r, lam)
    z = (r / s).dropna()
    last_vol = np.sqrt(((r ** 2).ewm(alpha=1 - lam, adjust=False).mean()).iloc[-1])
    return float(-np.quantile(z, 1 - level) * last_vol)


def var_backtest(r, level=0.99, window=500, method='historical'):
    """Rolling 1-day VaR forecasts plus a Kupiec POF test on the breaches."""
    r = pd.Series(r).dropna()
    f = {'historical': var_historical, 'normal': var_normal, 'cornish_fisher': var_cornish_fisher,
         'filtered_historical': var_filtered_historical}[method]
    fc = pd.Series(np.nan, index=r.index)
    vals = r.values
    for t in range(window, len(r)):
        fc.iloc[t] = f(vals[t - window:t], level)     # window ends at t-1, no peeking at day t
    hit = (r < -fc)[fc.notna()]
    n, x = len(hit), int(hit.sum())
    p = 1 - level
    phat = x / n
    lr = -2 * (np.log((1 - p) ** (n - x) * p ** x) - np.log((1 - phat) ** (n - x) * phat ** x)) if 0 < x < n else np.nan
    return {'forecasts': fc, 'breaches': x, 'observations': n, 'expected_breaches': n * p,
            'breach_rate': phat, 'kupiec_LR': lr, 'kupiec_p': float(1 - stats.chi2.cdf(lr, 1)) if lr == lr else np.nan}


def christoffersen(hits, level=0.99):
    """Kupiec (right number of breaches), Christoffersen independence (do breaches cluster?) and the
    conditional coverage test that combines both. hits is a 0/1 series of VaR breaches."""
    h = np.asarray(hits, int)
    n, x, p = len(h), int(h.sum()), 1 - level
    ph = x / n
    ll = lambda q, k, m: (m - k) * np.log(1 - q) + k * np.log(q) if 0 < q < 1 else 0.0
    lr_pof = -2 * (ll(p, x, n) - ll(ph, x, n))
    # transition counts: n01 = no breach then breach, etc.
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum()); n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum()); n11 = int(((prev == 1) & (cur == 1)).sum())
    p01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    p1 = (n01 + n11) / (n00 + n01 + n10 + n11)
    lr_ind = -2 * (ll(p1, n01 + n11, n00 + n01 + n10 + n11) - ll(p01, n01, n00 + n01) - ll(p11, n11, n10 + n11))
    lr_cc = lr_pof + lr_ind
    return {'breaches': x, 'expected': n * p, 'breach_rate': ph,
            'kupiec_p': float(1 - stats.chi2.cdf(lr_pof, 1)),
            'independence_p': float(1 - stats.chi2.cdf(lr_ind, 1)),
            'cond_coverage_p': float(1 - stats.chi2.cdf(lr_cc, 2)),
            'back_to_back_breaches': n11}


def basel_zone(hits, window=250):
    """Basel traffic light on the last 250 days at 99%: green 0-4 breaches, yellow 5-9, red 10+.
    Also the share of all rolling 250-day windows that ended in each zone."""
    h = pd.Series(hits).astype(int)
    roll = h.rolling(window).sum().dropna()
    zone = lambda k: 'green' if k <= 4 else ('yellow' if k <= 9 else 'red')
    shares = roll.map(zone).value_counts(normalize=True).reindex(['green', 'yellow', 'red']).fillna(0)
    return {'latest_zone': zone(int(roll.iloc[-1])), 'latest_breaches_250d': int(roll.iloc[-1]),
            **{f'share_{k}': float(v) for k, v in shares.items()}}


def compare_var_models(r, forecasts, level=0.99):
    """One row per VaR model: Kupiec, Christoffersen and Basel zone on the days every model has a forecast."""
    r = pd.Series(r)
    F = pd.DataFrame(forecasts).dropna()
    rr = r.reindex(F.index)
    rows = {}
    for k in F:
        hits = (rr < -F[k]).astype(int)
        rows[k] = {**christoffersen(hits, level), **basel_zone(hits), 'avg_VaR': float(F[k].mean())}
    return pd.DataFrame(rows).T


def _horizon_returns(r, horizon):
    r = np.asarray(r)
    if horizon == 1:
        return r
    g = np.log1p(r)
    c = np.concatenate([[0], np.cumsum(g)])
    return np.expm1(c[horizon:] - c[:-horizon])          # overlapping windows, so obs are not independent


# ---------------------------------
# 2) PERFORMANCE AND DRAWDOWNS
# ---------------------------------
def summary(r, rf=None):
    r = pd.Series(r).dropna()
    ex = r - (rf.reindex(r.index).fillna(0) if rf is not None else 0)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    yrs = len(r) / TD
    return {'CAGR': eq.iloc[-1] ** (1 / yrs) - 1, 'Volatility': r.std() * np.sqrt(TD),
            'Sharpe (excess)': ex.mean() / ex.std() * np.sqrt(TD), 'Max drawdown': dd.min(),
            'Worst day': r.min(), 'VaR 99% 1d (hist)': var_historical(r, 0.99),
            'ES 97.5% 1d (hist)': es_historical(r, 0.975), 'Skew': stats.skew(r), 'Excess kurtosis': stats.kurtosis(r)}


def drawdown_table(r, top=5):
    eq = (1 + pd.Series(r).dropna()).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    out, used = [], pd.Series(False, index=dd.index)
    for _ in range(top):
        cand = dd[~used]
        if cand.min() >= 0:
            break
        trough = cand.idxmin()
        start = eq.loc[:trough].idxmax()
        after = eq.loc[trough:]
        rec = after[after >= eq.loc[start]]
        end = rec.index[0] if len(rec) else None
        out.append({'Peak': start.date(), 'Trough': trough.date(), 'Recovered': end.date() if end is not None else 'not yet',
                    'Depth': float(dd.loc[trough]), 'Days to trough': int((dd.index.get_loc(trough) - dd.index.get_loc(start))),
                    'Days to recover': int(dd.index.get_loc(end) - dd.index.get_loc(trough)) if end is not None else None})
        used.loc[start:(end if end is not None else dd.index[-1])] = True
    return pd.DataFrame(out)


# ---------------------------------
# 3) RISK DECOMPOSITION
# ---------------------------------
def risk_contributions(w, cov):
    """Euler split of portfolio vol. Contributions add up to the vol exactly."""
    w = np.asarray(w, float)
    cov = np.asarray(cov, float)
    vol = np.sqrt(w @ cov @ w)
    mrc = cov @ w / vol
    crc = w * mrc
    return pd.DataFrame({'weight': w, 'marginal': mrc, 'contribution': crc, 'share': crc / vol}), float(vol)


def beta(r, factor):
    j = pd.concat([pd.Series(r), pd.Series(factor)], axis=1).dropna()
    c = np.cov(j.iloc[:, 0], j.iloc[:, 1])
    return float(c[0, 1] / c[1, 1])


# ---------------------------------
# 4) STRESS TESTS
# ---------------------------------
SCENARIOS = {
    'Taper tantrum (May-Jun 2013)': ('2013-05-21', '2013-06-24'),
    'SNB drops EUR/CHF floor (15 Jan 2015)': ('2015-01-15', '2015-01-15'),
    'Oil collapse (Jun 2014 - Feb 2016)': ('2014-06-20', '2016-02-11'),
    'Volmageddon (Feb 2018)': ('2018-01-26', '2018-02-08'),
    'COVID crash (Feb-Mar 2020)': ('2020-02-19', '2020-03-23'),
    'WTI below zero (Apr 2020)': ('2020-04-14', '2020-04-21'),
    'Inflation and rate shock (2022)': ('2022-01-03', '2022-10-12'),
    'Russia invades Ukraine (Feb-Mar 2022)': ('2022-02-23', '2022-03-08'),
}


def historical_stress(returns, weights, scenarios=SCENARIOS):
    """Cumulative book and asset returns over each named window. Weights rebalanced daily."""
    rows = []
    port = returns @ pd.Series(weights).reindex(returns.columns).fillna(0)
    for name, (a, b) in scenarios.items():
        seg = returns.loc[a:b]
        if seg.empty:
            continue
        row = {'Scenario': name, 'Start': seg.index[0].date(), 'End': seg.index[-1].date(), 'Days': len(seg),
               'Portfolio': float((1 + port.loc[a:b]).prod() - 1)}
        worst = ((1 + seg).prod() - 1).sort_values()
        row['Worst asset'] = f'{worst.index[0]} ({worst.iloc[0]:.1%})'
        row['Best asset'] = f'{worst.index[-1]} ({worst.iloc[-1]:.1%})'
        rows.append(row)
    return pd.DataFrame(rows)


def hypothetical_stress(weights, shocks):
    """Instant shock per asset (0.1 = +10%) -> book P&L."""
    w = pd.Series(weights)
    s = pd.Series(shocks).reindex(w.index).fillna(0)
    return float((w * s).sum())
