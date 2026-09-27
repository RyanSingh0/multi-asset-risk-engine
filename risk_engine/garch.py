"""GARCH(1,1) and DCC(1,1), fitted by maximum likelihood with scipy, no extra packages.

I work in percent returns inside the fits (numbers near 1 make the optimiser much happier) and convert
back to fractions on the way out.

GARCH(1,1):  s2[t] = omega + alpha * e[t-1]^2 + beta * s2[t-1],   e = r - mu
DCC(1,1):    Q[t] = (1 - a - b) * Qbar + a * z[t-1] z[t-1]' + b * Q[t-1],   R[t] = Q[t] scaled to a correlation matrix
where z are the GARCH-standardised residuals (Engle 2002, two-step estimator).
"""
import numpy as np
import pandas as pd
from scipy import optimize, signal, special, stats


# ---------------------------------
# 1) UNIVARIATE GARCH(1,1)
# ---------------------------------
def _variance_path(e, omega, alpha, beta, s0):
    # s2[t] = omega + alpha*e[t-1]^2 + beta*s2[t-1], started from s0. lfilter does the recursion in C.
    e2_prev = np.concatenate([[s0], e[:-1] ** 2])
    x = omega + alpha * e2_prev
    return signal.lfilter([1.0], [1.0, -beta], x, zi=[beta * s0])[0]


def _nll(theta, r, dist):
    mu, omega, alpha, beta = theta[:4]
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e10
    e = r - mu
    s2 = _variance_path(e, omega, alpha, beta, r.var())
    if np.any(s2 <= 0):
        return 1e10
    if dist == 'normal':
        return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2)
    nu = theta[4]
    if nu <= 2.05:
        return 1e10
    # Student-t scaled to unit variance, so s2 is still the conditional variance
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return -np.sum(c - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log1p(e ** 2 / (s2 * (nu - 2))))


def fit_garch(r, dist='t'):
    """Fit GARCH(1,1) to a return series (fractions). Returns a dict of parameters in percent units."""
    x = 100 * np.asarray(pd.Series(r).dropna(), float)
    v = x.var()
    start = [x.mean(), 0.05 * v, 0.08, 0.90] + ([8.0] if dist == 't' else [])
    bounds = [(None, None), (1e-8, 10 * v), (0, 0.5), (0, 0.9999)] + ([(2.1, 200)] if dist == 't' else [])
    res = optimize.minimize(_nll, start, args=(x, dist), method='L-BFGS-B', bounds=bounds)
    # L-BFGS-B often stops early on nu (flat likelihood), so I always polish with Nelder-Mead.
    # Checked against three different starting values of nu, all land on the same optimum.
    res = optimize.minimize(_nll, res.x, args=(x, dist), method='Nelder-Mead',
                            options={'maxiter': 20000, 'xatol': 1e-8, 'fatol': 1e-8})
    p = dict(zip(['mu', 'omega', 'alpha', 'beta', 'nu'], res.x))
    p['dist'] = dist
    p['loglik'] = -res.fun
    p['persistence'] = p['alpha'] + p['beta']
    p['long_run_vol'] = float(np.sqrt(p['omega'] / (1 - p['persistence']) * 252) / 100)   # annualised, fraction
    return p


def garch_filter(r, p):
    """Conditional vol (fraction) for each day given the params. Value on day t only uses returns up to t-1,
    so it is a genuine one-step-ahead forecast."""
    x = 100 * np.asarray(r, float)
    s2 = _variance_path(x - p['mu'], p['omega'], p['alpha'], p['beta'], x.var())
    return np.sqrt(s2) / 100


def std_t_quantile(q, nu):
    # quantile of a Student-t rescaled to unit variance
    return stats.t.ppf(q, nu) * np.sqrt((nu - 2) / nu)


def garch_var_forecasts(r, level=0.99, window=1000, refit=21, dist='t', quantile='model'):
    """Rolling one-day VaR from GARCH(1,1). Params re-estimated every `refit` days on the last `window` days,
    then held fixed while I filter forward. Positive number = loss.

    quantile='model' uses the fitted t (or normal). quantile='empirical' uses the actual 1% quantile of the
    standardised residuals in the fit window (filtered historical simulation on GARCH vol), which picks up
    the negative skew a symmetric t can't."""
    r = pd.Series(r).dropna()
    x = r.values
    out = pd.Series(np.nan, index=r.index)
    for t0 in range(window, len(x), refit):
        t1 = min(t0 + refit, len(x))
        p = fit_garch(x[t0 - window:t0], dist)
        s_all = garch_filter(x[t0 - window:t1], p)
        s = s_all[window:]                                          # forecasts for days t0..t1-1
        if quantile == 'empirical':
            z = (x[t0 - window:t0] - p['mu'] / 100) / s_all[:window]
            q = np.quantile(z, 1 - level)
        else:
            q = std_t_quantile(1 - level, p['nu']) if dist == 't' else stats.norm.ppf(1 - level)
        out.iloc[t0:t1] = -(p['mu'] / 100 + s * q)
    return out


# ---------------------------------
# 2) DCC(1,1)
# ---------------------------------
def _dcc_nll(ab, Z, Qbar):
    a, b = ab
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    T, N = Z.shape
    Q = Qbar.copy()
    nll = 0.0
    for t in range(T):
        if t > 0:
            z = Z[t - 1][:, None]
            Q = (1 - a - b) * Qbar + a * (z @ z.T) + b * Q
        d = 1 / np.sqrt(np.diag(Q))
        R = Q * np.outer(d, d)
        L = np.linalg.cholesky(R)
        y = np.linalg.solve(L, Z[t])
        nll += 2 * np.log(np.diag(L)).sum() + y @ y - Z[t] @ Z[t]
    return 0.5 * nll


def fit_dcc(returns, dist='normal'):
    """Two-step DCC: a GARCH(1,1) per asset, then (a, b) on the standardised residuals.
    Returns the per-asset params, a, b and Qbar."""
    X = pd.DataFrame(returns).dropna()
    params, Z = {}, []
    for c in X.columns:
        p = fit_garch(X[c], dist)
        params[c] = p
        Z.append((X[c].values - p['mu'] / 100) / garch_filter(X[c].values, p))
    Z = np.column_stack(Z)
    Qbar = np.corrcoef(Z.T)
    res = optimize.minimize(_dcc_nll, [0.02, 0.95], args=(Z, Qbar), method='L-BFGS-B',
                            bounds=[(0, 0.5), (0, 0.998)])
    a, b = res.x
    return {'garch': params, 'a': float(a), 'b': float(b), 'Qbar': Qbar, 'columns': list(X.columns),
            'loglik': -float(res.fun)}


def dcc_path(returns, fit):
    """Conditional vols and correlations for each day, one step ahead (day t uses data to t-1).
    Returns (vols T x N, correlations T x N x N)."""
    X = pd.DataFrame(returns)[fit['columns']].values
    T, N = X.shape
    V = np.column_stack([garch_filter(X[:, i], fit['garch'][c]) for i, c in enumerate(fit['columns'])])
    mu = np.array([fit['garch'][c]['mu'] / 100 for c in fit['columns']])
    Z = (X - mu) / V
    a, b, Qbar = fit['a'], fit['b'], fit['Qbar']
    Rs = np.empty((T, N, N))
    Q = Qbar.copy()
    for t in range(T):
        if t > 0:
            z = Z[t - 1][:, None]
            Q = (1 - a - b) * Qbar + a * (z @ z.T) + b * Q
        d = 1 / np.sqrt(np.diag(Q))
        Rs[t] = Q * np.outer(d, d)
    return V, Rs


def dcc_var_forecasts(returns, weights, level=0.99, window=1000, refit=126):
    """Rolling one-day portfolio VaR from DCC-GARCH. Refit every `refit` days on the last `window` days.

    Returns two VaR series, one with the normal quantile on the portfolio vol and one with the empirical
    quantile of the portfolio's standardised residuals in the fit window, plus the average pairwise
    correlation each day (useful on its own as a "diversification is breaking down" gauge)."""
    X = pd.DataFrame(returns).dropna()
    w = pd.Series(weights).reindex(X.columns).fillna(0).values
    var_n = pd.Series(np.nan, index=X.index)
    var_e = pd.Series(np.nan, index=X.index)
    avg_corr = pd.Series(np.nan, index=X.index)
    zn = stats.norm.ppf(1 - level)
    iu = np.triu_indices(X.shape[1], 1)
    rp = X.values @ w
    for t0 in range(window, len(X), refit):
        t1 = min(t0 + refit, len(X))
        fit = fit_dcc(X.iloc[t0 - window:t0])
        V, Rs = dcc_path(X.iloc[t0 - window:t1], fit)
        mu = np.array([fit['garch'][c]['mu'] / 100 for c in fit['columns']])
        H = Rs * V[:, :, None] * V[:, None, :]
        sp = np.sqrt(np.einsum('i,tij,j->t', w, H, w))
        ze = np.quantile((rp[t0 - window:t0] - w @ mu) / sp[:window], 1 - level)
        var_n.iloc[t0:t1] = -(w @ mu + sp[window:] * zn)
        var_e.iloc[t0:t1] = -(w @ mu + sp[window:] * ze)
        avg_corr.iloc[t0:t1] = Rs[window:, iu[0], iu[1]].mean(1)
    return var_n, var_e, avg_corr
