"""Engine tests. Each one checks something that should hold exactly or within a known tolerance.
Run from the Multi-Asset Risk Engine folder: python -m pytest -q tests"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from risk_engine import risk, simulation as sim, portfolio as pf, fx

rng = np.random.default_rng(1)   # fixed seed so the tolerances below stay stable


def test_normal_var_matches_closed_form():
    r = rng.normal(0.0005, 0.01, 200000)
    assert abs(risk.var_normal(r, 0.99) - (-(0.0005 + 0.01 * stats.norm.ppf(0.01)))) < 2e-4
    assert abs(risk.var_historical(r, 0.99) - risk.var_normal(r, 0.99)) < 5e-4
    assert abs(risk.es_historical(r, 0.975) - risk.es_normal(r, 0.975)) < 5e-4


def test_cornish_fisher_reduces_to_normal_for_gaussian():
    r = rng.normal(0, 0.01, 400000)
    assert abs(risk.var_cornish_fisher(r, 0.99) - risk.var_normal(r, 0.99)) < 3e-4


def test_cornish_fisher_larger_for_fat_left_tail():
    r = -rng.lognormal(0, 0.8, 200000) * 0.005 + 0.005
    assert risk.var_cornish_fisher(r, 0.99) > risk.var_normal(r, 0.99)


def test_risk_contributions_sum_to_vol():
    A = rng.normal(size=(500, 5)); cov = np.cov(A.T)
    w = rng.random(5); w /= w.sum()
    tab, vol = risk.risk_contributions(w, cov)
    assert np.isclose(tab.contribution.sum(), vol) and np.isclose(tab.share.sum(), 1)


def test_risk_parity_equalises_contributions():
    A = rng.normal(size=(1000, 6)) * np.array([1, 2, 3, 0.5, 1.5, 4]); cov = np.cov(A.T) + 0.1
    w = pf.risk_parity(cov)
    tab, _ = risk.risk_contributions(w, cov)
    assert np.allclose(tab.share, 1 / 6, atol=1e-6) and np.isclose(w.sum(), 1) and (w > 0).all()


def test_min_variance_beats_equal_weight():
    A = rng.normal(size=(1000, 4)) * np.array([1, 2, 3, 4]); cov = np.cov(A.T)
    w = pf.min_variance(cov)
    assert w @ cov @ w <= pf.equal_weight(cov) @ cov @ pf.equal_weight(cov) + 1e-12
    assert np.isclose(w.sum(), 1) and (w >= -1e-9).all()


def test_mc_gaussian_recovers_moments():
    R = rng.multivariate_normal([0.001, -0.0005], [[1e-4, 3e-5], [3e-5, 4e-4]], 5000)
    S = sim.mc_gaussian(R, n_paths=4000, horizon=50, seed=2).reshape(-1, 2)
    assert np.allclose(S.mean(0), R.mean(0), atol=2e-4)
    assert np.allclose(np.cov(S.T), np.cov(R.T), rtol=0.05)


def test_student_t_keeps_covariance_and_adds_kurtosis():
    R = rng.standard_t(5, size=(20000, 2)) * 0.01
    S = sim.mc_student_t(R, n_paths=2000, horizon=50, seed=3).reshape(-1, 2)
    assert np.allclose(np.cov(S.T), np.cov(R.T), rtol=0.05, atol=5e-6)
    assert stats.kurtosis(S[:, 0]) > 1


def test_block_bootstrap_uses_only_observed_rows():
    R = rng.normal(size=(300, 3))
    S = sim.block_bootstrap(R, n_paths=200, horizon=40, mean_block=10, seed=4).reshape(-1, 3)
    rows = {tuple(x) for x in np.round(R, 12)}
    assert all(tuple(x) in rows for x in np.round(S, 12))


def test_bootstrap_blocks_have_expected_length():
    idx = sim.stationary_bootstrap_indices(10000, 2000, 200, mean_block=20, seed=5)
    jumps = (np.diff(idx, axis=1) != 1).mean()
    assert abs(jumps - 1 / 20) < 0.005


def test_portfolio_paths_daily_rebalance_equals_weighted_returns():
    S = rng.normal(0, 0.01, (10, 30, 3)); w = np.array([0.5, 0.3, 0.2])
    p = sim.portfolio_paths(S, w)
    assert np.allclose(p[:, -1], np.prod(1 + S @ w, axis=1))


def test_min_variance_hedge_ratio():
    f = pd.Series(rng.normal(0, 0.01, 5000)); a = 0.7 * f + pd.Series(rng.normal(0, 0.005, 5000))
    h, red = fx.min_variance_hedge(a, f)
    assert abs(h - 0.7) < 0.02 and abs(red - 0.49 / 0.74) < 0.03   # expected R^2 = 0.7^2*1e-4 / (0.49e-4 + 0.25e-4)


def test_backtest_equal_weight_no_cost_matches_manual():
    idx = pd.bdate_range('2020-01-01', periods=600)
    R = pd.DataFrame(rng.normal(0.0003, 0.01, (600, 3)), index=idx, columns=list('abc'))
    port, W = pf.backtest(R, 'Equal weight', lookback=100, rebalance='M', cost_bp=0)
    s = W.index[0]; e = W.index[1]
    # rebuild the first holding period by hand: buy on day s, hold to e
    seg = R.loc[s:e].iloc[1:]
    val = (np.cumprod(1 + seg.values, axis=0) @ np.full(3, 1 / 3))
    manual = np.concatenate([[val[0] - 1], val[1:] / val[:-1] - 1])
    assert np.allclose(port.loc[seg.index].values, manual)


def test_var_backtest_calibrated_on_iid_normal():
    r = pd.Series(rng.normal(0, 0.01, 4000))
    # normal VaR on iid normal data should pass Kupiec
    bt = risk.var_backtest(r, 0.99, window=500, method='normal')
    assert bt['kupiec_p'] > 0.01


# ---------------------------------
# GARCH / DCC / VaR model backtests
# ---------------------------------
from risk_engine import garch


def _sim_garch(T, omega=0.02, alpha=0.07, beta=0.90, nu=6, seed=5):
    g = np.random.default_rng(seed)
    e = np.zeros(T); s2 = np.full(T, omega / (1 - alpha - beta))
    for t in range(1, T):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * g.standard_t(nu) * np.sqrt((nu - 2) / nu)
    return e / 100                                     # percent -> fraction, like real returns


def test_garch_recovers_parameters():
    p = garch.fit_garch(_sim_garch(20000), 't')
    assert abs(p['alpha'] - 0.07) < 0.02 and abs(p['beta'] - 0.90) < 0.03 and abs(p['nu'] - 6) < 1.5


def test_garch_forecast_uses_only_past():
    # changing the last return must not change the last forecast (it only uses returns up to t-1)
    r = _sim_garch(3000)
    p = garch.fit_garch(r[:2000])
    a = garch.garch_filter(r, p); r2 = r.copy(); r2[-1] *= 50
    assert np.isclose(a[-1], garch.garch_filter(r2, p)[-1])


def test_christoffersen_flags_clustered_breaches():
    g = np.random.default_rng(11)
    iid = (g.random(5000) < 0.01).astype(int)
    clustered = np.zeros(5000, int)
    for s in g.choice(np.arange(0, 4990, 10), 10, replace=False):
        clustered[s:s + 5] = 1                         # 50 breaches in 10 runs of 5
    assert risk.christoffersen(iid)['independence_p'] > 0.01
    assert risk.christoffersen(clustered)['independence_p'] < 0.001


def test_basel_zones():
    h = np.zeros(1000, int); h[-7:] = 1
    assert risk.basel_zone(h)['latest_zone'] == 'yellow'
    h[-12:] = 1
    assert risk.basel_zone(h)['latest_zone'] == 'red'


def test_dcc_on_constant_correlation():
    # iid normals with fixed correlation: DCC should find almost no dynamics and the right Qbar
    g = np.random.default_rng(3)
    C = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]])
    Z = g.multivariate_normal(np.zeros(3), C, 3000) * 0.01
    f = garch.fit_dcc(pd.DataFrame(Z, columns=list('abc')))
    assert f['a'] < 0.03 and np.allclose(f['Qbar'], C, atol=0.05)
