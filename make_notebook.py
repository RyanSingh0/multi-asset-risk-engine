# Writes 02_Multi_Asset_Risk_Report.ipynb. I edit the cells here, not in the notebook, then re-run this.
import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))
md("""# Multi-asset risk report

Risk report on one multi-asset book, using data I already had for the DXY and commodities papers: nine Bloomberg commodity TR sub-indices, the S&P 500, five CME currency futures and ICE Dollar Index futures, Dec 2010 to Dec 2025.

What's in here:
1. Data and the book
2. Portfolio construction: 1/N, inverse vol, min variance, risk parity, each with and without a 10% vol target
3. VaR and ES four ways, with a Kupiec backtest
3b. GARCH and DCC-GARCH VaR, and which model passes a bank-style backtest (Kupiec, Christoffersen, Basel traffic light)
4. Which assets carry the risk
5. Historical stress tests
6. One-month loss distribution: Gaussian MC, Student-t MC, block bootstrap
7. Bootstrap CIs on Sharpe
8. FX: dollar beta, hedge ratios, carry vs spot, correlation in stress

Needs the packages in `requirements.txt`. Run All takes about 4 minutes, most of it the rolling DCC refits in 3b.""")
code("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from risk_engine import data, risk, simulation as sim, portfolio as pf, fx, garch
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 30); pd.set_option('display.precision', 4)
R, rf = data.load_panel()
print(R.shape, R.index[0].date(), '->', R.index[-1].date())
R.describe().T[['mean', 'std', 'min', 'max']].assign(ann_vol=lambda x: x['std'] * np.sqrt(252))""")
md("## 1. The book\n40% equities, 30% commodities (six largest, equal weight), 30% currency futures vs the dollar (equal weight). I rebalance to these weights daily in the risk numbers.")
code("""assets = ['SPX', 'WTI', 'Brent', 'Gold', 'NatGas', 'Soybeans', 'Silver', 'EUR', 'GBP', 'CAD', 'JPY', 'CHF']
W = pd.Series({'SPX': 0.40, **{c: 0.05 for c in ['WTI', 'Brent', 'Gold', 'NatGas', 'Soybeans', 'Silver']},
               **{c: 0.06 for c in ['EUR', 'GBP', 'CAD', 'JPY', 'CHF']}})
X = R[assets]
port = X @ W
pd.DataFrame({'Book': risk.summary(port, rf), 'S&P 500': risk.summary(R.SPX, rf), 'BCOMTR': risk.summary(R.BCOMTR, rf)}).T""")
md("## 2. Portfolio construction\nEach scheme re-estimated monthly on the trailing year, 2 bp per unit of turnover. The vol-target versions scale the book to 10% annual vol, leverage capped at 2x, idle cash earns T-bills.")
code("""rows, curves = {}, {}
for s in pf.SCHEMES:
    for vt in [None, 0.10]:
        r, w = pf.backtest(X, s, lookback=252, rebalance='M', cost_bp=2, vol_target=vt, rf=rf)
        name = s + (' + 10% vol target' if vt else '')
        rows[name] = risk.summary(r, rf); curves[name] = (1 + r).cumprod()
tab = pd.DataFrame(rows).T
fig, ax = plt.subplots(figsize=(10, 4))
for k, v in curves.items():
    ax.plot(v.index, v, lw=1.2 if 'target' in k else 0.8, label=k)
ax.set_yscale('log'); ax.set_title('Growth of $1 by construction method'); ax.legend(fontsize=7, ncol=2, frameon=False)
ax.spines[['top', 'right']].set_visible(False); plt.show()
tab""")
md("## 3. VaR and expected shortfall\n1-day 99% VaR four ways, 97.5% ES (what Basel FRTB uses), and a rolling 500-day backtest with Kupiec's proportion-of-failures test.")
code("""m = {'Historical': risk.var_historical(port, .99), 'Normal': risk.var_normal(port, .99),
     'Cornish-Fisher': risk.var_cornish_fisher(port, .99), 'Filtered historical (EWMA)': risk.var_filtered_historical(port, .99)}
es = {'ES 97.5% historical': risk.es_historical(port, .975), 'ES 97.5% normal': risk.es_normal(port, .975)}
display(pd.Series(m, name='1-day 99% VaR (fraction of book)').to_frame().T); display(pd.Series(es).to_frame('value').T)
bt = pd.DataFrame({meth: {k: v for k, v in risk.var_backtest(port, .99, 500, meth).items() if k != 'forecasts'}
                   for meth in ['historical', 'normal', 'cornish_fisher']}).T
bt""")
md("""## 3b. Forecasting VaR: GARCH and DCC-GARCH
Eight one-day 99% VaR models, all forecast out of sample: each is estimated on the previous 1,000 days, GARCH refit every 21 days and DCC every 126. Then three checks a bank's model validation team would run:
* **Kupiec**: right number of breaches?
* **Christoffersen independence**: do breaches come in clusters? A model that breaches five days in a row is useless even if the yearly count is fine.
* **Basel traffic light**: share of rolling 250-day windows in the green (0-4 breaches), yellow (5-9) and red (10+) zone.

The "FHS" versions keep the GARCH or DCC volatility but take the tail quantile from the model's own standardised residuals instead of assuming a symmetric t or normal.""")
code("""F = {m: risk.var_backtest(port, .99, 1000, m)['forecasts'] for m in ['historical', 'normal', 'cornish_fisher', 'filtered_historical']}
F['garch_t'] = garch.garch_var_forecasts(port, .99, 1000, 21, 't')
F['garch_fhs'] = garch.garch_var_forecasts(port, .99, 1000, 21, 't', 'empirical')
F['dcc_garch'], F['dcc_garch_fhs'], dcc_corr = garch.dcc_var_forecasts(X, W, .99, 1000, 126)
cmp_tab = risk.compare_var_models(port, F, .99)
cmp_tab[['breaches', 'expected', 'kupiec_p', 'independence_p', 'cond_coverage_p', 'back_to_back_breaches',
         'share_green', 'share_yellow', 'share_red', 'avg_VaR']]""")
code("""# what the forecasts looked like through COVID
seg = slice('2020-01-15', '2020-06-30')
Fd = pd.DataFrame(F)
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(port.loc[seg].index, port.loc[seg], color='#9DB9D5', width=1.0, label='daily return of the book')
for k, c in [('historical', '#D62728'), ('garch_fhs', '#0B3C5D'), ('dcc_garch_fhs', '#E69F00')]:
    ax.plot(Fd.loc[seg].index, -Fd.loc[seg, k], lw=1.3, color=c, label=f'-VaR {k}')
ax.set_title('One-day 99% VaR vs realised returns, spring 2020'); ax.legend(frameon=False, fontsize=8, ncol=2)
ax.spines[['top', 'right']].set_visible(False); plt.show()
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.plot(dcc_corr.index, dcc_corr, color='#0B3C5D', lw=0.9); ax.set_title('DCC average pairwise correlation of the 12 assets')
ax.spines[['top', 'right']].set_visible(False); plt.show()""")
md("## 4. Where the risk comes from\nEuler split of book vol on the full-sample covariance. I expect the 40% equity sleeve to carry most of it.")
code("""cov = X.cov().values * 252
rc, vol = risk.risk_contributions(W.values, cov)
rc.index = assets
print(f'Portfolio volatility {vol:.2%}')
ax = rc[['weight', 'share']].plot.bar(figsize=(10, 3), color=['#9DB9D5', '#0B3C5D']); ax.set_title('Weight vs share of risk')
ax.spines[['top', 'right']].set_visible(False); plt.show()
rc""")
md("## 5. Historical stress tests")
code("""risk.historical_stress(X, W)""")
md("## 6. One-month loss distribution: Monte Carlo vs block bootstrap\nSame 21-day horizon, three generators. The bootstrap keeps fat tails and vol clustering with no distribution assumed. Gaussian is the baseline and usually understates the tail.")
code("""gens = {'Gaussian MC': sim.mc_gaussian(X.values, 20000, 21, seed=1),
        'Student-t MC': sim.mc_student_t(X.values, 20000, 21, seed=1),
        'Block bootstrap (21d blocks)': sim.block_bootstrap(X.values, 20000, 21, 21, seed=1)}
out, ends = {}, {}
for k, S in gens.items():
    p = sim.portfolio_paths(S, W.values); out[k] = sim.horizon_risk(p); ends[k] = p[:, -1] - 1
display(pd.DataFrame(out).T)
fig, ax = plt.subplots(figsize=(10, 3.5))
for k, e in ends.items():
    ax.hist(e, bins=200, histtype='step', density=True, label=k)
ax.set_xlim(-0.2, 0.15); ax.set_title('Simulated 21-day return of the book'); ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False); plt.show()""")
code("""# Fan chart, one year of block-bootstrap paths
S = sim.block_bootstrap(X.values, 5000, 252, 21, seed=7)
P = sim.portfolio_paths(S, W.values)
q = np.quantile(P, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
fig, ax = plt.subplots(figsize=(10, 3.5)); d = np.arange(1, 253)
ax.fill_between(d, q[0], q[4], color='#C4D8EC', label='5-95%'); ax.fill_between(d, q[1], q[3], color='#7FA7CF', label='25-75%')
ax.plot(d, q[2], color='#0B3C5D', label='median'); ax.axhline(1, color='black', lw=0.6)
ax.set_title('One-year value of $1: block-bootstrap fan chart'); ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False); plt.show()""")
md("## 7. How sure am I of a Sharpe ratio?\nBlock-bootstrap 95% intervals. If the interval includes zero, the backtest can't tell the strategy apart from no skill.")
code("""ci = {}
for name, r in {'Book': port, 'Risk parity': pf.backtest(X, 'Risk parity (ERC)', rf=rf)[0], 'S&P 500': R.SPX, 'BCOMTR': R.BCOMTR}.items():
    est, lo, hi, _ = sim.bootstrap_ci(r.values, sim.sharpe, n_boot=2000, mean_block=21, seed=3)
    ci[name] = {'Sharpe': est, '95% low': lo, '95% high': hi}
pd.DataFrame(ci).T""")
md("## 8. FX\n(a) Rolling 6-month beta of each asset to DX futures. (b) Min-variance hedge ratio of each currency against DX. (c) Carry vs spot in the currency futures returns. (d) Does diversification hold up in stress?")
code("""beta = fx.dollar_exposure(R[assets + ['DXY']], 'DXY', 126)
full = pd.Series({c: risk.beta(R[c], R.DXY) for c in assets}, name='full-sample beta to DXY')
fig, ax = plt.subplots(figsize=(10, 3.5))
for c in ['SPX', 'Gold', 'WTI', 'EUR', 'JPY']:
    ax.plot(beta.index, beta[c], lw=0.9, label=c)
ax.axhline(0, color='black', lw=0.6); ax.set_title('Rolling 126-day beta to the Dollar Index'); ax.legend(frameon=False, ncol=5)
ax.spines[['top', 'right']].set_visible(False); plt.show()
book_beta = risk.beta(port, R.DXY)
h = {c: fx.min_variance_hedge(R[c], R.DXY) for c in ['EUR', 'GBP', 'CAD', 'JPY', 'CHF', 'Gold', 'SPX']}
print(f'Book beta to the dollar: {book_beta:.2f}')
display(full.to_frame().T)
pd.DataFrame(h, index=['hedge ratio vs DXY', 'variance reduction']).T""")
code("""summ, _ = fx.carry_attribution(R[['EUR', 'GBP', 'CAD', 'JPY', 'CHF']], data.interest_differentials())
display(summ)
cr = fx.correlation_regimes(X, 'SPX')
print(f"Average pairwise correlation: calm {cr['calm']:.2f} vs stressed {cr['stressed']:.2f} (stress = top 20% of trailing S&P 500 volatility)")""")
md("""## Notes
* All the math lives in `risk_engine/` with its own tests. This notebook only calls it.
* Gaussian MC, Student-t MC (sample or EWMA cov) and the block bootstrap all return the same shape, so sections 6 and 7 reuse one set of functions.
* VaR / ES four ways + Kupiec, Euler risk split, historical stress tests.
* GARCH(1,1)-t and DCC(1,1) written from scratch with scipy, backtested out of sample with Kupiec, Christoffersen and the Basel traffic light.
* Construction: 1/N, inverse vol, min variance, risk parity, with vol targeting and costs.
* FX: dollar beta, min-variance hedge ratios, carry vs spot, correlation in stress.

Next: pull SPXT (S&P total return) and a bond future from Bloomberg; expected shortfall backtest (Acerbi-Szekely); factor model (dollar, carry, momentum, equity beta) for attribution; hook the engine into the SQL database and dashboard.""")
nb['cells'] = C
nb['metadata'] = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}}
nbf.write(nb, '02_Multi_Asset_Risk_Report.ipynb')
