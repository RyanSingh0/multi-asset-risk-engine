# Multi-Asset Risk Engine

A small risk engine I built on top of the data I already had from the DXY and commodities projects:
nine Bloomberg commodity total-return sub-indices, BCOMTR and the S&P 500 (from the commodities repo),
five CME currency futures, ICE Dollar Index futures and FRED rates (from the dollar index repo).

## Run
```
pip install -r requirements.txt
python -m pytest -q tests                  # 19 tests, ~5 s
```
Then Run All in `02_Multi_Asset_Risk_Report.ipynb` (about 4 minutes, mostly the rolling DCC refits). To change the report, edit `make_notebook.py` and re-run it.

## Modules (`risk_engine/`)
| Module | What's in it |
|---|---|
| `data.py` | Puts every return series on one calendar. Futures returns are within contract (roll day = 0) |
| `portfolio.py` | Equal weight, inverse vol, min variance, risk parity (ERC); monthly rebalanced backtest with costs and a vol-target overlay |
| `risk.py` | VaR and ES (historical, normal, Cornish-Fisher, filtered historical), Kupiec and Christoffersen backtests, Basel traffic light, drawdown table, Euler risk contributions, historical and hypothetical stress tests |
| `simulation.py` | Gaussian and Student-t Monte Carlo (sample or EWMA cov) and the Politis-Romano stationary bootstrap, all returning the same shape; horizon VaR/ES, drawdown probabilities, bootstrap CIs |
| `garch.py` | GARCH(1,1) (normal or Student-t) and two-step DCC(1,1) by maximum likelihood, written with scipy; rolling out-of-sample VaR forecasts |
| `fx.py` | Rolling dollar beta, min-variance hedge ratios, carry vs spot split for currency futures, correlation in calm vs stressed markets |

## What the first report found (Dec 2010 to Dec 2025)
- 40% equity / 30% commodity / 30% currency book: 10.8% volatility, 27% max drawdown, and **52% of its risk comes from the 40% equity weight**.
- Normal and historical 1-day 99% VaR both **fail the Kupiec backtest** (78 and 55 breaches vs 33 expected). Cornish-Fisher VaR is calibrated (33 breaches).
- 21-day block bootstrap gives a 99% ES of 11.1% vs 7.4% for Gaussian Monte Carlo, so the normal model understates tail risk by about a third.
- The book's beta to the dollar is -0.61. It's basically short the dollar.
- Average cross-asset correlation goes from 0.19 in calm markets to 0.25 in stressed ones.

### Which VaR model passes a bank-style backtest (Nov 2014 to Dec 2025, 2,796 out-of-sample days, 28 breaches expected)
| Model | Breaches | Kupiec p | Christoffersen independence p | Conditional coverage p | Avg VaR |
|---|---|---|---|---|---|
| Historical simulation | 33 | 0.35 | < 0.001 | < 0.001 | 1.86% |
| Normal | 54 | < 0.001 | < 0.001 | < 0.001 | 1.54% |
| Cornish-Fisher | 19 | 0.07 | < 0.001 | < 0.001 | 2.72% |
| Filtered historical (EWMA) | 28 | 0.99 | < 0.001 | < 0.001 | 1.82% |
| GARCH(1,1)-t | 44 | 0.005 | 0.005 | < 0.001 | 1.56% |
| GARCH(1,1), empirical tail | 31 | 0.57 | 0.004 | 0.014 | 1.70% |
| DCC-GARCH, normal | 50 | < 0.001 | 0.07 | < 0.001 | 1.42% |
| **DCC-GARCH, empirical tail** | **30** | **0.70** | **0.04** | **0.12** | **1.63%** |

- Historical simulation gets the count right but its breaches come in clusters, which is exactly what Christoffersen's test catches.
- GARCH-t still breaches too often: the book's left tail is heavier than a symmetric t allows, so the fitted t quantile is too small.
- DCC-GARCH with the tail taken from its own standardised residuals is the only model that passes conditional coverage at 5%, and it holds about 13% less VaR on average than historical simulation.

## Next
1. Add S&P 500 total return (SPXT), a Treasury future and IG credit to the Bloomberg pull.
2. Backtest expected shortfall too (Acerbi-Szekely), since Basel FRTB uses 97.5% ES.
3. Factor risk model (equity beta, dollar, carry, momentum, commodity curve) with attribution.
4. Serve results through the SQL database and dashboard in [backtest-reality-check](https://github.com/RyanSingh0/backtest-reality-check).

## Data
The engine reads data from my two research repos cloned next to this one: [commodity-tercile-strategies](https://github.com/RyanSingh0/commodity-tercile-strategies) (Bloomberg export in its `data/`, licensed, not public) and [dxy-fx-strategies](https://github.com/RyanSingh0/dxy-fx-strategies) (currency futures and FRED rates). The tests use simulated data, so `pytest` runs without any of it.
