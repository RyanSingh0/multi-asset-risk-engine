"""Loads every return series I have into one daily panel.

Where the data comes from (the research repos sit next to this one):
  commodity-tercile-strategies/data/Commodities_TR_Futures.xlsx   Bloomberg BCOM TR sub-indices, BCOMTR, SPX (price)
  dxy-fx-strategies/data/futures/*.csv               CME currency futures + ICE DX, daily nearby, 2010-12 to 2025-12
  dxy-fx-strategies/data/fred/rates.csv              T-bill and interbank rates

Futures returns are within contract. On the day the front contract changes I set the return to 0,
since that price jump is just the spread between two contracts.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent          # this repo
ROOT = BASE.parent                                      # folder holding all my research repos
COMM = ROOT / 'commodity-tercile-strategies'
if not COMM.exists():                                   # my local working folder has the older name
    COMM = ROOT / 'Commodities Research'
FUT = ROOT / 'dxy-fx-strategies' / 'data' / 'futures'
FRED = ROOT / 'dxy-fx-strategies' / 'data' / 'fred'

FX_FUTURES = {'E6': 'EUR', 'B6': 'GBP', 'D6': 'CAD', 'J6': 'JPY', 'S6': 'CHF', 'DX': 'DXY'}
COMMODITIES = {'BCOMCLTR': 'WTI', 'BCOMCOT': 'Brent', 'BCOMGCTR': 'Gold', 'BCOMNGTR': 'NatGas',
               'BCOMSYTR': 'Soybeans', 'BCOMSITR': 'Silver', 'BCOMALTR': 'Aluminium', 'BCOMCNTR': 'Corn',
               'BCOMHGTR': 'Copper'}
ASSET_CLASS = {**{v: 'Commodity' for v in COMMODITIES.values()}, 'BCOMTR': 'Commodity index',
               'SPX': 'Equity', **{v: 'FX' for v in FX_FUTURES.values()}}


def _futures_returns(code):
    d = pd.read_csv(FUT / f'{code}.csv', skipfooter=1, engine='python')
    d['Date'] = pd.to_datetime(d['Time'])
    d = d.sort_values('Date').set_index('Date')
    px = pd.to_numeric(d['Latest'], errors='coerce')
    r = px.pct_change()
    roll = d['Symbol'] != d['Symbol'].shift(1)
    r[roll] = 0.0                                      # roll day is a spread, not a return
    return r.iloc[1:]


def load_panel(start='2010-12-02', end='2025-12-30', include=('commodities', 'equity', 'fx', 'index')):
    """Daily simple returns, one column per asset, only on days every market traded.
    Commodities are total return (collateral in). SPX is PRICE only, no dividends.
    FX futures are excess returns: spot move plus the rate differential baked into the basis."""
    sys.path.insert(0, str(COMM))
    from commodity_research.loader import Data
    d = Data()
    cols = {}
    if 'commodities' in include:
        for k, v in COMMODITIES.items():
            cols[v] = d.PX[k].pct_change()
    if 'index' in include:
        cols['BCOMTR'] = d.BCOM.pct_change()
    if 'equity' in include:
        spx = d.raw['bench']['SPX Index|PX_LAST'].reindex(d.dates)
        cols['SPX'] = spx.pct_change()
    panel = pd.DataFrame(cols)
    if 'fx' in include:
        fx = pd.DataFrame({FX_FUTURES[c]: _futures_returns(c) for c in FX_FUTURES})
        panel = panel.join(fx, how='inner')
    panel = panel.loc[start:end].dropna(how='any')
    rf = risk_free().reindex(panel.index).ffill().fillna(0.0)
    return panel, rf


def risk_free():
    """Daily T-bill accrual (DTB3 / 252). Lagged a day so it's known in advance."""
    p = COMM / 'data' / 'DTB3.csv'
    tb = pd.read_csv(p, parse_dates=['observation_date'], index_col=0)['DTB3']
    tb = pd.to_numeric(tb, errors='coerce').ffill() / 100 / 252
    return tb.shift(1)


def interest_differentials():
    """3m interbank rate differentials, foreign minus US, annualised. Used for FX carry."""
    r = pd.read_csv(FRED / 'rates.csv', parse_dates=['Date'], index_col='Date').ffill()
    us = r['IR3TIB01USM156N']
    out = pd.DataFrame({'EUR': r['IR3TIB01EZM156N'] - us, 'GBP': r['IR3TIB01GBM156N'] - us,
                        'CAD': r['IR3TIB01CAM156N'] - us, 'JPY': r['IR3TIB01JPM156N'] - us,
                        'CHF': r['IR3TIB01CHM156N'] - us}) / 100
    return out
