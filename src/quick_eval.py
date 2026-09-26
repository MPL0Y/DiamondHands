"""Compact evaluation of an experiment id: dev OOS (2018-01 -> 2025-09) and post-lockbox year, with lots
(₹10k, realistic) and lot-free (strategy quality), at leverage multipliers."""
import numpy as np, pandas as pd
import src.data as D
from src.config import LOCKBOX_START, OOS_START
from src.backtest import sim, slice_arr, target_sim
from src.metrics import max_dd, daily_returns


def _stats(eq):
    eq = eq[eq.index >= eq.index[0]]
    if eq.iloc[0] <= 0:
        return dict(ret=np.nan, cagr=np.nan, dd=np.nan, sharpe=np.nan)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    f = eq.iloc[-1] / eq.iloc[0]
    r = daily_returns(eq)
    return dict(ret=f - 1, cagr=(f ** (1 / yrs) - 1) if f > 0 else -1, dd=max_dd(eq.values),
                sharpe=float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0)


def quick(cid, mults=(1.0,), lockbox=False, period=None):
    """period: 'dev' (2018-01 -> 2025-09), 'lockbox' (last year), 'full' (2018-01 -> 2026-09, all OOS)."""
    period = period or ("lockbox" if lockbox else "dev")
    lockbox = period in ("lockbox", "full")
    D.MODE["lockbox"] = lockbox
    from src.finalist import rebuild
    cfg, df, venue, pl, cts, st, sel, simfn = rebuild(cid)
    start = LOCKBOX_START if period == "lockbox" else OOS_START
    i0 = df.index.searchsorted(start)
    d = df.iloc[i0:]
    rows = []
    for m in mults:
        arr = slice_arr(st, i0, len(df))
        arr = dict(arr); arr["target"] = arr["target"] * m
        for lf in (False, True):
            r = (simfn or target_sim)(d, arr, venue, 0, lf, 0, cfg.get("trail", 0.0), cfg.get("band", 0.0))
            s = _stats(r["eq"]); s.update(id=cid, mult=m, lotfree=lf, period=period, liq=r["nliq"], final_inr=float(r["eq"].iloc[-1] * d.fx.iloc[-1]))
            rows.append(s)
    D.MODE["lockbox"] = False
    return pd.DataFrame(rows)
