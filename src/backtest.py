"""Glue: run a target series through the fast engine, walk-forward selection, gates."""
import math
import numpy as np
import pandas as pd

from src.engine_fast import simulate
from src.metrics import perf, daily_returns, max_dd, monte_carlo
from src.config import START_CAPITAL_INR, OOS_START, DEV_END


def sim(df, target, venue, mode=0, stop=None, tp=None, trail=0.0, band=0.0, lotfree=False, delay=0, lev_cap=None):
    """target: array aligned with df (signed leverage decided at bar close)."""
    n = len(df)
    tgt = np.asarray(target, float).copy()
    cap = lev_cap if lev_cap is not None else venue.max_leverage
    tgt = np.clip(tgt, -cap, cap)
    stp = np.zeros(n) if stop is None else np.nan_to_num(np.asarray(stop, float))
    tpp = np.zeros(n) if tp is None else np.nan_to_num(np.asarray(tp, float))
    if delay:
        tgt = np.concatenate([np.full(delay, np.nan), tgt[:-delay]])
        stp = np.concatenate([np.zeros(delay), stp[:-delay]])
        tpp = np.concatenate([np.zeros(delay), tpp[:-delay]])
    fund = df["fund"].to_numpy() if venue.funding else np.zeros(n)
    lot = 1e-9 if lotfree else venue.lot_btc * venue.min_lots
    out = simulate(df.open.to_numpy(), df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy(), fund,
                   tgt, stp, tpp, float(trail), df.month_id.to_numpy(), df.fx.to_numpy(), lot, venue.mmr,
                   venue.taker, venue.maker, venue.slippage, mode, START_CAPITAL_INR, float(band))
    eq, pos, m_pnl, m_liq, trades, nliq, fees, fundp, turn = out
    return {"eq": pd.Series(eq, df.index), "pos": pos, "m_pnl": m_pnl, "m_liq": m_liq, "trades": trades,
            "nliq": nliq, "fees": fees, "funding": fundp, "turnover": turn}


def month_index(df):
    m0 = df.month_id.iloc[0]
    ids = np.arange(m0, df.month_id.iloc[-1] + 1)
    return pd.PeriodIndex([pd.Period(year=i // 12, month=i % 12 + 1, freq="M") for i in ids])


def train_score(eq, how):
    r = daily_returns(eq)
    if len(r) < 30:
        return -np.inf
    if eq.iloc[-1] <= 0 or (eq <= 0).any():
        return -1e6
    mdd = max_dd(eq.values)
    if how == "sharpe":
        s = r.mean() / r.std() * math.sqrt(365) if r.std() > 0 else 0.0
    elif how == "calmar":
        yrs = len(r) / 365.25
        cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
        s = cagr / max(mdd, 0.05)
    else:  # median monthly return
        me = eq.resample("ME").last()
        mr = me.pct_change().dropna()
        s = float(mr.median()) if len(mr) else -np.inf
    if mdd > 0.5:
        s -= 10.0  # infeasible in training (risk limit)
    return s


def windows(start=OOS_START, end=None, test_months=6):
    from src.data import MODE
    from src.config import LOCKBOX_END
    if end is None:
        end = LOCKBOX_END if MODE["lockbox"] else DEV_END
    out = []
    a = start
    while a < end:
        b = min(a + pd.DateOffset(months=test_months), end)
        out.append((a, b))
        a = b
    return out


def target_sim(df, arr, venue, mode, lotfree=False, delay=0, trail=0.0, band=0.0):
    return sim(df, arr["target"], venue, mode, arr.get("stop"), arr.get("tp"), trail, band, lotfree, delay)


def slice_arr(arr, i, j):
    return {k: (None if v is None else v[i:j]) for k, v in arr.items()}


def walk_forward(df, cands, venue, score="median_month", train_years=3, test_months=6, min_train_days=120,
                 simfn=None, trail=0.0, band=0.0):
    """cands: list of dicts {params, arrays: {name: np.ndarray|None}}. Returns stitched arrays + selections.
    Selection for each OOS window uses only data strictly before that window (rolling train_years)."""
    simfn = simfn or target_sim
    idx = df.index
    keys = set().union(*[c["arrays"].keys() for c in cands])
    stitched = {k: None for k in keys}
    for k in keys:
        if any(c["arrays"].get(k) is not None for c in cands):
            stitched[k] = np.zeros(len(df))
    sel = []
    for a, b in windows(test_months=test_months):
        tr0 = max(idx[0], a - pd.DateOffset(years=train_years))
        i0, i1, i2 = idx.searchsorted(tr0), idx.searchsorted(a), idx.searchsorted(b)
        if len(cands) == 1 or (a - tr0).days < min_train_days:
            best = 0
        else:
            sub = df.iloc[i0:i1]
            scores = [train_score(simfn(sub, slice_arr(c["arrays"], i0, i1), venue, 0, True, 0, trail, band)["eq"], score)
                      for c in cands]
            best = int(np.nanargmax(scores))
        for k in keys:
            v = cands[best]["arrays"].get(k)
            if stitched[k] is not None and v is not None:
                stitched[k][i1:i2] = v[i1:i2]
        sel.append((str(a.date()), best))
    return stitched, sel


REGIMES = [("2018", "2018-01-01", "2019-01-01"), ("2019", "2019-01-01", "2020-01-01"),
           ("2020", "2020-01-01", "2021-01-01"), ("2021", "2021-01-01", "2022-01-01"),
           ("2022", "2022-01-01", "2023-01-01"), ("2023-24", "2023-01-01", "2025-01-01"),
           ("2025+", "2025-01-01", "2030-01-01")]


def evaluate(df_full, arrays, venue, trail=0.0, band=0.0, mc_runs=5000, extra=True, simfn=None, start=None):
    """Evaluate OOS arrays over [start (default OOS_START), end). Returns metrics dict + series."""
    simfn = simfn or target_sim
    i0 = df_full.index.searchsorted(OOS_START if start is None else start)
    df = df_full.iloc[i0:]
    arr = slice_arr(arrays, i0, len(df_full))
    comp = simfn(df, arr, venue, 0, False, 0, trail, band)
    wd = simfn(df, arr, venue, 1, False, 0, trail, band)
    lf = simfn(df, arr, venue, 1, True, 0, trail, band)
    mi = month_index(df)
    m = perf(comp["eq"], df.fx, wd["m_pnl"], mi, comp["trades"], lf["m_pnl"])
    m["exposure"] = float((np.asarray(comp["pos"]) != 0).mean())
    m["liquidations_compound"] = int(comp["nliq"])
    m["liquidations_withdraw"] = int(wd["nliq"])
    m["fees_usd"] = float(comp["fees"])
    m["funding_usd"] = float(comp["funding"])
    res = {"metrics": m, "comp": comp, "wd": wd, "lf": lf, "df": df, "months": mi}
    if not extra:
        return res
    r = daily_returns(comp["eq"])
    tr = comp["trades"][:, 2] if len(comp["trades"]) else np.array([])
    m.update(monte_carlo(r.values, tr, runs=mc_runs))
    wm = pd.Series(wd["m_pnl"], index=mi.to_timestamp().tz_localize("UTC"))
    reg = {}
    for name, a, b in REGIMES:
        a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        s_ = wm[(wm.index >= a) & (wm.index < b)]
        e = comp["eq"][(comp["eq"].index >= a) & (comp["eq"].index < b)]
        reg[name] = {"wd_sum_inr": float(s_.sum()), "wd_median_inr": float(s_.median()) if len(s_) else np.nan,
                     "comp_ret": float(e.iloc[-1] / e.iloc[0] - 1) if len(e) > 1 and e.iloc[0] > 0 else np.nan}
    pos_tot = sum(max(v["wd_sum_inr"], 0) for v in reg.values())
    top = max(reg.items(), key=lambda kv: kv[1]["wd_sum_inr"])
    m["regime_flag"] = bool(pos_tot > 0 and top[1]["wd_sum_inr"] / pos_tot > 0.5)
    m["regime_top"] = top[0]
    m["regimes"] = reg
    stv = venue.stressed(3.0)
    wd3 = simfn(df, arr, stv, 1, False, 0, trail, band)
    c3 = simfn(df, arr, stv, 0, False, 0, trail, band)
    m["stress3x_median_month_inr"] = float(np.nanmedian(wd3["m_pnl"]))
    m["stress3x_cagr"] = _cagr(c3["eq"])
    wdl = simfn(df_full, arrays, venue, 1, False, 1, trail, band)
    k0 = len(np.unique(df_full.month_id.to_numpy())) - len(mi)
    m["delay1_median_month_inr"] = float(np.nanmedian(wdl["m_pnl"][k0:]))
    cdl = simfn(df_full.iloc[i0 - 1:], slice_arr(arrays, i0 - 1, len(df_full)), venue, 0, False, 1, trail, band)
    m["delay1_cagr"] = _cagr(cdl["eq"])
    return res


def _cagr(eq):
    if eq.iloc[0] <= 0:
        return -1.0
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    f = eq.iloc[-1] / eq.iloc[0]
    return f ** (1 / yrs) - 1 if f > 0 else -1.0
