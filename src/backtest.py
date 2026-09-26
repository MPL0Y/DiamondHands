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


def windows(start=OOS_START, end=DEV_END, test_months=6):
    out = []
    a = start
    while a < end:
        b = min(a + pd.DateOffset(months=test_months), end)
        out.append((a, b))
        a = b
    return out


def walk_forward(df, cfg_targets, venue, score="median_month", train_years=3, test_months=6, min_train_days=120,
                 trail_of=None, band=0.0):
    """cfg_targets: list of dicts {params, target, stop, tp, trail}. Targets already include leverage.
    Returns stitched arrays + per-window selections. Only data before each window is used to select."""
    n = len(df)
    idx = df.index
    stitched = np.zeros(n)
    st_stop, st_tp, st_trail = np.zeros(n), np.zeros(n), np.zeros(n)
    sel = []
    for a, b in windows(test_months=test_months):
        tr0 = max(idx[0], a - pd.DateOffset(years=train_years))
        i0, i1, i2 = idx.searchsorted(tr0), idx.searchsorted(a), idx.searchsorted(b)
        if (a - tr0).days < min_train_days or len(cfg_targets) == 1:
            best = 0 if len(cfg_targets) == 1 else None
        else:
            best = None
        if best is None:
            sub = df.iloc[i0:i1]
            scores = []
            for k, c in enumerate(cfg_targets):
                r = sim(sub, c["target"][i0:i1], venue, 0,
                        None if c.get("stop") is None else c["stop"][i0:i1],
                        None if c.get("tp") is None else c["tp"][i0:i1], c.get("trail", 0.0), band, lotfree=True)
                scores.append(train_score(r["eq"], score))
            best = int(np.nanargmax(scores))
        c = cfg_targets[best]
        stitched[i1:i2] = c["target"][i1:i2]
        if c.get("stop") is not None:
            st_stop[i1:i2] = c["stop"][i1:i2]
        if c.get("tp") is not None:
            st_tp[i1:i2] = c["tp"][i1:i2]
        sel.append((str(a.date()), best))
    trails = {c.get("trail", 0.0) for c in cfg_targets}
    return stitched, st_stop, st_tp, sel


REGIMES = [("2018", "2018-01-01", "2019-01-01"), ("2019", "2019-01-01", "2020-01-01"),
           ("2020", "2020-01-01", "2021-01-01"), ("2021", "2021-01-01", "2022-01-01"),
           ("2022", "2022-01-01", "2023-01-01"), ("2023-24", "2023-01-01", "2025-01-01"),
           ("2025+", "2025-01-01", "2030-01-01")]


def evaluate(df_full, target, venue, stop=None, tp=None, trail=0.0, band=0.0, mc_runs=5000, extra=True):
    """Evaluate an OOS target over [OOS_START, end of df). Returns metrics dict + series."""
    i0 = df_full.index.searchsorted(OOS_START)
    df = df_full.iloc[i0:]
    tg = np.asarray(target)[i0:]
    sp = None if stop is None else np.asarray(stop)[i0:]
    tpp = None if tp is None else np.asarray(tp)[i0:]
    comp = sim(df, tg, venue, 0, sp, tpp, trail, band)
    wd = sim(df, tg, venue, 1, sp, tpp, trail, band)
    lf = sim(df, tg, venue, 1, sp, tpp, trail, band, lotfree=True)
    mi = month_index(df)
    m = perf(comp["eq"], df.fx, wd["m_pnl"], mi, comp["trades"], lf["m_pnl"])
    m["exposure"] = float((comp["pos"] != 0).mean())
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
    # regimes (withdraw-mode INR profit, compounding return)
    wm = pd.Series(wd["m_pnl"], index=mi.to_timestamp().tz_localize("UTC"))
    reg = {}
    for name, a, b in REGIMES:
        a, b = pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")
        s = wm[(wm.index >= a) & (wm.index < b)]
        e = comp["eq"][(comp["eq"].index >= a) & (comp["eq"].index < b)]
        reg[name] = {"wd_sum_inr": float(s.sum()), "wd_median_inr": float(s.median()) if len(s) else np.nan,
                     "comp_ret": float(e.iloc[-1] / e.iloc[0] - 1) if len(e) > 1 and e.iloc[0] > 0 else np.nan}
    pos_tot = sum(max(v["wd_sum_inr"], 0) for v in reg.values())
    top = max(reg.items(), key=lambda kv: kv[1]["wd_sum_inr"])
    m["regime_flag"] = bool(pos_tot > 0 and top[1]["wd_sum_inr"] / pos_tot > 0.5)
    m["regime_top"] = top[0]
    m["regimes"] = reg
    # stress 3x and delay +1 bar
    st = venue.stressed(3.0)
    wd3 = sim(df, tg, st, 1, sp, tpp, trail, band)
    c3 = sim(df, tg, st, 0, sp, tpp, trail, band)
    m["stress3x_median_month_inr"] = float(np.nanmedian(wd3["m_pnl"]))
    m["stress3x_cagr"] = _cagr(c3["eq"])
    wdl = sim(df_full, np.asarray(target), venue, 1, stop, tp, trail, band, delay=1)
    k0 = len(df_full.month_id.unique()) - len(mi)
    m["delay1_median_month_inr"] = float(np.nanmedian(wdl["m_pnl"][k0:]))
    cdl = sim(df_full, np.asarray(target), venue, 0, stop, tp, trail, band, delay=1)
    m["delay1_cagr"] = _cagr(cdl["eq"].iloc[i0:])
    return res


def _cagr(eq):
    if eq.iloc[0] <= 0:
        return -1.0
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    f = eq.iloc[-1] / eq.iloc[0]
    return f ** (1 / yrs) - 1 if f > 0 else -1.0
