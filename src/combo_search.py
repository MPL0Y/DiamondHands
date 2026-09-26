"""Walk-forward search over indicator combinations (see src/combo.py).

Usage (programmatic): from src.combo_search import prepare, wf_select
  prep = prepare("4h")                      # simulates every combo once (cached), returns R [days x combos]
  arrays, info = wf_select(prep, metric="sharpe", top_k=10, train_years=3, lev=1.0)
The resulting OOS arrays are logged through src.experiment via the custom strategy "combo_wf".
"""
import json, hashlib, time
import numpy as np
import pandas as pd

from src.config import PROC, OOS_START, EXP
from src.combo import library, enumerate_combos
from src.backtest import sim, windows
from src.venues import DELTA_INDIA
from src.data import MODE

CACHE = PROC / "combo_cache"
CACHE.mkdir(parents=True, exist_ok=True)
_PREP = {}


def _target(L, combo, side="long"):
    t = np.ones(len(next(iter(L.values()))), bool)
    for c in combo:
        t &= L[c]
    return t.astype(float)


def prepare(tf="4h", max_triples=6000, side="long"):
    from src.backtest import EXEC
    key = (tf, max_triples, MODE["lockbox"], side, EXEC["limit"])
    if key in _PREP:
        return _PREP[key]
    df, L = library(tf, side)
    combos = enumerate_combos(L, max_triples=max_triples)
    tag = hashlib.md5(json.dumps([tf, max_triples, len(df), str(df.index[-1]), sorted(L), side, EXEC["limit"]]).encode()).hexdigest()[:10]
    fp = CACHE / f"R_{tf}_{side}_{tag}.npz"
    sgn = 1.0 if side == "long" else -1.0
    days = df.index.floor("1D")
    last_of_day = np.r_[np.nonzero(days[1:] != days[:-1])[0], len(df) - 1]
    dix = days[last_of_day]
    if fp.exists():
        z = np.load(fp, allow_pickle=True)
        R, ntr, expo = z["R"], z["ntr"], z["expo"]
    else:
        t0 = time.time()
        R = np.zeros((len(dix), len(combos)), np.float32)
        ntr = np.zeros(len(combos), np.int32); expo = np.zeros(len(combos), np.float32)
        for j, cb in enumerate(combos):
            tg = _target(L, cb)
            r = sim(df, sgn * tg, DELTA_INDIA, 0, lotfree=True)
            eq = r["eq"].to_numpy()[last_of_day]
            prev = np.r_[r["eq"].iloc[0], eq[:-1]]
            R[:, j] = np.where(prev > 0, eq / prev - 1, 0.0)
            ntr[j] = int((np.diff(tg) > 0).sum()); expo[j] = tg.mean()
            if j % 2000 == 0:
                print(f"  {tf}: {j}/{len(combos)} combos simulated [{time.time()-t0:.0f}s]", flush=True)
        np.savez_compressed(fp, R=R, ntr=ntr, expo=expo)
    out = {"tf": tf, "df": df, "L": L, "combos": combos, "R": R, "days": dix, "ntr": ntr, "expo": expo, "side": side}
    _PREP[key] = out
    return out


def _metric(Rw, how, min_trades_mask=None):
    mu = Rw.mean(0); sd = Rw.std(0) + 1e-12
    if how == "sharpe":
        s = mu / sd
    elif how == "return":
        s = np.log1p(np.clip(Rw, -0.99, None)).sum(0)
    elif how == "sortino":
        dn = np.sqrt((np.minimum(Rw, 0) ** 2).mean(0)) + 1e-12
        s = mu / dn
    else:  # calmar
        eq = np.cumprod(1 + Rw.astype(np.float64), 0)
        dd = (1 - eq / np.maximum.accumulate(eq, 0)).max(0)
        s = np.log(eq[-1]) / np.maximum(dd, 0.05)
    if min_trades_mask is not None:
        s = np.where(min_trades_mask, s, -np.inf)
    return s


def wf_select(prep, metric="sharpe", top_k=10, train_years=3, lev=1.0, test_months=6, min_expo=0.05,
              step=0.25, diversify=False):
    """For each OOS window, rank all combos on the training window and average the top-k positions.
    Returns stitched target array (aligned with prep['df']) and per-window picks."""
    df, L, combos, R, dix = prep["df"], prep["L"], prep["combos"], prep["R"], prep["days"]
    idx = df.index
    target = np.zeros(len(df))
    picks = []
    for a, b in windows(test_months=test_months):
        tr0 = a - pd.DateOffset(years=train_years)
        d0, d1 = dix.searchsorted(tr0), dix.searchsorted(a)
        if d1 - d0 < 90:
            continue
        Rw = R[d0:d1]
        expo_w = (Rw != 0).mean(0)
        s = _metric(Rw, metric, expo_w >= min_expo)
        order = np.argsort(-s)
        chosen = []
        used = set()
        for j in order:
            if not np.isfinite(s[j]):
                break
            if diversify and any(c in used for c in combos[j]):
                continue
            chosen.append(j); used.update(combos[j])
            if len(chosen) >= top_k:
                break
        i1, i2 = idx.searchsorted(a), idx.searchsorted(b)
        if chosen:
            T = np.mean([_target(L, combos[j])[i1:i2] for j in chosen], axis=0)
            target[i1:i2] = np.round(T / step) * step * lev * (1.0 if prep.get("side", "long") == "long" else -1.0)
        picks.append((str(a.date()), [" & ".join(combos[j]) for j in chosen[:5]], [float(s[j]) for j in chosen[:5]]))
    return target, picks


# ------------------------------------------------------------------ experiment-framework integration
def combo_build(df, cfg, params_list):
    from src.config import EXP
    out = []
    for p in params_list:
        tf = cfg["tf"]
        pl = prepare(tf, p.get("triples", 6000), "long")
        tl, picks_l = wf_select(pl, p.get("long_metric", "return"), p.get("long_k", 5), p.get("long_ty", 4),
                                1.0, min_expo=p.get("min_expo", 0.05))
        tg = tl
        picks_s = []
        if p.get("short", True):
            ps = prepare(tf, p.get("triples", 6000), "short")
            ts, picks_s = wf_select(ps, p.get("short_metric", "sortino"), p.get("short_k", 5), p.get("short_ty", 4),
                                    1.0, min_expo=p.get("min_expo", 0.05))
            tg = tl + ts
        tg = tg * float(p.get("lev", 1.0))
        stop = np.full(len(df), float(p["stop"])) if p.get("stop") else None
        combo_build.last_picks = {"long": picks_l, "short": picks_s}
        out.append({"params": p, "arrays": {"target": tg, "stop": stop, "tp": None}})
    return out


def _register():
    from src.custom import CUSTOM
    from src.backtest import target_sim
    CUSTOM["combo_wf"] = {"build": combo_build, "sim": target_sim}


_register()
