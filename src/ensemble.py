"""Ensembles of sub-strategies.

Each component is an earlier experiment config. Its walk-forward-stitched OOS target is recomputed:
every window's parameters are picked using only prior data, so the component is causal. The
component target is then mapped onto the ensemble's (finer or equal) timeframe. A value decided at
the close of coarse bar t is placed on the fine bar that closes at the same instant, then
forward-filled. So it becomes tradable only at the next fine bar's open, which is the same instant
as the next coarse open.

Combination schemes (the ensemble's walk-forward picks among them using prior data only):
  equal     : mean of component targets
  invvol    : weights proportional to 1/rolling-vol of each component's 1x lot-free returns (causal, lagged)
  vote      : long only when >= k components are long
"""
import json
import numpy as np
import pandas as pd

from src.config import EXP
from src.data import bars
from src.backtest import walk_forward, sim
from src.venues import VENUES


def component_target(cid, tf_out):
    from src.experiment import build_targets, expand_grid
    from src.custom import CUSTOM
    cfg = json.loads((EXP / "configs" / f"{cid}.json").read_text())
    if cfg["strategy"] in CUSTOM:
        raise ValueError("custom sims cannot be ensemble components")
    df = bars(cfg["tf"])
    venue = VENUES[cfg.get("venue", "delta_india_perp")]
    cts = build_targets(df, cfg, expand_grid(cfg.get("grid", {}), cfg.get("fixed", {})))
    st, _ = walk_forward(df, cts, venue, cfg.get("score", "median_month"), cfg.get("train_years", 3),
                         cfg.get("test_months", 6), trail=cfg.get("trail", 0.0), band=cfg.get("band", 0.0))
    t = pd.Series(st["target"], df.index)
    if cfg.get("lev") not in (None, "kelly"):
        t = t / float(cfg["lev"])            # components enter at unit exposure; the ensemble applies leverage
    dfo = bars(tf_out)
    step_in = df.index[1] - df.index[0]
    step_out = dfo.index[1] - dfo.index[0]
    if step_in == step_out:
        return t.reindex(dfo.index).fillna(0.0).to_numpy()
    assert step_in > step_out, "components must be at the ensemble TF or coarser"
    decided = t.copy()
    decided.index = t.index + step_in - step_out        # fine bar whose close == coarse close
    return decided.reindex(dfo.index, method="ffill").fillna(0.0).to_numpy()


def ens_build(df, cfg, params_list):
    comps = {}
    out = []
    venue = VENUES[cfg.get("venue", "delta_india_perp")]
    for p in params_list:
        ids = p["components"]
        for c in ids:
            if c not in comps:
                comps[c] = component_target(c, cfg["tf"])
        T = np.vstack([comps[c] for c in ids])
        w = p.get("weighting", "equal")
        if w == "equal":
            tg = T.mean(0)
        elif w == "vote":
            tg = ((T > 0).sum(0) >= p.get("k", 2)).astype(float)
        else:  # invvol, causal: vol of each component's 1x returns over the trailing window, lagged one bar
            R = []
            for row in T:
                r = sim(df, row, venue, 0, lotfree=True)["eq"]
                R.append(r.pct_change().fillna(0.0).to_numpy())
            R = np.vstack(R)
            n = int(p.get("vol_n", 24 * 60))
            vol = np.vstack([pd.Series(r).rolling(n, min_periods=n // 4).std().shift(1).to_numpy() for r in R])
            iv = 1.0 / np.where(vol > 0, vol, np.nan)
            wts = iv / np.nansum(iv, 0)
            tg = np.nansum(np.nan_to_num(wts, nan=1.0 / len(ids)) * T, 0)
        q = p.get("step", 0.25)
        tg = np.round(tg / q) * q                      # quantise to limit churn
        L = float(p.get("lev", 1.0))
        out.append({"params": p, "arrays": {"target": tg * L, "stop": None, "tp": None}})
    return out


def ens_sim(df, arr, venue, mode, lotfree=False, delay=0, trail=0.0, band=0.0):
    return sim(df, arr["target"], venue, mode, None, None, trail, band, lotfree, delay)
