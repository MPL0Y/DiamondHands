"""Finalist gates beyond the per-experiment ones:
- plateau test (each numeric parameter perturbed ±20% in every walk-forward window)
- fast engine vs event-driven engine agreement (bar mode and 1-minute intrabar replay)
- leverage frontier (median monthly INR vs P(ruin) / maxDD for 0.25x..20x)
"""
import json
import numpy as np
import pandas as pd

from src.config import EXP, OOS_START, REPORTS
from src.data import bars, one_minute
from src.backtest import walk_forward, evaluate, target_sim, slice_arr, windows, month_index, sim
from src.experiment import build_targets, expand_grid, kelly_scale
from src.custom import CUSTOM
from src.engine_event import run_event
from src.metrics import daily_returns, monte_carlo, max_dd
from src.venues import VENUES


def rebuild(cid):
    import src.backtest as BT
    cfg = json.loads((EXP / "configs" / f"{cid}.json").read_text())
    BT.EXEC["limit"] = 1 if cfg.get("exec") == "limit" else 0
    df = bars(cfg["tf"])
    venue = VENUES[cfg.get("venue", "delta_india_perp")]
    pl = expand_grid(cfg.get("grid", {}), cfg.get("fixed", {}))
    custom = CUSTOM.get(cfg["strategy"])
    simfn = custom["sim"] if custom else target_sim
    cts = custom["build"](df, cfg, pl) if custom else build_targets(df, cfg, pl)
    st, sel = walk_forward(df, cts, venue, cfg.get("score", "median_month"), cfg.get("train_years", 3),
                           cfg.get("test_months", 6), simfn=simfn, trail=cfg.get("trail", 0.0), band=cfg.get("band", 0.0))
    if cfg.get("lev") == "kelly":
        st = kelly_scale(df, st, cts, sel, venue, cfg, simfn)
    if cfg.get("exec") == "limit_eval":
        BT.EXEC["limit"] = 1
    return cfg, df, venue, pl, cts, st, sel, simfn


def _perturb(v, f):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return v
    if isinstance(v, int):
        r = int(round(v * f))
        return max(1, r) if v > 0 else r
    return v * f


def plateau(cid):
    cfg, df, venue, pl, cts, st, sel, simfn = rebuild(cid)
    base = evaluate(df, st, venue, cfg.get("trail", 0.0), cfg.get("band", 0.0), extra=False, simfn=simfn)
    b_sum = float(np.nansum(base["wd"]["m_pnl"]))
    b_med = base["metrics"]["median_month_inr"]
    keys = [k for k in cfg.get("grid", {}) if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in cfg["grid"][k])]
    keys += [k for k, v in cfg.get("fixed", {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    custom = CUSTOM.get(cfg["strategy"])
    rows = []
    for k in keys:
        for f in (0.8, 1.2):
            stp = {kk: (None if v is None else np.zeros(len(df))) for kk, v in st.items()}
            for (a, b), (_, j) in zip(windows(test_months=cfg.get("test_months", 6)), sel):
                p = dict(pl[j]); p[k] = _perturb(p[k], f)
                import src.backtest as BT
                if cfg.get("exec") == "limit_eval":
                    BT.EXEC["limit"] = 0          # selection is costed at market in this design
                c = (custom["build"](df, cfg, [p]) if custom else build_targets(df, cfg, [p]))[0]
                if cfg.get("exec") == "limit_eval":
                    BT.EXEC["limit"] = 1          # ... and executed with limit orders
                i1, i2 = df.index.searchsorted(a), df.index.searchsorted(b)
                for kk in stp:
                    if stp[kk] is not None and c["arrays"].get(kk) is not None:
                        stp[kk][i1:i2] = c["arrays"][kk][i1:i2]
            if cfg.get("lev") == "kelly":
                ratio = np.where(np.abs(st["target"]) > 0, st["target"], np.nan)
            r = evaluate(df, stp, venue, cfg.get("trail", 0.0), cfg.get("band", 0.0), extra=False, simfn=simfn)
            s = float(np.nansum(r["wd"]["m_pnl"]))
            rows.append({"param": k, "factor": f, "wd_total_inr": s, "median_inr": r["metrics"]["median_month_inr"],
                         "retention": s / b_sum if b_sum > 0 else np.nan, "max_dd": r["metrics"]["max_dd"]})
    out = pd.DataFrame(rows)
    ok = bool(len(out) == 0 or (out.retention >= 0.7).all())
    return {"base_total_inr": b_sum, "base_median_inr": b_med, "table": out, "pass": ok}


def engines_agree(cid, one_min=True):
    cfg, df, venue, pl, cts, st, sel, simfn = rebuild(cid)
    if cfg["strategy"] in CUSTOM and cfg["strategy"] != "ensemble":
        return {"note": "custom simulator — event verification via its own 1m run", "pass": None}
    i0 = df.index.searchsorted(OOS_START)
    d = df.iloc[i0:]
    arr = slice_arr(st, i0, len(df))
    tgt = pd.Series(arr["target"], d.index)
    stop = None if arr.get("stop") is None else pd.Series(arr["stop"], d.index)
    tp = None if arr.get("tp") is None else pd.Series(arr["tp"], d.index)
    tr, bd = cfg.get("trail", 0.0), cfg.get("band", 0.0)
    out = {}
    for mode, m in (("compound", 0), ("withdraw", 1)):
        f = sim(d, arr["target"], venue, m, arr.get("stop"), arr.get("tp"), tr, bd)
        import src.backtest as BT
        xl = BT.EXEC["limit"]
        e = run_event(d, tgt, venue, mode, fx=d.fx, fund=d.fund if venue.funding else None, stops=stop, tps=tp, trail=tr, band=bd,
                      exec_limit=xl)
        if m == 0:
            fv, ev = f["eq"].iloc[-1] * d.fx.iloc[-1], e["equity"].iloc[-1] * d.fx.iloc[-1]
        else:
            fv, ev = float(np.nansum(f["m_pnl"])), float(e["months"].sum())
        out[mode] = {"fast": fv, "event": ev, "rel_diff": abs(fv - ev) / max(abs(fv), 1e-9)}
        if one_min:
            e1 = run_event(d, tgt, venue, mode, fx=d.fx, fund=d.fund if venue.funding else None, stops=stop, tps=tp,
                           trail=tr, band=bd, sub_bars=one_minute(), exec_limit=xl)
            v1 = e1["equity"].iloc[-1] * d.fx.iloc[-1] if m == 0 else float(e1["months"].sum())
            out[mode]["event_1m"] = v1
            out[mode]["liq_1m"] = e1["liquidations"]
            out[mode]["rel_diff_1m"] = abs(fv - v1) / max(abs(fv), 1e-9)
    out["pass"] = all(out[k]["rel_diff"] <= 0.02 for k in ("compound", "withdraw"))
    return out


def frontier(cid, levels=(0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7, 10, 15, 20), mc_runs=5000):
    cfg, df, venue, pl, cts, st, sel, simfn = rebuild(cid)
    L0 = 1.0 if cfg.get("lev") in (None, "kelly") else float(cfg["lev"])
    rows = []
    for L in levels:
        s2 = dict(st)
        if "target" in s2:
            s2["target"] = st["target"] / L0 * L
        elif "lev" in s2:
            s2["lev"] = st["lev"] / L0 * L
        r = evaluate(df, s2, venue, cfg.get("trail", 0.0), cfg.get("band", 0.0), extra=False, simfn=simfn)
        m = r["metrics"]
        mc = monte_carlo(daily_returns(r["comp"]["eq"]).values, np.array([]), runs=mc_runs)
        rows.append({"lev": L, "median_month_inr": m["median_month_inr"], "pct_months_1L": m["pct_months_1L"],
                     "max_dd": m["max_dd"], "p_ruin": mc["mc_p_ruin"], "p_dd50": mc["mc_p_dd50"], "cagr": m["oos_cagr"],
                     "liq_withdraw": m["liquidations_withdraw"], "median_ret_lotfree": m["median_month_ret_lotfree"],
                     "within_limits": bool(m["max_dd"] <= 0.5 and mc["mc_p_ruin"] <= 0.05)})
    return pd.DataFrame(rows)
