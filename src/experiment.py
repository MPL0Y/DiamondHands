"""Experiment runner: config -> walk-forward OOS -> gates -> leaderboard row -> git commit.

Config (JSON in experiments/configs/<ID>.json):
{ "id": "E0001", "family": "trend", "strategy": "ma_cross", "tf": "1d",
  "grid": {"fast": [10,20], "slow": [50,100]},  "fixed": {"side": "long"},
  "lev": 1.0 | "kelly",  "kelly_cap": 5, "venue": "delta_india_perp",
  "overlay": {"vol_target": 0.6, "vol_n": 30, "dd_delever": [0.2, 0.5]},
  "trail": 0.0, "band": 0.0, "score": "median_month", "train_years": 3, "test_months": 6,
  "hypothesis": "...", "lesson": "..." }
"""
import argparse, itertools, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import EXP, ROOT, OOS_START, START_CAPITAL_INR, TARGET_MONTHLY_INR
from src.data import bars
from src.backtest import sim, walk_forward, evaluate, windows, month_index, target_sim, slice_arr
from src.custom import CUSTOM
from src.metrics import daily_returns, deflated_sharpe, pbo_cscv
from src.strategies import REGISTRY, rvol, bpy
import src.ml  # registers ML strategies
import src.options  # registers options custom sim
from src.venues import VENUES

LB = EXP / "leaderboard.csv"
STATE = EXP / "state.json"
TRIALS = EXP / "trials.csv"

COLS = ["id", "family", "strategy", "tf", "params", "lev", "venue", "N_so_far", "n_configs", "oos_cagr",
        "median_month_inr", "pct_months_1L", "max_dd", "sharpe", "sortino", "calmar", "dsr", "pbo",
        "n_trades", "exposure", "p_ruin", "p_dd50", "req_capital_inr", "worst_month_inr", "mean_month_inr",
        "compound_first_1L", "final_equity_inr", "stress3x_median_inr", "delay1_median_inr", "delay1_cagr",
        "liq_withdraw", "regime_flag", "regime_top", "status", "hypothesis", "lesson", "timestamp"]


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"N": 0, "n_experiments": 0}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))


def expand_grid(grid, fixed):
    keys = list(grid)
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = dict(fixed)
        p.update(dict(zip(keys, vals)))
        out.append(p)
    return out or [dict(fixed)]


def apply_overlay(df, base, ov):
    t = np.asarray(base, float).copy()
    if not ov:
        return t
    if ov.get("vol_target"):
        v = rvol(df.close.to_numpy(), ov.get("vol_n", 30), bpy(df))
        sc = np.nan_to_num(ov["vol_target"] / v, nan=0.0)
        sc = np.clip(sc, 0, ov.get("vt_cap", 3.0))
        q = ov.get("vt_step", 0.25)
        sc = np.floor(sc / q) * q
        t = t * sc
    return t


def dd_delever(df, target, venue, levels):
    """Scale exposure down while the strategy's own (1x, lot-free) equity is in drawdown.
    levels = [dd1, dd2]: full size below dd1, half size between dd1 and dd2, flat beyond dd2.
    Uses the equity up to bar t's close -> decision for t+1: causal."""
    r = sim(df, np.sign(target) * np.minimum(np.abs(target), 1.0), venue, 0, lotfree=True)
    eq = r["eq"].to_numpy()
    pk = np.maximum.accumulate(np.where(eq > 0, eq, np.nan))
    dd = np.nan_to_num(1 - eq / pk, nan=1.0)
    f = np.where(dd < levels[0], 1.0, np.where(dd < levels[1], 0.5, 0.0))
    return target * f


def build_targets(df, cfg, params_list):
    fn = REGISTRY[cfg["strategy"]]
    venue = VENUES[cfg.get("venue", "delta_india_perp")]
    lev = cfg.get("lev", 1.0)
    out = []
    for p in params_list:
        s = fn(df, **p)
        base = apply_overlay(df, s["target"], cfg.get("overlay"))
        L = 1.0 if lev == "kelly" else float(lev)
        tgt = base * L
        if cfg.get("overlay", {}).get("dd_delever"):
            tgt = dd_delever(df, tgt, venue, cfg["overlay"]["dd_delever"])
        out.append({"params": p, "arrays": {"target": tgt, "stop": s.get("stop"), "tp": s.get("tp")}})
    return out


def kelly_scale(df, stitched, cfg_targets, sel, venue, cfg, simfn):
    """Half-Kelly leverage per OOS window from the selected config's TRAIN returns (capped)."""
    cap = cfg.get("kelly_cap", 5.0)
    idx = df.index
    out = stitched["target"].copy()
    ws = windows(test_months=cfg.get("test_months", 6))
    for (a, b), (_, k) in zip(ws, sel):
        tr0 = max(idx[0], a - pd.DateOffset(years=cfg.get("train_years", 3)))
        i0, i1, i2 = idx.searchsorted(tr0), idx.searchsorted(a), idx.searchsorted(b)
        c = cfg_targets[k]
        r = simfn(df.iloc[i0:i1], slice_arr(c["arrays"], i0, i1), venue, 0, True, 0, cfg.get("trail", 0.0), cfg.get("band", 0.0))
        d = daily_returns(r["eq"])
        mu, var = d.mean(), d.var()
        f = 0.5 * mu / var if var > 0 else 0.0
        f = float(np.clip(f, 0.0, cap))
        out[i1:i2] = stitched["target"][i1:i2] * f
    stitched = dict(stitched)
    stitched["target"] = out
    return stitched


def run(cfg, commit=True, verbose=True):
    t0 = time.time()
    st = load_state()
    df = bars(cfg["tf"])
    venue = VENUES[cfg.get("venue", "delta_india_perp")]
    params_list = expand_grid(cfg.get("grid", {}), cfg.get("fixed", {}))
    custom = CUSTOM.get(cfg["strategy"])
    if custom:
        cts = custom["build"](df, cfg, params_list)
        simfn = custom["sim"]
    else:
        cts = build_targets(df, cfg, params_list)
        simfn = target_sim
    trail, band = cfg.get("trail", 0.0), cfg.get("band", 0.0)
    stitched, sel = walk_forward(df, cts, venue, cfg.get("score", "median_month"), cfg.get("train_years", 3),
                                 cfg.get("test_months", 6), simfn=simfn, trail=trail, band=band)
    if cfg.get("lev") == "kelly":
        stitched = kelly_scale(df, stitched, cts, sel, venue, cfg, simfn)
    res = evaluate(df, stitched, venue, trail, band, simfn=simfn)
    res["stitched"], res["sel"], res["cands"] = stitched, sel, cts
    m = res["metrics"]
    # trial bookkeeping: OOS-period daily Sharpe of every config (for DSR variance, PBO matrix)
    i0 = df.index.searchsorted(OOS_START)
    R, srs = [], []
    for c in cts:
        r = simfn(df.iloc[i0:], slice_arr(c["arrays"], i0, len(df)), venue, 0, True, 0, trail, band)
        d = daily_returns(r["eq"]).to_numpy()
        R.append(d)
        srs.append(d.mean() / d.std() if d.std() > 0 else 0.0)
    R = np.array(R).T
    st["N"] += len(cts)
    st["n_experiments"] += 1
    with open(TRIALS, "a") as f:
        if f.tell() == 0:
            f.write("id,k,sr_daily\n")
        for k, s in enumerate(srs):
            f.write(f"{cfg['id']},{k},{s:.6f}\n")
    allsr = pd.read_csv(TRIALS)["sr_daily"].to_numpy()
    d = daily_returns(res["comp"]["eq"]).to_numpy()
    sr_d = d.mean() / d.std() if d.std() > 0 else 0.0
    dsr, sr0 = deflated_sharpe(sr_d, len(d), m["skew"], m["kurt"], st["N"], float(np.var(allsr)))
    m["dsr"], m["sr0_daily"] = dsr, sr0
    m["pbo"] = pbo_cscv(R) if R.shape[1] >= 4 else float("nan")
    # status
    status = "ok"
    if m["max_dd"] > 0.5 or m["mc_p_ruin"] > 0.05:
        status = f"rejected: too risky (maxDD {m['max_dd']:.0%}, P(ruin) {m['mc_p_ruin']:.1%})"
    elif m["n_trades"] < 30:
        status = "rejected: <30 OOS trades"
    elif m["n_trades"] < 100:
        status = "ok (low-freq: 30-99 trades)"
    alarms = []
    if m["sharpe"] > 3 and cfg["tf"] in ("1d",):
        alarms.append("SANITY: Sharpe>3 on daily")
    if m["median_month_ret_lotfree"] > 0.5:
        alarms.append("SANITY: >50%/month")
    if alarms:
        status += " | " + "; ".join(alarms)
    selc = pd.Series([s for _, s in sel]).value_counts()
    sel_desc = "; ".join(f"{json.dumps(params_list[k], separators=(',', ':'))}x{v}" for k, v in selc.items())
    grid_desc = json.dumps({"grid": cfg.get("grid", {}), "fixed": cfg.get("fixed", {}),
                            "overlay": cfg.get("overlay"), "trail": cfg.get("trail", 0), "score": cfg.get("score")},
                           separators=(",", ":"))
    row = {"id": cfg["id"], "family": cfg["family"], "strategy": cfg["strategy"], "tf": cfg["tf"],
           "params": grid_desc + " | selected: " + sel_desc, "lev": cfg.get("lev", 1.0), "venue": venue.name,
           "N_so_far": st["N"], "n_configs": len(cts), "oos_cagr": m["oos_cagr"],
           "median_month_inr": m["median_month_inr"], "pct_months_1L": m["pct_months_1L"], "max_dd": m["max_dd"],
           "sharpe": m["sharpe"], "sortino": m["sortino"], "calmar": m["calmar"], "dsr": m["dsr"], "pbo": m["pbo"],
           "n_trades": m["n_trades"], "exposure": m["exposure"], "p_ruin": m["mc_p_ruin"], "p_dd50": m["mc_p_dd50"],
           "req_capital_inr": m["req_capital_inr"], "worst_month_inr": m["worst_month_inr"],
           "mean_month_inr": m["mean_month_inr"], "compound_first_1L": m["compound_first_1L_month"],
           "final_equity_inr": m["final_equity_inr"], "stress3x_median_inr": m["stress3x_median_month_inr"],
           "delay1_median_inr": m["delay1_median_month_inr"], "delay1_cagr": m["delay1_cagr"],
           "liq_withdraw": m["liquidations_withdraw"], "regime_flag": m["regime_flag"], "regime_top": m["regime_top"],
           "status": status, "hypothesis": cfg.get("hypothesis", ""), "lesson": cfg.get("lesson", ""),
           "timestamp": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    new = not LB.exists()
    pd.DataFrame([row], columns=COLS).to_csv(LB, mode="a", header=new, index=False)
    save_state(st)
    (EXP / "configs" / f"{cfg['id']}.json").write_text(json.dumps(cfg, indent=1))
    det = EXP / "details"
    det.mkdir(exist_ok=True)
    (det / f"{cfg['id']}.json").write_text(json.dumps({k: v for k, v in m.items()}, indent=1, default=str))
    if verbose:
        print(f"{cfg['id']} {cfg['strategy']:>14} {cfg['tf']:>3} lev={cfg.get('lev',1)} | med/mo ₹{m['median_month_inr']:>9,.0f} "
              f"CAGR {m['oos_cagr']:6.1%} DD {m['max_dd']:5.1%} SR {m['sharpe']:5.2f} tr {m['n_trades']:5d} "
              f"ruin {m['mc_p_ruin']:5.1%} req ₹{m['req_capital_inr']:,.0f} | {status} [{time.time()-t0:.0f}s]", flush=True)
    if commit:
        subprocess.run(["git", "add", "experiments"], cwd=ROOT, check=False)
        subprocess.run(["git", "commit", "-q", "-m", f"{cfg['id']}: {cfg['strategy']} {cfg['tf']} lev={cfg.get('lev',1)} "
                        f"med ₹{m['median_month_inr']:.0f}/mo, DD {m['max_dd']:.0%} [{status[:40]}]"], cwd=ROOT, check=False)
    return row, res


def next_id():
    s = load_state()
    return f"E{s['n_experiments'] + 1:04d}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    a = ap.parse_args()
    run(json.loads(Path(a.config).read_text()))
