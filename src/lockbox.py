"""Section 8 — run each frozen finalist ONCE on the sealed lockbox (2025-09-26 → 2026-09-26).
The frozen procedure is unchanged: the walk-forward simply continues. Every 6-month window's parameters
(and every ML refit) use only data before that window. Results are written verbatim to
reports/lockbox_results.json/.md. Refuses to run twice unless --force is given (and logs it)."""
import argparse, json
import numpy as np
import pandas as pd

from src.config import REPORTS, LOCKBOX_START, LOCKBOX_END, ROOT
import src.data as D
from src.metrics import daily_returns, monte_carlo, max_dd

FINALISTS = ["E0449", "E0425", "E0395"]


def log(msg):
    with open(REPORTS / "lockbox_access.log", "a") as f:
        f.write(f"{pd.Timestamp.now(tz='UTC')} {msg}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_js = REPORTS / "lockbox_results.json"
    if out_js.exists() and not a.force:
        raise SystemExit("lockbox already evaluated — refusing to run again (Section 8: run ONCE)")
    D.MODE["lockbox"] = True
    log(f"EVALUATE lockbox finalists={FINALISTS}")
    from src.finalist import rebuild
    from src.backtest import evaluate
    from src.engine_event import run_event
    res = {}
    for cid in FINALISTS:
        cfg, df, venue, pl, cts, st, sel, simfn = rebuild(cid)
        r = evaluate(df, st, venue, cfg.get("trail", 0.0), cfg.get("band", 0.0), simfn=simfn, start=LOCKBOX_START)
        m = r["metrics"]
        wm = pd.Series(r["wd"]["m_pnl"], index=r["months"].astype(str))
        # 1-minute event replay through the lockbox (includes the 10-Oct-2025 crash)
        i0 = df.index.searchsorted(LOCKBOX_START)
        d = df.iloc[i0:]
        tg = pd.Series(st["target"][i0:], d.index)
        e1 = run_event(d, tg, venue, "compound", fx=d.fx, fund=d.fund, sub_bars=D.one_minute(lockbox=True))
        fast_final = float(r["comp"]["eq"].iloc[-1])
        oct10 = r["comp"]["eq"]["2025-10-09":"2025-10-12"]
        res[cid] = {
            "selections_in_lockbox": [(w, pl[k]) for w, k in sel if w >= "2025-07-01"],
            "median_month_inr": m["median_month_inr"], "mean_month_inr": m["mean_month_inr"],
            "pct_months_1L": m["pct_months_1L"], "worst_month_inr": m["worst_month_inr"],
            "total_withdraw_inr": float(np.nansum(r["wd"]["m_pnl"])),
            "compound_return": float(r["comp"]["eq"].iloc[-1] / r["comp"]["eq"].iloc[0] - 1),
            "max_dd": m["max_dd"], "sharpe": m["sharpe"], "n_trades": m["n_trades"], "exposure": m["exposure"],
            "liquidations_withdraw": m["liquidations_withdraw"], "mc_p_ruin": m["mc_p_ruin"],
            "stress3x_median_inr": m["stress3x_median_month_inr"], "delay1_median_inr": m["delay1_median_month_inr"],
            "monthly_withdraw_inr": {k: float(v) for k, v in wm.items()},
            "event_1m_final_equity_usd": float(e1["equity"].iloc[-1]), "fast_final_equity_usd": fast_final,
            "event_1m_liquidations": e1["liquidations"],
            "oct10_2025_equity_usd": {str(k): float(v) for k, v in oct10.resample("1D").last().items()},
            "compound_first_1L": m["compound_first_1L_month"],
        }
        print(cid, json.dumps({k: v for k, v in res[cid].items() if k != "monthly_withdraw_inr"}, default=str)[:800])
    out_js.write_text(json.dumps(res, indent=1, default=str))
    log("EVALUATE done; results written verbatim to reports/lockbox_results.json")


if __name__ == "__main__":
    main()
