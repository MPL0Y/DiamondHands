"""Run finalist gates, leverage frontiers and charts.
  python -m src.finalize --finalists E0xxx E0yyy --frontier E0a E0b E0c E0d E0e
Outputs reports/finalists/<id>.json, reports/figures/*.png, reports/frontier_<id>.csv
"""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import REPORTS, FIG, OOS_START
from src.backtest import evaluate
from src.finalist import rebuild, plateau, engines_agree, frontier
from src.metrics import daily_returns

OUT = REPORTS / "finalists"
OUT.mkdir(parents=True, exist_ok=True)
C1, C2, C3 = "#2f6db5", "#c2452d", "#6b7280"


def charts(cid, res):
    comp, df = res["comp"], res["df"]
    eq_inr = comp["eq"] * df.fx
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].plot(eq_inr.resample("1D").last(), color=C1, lw=1.2)
    ax[0].set_yscale("log"); ax[0].set_ylabel("equity ₹ (compounding, log)")
    ax[0].set_title(f"{cid} — walk-forward OOS equity from ₹10,000")
    pk = eq_inr.cummax(); dd = 1 - eq_inr / pk
    ax[1].fill_between(dd.resample("1D").max().index, 0, -dd.resample("1D").max(), color=C2, alpha=0.6)
    ax[1].axhline(-0.5, color=C3, ls="--", lw=0.8); ax[1].set_ylabel("drawdown")
    for a in ax:
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / f"{cid}_equity_dd.png", dpi=120); plt.close(fig)
    wm = pd.Series(res["wd"]["m_pnl"], index=res["months"].astype(str))
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.bar(range(len(wm)), wm.values, color=np.where(wm.values >= 0, C1, C2), width=0.85)
    ax.axhline(wm.median(), color=C3, ls="--", lw=0.8, label=f"median ₹{wm.median():,.0f}")
    ax.set_xticks(range(0, len(wm), 6)); ax.set_xticklabels(wm.index[::6], rotation=45, fontsize=7)
    ax.set_ylabel("₹ profit / month"); ax.set_title(f"{cid} — withdraw mode: monthly ₹ P&L on ₹10,000"); ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / f"{cid}_monthly.png", dpi=120); plt.close(fig)


def mc_chart(cid, r_daily):
    rng = np.random.default_rng(11)
    n = len(r_daily); term, mdd = [], []
    for _ in range(5000):
        idx = np.empty(365, dtype=np.int64); i = 0
        while i < 365:
            s = rng.integers(0, n); L = min(rng.geometric(0.1), 365 - i)
            idx[i:i + L] = (s + np.arange(L)) % n; i += L
        p = np.cumprod(1 + r_daily[idx]); pk = np.maximum.accumulate(np.r_[1.0, p])
        term.append(p[-1]); mdd.append(np.max(1 - np.r_[1.0, p] / pk))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    ax[0].hist(np.array(term) * 10000, bins=80, color=C1); ax[0].set_title("12-month terminal equity (₹, from ₹10k)")
    ax[0].axvline(np.percentile(term, 5) * 10000, color=C2, ls="--", label="5th pct"); ax[0].legend(frameon=False)
    ax[1].hist(mdd, bins=60, color=C2); ax[1].axvline(0.5, color=C3, ls="--"); ax[1].set_title("12-month max drawdown")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{cid} — block-bootstrap Monte Carlo (5,000 paths)"); fig.tight_layout()
    fig.savefig(FIG / f"{cid}_montecarlo.png", dpi=120); plt.close(fig)


def frontier_chart(frs):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for cid, f in frs.items():
        ax[0].plot(f.p_ruin * 100, f.median_month_inr, marker="o", ms=3, label=cid)
        for _, r in f.iterrows():
            if r.lev in (1, 3, 10, 20):
                ax[0].annotate(f"{r.lev:g}x", (r.p_ruin * 100, r.median_month_inr), fontsize=6)
        ax[1].plot(f.lev, f.max_dd * 100, marker="o", ms=3, label=cid)
    ax[0].axvline(5, color=C3, ls="--", lw=0.8); ax[0].set_xlabel("P(ruin) within 12 months, %"); ax[0].set_ylabel("median monthly ₹ (withdraw, ₹10k)")
    ax[1].axhline(50, color=C3, ls="--", lw=0.8); ax[1].set_xscale("log"); ax[1].set_xlabel("leverage"); ax[1].set_ylabel("OOS max DD %")
    for a in ax:
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Leverage frontier (walk-forward OOS 2018-01 → 2025-09)"); fig.tight_layout()
    fig.savefig(FIG / "leverage_frontier.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalists", nargs="*", default=[])
    ap.add_argument("--frontier", nargs="*", default=[])
    ap.add_argument("--no-1m", action="store_true")
    a = ap.parse_args()
    frs = {}
    for cid in a.frontier:
        f = frontier(cid)
        f.to_csv(REPORTS / f"frontier_{cid}.csv", index=False)
        frs[cid] = f
        print(cid, "\n", f.to_string(index=False))
    if frs:
        frontier_chart(frs)
    for cid in a.finalists:
        cfg, df, venue, pl, cts, st, sel, simfn = rebuild(cid)
        res = evaluate(df, st, venue, cfg.get("trail", 0.0), cfg.get("band", 0.0), simfn=simfn)
        charts(cid, res)
        mc_chart(cid, daily_returns(res["comp"]["eq"]).values)
        pt = plateau(cid)
        ea = engines_agree(cid, one_min=not a.no_1m)
        d = json.loads((REPORTS.parent / "experiments" / "details" / f"{cid}.json").read_text())
        out = {"id": cid, "config": cfg, "selections": [(w, pl[k]) for w, k in sel], "metrics": d,
               "plateau": {"pass": pt["pass"], "base_total_inr": pt["base_total_inr"], "table": pt["table"].to_dict("records")},
               "engines": ea,
               "monthly_withdraw_inr": dict(zip(res["months"].astype(str), map(float, res["wd"]["m_pnl"])))}
        (OUT / f"{cid}.json").write_text(json.dumps(out, indent=1, default=str))
        print(cid, "plateau pass", pt["pass"], "engines", json.dumps(ea, default=str)[:400])


if __name__ == "__main__":
    main()
