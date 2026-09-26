"""PBO via CSCV across a whole FAMILY of experiments: each experiment's stitched OOS daily return series is one
'configuration'. It measures how often the in-sample-best member of the family underperforms the median OOS."""
import sys, json
import numpy as np, pandas as pd
from src.finalist import rebuild
from src.backtest import evaluate
from src.metrics import daily_returns, pbo_cscv

def family_pbo(ids):
    R = []
    for cid in ids:
        cfg, df, venue, pl, cts, st, sel, simfn = rebuild(cid)
        r = evaluate(df, st, venue, cfg.get("trail", 0.0), cfg.get("band", 0.0), extra=False, simfn=simfn)
        R.append(daily_returns(r["comp"]["eq"]).rename(cid))
    M = pd.concat(R, axis=1).fillna(0.0)
    return pbo_cscv(M.to_numpy()), M.shape

if __name__ == "__main__":
    lb = pd.read_csv("experiments/leaderboard.csv")
    fam = sys.argv[1]
    ids = lb[(lb.family == fam) & ~lb.status.str.startswith("SUPERSEDED")].id.tolist()
    p, sh = family_pbo(ids)
    print(json.dumps({"family": fam, "n": len(ids), "pbo": p, "shape": sh}))
