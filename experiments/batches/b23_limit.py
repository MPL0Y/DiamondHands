import json
from src.experiment import run, next_id
H = ("Execution: post a maker limit at the signal bar's close (fee 0.0236% vs ~0.11% taker+slippage). Fills only if the next bar "
     "trades >=1bp through the limit; otherwise a market order at the following open. Walk-forward selection also sees the lower costs.")
base = {"E0528": None, "E0550": None, "E0449": None, "E0543": None, "E0558": None}
for cid in base:
    cfg = json.load(open(f"experiments/configs/{cid}.json"))
    cfg["id"] = next_id(); cfg["exec"] = "limit"
    cfg["hypothesis"] = H + f" [limit-execution rerun of {cid}]"
    run(cfg)
for lev in (1.25, 1.5):
    cfg = json.load(open("experiments/configs/E0528.json")); cfg["id"] = next_id(); cfg["exec"] = "limit"
    cfg["fixed"] = dict(cfg["fixed"], lev=lev); cfg["hypothesis"] = H + f" [combo L/S at {lev}x]"
    run(cfg)
