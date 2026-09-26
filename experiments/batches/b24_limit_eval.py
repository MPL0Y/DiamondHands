import json
from src.experiment import run, next_id
H = ("Execution variant: walk-forward selection costed at taker+slippage (conservative), trades executed as maker limits "
     "(fill only if price trades >=1bp through; otherwise market at the next open).")
for cid, levs in (("E0528", (1.0, 1.25, 1.5)), ("E0550", (1.0, 1.5, 2.0)), ("E0543", (1.0, 1.5, 1.75, 2.0)), ("E0558", (1.0, 1.5, 2.0)), ("E0449", (1.0,))):
    for lev in levs:
        cfg = json.load(open(f"experiments/configs/{cid}.json")); cfg["id"] = next_id(); cfg["exec"] = "limit_eval"
        cfg["fixed"] = dict(cfg["fixed"], lev=lev * cfg["fixed"].get("lev", 1.0))
        cfg["hypothesis"] = H + f" [{cid} x{lev}]"
        run(cfg)
