import json
from src.experiment import run, next_id
for cid, lev, tag in (("E0595", 1.75, "max-profit blend"), ("E0604", 2.0, "balanced blend")):
    cfg = json.load(open(f"experiments/configs/{cid}.json")); cfg["id"] = next_id()
    cfg["fixed"] = dict(cfg["fixed"], lev=lev * cfg["fixed"].get("lev", 1.0))
    cfg["hypothesis"] = f"Finalist ({tag}): {cid} at {lev}x, chosen as the highest full-period profit with DD<=50%."
    run(cfg)
