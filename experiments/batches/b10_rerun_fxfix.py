"""Re-run every experiment E0001-E0212 after the withdraw-mode FX fix (P&L excludes FX drift on idle capital)."""
import json
import pandas as pd
from src.experiment import run, next_id, LB
lb = pd.read_csv(LB)
mask = lb["id"].str[1:].astype(int) <= 212
lb.loc[mask & ~lb["status"].str.startswith("SUPERSEDED"), "status"] = "SUPERSEDED(fx-drift bug; rerun below) " + lb.loc[mask, "status"]
lb.to_csv(LB, index=False)
for i in range(1, 213):
    cfg = json.load(open(f"experiments/configs/E{i:04d}.json"))
    cfg["hypothesis"] = f"[rerun of E{i:04d} after FX fix] " + cfg.get("hypothesis", "")
    cfg["id"] = next_id()
    run(cfg)
