from src.experiment import run, next_id
H = ("Blend 4 (limit execution): combine the full-period winner (combo L/S + E0449) with the multi-rule combo system, which was "
     "strongest in the recent bear year. Goal: keep the long-run profit and add recent robustness.")
S = [["E0528", "E0529", "E0530", "E0449"], ["E0543", "E0550"], ["E0528", "E0550", "E0449"], ["E0543", "E0550", "E0528"]]
for ids in S:
    run({"id": next_id(), "family": "ensemble", "strategy": "ensemble", "tf": "1h", "lev": 1.0, "venue": "delta_india_perp",
         "exec": "limit_eval", "hypothesis": H + f" {ids}", "grid": {"weighting": ["equal", "invvol"]},
         "fixed": {"components": ids, "lev": 1.0, "step": 0.25}})
