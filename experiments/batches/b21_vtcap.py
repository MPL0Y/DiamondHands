from src.experiment import run, next_id
H = ("De-risking overlay: exposure scaled by min(1, target/realised vol). It cuts size in turbulent periods and never adds "
     "leverage in calm ones, which should lower max DD so a higher leverage multiplier fits under the 50% cap.")
for comp in (["E0543"], ["E0558"], ["E0550"]):
    for vt in (0.5, 0.7, 0.9):
        run({"id": next_id(), "family": "ensemble", "strategy": "ensemble", "tf": "1h", "lev": 1.0, "venue": "delta_india_perp",
             "hypothesis": H + f" {comp} vt={vt}", "grid": {"weighting": ["equal"]},
             "fixed": {"components": comp, "lev": 1.0, "step": 0.25, "vol_target": vt, "vt_cap": 1.0}})
