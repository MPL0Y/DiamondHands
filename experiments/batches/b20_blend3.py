from src.experiment import run, next_id
H = ("Blend 3: maximise risk-adjusted return, which leverage then scales up to the DD cap. Mix the robust multi-rule combo system "
     "with E0449 and its components, and add a vol-target overlay (exposure ~ target/realised vol).")
S = [
 (["E0528", "E0529", "E0530", "E0449"], {}),
 (["E0528", "E0529", "E0530", "E0301", "E0292", "E0420"], {}),
 (["E0528", "E0532", "E0449"], {}),
 (["E0550", "E0449"], {}),
 (["E0543"], {"vol_target": 0.6, "vt_cap": 2.0}),
 (["E0543"], {"vol_target": 0.9, "vt_cap": 2.5}),
 (["E0550"], {"vol_target": 0.6, "vt_cap": 2.0}),
 (["E0550"], {"vol_target": 0.9, "vt_cap": 2.5}),
 (["E0550", "E0449"], {"vol_target": 0.7, "vt_cap": 2.5}),
]
for ids, extra in S:
    fx = {"components": ids, "lev": 1.0, "step": 0.25}; fx.update(extra)
    run({"id": next_id(), "family": "ensemble", "strategy": "ensemble", "tf": "1h", "lev": 1.0, "venue": "delta_india_perp",
         "hypothesis": H + f" Components {ids} {extra}.", "grid": {"weighting": ["equal", "invvol"]}, "fixed": fx})
