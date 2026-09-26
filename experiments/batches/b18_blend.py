from src.experiment import run, next_id
H = ("Blend: the combo long/short system (E0528, 4h) and the earlier best ensemble (E0449, 1h) use different signals. "
     "Averaging them should smooth equity, so more leverage fits under the 50% DD limit and profit rises.")
ex = []
for lev in [1.0, 1.25, 1.5, 1.75]:
    ex.append(dict(grid={"weighting": ["equal", "invvol"]}, fixed={"components": ["E0528", "E0449"], "lev": lev, "step": 0.25}))
for lev in [1.0, 1.5]:
    ex.append(dict(grid={"weighting": ["equal", "invvol"]}, fixed={"components": ["E0528", "E0301", "E0292", "E0420"], "lev": lev, "step": 0.25}))
for e in ex:
    run({"id": next_id(), "family": "ensemble", "strategy": "ensemble", "tf": "1h", "lev": 1.0, "venue": "delta_india_perp", "hypothesis": H, **e})
