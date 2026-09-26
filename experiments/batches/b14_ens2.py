from src.experiment import run, next_id
H = "Ensemble round 2: E0449 (FL1h+VR4h+MLM4h at 2x) leads. Probe leverage around it, add the turn-of-month component (E0425) and the 1d funding-level signal, and test robustness to dropping a component."
D = ["E0301", "E0292", "E0420"]
ex = []
for lev in [1.75, 2.25, 2.5, 3.0]:
    ex.append(dict(grid={"weighting": ["equal", "invvol"]}, fixed={"components": D, "lev": lev, "step": 0.25}))
for ids in (["E0301", "E0292"], ["E0301", "E0420"], ["E0292", "E0420"], D + ["E0425"], D + ["E0311"], D + ["E0425", "E0344"]):
    for lev in [2.0, 2.5]:
        ex.append(dict(grid={"weighting": ["equal", "invvol"]}, fixed={"components": ids, "lev": lev, "step": 0.25}))
ex.append(dict(grid={"weighting": ["equal", "invvol"], "lev": [1.5, 2.0, 2.5]}, fixed={"components": D, "step": 0.25}))
ex.append(dict(grid={"weighting": ["equal"]}, fixed={"components": D, "lev": 2.0, "step": 0.5}))
for e in ex:
    run({"id": next_id(), "family": "ensemble", "strategy": "ensemble", "tf": "1h", "lev": 1.0, "venue": "delta_india_perp", "hypothesis": H, **e})
