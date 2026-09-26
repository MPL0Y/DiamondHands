from src.experiment import run, next_id
H = "Ensemble: signals from different families (funding, vol-regime trend, supertrend, ML meta, daily MR) are weakly correlated. Averaging them should lift Sharpe and cut DD, allowing more leverage within limits and so a higher median month."
# rerun IDs (old+212): FL1h=E0301, FC1h=E0299, VR4h=E0292, VR1h=E0289, ST1d=E0344, MLM4h=E0420, TS1d=E0226, DC1h=E0241, RSI1d=E0262, FC4h=E0304
SETS = {
 "A": ["E0301", "E0299", "E0292"],
 "B": ["E0301", "E0292", "E0344"],
 "C": ["E0301", "E0299", "E0292", "E0344", "E0226"],
 "D": ["E0301", "E0292", "E0420"],
 "E": ["E0301", "E0299", "E0292", "E0344", "E0420", "E0262"],
 "F": ["E0299", "E0304", "E0301"],
 "G": ["E0292", "E0289", "E0344", "E0226", "E0241"],
}
ex = []
for name, ids in SETS.items():
    for lev in [1.0, 1.5, 2.0]:
        ex.append(dict(grid={"weighting": ["equal", "invvol"]}, fixed={"components": ids, "lev": lev, "step": 0.25}))
for name in ["A", "C", "E"]:
    ex.append(dict(grid={"k": [1, 2, 3]}, fixed={"components": SETS[name], "weighting": "vote", "lev": 1.0}))
for e in ex:
    run({"id": next_id(), "family": "ensemble", "strategy": "ensemble", "tf": "1h", "lev": 1.0, "venue": "delta_india_perp", "hypothesis": H, **e})
