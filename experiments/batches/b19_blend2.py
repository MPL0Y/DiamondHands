from src.experiment import run, next_id
H = ("Blend 2: combine the combo L/S system with diversifying components other than E0449, which failed in the "
     "post-lockbox year: other-timeframe combo systems, other selection rules, turn-of-month, funding filter, vol-regime.")
S = {
 "multiTF": ["E0528", "E0532", "E0536"],
 "multiRule": ["E0528", "E0529", "E0530"],
 "combo+tom+fund": ["E0528", "E0425", "E0301"],
 "combo+vr+fund": ["E0528", "E0292", "E0301"],
 "combo+tom": ["E0528", "E0425"],
 "combo+multiTF+tom": ["E0528", "E0532", "E0425"],
}
for name, ids in S.items():
    run({"id": next_id(), "family": "ensemble", "strategy": "ensemble", "tf": "1h", "lev": 1.0, "venue": "delta_india_perp",
         "hypothesis": H + f" Set {name}.", "grid": {"weighting": ["equal", "invvol"]}, "fixed": {"components": ids, "lev": 1.0, "step": 0.25}})
