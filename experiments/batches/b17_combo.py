from src.experiment import run, next_id, load_state, save_state
H = ("Indicator combinations: AND of 1-3 conditions from a 76-condition library (trend/momentum/volatility/volume/"
     "funding/sentiment/calendar, own TF + daily). Walk-forward picks the top-k combos per 6-month window on "
     "prior data only and averages their positions. A mirrored short-side library covers bear markets.")
ex = []
for tf in ["4h", "1h", "1d"]:
    ex.append(dict(tf=tf, fixed={"long_metric": "return", "long_k": 5, "long_ty": 4, "short": False}))
    ex.append(dict(tf=tf, fixed={"long_metric": "return", "long_k": 5, "long_ty": 4, "short_metric": "sortino", "short_k": 1, "short_ty": 4}))
    ex.append(dict(tf=tf, fixed={"long_metric": "sortino", "long_k": 5, "long_ty": 4, "short_metric": "sortino", "short_k": 5, "short_ty": 4}))
    ex.append(dict(tf=tf, fixed={"long_metric": "return", "long_k": 20, "long_ty": 3, "short_metric": "sortino", "short_k": 20, "short_ty": 3}))
for lev in [1.25, 1.5]:
    ex.append(dict(tf="4h", fixed={"long_metric": "return", "long_k": 5, "long_ty": 4, "short_metric": "sortino", "short_k": 1, "short_ty": 4, "lev": lev}))
for stop in [0.05, 0.10]:
    ex.append(dict(tf="4h", fixed={"long_metric": "return", "long_k": 5, "long_ty": 4, "short_metric": "sortino", "short_k": 1, "short_ty": 4, "stop": stop}))
for e in ex:
    run({"id": next_id(), "family": "combo", "strategy": "combo_wf", "lev": 1.0, "venue": "delta_india_perp", "hypothesis": H, **e})
