from src.experiment import run, next_id
H = "Options: BTC implied vol has historically exceeded realised vol (variance risk premium). Selling weekly OTM options on real Deribit prints collects premium in most weeks, which should give a high median month; crash weeks are the risk."
ex = []
for st in ["put", "strangle", "call"]:
    for lev in [0.5, 1.0, 2.0, 3.0, 5.0]:
        ex.append(dict(grid={"delta": [0.05, 0.1, 0.15, 0.2, 0.3]}, fixed={"structure": st, "lev": lev}))
for st in ["put", "strangle"]:
    ex.append(dict(grid={"delta": [0.05, 0.1, 0.2], "lev": [0.5, 1.0, 2.0, 3.0]}, fixed={"structure": st}))
    ex.append(dict(grid={"delta": [0.05, 0.1, 0.2], "lev": [0.5, 1.0, 2.0, 3.0]}, fixed={"structure": st}, score="sharpe"))
for d in [0.1, 0.2]:
    ex.append(dict(fixed={"structure": "put", "delta": d, "lev": 1.0}))
for e in ex:
    run({"id": next_id(), "family": "options", "strategy": "options", "tf": "1h", "lev": 1.0, "venue": "delta_india_perp", "hypothesis": H, **e})
