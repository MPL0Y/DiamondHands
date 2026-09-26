from src.experiment import run, next_id
M = {"1d": 1, "4h": 6, "1h": 24}
H = "Trend: BTC shows time-series momentum (Moskowitz et al. 2012; Liu & Tsyvinski 2021). Trend-following should cut bear-market drawdowns versus buy-and-hold."
exps = []
for tf in ["1d", "4h", "1h"]:
    m = M[tf]
    for kind in ["sma", "ema"]:
        for side in ["long", "both"]:
            exps.append(dict(strategy="ma_cross", tf=tf, grid={"fast": [5*m, 10*m, 20*m], "slow": [30*m, 50*m, 100*m, 200*m]},
                             fixed={"kind": kind, "side": side}))
    for side in ["long", "both"]:
        exps.append(dict(strategy="donchian", tf=tf, grid={"entry": [10*m, 20*m, 55*m], "exit": [5*m, 10*m, 20*m]}, fixed={"side": side}))
    exps.append(dict(strategy="donchian", tf=tf, grid={"entry": [10*m, 20*m, 55*m], "exit": [5*m, 10*m, 20*m], "atr_stop": [2.0, 3.0]}, fixed={"side": "long"}))
    for side in ["long", "both"]:
        exps.append(dict(strategy="tsmom", tf=tf, grid={"lookbacks": [(7*m, 30*m, 90*m), (14*m, 60*m, 180*m), (30*m, 90*m, 180*m), (3*m, 7*m, 30*m)], "thresh": [0.0, 0.5]}, fixed={"side": side}))
for e in exps:
    run({"id": next_id(), "family": "trend", "lev": 1.0, "venue": "delta_india_perp", "hypothesis": H, **e})
