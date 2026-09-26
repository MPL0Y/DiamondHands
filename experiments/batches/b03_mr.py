from src.experiment import run, next_id
H = "Mean reversion: short-horizon BTC returns show reversal after extremes (e.g. Wen et al. 2022 intraday; practitioner RSI(2) evidence). Oversold entries should earn a positive median."
exps = []
for tf in ["1h", "4h", "1d", "15m"]:
    for side in ["long", "both"]:
        exps.append(dict(strategy="rsi_mr", tf=tf, grid={"n": [2, 7, 14], "lo": [10, 20, 30], "exit_mid": [50, 65]}, fixed={"side": side, "hi": 80}))
    exps.append(dict(strategy="rsi_mr", tf=tf, grid={"n": [2, 7, 14], "lo": [10, 20, 30], "trend_filter": [200, 800]}, fixed={"side": "long", "hi": 80, "exit_mid": 55}))
    for side in ["long", "both"]:
        exps.append(dict(strategy="bb_mr", tf=tf, grid={"n": [20, 50, 100], "k": [1.5, 2.0, 2.5, 3.0]}, fixed={"side": side}))
    exps.append(dict(strategy="bb_mr", tf=tf, grid={"n": [20, 50], "k": [2.0, 2.5, 3.0], "stop": [0.03, 0.06]}, fixed={"side": "long"}))
    exps.append(dict(strategy="zscore_mr", tf=tf, grid={"n": [24, 48, 96, 168], "z_in": [1.5, 2.0, 2.5], "z_out": [0.0, 0.5]}, fixed={"side": "both"}))
for tf, lbs, hs in [("1h", [2, 4, 8, 24], [4, 12, 24]), ("15m", [4, 8, 16], [8, 24, 48]), ("4h", [1, 2, 6], [2, 6, 12])]:
    for side in ["long", "both"]:
        exps.append(dict(strategy="intraday_rev", tf=tf, grid={"lookback": lbs, "thresh": [0.02, 0.04, 0.06], "hold": hs}, fixed={"side": side}))
for e in exps:
    run({"id": next_id(), "family": "mean_reversion", "lev": 1.0, "venue": "delta_india_perp", "hypothesis": H, **e})
