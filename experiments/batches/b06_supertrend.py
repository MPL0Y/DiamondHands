from src.experiment import run, next_id
H = "User request: TradingView Supertrend (ATR 10, x3.0, hl2, RMA ATR) on daily bars. Long on up-trend. Variants add flat vs short on down-trend, a WF-tuned grid, vol-targeting and leverage."
ex = [
  dict(strategy="supertrend", tf="1d", fixed={"period": 10, "mult": 3.0, "side": "long"}),
  dict(strategy="supertrend", tf="1d", fixed={"period": 10, "mult": 3.0, "side": "both"}),
  dict(strategy="supertrend", tf="1d", fixed={"period": 10, "mult": 3.0, "side": "long", "change_atr": False}),
  dict(strategy="supertrend", tf="1d", grid={"period": [7, 10, 14, 20], "mult": [2.0, 2.5, 3.0, 3.5, 4.0]}, fixed={"side": "long"}),
  dict(strategy="supertrend", tf="1d", grid={"period": [7, 10, 14, 20], "mult": [2.0, 2.5, 3.0, 3.5, 4.0]}, fixed={"side": "both"}),
  dict(strategy="supertrend", tf="1d", grid={"period": [7, 10, 14, 20], "mult": [2.0, 2.5, 3.0, 3.5, 4.0]}, fixed={"side": "long"}, score="sharpe"),
  dict(strategy="supertrend", tf="1d", fixed={"period": 10, "mult": 3.0, "side": "long"}, overlay={"vol_target": 0.5, "vol_n": 30, "vt_cap": 2.0}),
  dict(strategy="supertrend", tf="1d", fixed={"period": 10, "mult": 3.0, "side": "long"}, lev=2.0),
]
for e in ex:
    run({"id": next_id(), "family": "trend", "venue": "delta_india_perp", "hypothesis": H, "lev": e.pop("lev", 1.0), **e})
