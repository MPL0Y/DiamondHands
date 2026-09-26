from src.experiment import run, next_id
H = "Sizing overlay: vol-targeting / DD-delevering / half-Kelly should cut the drawdowns of the best raw signals below 50%, so leverage can raise the median month within the risk limits."
BASE = {
 "fc1h": dict(family="funding", strategy="funding_contra", tf="1h", grid={"window": [7, 21, 60], "z_in": [1.5, 2.0, 2.5, 3.0], "hold": [24, 72, 168]}, fixed={"side": "both"}),
 "fl1d": dict(family="funding", strategy="funding_level", tf="1d", grid={"hi": [0.00005, 0.0001, 0.0002, 0.0004, 0.001], "lo": [-1, 0.0]}, fixed={"base": "none"}),
 "vr4h": dict(family="volatility", strategy="vol_regime", tf="4h", grid={"fast": [60, 120], "slow": [300, 600], "vol_q": [0.5, 0.7, 0.85]}, fixed={"mode": "trend_lowvol", "vol_n": 20, "q_window": 2190}),
 "vr1d": dict(family="volatility", strategy="vol_regime", tf="1d", grid={"fast": [10, 20], "slow": [50, 100], "vol_q": [0.5, 0.7, 0.85]}, fixed={"mode": "trend_lowvol", "vol_n": 20, "q_window": 365}),
 "ir15": dict(family="mean_reversion", strategy="intraday_rev", tf="15m", grid={"lookback": [4, 8, 16], "thresh": [0.02, 0.04, 0.06], "hold": [8, 24, 48]}, fixed={"side": "long"}),
 "st1d": dict(family="trend", strategy="supertrend", tf="1d", grid={"period": [7, 10, 14, 20], "mult": [2.0, 2.5, 3.0, 3.5, 4.0]}, fixed={"side": "long"}),
}
OV = [
 dict(lev=1.0, overlay={"vol_target": 0.6, "vol_n": 30, "vt_cap": 3.0}),
 dict(lev=1.0, overlay={"vol_target": 1.0, "vol_n": 30, "vt_cap": 4.0}),
 dict(lev=1.0, overlay={"vol_target": 1.0, "vol_n": 30, "vt_cap": 4.0, "dd_delever": [0.2, 0.4]}),
 dict(lev=2.0, overlay={}),
 dict(lev="kelly", kelly_cap=4.0, overlay={}),
]
for k, b in BASE.items():
    for ov in OV:
        cfg = dict(b); cfg.update(ov)
        n = b["tf"]
        if n in ("1h", "15m", "4h") and ov.get("overlay", {}).get("vol_n"):
            cfg["overlay"] = dict(ov["overlay"]); cfg["overlay"]["vol_n"] = {"1h": 24*30, "15m": 96*30, "4h": 6*30}[n]
        run({"id": next_id(), "venue": "delta_india_perp", "hypothesis": H, **cfg})
