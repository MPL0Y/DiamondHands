from src.experiment import run, next_id
H = "Fractional leverage: the highest-median funding/grid signals fail only on DD (51-70%). Scaling exposure to 0.5-0.85x should bring OOS DD under 50% while keeping most of the median."
B = {
 "fc1h": dict(family="funding", strategy="funding_contra", tf="1h", grid={"window": [7, 21, 60], "z_in": [1.5, 2.0, 2.5, 3.0], "hold": [24, 72, 168]}, fixed={"side": "both"}),
 "fc4h": dict(family="funding", strategy="funding_contra", tf="4h", grid={"window": [7, 21, 60], "z_in": [1.5, 2.0, 2.5, 3.0], "hold": [6, 18, 42]}, fixed={"side": "both"}),
 "fl1d": dict(family="funding", strategy="funding_level", tf="1d", grid={"hi": [0.00005, 0.0001, 0.0002, 0.0004, 0.001], "lo": [-1, 0.0]}, fixed={"base": "none"}),
 "fl1h": dict(family="funding", strategy="funding_level", tf="1h", grid={"hi": [0.00005, 0.0001, 0.0002, 0.0004, 0.001], "lo": [-1, 0.0]}, fixed={"base": "none"}),
 "gr5": dict(family="grid", strategy="grid", tf="5m", grid={"spacing": [0.005, 0.01, 0.02, 0.03], "levels": [3, 5, 10]}, fixed={"filter": "none"}),
}
for k, b in B.items():
    for lev in [0.5, 0.7, 0.85]:
        cfg = {kk: (dict(v) if isinstance(v, dict) else v) for kk, v in b.items()}
        if k.startswith("gr"):
            cfg["fixed"]["lev"] = lev; L = 1.0
        else:
            L = lev
        run({"id": next_id(), "venue": "delta_india_perp", "hypothesis": H, "lev": L, **cfg})
