from src.experiment import run, next_id
H = "ML: gradient-boosted trees on price/vol/funding/sentiment features. Purged, embargoed CV; retrained every 6 months; OOS by construction. A 52-53% hit rate on hourly-to-daily horizons might beat costs after thresholding. Meta-labeling should filter a trend primary's bad trades (Lopez de Prado 2018)."
ex = []
for tf, Hs in [("1h", [6, 24, 72]), ("4h", [3, 6, 18]), ("1d", [1, 3, 7])]:
    for h in Hs:
        for side in ["long", "both"]:
            ex.append(dict(strategy="ml_dir", tf=tf, grid={"thr": [0.52, 0.54, 0.56, 0.58, 0.6]}, fixed={"H": h, "side": side}))
for tf, Hs, f, s in [("1h", [24, 72], 480, 2400), ("4h", [6, 18], 60, 300), ("1d", [3, 7], 20, 100)]:
    for h in Hs:
        ex.append(dict(strategy="ml_meta", tf=tf, grid={"thr": [0.45, 0.5, 0.55, 0.6]}, fixed={"H": h, "primary": "ma", "fast": f, "slow": s}))
ex.append(dict(strategy="ml_meta", tf="1d", grid={"thr": [0.45, 0.5, 0.55, 0.6]}, fixed={"H": 7, "primary": "st", "fast": 10, "slow": 30}))
ex.append(dict(strategy="ml_meta", tf="1d", grid={"thr": [0.45, 0.5, 0.55, 0.6]}, fixed={"H": 3, "primary": "st", "fast": 10, "slow": 30}))
for e in ex:
    run({"id": next_id(), "family": "ml", "lev": 1.0, "venue": "delta_india_perp", "hypothesis": H, **e})
