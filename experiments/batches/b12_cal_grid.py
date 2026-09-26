from src.experiment import run, next_id
HC = "Calendar 2: turn-of-month flows (Kaiser 2019) and US-session drift, with a trend filter so the effects are only harvested in up-regimes."
HG = "Grid 2: trend-filtered grids (run only above or below the daily SMA), wide spacing, few levels. Tests whether grid drawdowns are trend-driven and can be avoided."
ex = []
ex.append(("calendar", HC, dict(strategy="calendar", tf="1d", grid={"tom": [1, 2, 3, 4, 5]}, fixed={"side": "long"})))
ex.append(("calendar", HC, dict(strategy="calendar", tf="1d", grid={"tom": [1, 2, 3, 5], "trend": [50, 100, 200]}, fixed={"side": "long"})))
ex.append(("calendar", HC, dict(strategy="calendar", tf="1h", grid={"hours": [tuple(range(13, 21)), tuple(range(14, 22)), tuple(range(0, 8)), tuple(range(20, 24))], "trend": [24*50, 24*100]}, fixed={"side": "long"})))
ex.append(("calendar", HC, dict(strategy="calendar", tf="1h", grid={"weekdays": [(0, 1, 2, 3, 4), (5, 6), (0, 1, 2, 3, 4, 5, 6)], "trend": [24*20, 24*50, 24*100]}, fixed={"side": "long"})))
ex.append(("calendar", HC, dict(strategy="calendar", tf="4h", grid={"hours": [(0, 4), (8, 12), (12, 16), (16, 20), (20,)], "trend": [6*50, 6*100]}, fixed={"side": "long"})))
ex.append(("calendar", HC, dict(strategy="calendar", tf="1d", grid={"weekdays": [(0, 1, 2, 3, 4), (5, 6), (0, 1, 2, 3), (4, 5, 6)], "trend": [50, 100, 200]}, fixed={"side": "long"})))
for filt in ["above", "below"]:
    for lev in [0.5, 1.0]:
        ex.append(("grid", HG, dict(strategy="grid", tf="15m", grid={"spacing": [0.01, 0.02, 0.04], "levels": [3, 5], "sma_days": [50, 100, 200]}, fixed={"lev": lev, "filter": filt})))
for lev in [0.3, 0.5, 0.7]:
    ex.append(("grid", HG, dict(strategy="grid", tf="15m", grid={"spacing": [0.01, 0.02, 0.03, 0.05], "levels": [2, 3, 5]}, fixed={"lev": lev, "filter": "none"})))
for fam, h, e in ex:
    run({"id": next_id(), "family": fam, "lev": 1.0, "venue": "delta_india_perp", "hypothesis": h, **e})
