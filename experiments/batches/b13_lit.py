from src.experiment import run, next_id
HQ = "Literature (QuantPedia): long BTC 21:00-23:00 UTC -> 40.6%/yr, Calmar 1.79. Test with Delta costs and a walk-forward over neighbouring windows/weekdays."
HT = "Turn-of-month (TOM) follow-up: E0425 leads within limits. Test TOM at 4h resolution, fractional/extra leverage, and combined with a funding-level filter."
ex = []
ex.append((HQ, "calendar", dict(strategy="calendar", tf="1h", grid={"hours": [(21, 22), (22,), (21,), (20, 21, 22), (21, 22, 23)]}, fixed={"side": "long"})))
ex.append((HQ, "calendar", dict(strategy="calendar", tf="1h", grid={"weekdays": [(3, 4, 5, 6), (4,), (0, 1, 2, 3, 4, 5, 6)]}, fixed={"side": "long", "hours": (21, 22)})))
ex.append((HQ, "calendar", dict(strategy="calendar", tf="1h", grid={"hours": [(21, 22), (20, 21, 22)], "trend": [24 * 50, 24 * 100]}, fixed={"side": "long"})))
for lev in [0.5, 0.75, 1.25, 1.5]:
    ex.append((HT, "calendar", dict(strategy="calendar", tf="1d", grid={"tom": [1, 2, 3, 4, 5]}, fixed={"side": "long"}, lev=lev)))
ex.append((HT, "calendar", dict(strategy="calendar", tf="4h", grid={"tom": [1, 2, 3, 4, 5]}, fixed={"side": "long"})))
ex.append((HT, "calendar", dict(strategy="calendar", tf="1d", grid={"tom": [1, 2, 3, 4, 5]}, fixed={"side": "long"}, score="sharpe")))
ex.append((HT, "calendar", dict(strategy="calendar", tf="1d", grid={"tom": [2, 3, 4], "weekdays": [(0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4)]}, fixed={"side": "long"})))
for h, fam, e in ex:
    lev = e.pop("lev", 1.0)
    run({"id": next_id(), "family": fam, "lev": lev, "venue": "delta_india_perp", "hypothesis": h, **e})
