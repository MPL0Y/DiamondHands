from src.experiment import run, next_id
HC = "Funding carry: long spot + short perp collects positive funding (BTC perps averaged ~10-15%/yr, Binance 2020-25). It is delta-neutral, so it should give a very high share of positive months."
HG = "Grid trading: BTC mean-reverts intraday, so a grid of resting limit orders earns spacing minus maker fees per round trip. Trends are the risk, capped by a stop below the grid."
ex = []
for levp in [1.5, 2, 3, 5]:
    ex.append(("funding_carry", HC, dict(strategy="carry", tf="1h", grid={"window": [3, 7, 21], "enter": [0.0, 0.0001, 0.0002]}, fixed={"levp": levp})))
for levp in [2, 3]:
    ex.append(("funding_carry", HC, dict(strategy="carry", tf="4h", grid={"window": [3, 7, 21, 60], "enter": [0.00005, 0.00015, 0.0003]}, fixed={"levp": levp})))
for tf in ["5m", "15m"]:
    for lev in [1, 2, 3]:
        ex.append(("grid", HG, dict(strategy="grid", tf=tf, grid={"spacing": [0.005, 0.01, 0.02, 0.03], "levels": [3, 5, 10]}, fixed={"lev": lev, "filter": "none"})))
    ex.append(("grid", HG, dict(strategy="grid", tf=tf, grid={"spacing": [0.005, 0.01, 0.02], "levels": [5, 10], "sma_days": [50, 100, 200]}, fixed={"lev": 1, "filter": "above"})))
    ex.append(("grid", HG, dict(strategy="grid", tf=tf, grid={"spacing": [0.005, 0.01, 0.02], "levels": [5, 10], "sma_days": [50, 100, 200]}, fixed={"lev": 2, "filter": "above"})))
for fam, h, e in ex:
    run({"id": next_id(), "family": fam, "lev": 1.0, "venue": "delta_india_perp", "hypothesis": h, **e})
