from src.experiment import run, next_id
H = "Benchmark top-up: passive BTC exposure at 0.5-5x, spot vs perp, never/banded rebalancing. Defines what 'no skill' earns at each risk level."
ex = []
for lev in [0.5, 1.5, 5.0]:
    ex.append(dict(venue="delta_india_perp", lev=lev, band=0.0))
for lev in [0.5, 1.0, 1.5, 2.0, 3.0]:
    for band in [0.1, 0.5]:
        ex.append(dict(venue="delta_india_perp", lev=lev, band=band))
for tf in ["1h", "4h"]:
    ex.append(dict(venue="coindcx_spot", lev=1.0, band=0.0, tf=tf))
ex.append(dict(venue="coindcx_perp", lev=1.0, band=0.0))
for e in ex:
    tf = e.pop("tf", "1d")
    run({"id": next_id(), "family": "benchmark", "strategy": "hold", "tf": tf, "hypothesis": H, **e})
