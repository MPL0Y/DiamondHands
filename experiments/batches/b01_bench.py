from src.experiment import run, next_id
H = "Benchmark: passive exposure sets the bar every active strategy must beat."
for venue, lev, band, tf in [("coindcx_spot",1,0,"1d"),("delta_india_perp",1,0,"1d"),("delta_india_perp",2,0,"1d"),
                             ("delta_india_perp",3,0,"1d"),("delta_india_perp",2,0.25,"1d"),("delta_india_perp",3,0.25,"1d")]:
    run({"id": next_id(), "family": "benchmark", "strategy": "hold", "tf": tf, "lev": lev, "venue": venue, "band": band,
         "hypothesis": H + f" venue={venue} lev={lev} rebalance band={band}"})
