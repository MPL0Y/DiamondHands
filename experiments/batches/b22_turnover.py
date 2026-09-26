from src.experiment import run, next_id
H = ("Turnover: the E0543 blend turns over 2,216x equity in 8.7 years, and costs cut CAGR from ~161% (zero cost) to 98%. "
     "Coarser steps, hysteresis and a 4h/24h decision cadence should keep most of the signal at a fraction of the cost.")
V = [dict(step=0.5), dict(hyst=0.5), dict(hyst=0.75), dict(cadence=4), dict(cadence=4, hyst=0.5), dict(cadence=24),
     dict(cadence=24, hyst=0.5), dict(step=0.5, cadence=4), dict(cadence=8, hyst=0.5)]
for v in V:
    fx = {"components": ["E0528", "E0449"], "lev": 1.0, "step": 0.25}; fx.update(v)
    run({"id": next_id(), "family": "ensemble", "strategy": "ensemble", "tf": "1h", "lev": 1.0, "venue": "delta_india_perp",
         "hypothesis": H + f" {v}", "grid": {"weighting": ["equal", "invvol"]}, "fixed": fx})
