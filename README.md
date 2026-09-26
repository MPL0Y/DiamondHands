# DiamondHands — BTC strategy research (₹10,000 → ₹1,00,000/month?)

This repo runs a rigorous, leakage-audited search for a BTC strategy that could earn ₹1,00,000/month
from ₹10,000 of capital. It also ships a signals-only generator for the best strategy found.
**Read `reports/final_report.md` and `reports/ceiling.md` for the answer.** Short version: the target
(~1,000%/month) is not achievable within the risk limits. The reports quantify the ceiling and the
capital the best strategy would actually need.

## Reproduce from a clean machine

```bash
git clone <this repo> && cd DiamondHands
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 1. data (development window only; the last 12 months are a sealed lockbox)
python -m src.download          # Binance spot+perp 1m, funding, BitMEX funding, Bitstamp, USDINR (FRED), Fear&Greed
python -m src.build_data        # cleans, resamples, writes data/processed/*.parquet + reports/data_quality.md

# 2. experiments (each appends a row to experiments/leaderboard.csv and git-commits)
for b in experiments/batches/b*.py; do python -m experiments.batches.$(basename $b .py); done

# 3. finalists: gates, leverage frontier, lockbox, report
python -m src.finalize          # plateau, engine agreement (bar + 1-minute), frontier, charts, reports
python -m src.download --lockbox && python -m src.build_lockbox && python -m src.lockbox   # Section 8, run once

# 4. live signal (signals only — never places orders, needs no API keys)
python signals/generate_signal.py            # prints action, size in ₹ and BTC, leverage, stop, target, next check
python signals/generate_signal.py --paper    # also appends to signals/paper_log.csv (hypothetical fills)
```

Notes:
- `data.binance.vision` (bulk history) is reachable from India. `fapi.binance.com` / Bybit may be geo-blocked,
  so the live generator uses Delta Exchange India's public API plus OKX public funding.
- Experiments E0001–E0212 were superseded after a bug fix: the withdraw-mode P&L had been counting
  USDINR drift on idle capital. They were re-run as E0213–E0424. Both sets stay in the leaderboard, and N counts both.

## Layout

| path | what |
|---|---|
| `src/download.py`, `src/build_data.py` | data acquisition, cleaning, quality report |
| `src/engine_fast.py` | numba array engine (search) |
| `src/engine_event.py` | independent event-driven engine (verification, optional 1-minute intrabar replay) |
| `src/backtest.py` | walk-forward selection, evaluation, gates (MC, regimes, stress, delay) |
| `src/strategies.py`, `src/ml.py`, `src/custom.py`, `src/ensemble.py` | strategy families |
| `src/experiment.py` | experiment runner → `experiments/leaderboard.csv`, `experiments/configs/`, `experiments/details/` |
| `src/finalist.py` | plateau test, engine agreement, leverage frontier |
| `src/venues.py` | venue cost models (Delta Exchange India, CoinDCX) |
| `reports/` | data quality, venue research, final report, ceiling, figures |
| `signals/generate_signal.py` | live signal + paper-trading logger |
