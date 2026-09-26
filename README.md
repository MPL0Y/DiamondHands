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

# 2. experiments, in order (each appends a row to experiments/leaderboard.csv and git-commits).
#    IDs are sequential, so later batches (ensembles) reference earlier IDs: run them in this order.
python -m src.options           # downloads real Deribit weekly option prints (needed by b15)
for b in $(ls experiments/batches/b*.py | sort); do python -m experiments.batches.$(basename $b .py); done
python -m src.dca               # DCA benchmark (reported in final_report.md)
python -m src.family_pbo ensemble; python -m src.family_pbo calendar

# 3. finalists: plateau, engine agreement (bar + 1-minute replay), leverage frontier, charts
python -m src.finalize --frontier E0449 E0425 E0510 E0395 E0453 --finalists E0449 E0425 E0510
git checkout lockbox-freeze     # the frozen code
python -m src.download --lockbox && python -m src.build_lockbox && python -m src.lockbox   # Section 8, runs ONCE

# 4. live signal for E0449 (signals only: never places orders, needs no API keys). Current-window params: signals/current_selections.json
python signals/generate_signal.py            # prints action, size in ₹ and BTC, leverage, stop, target, next check
python signals/generate_signal.py --paper    # also appends to signals/paper_log.csv (hypothetical fills)
```

## Result in one paragraph
Best risk-compliant strategy (E0449, an ensemble of a funding filter, a vol-regime trend and an ML meta-label on
Delta Exchange India's BTC perp at up to 2×): walk-forward OOS 2018–2025 median **₹142/month on ₹10,000**
(CAGR 63%, max DD 49%, 0% of months ≥ ₹1L). Lockbox year: **−24.5%**. Capital needed for ₹1L/month at the OOS median ≈ ₹44 lakh.
The ₹1L-on-₹10k target is ~700× beyond anything that respects the risk limits (`reports/ceiling.md`).

Notes:
- `fapi.binance.com` and Bybit may be geo-blocked, so the live generator uses only public, keyless endpoints:
  Binance's market-data mirror `data-api.binance.vision` (klines), `data.binance.vision` monthly funding files plus OKX public
  funding for the current month, alternative.me Fear & Greed, and Delta Exchange India's ticker for the mark price.
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
