# Ceiling: what ₹10,000 in BTC can earn per month (stop condition B)

## Answer

| | Walk-forward OOS 2018-01 → 2025-09 | Lockbox 2025-09-26 → 2026-09-25 |
|---|---|---|
| **Best median monthly profit on ₹10,000 within the risk limits** (max DD ≤ 50%, P(ruin) ≤ 5%) | **₹142** (E0449 ensemble, 2×) | ₹0 (E0449); best finalist −₹11 (E0425) |
| Share of months ≥ ₹1,00,000 | 0 of 93 (best single month ₹7,069) | 0 of 13 |
| Target (median ₹1,00,000) achieved? | **No: 0.14% of target** | **No** |
| **Capital required for ₹1,00,000/month** at the best OOS median return (2.25%/month, lot-free) | **≈ ₹44 lakh** | not estimable (median return ≤ 0) |
| Runner-up with the least overfitting (turn-of-month, E0425): capital for ₹1L/month | ≈ ₹1.04 crore (0.96%/month) | — |

The target implies ~1,000% per month, **≈700× the best risk-compliant median found**. Nothing within the Section 4–5
rules comes within two orders of magnitude.

## Why this is the ceiling (evidence)

**1. Every family was searched with ≥ 20 variants: 526 experiments, 6,549 configurations.**

| family | variants | within limits | best id (within limits) | median ₹/mo | max DD | best median at any risk | id | its max DD |
|:---|---:|---:|:---|---:|:---|---:|:---|:---|
| ensemble | 42 | 14 | E0449 | 142 | 49% | 375 | E0489 | 65% |
| calendar | 32 | 8 | E0467 | 135 | 49% | 135 | E0467 | 49% |
| options (real Deribit prints) | 21 | 11 | E0510 | 130 | 50% | 514 | E0499 | 100% (liquidated) |
| funding (contrarian / level) | 37 | 10 | E0395 | 107 | 46% | 223 | E0311 | 63% |
| grid | 20 | 13 | E0334 | 0 | 45% | 0 | E0334 | 45% |
| mean reversion | 39 | 5 | E0248 | 0 | 42% | 96 | E0260 | 64% |
| ML (LightGBM, meta-labeling) | 26 | 4 | E0420 | 0 | 47% | 0 | E0407 | 65% |
| trend (incl. Supertrend) | 42 | 3 | E0226 | 0 | 44% | 95 | E0238 | 77% |
| volatility | 27 | 6 | E0284 | 0 | 46% | 10 | E0281 | 65% |
| benchmark | 22 | 0 | – | – | – | 64 | E0525 | 82% |
| funding carry (cash-and-carry) | 6 | 0 | – | – | – | 0 | E0331 | 78% |

Even **ignoring the risk limits**, the highest median month of any valid experiment is ₹514 (short weekly options at
5×, which was liquidated: −100%). No configuration of any kind reached ₹1,00,000 in its median month. **None of the 314 valid experiments had even one
month ≥ ₹1,00,000** in 93 OOS months. Only the 10–20× frontier probes produced ₹1L months (1–3% of months), and they were
liquidated in most other months.

**2. The leverage frontier is mapped for the top 5 (`reports/frontier_*.csv`, `figures/leverage_frontier.png`).** Leverage
is the only lever that scales ₹ profit, and it is capped by the drawdown limit at about 1× of each strategy's own scale.
From 5× up, every top-5 strategy is liquidated in the 2020–2022 crash wicks (verified on 1-minute data), and its median
withdraw-mode month collapses toward −₹10,000 at 20×. More leverage makes ₹ outcomes worse, not better.

**3. The leaderboard stopped improving.** The best within-limits median was set by E0449 (₹142). The following
**77 consecutive experiments** (E0450–E0526) did not beat it by > 5%. They covered ensemble variants, literature
tests, the options family and benchmarks.

**4. Structural reasons the ceiling is low.**
- **Volatility vs the DD limit.** BTC's annualised vol is ~60–80%. Any strategy exposed to it for a meaningful share of the time
  hits a ~50% drawdown at around 1× over 2018–2025. At 1×, BTC buy-and-hold's own median month was +0.64% (₹64 on ₹10k, E0213).
- **Costs vs short-horizon edges.** Delta India taker 0.059% + 0.05% slippage ≈ 0.22% per round trip. The published intraday
  effects (0.58 bp/min turn-of-candle, 21–23 UTC drift) are smaller than that, and they tested negative after costs.
- **Carry is fee-dominated at ₹10k in India.** Funding ≈ 10–15%/yr gross is eaten by 0.59% spot fees + ~$1 transfers.
- **Lot granularity.** On ₹10k (~$113) one 0.001 BTC contract (~$84–110) is already ~1×. Sizing is coarse and sub-1× positions round to zero.
- **Significance.** Deflated Sharpe ≈ 0 for every finalist at N = 6,549. The best OOS results are consistent with
  the best of many near-zero-edge strategies, and the lockbox (−24.5% for #1) agrees.

## What ₹1,00,000/month would realistically take
At the best risk-compliant OOS median (2.25%/month, which the lockbox did **not** confirm), the requirement is ≈ ₹44 lakh of
capital. At a more conservative 1%/month (turn-of-month, the least overfit family), it is ≈ ₹1 crore. On the lockbox evidence,
no amount of capital was sufficient in the most recent year.
