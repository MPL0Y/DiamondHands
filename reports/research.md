# Literature → hypotheses (every claimed return is a hypothesis, tested here)

| Source | Claim | Tested as | Result (see leaderboard) |
|---|---|---|---|
| Moskowitz, Ooi & Pedersen (2012); Liu & Tsyvinski (2021) "Risks and Returns of Cryptocurrency" | time-series momentum in BTC | `tsmom`, `ma_cross`, `donchian`, `supertrend` (1h/4h/1d) | positive CAGR (up to ~22%) but DD 44–80% at 1x; median month ≈ ₹0 |
| Moreira & Muir (2017) vol-managed portfolios | scaling by inverse vol raises Sharpe | `vol_target` overlays, `vol_regime` | small Sharpe gains; DD not reliably < 50% |
| Baur et al. (2019), [SSRN 3088472](https://www.ssrn.com/abstract=3088472) | time-of-day/weekday effects are time-varying, not persistent | `calendar` hours/weekdays | confirmed: no persistent intraday effect survives costs |
| [QuantPedia: Seasonality of Bitcoin](https://quantpedia.com/the-seasonality-of-bitcoin/) | long 21:00–23:00 UTC → 40.6%/yr, Calmar 1.79 | `calendar` hours=(21,22) and neighbours | see E-rows "calendar 21-23" |
| [Turn-of-the-month in cryptocurrencies](https://www.researchgate.net/publication/359327404_Turn-of-the-month_effect_in_cryptocurrencies) | TOM returns significantly > non-TOM (2015–2021) | `calendar` tom=1..5 (+trend filter) | see calendar rows |
| [Turn-of-the-candle effect](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10015199/) | +0.58 bp/min at 15-min candle turns | not tradable: 0.58 bp ≪ 21.8 bp round-trip taker+slippage | rejected a priori (cost arithmetic) |
| [Concretum: intraday trend seasonality](https://concretumgroup.com/seasonality-in-bitcoin-intraday-trend-trading/) | Monday Asia-open trend pickup | `calendar` weekday/hour sets on 1h | see calendar rows |
| Schmeling, Schrimpf & Todorov (2023) crypto carry; [Fundamentals of Perpetual Futures (arXiv 2212.06888)](https://arxiv.org/pdf/2212.06888) | funding carry earns ~10%+/yr; high funding predicts lower returns | `carry`, `funding_contra`, `funding_level` | carry is fee-dominated at ₹10k in India; funding-level filter is the best signal found |
| [Predictability of Funding Rates (SSRN 5576424)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424) | funding is predictable out of sample | funding z-score features in `ml_dir` | adds little OOS |
| López de Prado (2018) meta-labeling; Bailey & López de Prado (2014) DSR; Bailey et al. (2016) PBO | methodology | `ml_meta`, DSR, CSCV-PBO in every leaderboard row | — |
| Crabel / L. Williams volatility breakout | range expansion starts trends | `vol_breakout` | negative after costs on 15m–4h; flat on 1d |
