# India-accessible venues (researched 2026-09-26)

| | Delta Exchange India (chosen) | CoinDCX Futures | CoinDCX Spot (INR) | CoinSwitch / Mudrex |
|---|---|---|---|---|
| Product | BTCUSD perpetual, USD-quoted, **INR-margined** | B-BTC_USDT perpetual | BTC/INR spot | Mudrex: USDT perps via partner; CoinSwitch PRO: futures |
| Taker / maker | 0.05% / 0.02% + 18% GST → **0.059% / 0.0236%** | 0.075% / 0.025% (+GST) | ~0.5% + GST | higher than Delta at retail tiers |
| Contract / min size | **0.001 BTC per contract, min 1** | 0.001 BTC | fractional | — |
| Leverage / margin | up to 200x; IM 0.5%, **MM 0.25%**; isolated liq. penalty 0.05% | up to 20x on BTC for most users | 1x | — |
| Funding | every 8h (05:30/13:30/21:30 IST = 00/08/16 UTC), premium + 0.01%/8h interest, cap 1% (clamp 0.05) | 8h | none | — |
| INR rails | INR deposit/withdraw only; **fixed ₹85 per USD** for balances | INR deposit, USDT wallet | INR | INR |

Sources: live product spec from `https://api.india.delta.exchange/v2/products/BTCUSD` (queried 2026-09-26:
contract_value 0.001, taker_commission_rate 0.0005, maker 0.0002, initial_margin 0.5%, maintenance_margin 0.25%,
isolated_liq_penalty_factor 0.0005, rate_exchange_interval 28800s);
[Delta fee article](https://www.delta.exchange/support/solutions/articles/80001177864-fees-on-options-and-futures-trading);
[Delta India perpetual guide](https://guides.delta.exchange/delta-exchange-india-user-guide/derivatives-guide/docs);
[Delta India USD-INR rate](https://guides.delta.exchange/delta-exchange-india-user-guide/account-setup/usd-inr-rate);
[CoinDCX futures fees](https://coindcx.com/blog/crypto-futures-trading/how-much-it-cost-to-trade-in-crypto-futures/);
[TradersUnion CoinDCX fees](https://tradersunion.com/brokers/crypto/view/coindcx/fees/).

## Modelling choices
- All directional strategies are simulated on **Delta Exchange India** costs: taker 0.059% plus 0.05% slippage per
  side, maker 0.0236% for resting limit orders. Lots are 0.001 BTC, MM 0.25%, and liquidation loses the whole margin.
- **Lot size binds at ₹10,000.** ₹10k ≈ $113–156 over 2018–2025. At BTC $110k one contract is ~$110, so
  the smallest possible position is already ~1x. Sub-1x sizing rounds to zero in 2024–25. The simulator enforces this.
- Funding history: Delta publishes no bulk history, so **Binance BTCUSDT funding** (2020+) and **BitMEX XBTUSD** (pre-2020 and
  the Sep-2025 tail) are used. Both follow the same premium + 0.01% interest formula.
- INR conversion uses historical FRED DEXINUS. Delta itself uses a fixed ₹85/USD, so on Delta a strategy's INR
  *return* equals its USD return. Monthly profit is USD profit × that month's USDINR, and FX drift on idle capital is excluded.
- Cash-and-carry needs a spot leg. The only INR spot option modelled is CoinDCX (≈0.59% incl. GST, plus ~$1 per
  cross-venue transfer). At ₹10k those costs exceed the funding collected (see the carry experiments).
- Taxes ignored per instructions (1% TDS, 30% tax on VDA gains, 18% GST on fees is **included** since it is a fee).
