"""Venue cost models. Researched 2026-09-26 (sources in reports/venues.md).

All fee rates are fractions of notional and include GST where applicable.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Venue:
    name: str
    taker: float           # per side, fraction of notional (incl. GST)
    maker: float
    slippage: float        # per side baseline
    lot_btc: float         # contract / lot step in BTC
    min_lots: int
    mmr: float             # maintenance margin rate
    max_leverage: float
    funding: bool          # perp funding charged
    liq_fee: float = 0.0005  # isolated liquidation penalty (fraction of notional)

    def stressed(self, k=3.0):
        return replace(self, taker=self.taker * k, maker=self.maker * k, slippage=self.slippage * k,
                       name=f"{self.name}_x{k:g}")


# Delta Exchange India BTCUSD perpetual (api.india.delta.exchange/v2/products/BTCUSD):
# contract 0.001 BTC, taker 0.05% maker 0.02% + 18% GST, IM 0.5% MM 0.25% (200x), 8h funding,
# isolated liquidation penalty 0.05%. INR wallet at fixed 85 INR/USD.
DELTA_INDIA = Venue("delta_india_perp", taker=0.0005 * 1.18, maker=0.0002 * 1.18, slippage=0.0005,
                    lot_btc=0.001, min_lots=1, mmr=0.0025, max_leverage=200, funding=True)

# CoinDCX futures (B-BTC_USDT perp): taker 0.075% maker 0.025% (+18% GST), 0.001 BTC min qty.
COINDCX_PERP = Venue("coindcx_perp", taker=0.00075 * 1.18, maker=0.00025 * 1.18, slippage=0.0005,
                     lot_btc=0.001, min_lots=1, mmr=0.005, max_leverage=100, funding=True)

# CoinDCX spot BTC/INR: ~0.5% taker incl GST; 1% TDS ignored (taxes ignored). Fractional BTC allowed.
COINDCX_SPOT = Venue("coindcx_spot", taker=0.005 * 1.18, maker=0.005 * 1.18, slippage=0.0005,
                     lot_btc=1e-6, min_lots=1, mmr=0.0, max_leverage=1, funding=False)

VENUES = {v.name: v for v in (DELTA_INDIA, COINDCX_PERP, COINDCX_SPOT)}
