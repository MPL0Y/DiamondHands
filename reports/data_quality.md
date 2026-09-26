# Data quality report

Development window ends (exclusive) at 2025-09-26 00:00:00+00:00 — lockbox sealed after that.

## Binance spot BTCUSDT 1m

- Range: 2017-08-17 04:00:00+00:00 → 2025-09-25 23:59:00+00:00
- Raw rows 4,256,479; duplicate timestamps removed: 1
- Timestamps not aligned to the UTC minute: 43204 (constant sub-minute offset in 2017-12-04→12-18 and 2018-02-08→10 files; floored to the minute — without this those 16 days look like an outage)
- Rows with invalid OHLC (≤0, high<max(o,c), low>min(o,c)) dropped: 0
- Expected minutes 4,265,040; missing 8,562 (0.201%) in 34 gaps
- Largest gaps (exchange outages / maintenance):

  - 2018-02-08 00:29:00+00:00 → 2018-02-09 09:58:00+00:00 (2010 min)
  - 2018-06-26 02:00:00+00:00 → 2018-06-26 11:59:00+00:00 (600 min)
  - 2019-05-15 03:00:00+00:00 → 2019-05-15 12:59:00+00:00 (600 min)
  - 2019-08-15 02:00:00+00:00 → 2019-08-15 09:59:00+00:00 (480 min)
  - 2018-07-04 00:23:00+00:00 → 2018-07-04 07:59:00+00:00 (457 min)
  - 2017-09-06 16:00:00+00:00 → 2017-09-06 22:59:00+00:00 (420 min)
  - 2018-11-14 02:00:00+00:00 → 2018-11-14 08:59:00+00:00 (420 min)
  - 2019-03-12 02:00:00+00:00 → 2019-03-12 07:59:00+00:00 (360 min)
  - 2020-02-19 11:36:00+00:00 → 2020-02-19 17:29:00+00:00 (354 min)
  - 2021-04-25 04:01:00+00:00 → 2021-04-25 08:44:00+00:00 (284 min)
  - 2021-08-13 02:00:00+00:00 → 2021-08-13 06:29:00+00:00 (270 min)
  - 2020-12-21 14:10:00+00:00 → 2020-12-21 17:59:00+00:00 (230 min)

- 1m spike-and-revert prints (>8% move reversed next minute): none
- 1m bars with wicks >10% beyond body: none

## Binance perp BTCUSDT 1m

- Range: 2020-01-01 00:00:00+00:00 → 2025-09-25 23:59:00+00:00
- Raw rows 3,016,800; duplicate timestamps removed: 0
- Timestamps not aligned to the UTC minute: 0 (constant sub-minute offset in 2017-12-04→12-18 and 2018-02-08→10 files; floored to the minute — without this those 16 days look like an outage)
- Rows with invalid OHLC (≤0, high<max(o,c), low>min(o,c)) dropped: 0
- Expected minutes 3,016,800; missing 0 (0.000%) in 0 gaps
- Largest gaps (exchange outages / maintenance):


- 1m spike-and-revert prints (>8% move reversed next minute): none
- 1m bars with wicks >10% beyond body: [('2020-03-12 10:48:00+00:00', 0.123), ('2021-04-18 03:35:00+00:00', 0.149), ('2021-05-19 13:20:00+00:00', 0.113), ('2021-05-19 13:29:00+00:00', 0.11), ('2021-07-26 01:01:00+00:00', 0.192), ('2021-07-26 01:02:00+00:00', 0.104)]

## Funding rates

- BitMEX XBTUSD 2016-05-14 12:00:00+00:00 → 2025-09-25 20:00:00+00:00 (4014 prints), Binance BTCUSDT 2020-01-01 00:00:00+00:00 → 2025-09-25 20:00:00+00:00 (6210 prints)
- Interval not 8h: 23 occurrences (max gap 24h)
- Rate range -0.00682 … 0.01125; mean 0.000133/8h (≈14.6%/yr paid by longs)
- Cross-check BitMEX XBTUSD vs Binance BTCUSDT on 0 overlapping prints: corr nan, mean nan vs nan. Bybit and the Binance fapi are geo-blocked from this environment.
- Delta Exchange India publishes no downloadable history; Binance is used as its proxy (same 8h schedule, same premium+0.01% interest formula).

## USDINR (FRED DEXINUS, noon NY buying rate)

- 1973-01-02 → 2026-09-18, forward-filled over weekends/holidays; range 7.19–96.82

## Bitstamp BTCUSD 1d spliced with Binance at 2017-08-17 00:00:00+00:00

- Bitstamp rows 2,191 from 2011-08-18 00:00:00+00:00; invalid rows 0; zero-volume bars 33 (thin 2011–2013 liquidity: treat pre-2014 fills as optimistic; used for training history only, never OOS)

## Bitstamp BTCUSD 4h spliced with Binance at 2017-08-17 04:00:00+00:00

- Bitstamp rows 13,144 from 2011-08-18 12:00:00+00:00; invalid rows 0; zero-volume bars 887 (thin 2011–2013 liquidity: treat pre-2014 fills as optimistic; used for training history only, never OOS)

## Fear & Greed (alternative.me)

- 2018-02-01 → 2025-09-25, 2790 rows; value for day D is published shortly after 00:00 UTC on D. Used only with a 1-day lag (known at the D+1 daily open).

## Handling rules

- Outage minutes are NOT forward-filled at 1m. Resampled bars with no trades carry the prior close and zero volume.
- No price 'repair' is applied. Spikes listed above are genuine exchange prints and stay in the data, so they can trigger stops or liquidations. That is conservative.
- All timestamps are UTC bar-open times. Bars are [open, open+Δ).
