"""Build clean parquet datasets from raw downloads + write reports/data_quality.md.

Outputs (data/processed/):
  spot_1m.parquet, perp_1m.parquet            (UTC, 1-minute, dev window only)
  spot_{5m,15m,1h,4h,1d}.parquet, perp_*.parquet
  funding.parquet    8h funding rates (BitMEX XBTUSD before Binance data exists, then Binance)
  usdinr.parquet     daily USDINR (FRED DEXINUS, forward-filled over holidays)
  daily_long.parquet Bitstamp BTCUSD daily 2011-> spliced with Binance from 2017-08-17
  h4_long.parquet    same for 4h
  fng.parquet        Fear & Greed (value for day D published ~00:00 UTC D; usable from D+1 open)
"""
import glob, io, json, zipfile
import numpy as np
import pandas as pd

from src.config import RAW, PROC, DEV_END, REPORTS, LOCKBOX_DIR

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "trades", "tb_base", "tb_quote", "ignore"]
RULES = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def read_kline_zip(path):
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    df = pd.read_csv(io.BytesIO(raw), header=None, names=COLS, low_memory=False)
    if not str(df.iloc[0, 0]).strip().isdigit():  # header row in some futures files
        df = df.iloc[1:]
    df = df.astype({"open_time": "int64"})
    ot = df["open_time"].to_numpy()
    # Binance spot switched to microsecond timestamps in 2025
    ts = np.where(ot > 10**14, ot // 1000, ot)
    out = pd.DataFrame({
        "open": df["open"].astype(float).to_numpy(),
        "high": df["high"].astype(float).to_numpy(),
        "low": df["low"].astype(float).to_numpy(),
        "close": df["close"].astype(float).to_numpy(),
        "volume": df["volume"].astype(float).to_numpy(),
        "trades": df["trades"].astype(float).to_numpy(),
        "tb_base": df["tb_base"].astype(float).to_numpy(),
    }, index=pd.to_datetime(ts, unit="ms", utc=True))
    out.attrs["misaligned"] = int((out.index.second != 0).sum() + (out.index.microsecond != 0).sum())
    out.index = out.index.floor("1min")  # some 2017-12/2018-02 files carry a constant +20.8s/+14.8s offset
    return out


def load_klines(folder, start=None, end=None):
    files = sorted(glob.glob(str(folder / "*.zip")))
    parts = [read_kline_zip(f) for f in files]
    q = {"misaligned_fixed": sum(p.attrs["misaligned"] for p in parts)}
    df = pd.concat(parts).sort_index()
    q["rows_raw"] = len(df)
    q["duplicates"] = int(df.index.duplicated().sum())
    df = df[~df.index.duplicated(keep="first")]
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index < end]
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df.high < df[["open", "close"]].max(axis=1) - 1e-9) \
        | (df.low > df[["open", "close"]].min(axis=1) + 1e-9)
    q["bad_ohlc_rows"] = int(bad.sum())
    df = df[~bad]
    full = pd.date_range(df.index[0], df.index[-1], freq="1min")
    missing = full.difference(df.index)
    q["first"], q["last"] = str(df.index[0]), str(df.index[-1])
    q["expected_minutes"], q["missing_minutes"] = len(full), len(missing)
    # contiguous gaps
    gaps = []
    if len(missing):
        m = pd.Series(missing)
        grp = (m.diff() != pd.Timedelta("1min")).cumsum()
        for _, g in m.groupby(grp):
            gaps.append((g.iloc[0], g.iloc[-1], len(g)))
    gaps.sort(key=lambda x: -x[2])
    q["gaps_top"] = gaps[:12]
    q["n_gaps"] = len(gaps)
    # spike detection: 1m move >8% that reverts >80% on the next minute
    r = np.log(df.close).diff()
    spikes = (r.abs() > 0.08) & (r.shift(-1) * r < 0) & (r.shift(-1).abs() > 0.8 * r.abs())
    q["spike_reversals_1m"] = [str(t) for t in df.index[spikes]]
    wick = np.log(df.high / df[["open", "close"]].max(axis=1)).clip(lower=0) + \
        np.log(df[["open", "close"]].min(axis=1) / df.low).clip(lower=0)
    q["wicks_gt_10pct"] = [(str(t), round(float(v), 3)) for t, v in wick[wick > 0.10].items()]
    return df, q


def resample(df, rule):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
           "trades": "sum", "tb_base": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    # empty bars (full outages): carry previous close, zero volume
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["n_min"] = df["close"].resample(rule, label="left", closed="left").count()
    return out


def build_funding():
    rows = []
    for f in sorted(glob.glob(str(RAW / "funding" / "*.zip"))):
        with zipfile.ZipFile(f) as z:
            d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
        rows.append(d)
    b = pd.concat(rows)
    b.index = pd.to_datetime(b["calc_time"], unit="ms", utc=True).dt.floor("1h")
    b = b["last_funding_rate"].astype(float).rename("rate")
    m = pd.read_csv(RAW / "bitmex_xbtusd_funding.csv")
    m.index = pd.to_datetime(m["timestamp"], utc=True).dt.floor("1h")
    m = m["fundingRate"].astype(float).rename("rate")
    mx = m.copy()
    # cross-check: BitMEX vs Binance overlap
    j = pd.concat([mx.rename("bitmex"), b.rename("binance")], axis=1, join="inner")
    build_funding.xcheck = (len(j), float(j.corr().iloc[0, 1]), float(j.bitmex.mean()), float(j.binance.mean()))
    m = pd.concat([m[m.index < b.index.min()], m[m.index > b.index.max()]])  # BitMEX fills pre-2020 and Sep-2025 tail
    f = pd.concat([m, b]).sort_index()
    f = f[~f.index.duplicated()]
    f = f[f.index < DEV_END]
    src = pd.Series(np.where((f.index < b.index.min()) | (f.index > b.index.max()), "bitmex", "binance"), index=f.index, name="src")
    return pd.concat([f, src], axis=1)


def build_usdinr():
    d = pd.read_csv(RAW / "usdinr_fred.csv")
    d.columns = ["date", "usdinr"]
    d["usdinr"] = pd.to_numeric(d["usdinr"], errors="coerce")
    d.index = pd.to_datetime(d["date"]).dt.tz_localize("UTC")
    s = d["usdinr"].asfreq("1D").ffill()
    return s.to_frame()


def build_bitstamp(step):
    d = pd.read_csv(RAW / f"bitstamp_btcusd_{step}.csv")
    d.index = pd.to_datetime(d["timestamp"].astype(int), unit="s", utc=True)
    d = d[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
    return d


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    rep = ["# Data quality report", "", f"Development window ends (exclusive) at {DEV_END} — lockbox sealed after that.", ""]
    for name in ("spot", "perp"):
        df, q = load_klines(RAW / name, end=DEV_END)
        df.to_parquet(PROC / f"{name}_1m.parquet")
        for k, rule in RULES.items():
            resample(df, rule).to_parquet(PROC / f"{name}_{k}.parquet")
        rep += [f"## Binance {name} BTCUSDT 1m", "",
                f"- Range: {q['first']} → {q['last']}",
                f"- Raw rows {q['rows_raw']:,}; duplicate timestamps removed: {q['duplicates']}",
                f"- Timestamps not aligned to the UTC minute: {q['misaligned_fixed']} (constant sub-minute offset in "
                "2017-12-04→12-18 and 2018-02-08→10 files; floored to the minute — without this those 16 days look like an outage)",
                f"- Rows with invalid OHLC (≤0, high<max(o,c), low>min(o,c)) dropped: {q['bad_ohlc_rows']}",
                f"- Expected minutes {q['expected_minutes']:,}; missing {q['missing_minutes']:,} "
                f"({100*q['missing_minutes']/q['expected_minutes']:.3f}%) in {q['n_gaps']} gaps",
                "- Largest gaps (exchange outages / maintenance):", ""]
        rep += [f"  - {a} → {b} ({n} min)" for a, b, n in q["gaps_top"]]
        rep += ["", f"- 1m spike-and-revert prints (>8% move reversed next minute): {q['spike_reversals_1m'] or 'none'}",
                f"- 1m bars with wicks >10% beyond body: {q['wicks_gt_10pct'][:20] or 'none'}", ""]
        print(name, q["missing_minutes"], q["n_gaps"])
    f = build_funding()
    f.to_parquet(PROC / "funding.parquet")
    gaps = f.index.to_series().diff().dt.total_seconds().div(3600)
    rep += ["## Funding rates", "",
            f"- BitMEX XBTUSD {f[f.src=='bitmex'].index.min()} → {f[f.src=='bitmex'].index.max()} "
            f"({(f.src=='bitmex').sum()} prints), Binance BTCUSDT {f[f.src=='binance'].index.min()} → {f.index.max()} "
            f"({(f.src=='binance').sum()} prints)",
            f"- Interval not 8h: {int((gaps.dropna()!=8).sum())} occurrences (max gap {gaps.max():.0f}h)",
            f"- Rate range {f.rate.min():.5f} … {f.rate.max():.5f}; mean {f.rate.mean():.6f}/8h "
            f"(≈{f.rate.mean()*3*365*100:.1f}%/yr paid by longs)",
            f"- Cross-check BitMEX XBTUSD vs Binance BTCUSDT on {build_funding.xcheck[0]} overlapping prints: "
            f"corr {build_funding.xcheck[1]:.2f}, mean {build_funding.xcheck[2]:.6f} vs {build_funding.xcheck[3]:.6f}. "
            "Bybit and the Binance fapi are geo-blocked from this environment.",
            "- Delta Exchange India publishes no downloadable history; Binance is used as its proxy (same 8h schedule, "
            "same premium+0.01% interest formula).", ""]
    u = build_usdinr()
    u.to_parquet(PROC / "usdinr.parquet")
    rep += ["## USDINR (FRED DEXINUS, noon NY buying rate)", "",
            f"- {u.index.min().date()} → {u.index.max().date()}, forward-filled over weekends/holidays; "
            f"range {u.usdinr.min():.2f}–{u.usdinr.max():.2f}", ""]
    for step, nm in ((86400, "daily_long"), (14400, "h4_long")):
        bs = build_bitstamp(step)
        rule = "1d" if step == 86400 else "4h"
        bn = pd.read_parquet(PROC / f"spot_{rule}.parquet")[["open", "high", "low", "close", "volume"]]
        cut = bn.index[0]
        bs = bs[bs.index < cut]
        bad = int(((bs.high < bs.low) | (bs.close <= 0)).sum())
        zero_vol = int((bs.volume == 0).sum())
        lg = pd.concat([bs, bn])
        lg["src"] = np.where(lg.index < cut, "bitstamp", "binance")
        lg = lg[lg.index < DEV_END]
        lg.to_parquet(PROC / f"{nm}.parquet")
        rep += [f"## Bitstamp BTCUSD {rule} spliced with Binance at {cut}", "",
                f"- Bitstamp rows {len(bs):,} from {bs.index.min()}; invalid rows {bad}; zero-volume bars {zero_vol} "
                "(thin 2011–2013 liquidity: treat pre-2014 fills as optimistic; used for training history only, never OOS)", ""]
    fj = json.loads((RAW / "fng.json").read_text())["data"]
    fg = pd.DataFrame(fj)
    fg.index = pd.to_datetime(fg["timestamp"].astype(int), unit="s", utc=True)
    fg = fg[["value"]].astype(float).sort_index()
    fg = fg[fg.index < DEV_END]
    fg.to_parquet(PROC / "fng.parquet")
    rep += ["## Fear & Greed (alternative.me)", "",
            f"- {fg.index.min().date()} → {fg.index.max().date()}, {len(fg)} rows; value for day D is published "
            "shortly after 00:00 UTC on D. Used only with a 1-day lag (known at the D+1 daily open).", ""]
    rep += ["## Handling rules", "",
            "- Outage minutes are NOT forward-filled at 1m. Resampled bars with no trades carry the prior close and zero volume.",
            "- No price 'repair' is applied. Spikes listed above are genuine exchange prints and stay in the data, "
            "so they can trigger stops or liquidations. That is conservative.",
            "- All timestamps are UTC bar-open times. Bars are [open, open+Δ).", ""]
    (REPORTS / "data_quality.md").write_text("\n".join(rep))


if __name__ == "__main__":
    main()
