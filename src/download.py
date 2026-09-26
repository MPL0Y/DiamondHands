"""Download all raw data for the DEVELOPMENT window (strictly before the lockbox).

Sources
- Binance spot BTCUSDT 1m klines (data.binance.vision), 2017-08 -> 2025-09-25
- Binance USD-M perp BTCUSDT 1m klines (data.binance.vision), 2020-01 -> 2025-09-25
- Binance USD-M BTCUSDT funding rate history (data.binance.vision), 2020-01 ->
- BitMEX XBTUSD funding history (2016 ->) to cover pre-2020 perp funding
- Bitstamp BTCUSD daily + 4h OHLC (2011 ->) via public API
- USDINR daily (FRED DEXINUS)
- Crypto Fear & Greed index (alternative.me)

Usage: python -m src.download [--lockbox]   (--lockbox only in Section 8)
"""
import argparse, io, json, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import pandas as pd
import requests

from src.config import RAW, LOCKBOX_DIR, LOCKBOX_START, LOCKBOX_END, ROOT

S = requests.Session()
BV = "https://data.binance.vision/data"


def get(url, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except Exception as e:  # network retry with backoff
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def months(start, end_excl):
    d = date(start.year, start.month, 1)
    while d < end_excl.replace(day=1):
        yield d
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def fetch_zip(url, out):
    if out.exists():
        return out
    c = get(url)
    if c is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(c)
    return out


def binance_klines(kind, sym, start, end_excl, dest):
    """kind: 'spot' or 'futures/um'. Monthly files for full months, daily for the tail."""
    jobs = []
    for m in months(start, end_excl):
        f = f"{sym}-1m-{m:%Y-%m}.zip"
        jobs.append((f"{BV}/{kind}/monthly/klines/{sym}/1m/{f}", dest / f))
    d = end_excl.replace(day=1)
    while d < end_excl:
        f = f"{sym}-1m-{d:%Y-%m-%d}.zip"
        jobs.append((f"{BV}/{kind}/daily/klines/{sym}/1m/{f}", dest / f))
        d += timedelta(days=1)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: fetch_zip(*j), jobs))
    missing = [j[0] for j, r in zip(jobs, res) if r is None]
    print(kind, sym, "files:", len(jobs), "missing:", len(missing), missing[:5])


def binance_funding(start, end_excl, dest):
    jobs = []
    for m in months(start, end_excl):
        f = f"BTCUSDT-fundingRate-{m:%Y-%m}.zip"
        jobs.append((f"{BV}/futures/um/monthly/fundingRate/BTCUSDT/{f}", dest / f))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: fetch_zip(*j), jobs))


def bitmex_funding(dest, end):
    out = dest / "bitmex_xbtusd_funding.csv"
    if out.exists():
        return
    rows, start = [], 0
    while True:
        u = f"https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=500&start={start}&reverse=false"
        js = json.loads(get(u))
        if not js:
            break
        rows += js
        start += len(js)
        if pd.Timestamp(js[-1]["timestamp"]) >= end:
            break
        time.sleep(1.2)
    df = pd.DataFrame(rows)[["timestamp", "fundingRate"]]
    df.to_csv(out, index=False)
    print("bitmex funding rows", len(df))


def bitstamp(dest, step, end):
    out = dest / f"bitstamp_btcusd_{step}.csv"
    if out.exists():
        return
    rows, t = [], 1313000000  # 2011-08
    endts = int(end.timestamp())
    while t < endts:
        u = f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step={step}&limit=1000&start={t}"
        js = json.loads(get(u))["data"]["ohlc"]
        if not js:
            break
        rows += js
        nt = int(js[-1]["timestamp"]) + step
        if nt <= t:
            break
        t = nt
        time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates("timestamp")
    df.to_csv(out, index=False)
    print("bitstamp", step, len(df))


def fred_usdinr(dest):
    c = get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXINUS")
    (dest / "usdinr_fred.csv").write_bytes(c)


def fng(dest):
    c = get("https://api.alternative.me/fng/?limit=0&format=json")
    (dest / "fng.json").write_bytes(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lockbox", action="store_true")
    a = ap.parse_args()
    if a.lockbox:
        with open(ROOT / "reports" / "lockbox_access.log", "a") as f:
            f.write(f"{pd.Timestamp.utcnow()} DOWNLOAD lockbox raw data\n")
        dest = LOCKBOX_DIR / "raw"
        s, e = LOCKBOX_START.date(), LOCKBOX_END.date()
        # full months from lockbox start month; the dev part of Sep 2025 is dropped at build time
        binance_klines("spot", "BTCUSDT", s, e, dest / "spot")
        binance_klines("futures/um", "BTCUSDT", s, e, dest / "perp")
        binance_funding(s, e, dest / "funding")
        return
    end = LOCKBOX_START.date()
    binance_klines("spot", "BTCUSDT", date(2017, 8, 1), end, RAW / "spot")
    binance_klines("futures/um", "BTCUSDT", date(2020, 1, 1), end, RAW / "perp")
    binance_funding(date(2020, 1, 1), end, RAW / "funding")
    bitmex_funding(RAW, LOCKBOX_START)
    bitstamp(RAW, 86400, LOCKBOX_START)
    bitstamp(RAW, 14400, LOCKBOX_START)
    fred_usdinr(RAW)
    fng(RAW)


if __name__ == "__main__":
    main()
