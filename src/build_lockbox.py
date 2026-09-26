"""Section 8 only. Build processed datasets covering dev + lockbox (so indicators have warm-up history).
Every call is logged to reports/lockbox_access.log. Evaluation code reports only [LOCKBOX_START, LOCKBOX_END)."""
import glob, io, json, zipfile
import numpy as np
import pandas as pd

from src.config import RAW, PROC, LOCKBOX_DIR, LOCKBOX_START, LOCKBOX_END, ROOT
from src.build_data import read_kline_zip, resample, RULES
from src.download import get


def log(msg):
    with open(ROOT / "reports" / "lockbox_access.log", "a") as f:
        f.write(f"{pd.Timestamp.now(tz='UTC')} {msg}\n")


def main():
    log("BUILD lockbox processed data")
    out = LOCKBOX_DIR / "processed"
    out.mkdir(parents=True, exist_ok=True)
    rep = ["# Lockbox data quality", ""]
    for name in ("spot", "perp"):
        dev = pd.read_parquet(PROC / f"{name}_1m.parquet")
        files = sorted(glob.glob(str(LOCKBOX_DIR / "raw" / name / "*.zip")))
        lb = pd.concat([read_kline_zip(f) for f in files]).sort_index()
        lb = lb[~lb.index.duplicated()]
        lb = lb[(lb.index >= LOCKBOX_START) & (lb.index < LOCKBOX_END)]
        full = pd.date_range(LOCKBOX_START, lb.index[-1], freq="1min")
        rep.append(f"- {name}: {lb.index[0]} → {lb.index[-1]}, missing minutes {len(full.difference(lb.index))}")
        df = pd.concat([dev, lb])
        df = df[~df.index.duplicated()]
        df.to_parquet(out / f"{name}_1m.parquet")
        for k, rule in RULES.items():
            resample(df, rule).to_parquet(out / f"{name}_{k}.parquet")
    # funding: Binance monthly files + BitMEX for any tail not yet published monthly
    rows = []
    for f in sorted(glob.glob(str(LOCKBOX_DIR / "raw" / "funding" / "*.zip"))):
        with zipfile.ZipFile(f) as z:
            rows.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0]))))
    b = pd.concat(rows)
    b.index = pd.to_datetime(b["calc_time"], unit="ms", utc=True).dt.floor("1h")
    b = b["last_funding_rate"].astype(float).rename("rate")
    bm = []
    start = LOCKBOX_START
    while True:
        js = json.loads(get(f"https://www.bitmex.com/api/v1/funding?symbol=XBTUSD&count=500&startTime={start.strftime('%Y-%m-%dT%H:%M:%S.000Z')}&reverse=false"))
        if not js:
            break
        bm += js
        start = pd.Timestamp(js[-1]["timestamp"]) + pd.Timedelta("1s")
        if len(js) < 500:
            break
    m = pd.DataFrame(bm)
    m.index = pd.to_datetime(m["timestamp"], utc=True).dt.floor("1h")
    m = m["fundingRate"].astype(float).rename("rate")
    devf = pd.read_parquet(PROC / "funding.parquet")["rate"]
    tail = m[m.index > b.index.max()]
    f = pd.concat([devf, b[b.index >= LOCKBOX_START], tail]).sort_index()
    f = f[~f.index.duplicated()]
    f = f[f.index < LOCKBOX_END].to_frame()
    f.to_parquet(out / "funding.parquet")
    rep.append(f"- funding prints in lockbox: {(f.index >= LOCKBOX_START).sum()} (Binance to {b.index.max()}, BitMEX after)")
    fj = json.loads(get("https://api.alternative.me/fng/?limit=0&format=json"))["data"]
    fg = pd.DataFrame(fj)
    fg.index = pd.to_datetime(fg["timestamp"].astype(int), unit="s", utc=True)
    fg = fg[["value"]].astype(float).sort_index()
    fg[fg.index < LOCKBOX_END].to_parquet(out / "fng.parquet")
    fx = pd.read_parquet(PROC / "usdinr.parquet")
    fx.to_parquet(out / "usdinr.parquet")
    (ROOT / "reports" / "lockbox_data_quality.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
