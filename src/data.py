"""Load processed bars at a timeframe and assemble the per-bar arrays the engines need."""
from functools import lru_cache
import numpy as np
import pandas as pd

from src.config import PROC, DEV_END, LOCKBOX_DIR, ROOT

PERP_FROM = pd.Timestamp("2020-01-01", tz="UTC")


@lru_cache(maxsize=16)
def bars(tf: str, lockbox: bool = False) -> pd.DataFrame:
    """Execution bars: Binance spot before 2020-01-01, Binance USD-M perp afterwards
    (Delta India trades a perp; perp wicks are larger, which is conservative for liquidation).
    Adds: fund (sum of 8h funding stamped inside the bar), fx (USDINR), month_id, spot_close."""
    base = LOCKBOX_DIR / "processed" if lockbox else PROC
    if lockbox:
        with open(ROOT / "reports" / "lockbox_access.log", "a") as f:
            f.write(f"{pd.Timestamp.now(tz='UTC')} LOAD lockbox bars tf={tf}\n")
    spot = pd.read_parquet(base / f"spot_{tf}.parquet")
    perp = pd.read_parquet(base / f"perp_{tf}.parquet")
    ex = pd.concat([spot[spot.index < PERP_FROM], perp[perp.index >= PERP_FROM]])
    ex = ex[~ex.index.duplicated()]
    if not lockbox:
        ex = ex[ex.index < DEV_END]
    df = ex[["open", "high", "low", "close", "volume", "tb_base"]].copy()
    df["spot_close"] = spot["close"].reindex(df.index).ffill()
    fund = pd.read_parquet(base / "funding.parquet")["rate"]
    step = df.index[1] - df.index[0]
    pos = np.searchsorted(df.index.values, fund.index.values, side="right") - 1
    fsum = np.zeros(len(df))
    ok = (pos >= 0) & (fund.index.values < (df.index[-1] + step).to_datetime64())
    np.add.at(fsum, pos[ok], fund.values[ok])
    df["fund"] = fsum
    fx = pd.read_parquet(PROC / "usdinr.parquet")["usdinr"]
    if lockbox:
        fxl = pd.read_parquet(base / "usdinr.parquet")["usdinr"]
        fx = pd.concat([fx, fxl[fxl.index > fx.index.max()]])
    df["fx"] = fx.reindex(df.index.floor("1D")).ffill().bfill().to_numpy()
    df["month_id"] = (df.index.year * 12 + df.index.month - 1).astype(np.int64)
    return df


@lru_cache(maxsize=4)
def long_daily(tf="1d"):
    """Bitstamp-spliced long history (2011+) for training daily/4h models."""
    nm = "daily_long" if tf == "1d" else "h4_long"
    return pd.read_parquet(PROC / f"{nm}.parquet")


@lru_cache(maxsize=2)
def funding_series():
    return pd.read_parquet(PROC / "funding.parquet")["rate"]


@lru_cache(maxsize=2)
def fng():
    return pd.read_parquet(PROC / "fng.parquet")["value"]


@lru_cache(maxsize=2)
def one_minute(lockbox=False):
    base = LOCKBOX_DIR / "processed" if lockbox else PROC
    s = pd.read_parquet(base / "spot_1m.parquet")[["open", "high", "low", "close"]]
    p = pd.read_parquet(base / "perp_1m.parquet")[["open", "high", "low", "close"]]
    m = pd.concat([s[s.index < PERP_FROM], p[p.index >= PERP_FROM]])
    return m[~m.index.duplicated()]
