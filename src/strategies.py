"""Strategy signal library. Every function is causal: output[t] uses data up to and including bar t's
CLOSE only, and is executed by the engine at bar t+1's open.

Signature: fn(df, **params) -> dict(target=np.ndarray[-1..1 base exposure], stop=?, tp=?)
Leverage and sizing overlays are applied by src.experiment.
"""
import numpy as np
import pandas as pd
from numba import njit

from src.data import funding_series, fng


# ---------------------------------------------------------------- indicators
def sma(x, n):
    return pd.Series(x).rolling(int(n), min_periods=int(n)).mean().to_numpy()


def ema(x, n):
    return pd.Series(x).ewm(span=int(n), adjust=False, min_periods=int(n)).mean().to_numpy()


def rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.clip(d, 0, None)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = pd.Series(np.clip(-d, 0, None)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).to_numpy()


def atr(df, n):
    pc = df.close.shift(1)
    tr = np.maximum(df.high - df.low, np.maximum((df.high - pc).abs(), (df.low - pc).abs()))
    return tr.rolling(int(n), min_periods=int(n)).mean().to_numpy()


def rvol(c, n, bars_per_year):
    r = np.log(pd.Series(c)).diff()
    return (r.rolling(int(n), min_periods=int(n)).std() * np.sqrt(bars_per_year)).to_numpy()


def bpy(df):
    step = (df.index[1] - df.index[0]).total_seconds()
    return 365.25 * 86400 / step


def nz(x):
    return np.nan_to_num(np.asarray(x, float))


@njit(cache=True)
def _latch(enter_long, exit_long, enter_short, exit_short):
    n = len(enter_long)
    out = np.zeros(n)
    s = 0.0
    for t in range(n):
        if s == 0.0:
            if enter_long[t]:
                s = 1.0
            elif enter_short[t]:
                s = -1.0
        elif s > 0:
            if enter_short[t]:
                s = -1.0
            elif exit_long[t]:
                s = 0.0
        else:
            if enter_long[t]:
                s = 1.0
            elif exit_short[t]:
                s = 0.0
        out[t] = s
    return out


@njit(cache=True)
def _hold_n(trigger, direction, hold):
    """Enter on trigger, hold for `hold` bars (time stop), re-trigger extends."""
    n = len(trigger)
    out = np.zeros(n)
    left = 0
    d = 0.0
    for t in range(n):
        if trigger[t]:
            left = hold
            d = direction[t]
        if left > 0:
            out[t] = d
            left -= 1
    return out


def _sides(long_sig, short_sig, side):
    if side == "long":
        return long_sig
    if side == "short":
        return short_sig
    return long_sig + short_sig


# ---------------------------------------------------------------- benchmarks
def hold(df, **p):
    return {"target": np.ones(len(df))}


# ---------------------------------------------------------------- trend
def ma_cross(df, fast=20, slow=100, kind="sma", side="long"):
    c = df.close.to_numpy()
    f = (ema if kind == "ema" else sma)(c, fast)
    s = (ema if kind == "ema" else sma)(c, slow)
    up = nz(f > s).astype(float)
    dn = -nz(f < s).astype(float)
    return {"target": _sides(up, dn, side) * ~np.isnan(s)}


def donchian(df, entry=20, exit=10, side="long", atr_stop=0.0):
    hh = df.high.rolling(int(entry)).max().shift(1).to_numpy()
    ll = df.low.rolling(int(entry)).min().shift(1).to_numpy()
    xl = df.low.rolling(int(exit)).min().shift(1).to_numpy()
    xh = df.high.rolling(int(exit)).max().shift(1).to_numpy()
    c = df.close.to_numpy()
    el, es = nz(c > hh).astype(bool), nz(c < ll).astype(bool)
    if side == "long":
        es = np.zeros_like(el)
    if side == "short":
        el = np.zeros_like(es)
    t = _latch(el, nz(c < xl).astype(bool), es, nz(c > xh).astype(bool))
    out = {"target": t}
    if atr_stop > 0:
        out["stop"] = nz(atr_stop * atr(df, 20) / df.close.to_numpy())
    return out


def tsmom(df, lookbacks=(24, 72, 168, 336), side="long", thresh=0.0):
    c = df.close
    sig = np.zeros(len(df))
    for L in lookbacks:
        sig += np.sign(nz(np.log(c / c.shift(int(L)))))
    sig /= len(lookbacks)
    sig[np.abs(sig) < thresh] = 0.0
    if side == "long":
        sig = np.clip(sig, 0, None)
    return {"target": sig}


# ---------------------------------------------------------------- mean reversion
def rsi_mr(df, n=14, lo=30, hi=70, exit_mid=50, side="long", trend_filter=0):
    r = rsi(df.close.to_numpy(), int(n))
    el, xl = r < lo, r > exit_mid
    es, xs = r > hi, r < exit_mid
    if trend_filter:
        m = sma(df.close.to_numpy(), trend_filter)
        el = el & nz(df.close.to_numpy() > m).astype(bool)
        es = es & nz(df.close.to_numpy() < m).astype(bool)
    if side == "long":
        es = np.zeros_like(el)
    if side == "short":
        el = np.zeros_like(es)
    return {"target": _latch(el, xl, es, xs)}


def bb_mr(df, n=20, k=2.0, side="long", stop=0.0):
    c = df.close.to_numpy()
    m = sma(c, n)
    sd = pd.Series(c).rolling(int(n)).std().to_numpy()
    el, es = nz(c < m - k * sd).astype(bool), nz(c > m + k * sd).astype(bool)
    xl, xs = nz(c >= m).astype(bool), nz(c <= m).astype(bool)
    if side == "long":
        es = np.zeros_like(el)
    if side == "short":
        el = np.zeros_like(es)
    out = {"target": _latch(el, xl, es, xs)}
    if stop > 0:
        out["stop"] = np.full(len(c), stop)
    return out


def zscore_mr(df, n=48, z_in=2.0, z_out=0.0, side="both"):
    c = np.log(df.close.to_numpy())
    m = sma(c, n)
    sd = pd.Series(c).rolling(int(n)).std().to_numpy()
    z = nz((c - m) / sd)
    el, es = z < -z_in, z > z_in
    xl, xs = z > -z_out, z < z_out
    if side == "long":
        es = np.zeros_like(el)
    if side == "short":
        el = np.zeros_like(es)
    return {"target": _latch(el, xl, es, xs)}


def intraday_rev(df, lookback=4, thresh=0.03, hold=6, side="long"):
    """After a sharp drop over `lookback` bars, go long for `hold` bars (and the mirror for shorts)."""
    r = nz(np.log(df.close / df.close.shift(int(lookback))))
    trig_l, trig_s = r < -thresh, r > thresh
    if side == "long":
        trig_s[:] = False
    if side == "short":
        trig_l[:] = False
    trig = trig_l | trig_s
    d = np.where(trig_l, 1.0, -1.0)
    return {"target": _hold_n(trig, d, int(hold))}


# ---------------------------------------------------------------- volatility
def vol_breakout(df, n=20, k=1.0, hold=24, side="long"):
    """Close moves more than k*ATR from previous close -> ride the move for `hold` bars."""
    a = atr(df, n)
    d = df.close.diff().to_numpy()
    up, dn = nz(d > k * a).astype(bool), nz(d < -k * a).astype(bool)
    if side == "long":
        dn[:] = False
    trig = up | dn
    return {"target": _hold_n(trig, np.where(up, 1.0, -1.0), int(hold))}


def vol_regime(df, fast=20, slow=100, vol_n=30, vol_q=0.7, q_window=720, mode="trend_lowvol"):
    """Trend signal gated by volatility regime (percentile of realised vol over q_window bars)."""
    c = df.close.to_numpy()
    tr = nz(sma(c, fast) > sma(c, slow)).astype(float)
    v = pd.Series(rvol(c, vol_n, 1))
    pct = v.rolling(int(q_window), min_periods=int(q_window) // 2).rank(pct=True).to_numpy()
    hi = nz(pct > vol_q).astype(bool)
    if mode == "trend_lowvol":
        t = tr * ~hi
    elif mode == "trend_highvol":
        t = tr * hi
    else:  # long only in low vol regardless of trend
        t = (~hi).astype(float)
    return {"target": t}


# ---------------------------------------------------------------- funding
def _funding_on_bars(df, lag_prints=1):
    """Latest KNOWN funding rate at each bar close. A print stamped at T is known at T; we additionally
    lag by `lag_prints` bars for safety (the print for 08:00 is only final at 08:00)."""
    f = funding_series()
    s = f.reindex(df.index.union(f.index)).ffill().reindex(df.index)
    return nz(s.shift(lag_prints).to_numpy())


def funding_contra(df, window=21, z_in=2.0, side="both", hold=24):
    """Extreme positive funding (crowded longs) -> short; extreme negative -> long. Rolling z-score of funding."""
    fr = pd.Series(_funding_on_bars(df))
    m = fr.rolling(int(window) * 3, min_periods=10).mean()
    sd = fr.rolling(int(window) * 3, min_periods=10).std()
    z = nz((fr - m) / sd)
    long_t, short_t = z < -z_in, z > z_in
    if side == "long":
        short_t[:] = False
    if side == "short":
        long_t[:] = False
    trig = long_t | short_t
    return {"target": _hold_n(trig, np.where(long_t, 1.0, -1.0), int(hold))}


def funding_level(df, lo=0.0, hi=0.0003, base="trend", fast=20, slow=100):
    """Trend long, but only while funding is below `hi` (avoid paying crowded carry); flat otherwise."""
    fr = _funding_on_bars(df)
    c = df.close.to_numpy()
    tr = nz(sma(c, fast) > sma(c, slow)).astype(float) if base == "trend" else np.ones(len(df))
    return {"target": tr * (fr < hi) * (fr > lo - 1)}


# ---------------------------------------------------------------- calendar
def calendar(df, hours=(), weekdays=(0, 1, 2, 3, 4, 5, 6), side="long", tom=0, trend=0):
    """Hold during selected UTC hours/weekdays. The target for bar t+1 is known at t's close.
    tom>0: turn-of-month — only the last `tom` and first `tom` calendar days of each month.
    trend>0: only when close > SMA(trend) (known at close)."""
    nxt = df.index + (df.index[1] - df.index[0])
    ok = np.isin(nxt.weekday, list(weekdays))
    if hours:
        ok &= np.isin(nxt.hour, list(hours))
    if tom:
        dim = nxt.days_in_month
        ok &= (nxt.day <= tom) | (nxt.day > dim - tom)
    if trend:
        ok &= nz(df.close.to_numpy() > sma(df.close.to_numpy(), trend)).astype(bool)
    t = ok.astype(float)
    return {"target": t if side == "long" else -t}


def weekend_flat(df, fast=20, slow=100):
    nxt = df.index + (df.index[1] - df.index[0])
    c = df.close.to_numpy()
    tr = nz(sma(c, fast) > sma(c, slow)).astype(float)
    return {"target": tr * (nxt.weekday < 5)}


# ---------------------------------------------------------------- registry
REGISTRY = {
    "hold": hold, "ma_cross": ma_cross, "donchian": donchian, "tsmom": tsmom,
    "rsi_mr": rsi_mr, "bb_mr": bb_mr, "zscore_mr": zscore_mr, "intraday_rev": intraday_rev,
    "vol_breakout": vol_breakout, "vol_regime": vol_regime,
    "funding_contra": funding_contra, "funding_level": funding_level,
    "calendar": calendar, "weekend_flat": weekend_flat,
}


# ---------------------------------------------------------------- user-supplied: TradingView "Supertrend" (Pine v4)
@njit(cache=True)
def _supertrend(h, l, c, atr_, mult):
    n = len(c)
    trend = np.ones(n)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    for t in range(n):
        src = (h[t] + l[t]) / 2.0
        u = src - mult * atr_[t]
        d = src + mult * atr_[t]
        u1 = up[t - 1] if t > 0 and not np.isnan(up[t - 1]) else u
        d1 = dn[t - 1] if t > 0 and not np.isnan(dn[t - 1]) else d
        if t > 0 and c[t - 1] > u1:
            u = max(u, u1)
        if t > 0 and c[t - 1] < d1:
            d = min(d, d1)
        up[t] = u
        dn[t] = d
        tr_prev = trend[t - 1] if t > 0 else 1.0
        tr = tr_prev
        if tr_prev == -1.0 and c[t] > d1:
            tr = 1.0
        elif tr_prev == 1.0 and c[t] < u1:
            tr = -1.0
        trend[t] = tr
    return trend, up, dn


def pine_atr(df, n, rma=True):
    """Pine atr(): RMA of true range (alpha=1/n, seeded with SMA of first n); changeATR=false -> SMA."""
    pc = df.close.shift(1)
    tr = np.maximum(df.high - df.low, np.maximum((df.high - pc).abs(), (df.low - pc).abs()))
    tr.iloc[0] = df.high.iloc[0] - df.low.iloc[0]
    if not rma:
        return tr.rolling(int(n)).mean().to_numpy()
    x = tr.to_numpy()
    out = np.full(len(x), np.nan)
    n = int(n)
    if len(x) >= n:
        out[n - 1] = x[:n].mean()
        for i in range(n, len(x)):
            out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def supertrend(df, period=10, mult=3.0, change_atr=True, side="long"):
    a = pine_atr(df, period, change_atr)
    valid = ~np.isnan(a)
    trend, up, dn = _supertrend(df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy(), np.nan_to_num(a), float(mult))
    trend = np.where(valid, trend, 0.0)
    t = np.where(trend > 0, 1.0, 0.0 if side == "long" else -1.0)
    return {"target": t * valid}


REGISTRY["supertrend"] = supertrend
