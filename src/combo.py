"""Indicator-combination search.

A library of ~70 causal long-side CONDITIONS (booleans known at a bar's close) is built on a base timeframe.
Daily-timeframe conditions are mapped onto the base bars without look-ahead (a daily value becomes usable on the
base bar that closes at the daily close). A COMBINATION is an AND of 1-3 conditions: be long while all hold,
flat otherwise (enter at the next bar open when the AND turns true, exit at the next open when it turns false).

Every combination is simulated once over the whole development window: 1x, lot-free, full Delta India costs
(taker + slippage + funding). Its daily returns go into a matrix R [days x combos]. The walk-forward then
picks, for each 6-month OOS window, the top-k combinations by a training metric computed on the previous
`train_years` only. It averages their positions (an ensemble of the k best). The stitched OOS positions
are re-simulated with the real engine (lots, liquidation, withdraw/compound modes).
"""
import itertools
import numpy as np
import pandas as pd
from numba import njit

from src.data import bars, funding_series, fng
from src.strategies import sma, ema, rsi, _supertrend, pine_atr, nz

# ------------------------------------------------------------------ indicator helpers


def _rolling_max(x, n):
    return pd.Series(x).rolling(int(n), min_periods=int(n)).max().to_numpy()


def _rolling_min(x, n):
    return pd.Series(x).rolling(int(n), min_periods=int(n)).min().to_numpy()


def wma(x, n):
    w = np.arange(1, n + 1, dtype=float)
    return pd.Series(x).rolling(n, min_periods=n).apply(lambda a: np.dot(a, w) / w.sum(), raw=True).to_numpy()


def hma(x, n):
    return wma(2 * wma(x, n // 2) - wma(x, n), int(np.sqrt(n)))


def macd(c, f=12, s=26, sig=9):
    m = ema(c, f) - ema(c, s)
    return m, ema(np.nan_to_num(m), sig)


def adx(df, n=14):
    h, l, c = df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy()
    up = np.diff(h, prepend=h[0]); dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = pine_atr(df, n)
    rma = lambda x: pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().to_numpy()
    pdi = 100 * rma(pdm) / a; mdi = 100 * rma(mdm) / a
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return rma(np.nan_to_num(dx)), pdi, mdi


def stoch(df, n=14, d=3):
    hh = _rolling_max(df.high.to_numpy(), n); ll = _rolling_min(df.low.to_numpy(), n)
    k = 100 * (df.close.to_numpy() - ll) / (hh - ll)
    return k, sma(np.nan_to_num(k, nan=50), d)


def cci(df, n=20):
    tp = ((df.high + df.low + df.close) / 3).to_numpy()
    m = sma(tp, n)
    md = pd.Series(tp).rolling(n).apply(lambda a: np.mean(np.abs(a - a.mean())), raw=True).to_numpy()
    return (tp - m) / (0.015 * md)


def mfi(df, n=14):
    tp = ((df.high + df.low + df.close) / 3).to_numpy()
    mf = tp * df.volume.to_numpy()
    d = np.diff(tp, prepend=tp[0])
    pos = pd.Series(np.where(d > 0, mf, 0.0)).rolling(n).sum(); neg = pd.Series(np.where(d < 0, mf, 0.0)).rolling(n).sum()
    return (100 - 100 / (1 + pos / neg.replace(0, np.nan))).to_numpy()


@njit(cache=True)
def _psar(h, l, af0=0.02, afmax=0.2):
    n = len(h)
    up = np.ones(n)
    sar = l[0]; ep = h[0]; af = af0; bull = True
    for t in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, l[t - 1], l[t - 2] if t > 1 else l[t - 1])
            if l[t] < sar:
                bull = False; sar = ep; ep = l[t]; af = af0
            elif h[t] > ep:
                ep = h[t]; af = min(af + af0, afmax)
        else:
            sar = max(sar, h[t - 1], h[t - 2] if t > 1 else h[t - 1])
            if h[t] > sar:
                bull = True; sar = ep; ep = h[t]; af = af0
            elif l[t] < ep:
                ep = l[t]; af = min(af + af0, afmax)
        up[t] = 1.0 if bull else 0.0
    return up


def kama(c, n=10, f=2, s=30):
    ch = np.abs(c - np.roll(c, n)); vol = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().to_numpy()
    er = np.nan_to_num(ch / vol); sc = (er * (2 / (f + 1) - 2 / (s + 1)) + 2 / (s + 1)) ** 2
    out = np.full(len(c), np.nan); out[n] = c[n]
    for i in range(n + 1, len(c)):
        out[i] = out[i - 1] + sc[i] * (c[i] - out[i - 1])
    return out


def pct_rank(x, n):
    return pd.Series(x).rolling(n, min_periods=n // 2).rank(pct=True).to_numpy()


# ------------------------------------------------------------------ condition library

def conditions_for(df, prefix=""):
    """Long-side conditions on one timeframe. Returns dict name -> bool ndarray (NaN-safe: False when unknown)."""
    c = df.close.to_numpy(); o = df.open.to_numpy(); v = df.volume.to_numpy()
    C = {}
    B = lambda x: np.nan_to_num(np.asarray(x, float), nan=0.0).astype(bool)
    for n in (20, 50, 100, 200):
        C[f"close>sma{n}"] = B(c > sma(c, n))
    for f, s in ((12, 26), (20, 50), (50, 200)):
        C[f"ema{f}>ema{s}"] = B(ema(c, f) > ema(c, s))
    m, sg = macd(c)
    C["macd>signal"] = B(m > sg); C["macd>0"] = B(m > 0)
    for per, mu in ((10, 3.0), (10, 2.0), (20, 3.0)):
        a = pine_atr(df, per)
        tr, _, _ = _supertrend(df.high.to_numpy(), df.low.to_numpy(), c, np.nan_to_num(a), mu)
        C[f"supertrend{per}x{mu:g}"] = B((tr > 0) & ~np.isnan(a))
    for n in (20, 55):
        hh = _rolling_max(df.high.to_numpy(), n); ll = _rolling_min(df.low.to_numpy(), n)
        C[f"close>donchmid{n}"] = B(c > (hh + ll) / 2)
    ax, pdi, mdi = adx(df)
    C["di+>di-"] = B(pdi > mdi); C["adx>20&di+"] = B((ax > 20) & (pdi > mdi)); C["adx>25"] = B(ax > 25)
    ten = (_rolling_max(df.high.to_numpy(), 9) + _rolling_min(df.low.to_numpy(), 9)) / 2
    kij = (_rolling_max(df.high.to_numpy(), 26) + _rolling_min(df.low.to_numpy(), 26)) / 2
    sa = pd.Series((ten + kij) / 2).shift(26).to_numpy()
    sb = pd.Series((_rolling_max(df.high.to_numpy(), 52) + _rolling_min(df.low.to_numpy(), 52)) / 2).shift(26).to_numpy()
    C["close>cloud"] = B(c > np.fmax(sa, sb)); C["tenkan>kijun"] = B(ten > kij)
    C["psar_up"] = _psar(df.high.to_numpy(), df.low.to_numpy()).astype(bool)
    h = hma(c, 21); C["hma21_up"] = B(h > np.roll(h, 1))
    k = kama(c); C["kama_up"] = B(k > np.roll(k, 1))
    for n in (42, 180):
        C[f"roc{n}>0"] = B(c > np.roll(c, n))
    ha_c = (df.open + df.high + df.low + df.close).to_numpy() / 4
    ha_o = pd.Series((o + c) / 2).shift(1).ewm(alpha=0.5, adjust=False).mean().to_numpy()
    C["heikin_green"] = B(ha_c > ha_o)
    r14 = rsi(c, 14); r2 = rsi(c, 2)
    C["rsi14>50"] = B(r14 > 50); C["rsi14<70"] = B(r14 < 70); C["rsi14<30"] = B(r14 < 30); C["rsi2<10"] = B(r2 < 10)
    C["rsi14_40_70"] = B((r14 > 40) & (r14 < 70))
    kk, dd = stoch(df); C["stoch_k>d"] = B(kk > dd); C["stoch<20"] = B(kk < 20)
    cc = cci(df); C["cci>0"] = B(cc > 0); C["cci<-100"] = B(cc < -100)
    mf = mfi(df); C["mfi>50"] = B(mf > 50); C["mfi<20"] = B(mf < 20)
    obv = np.cumsum(np.sign(np.diff(c, prepend=c[0])) * v); C["obv>sma20"] = B(obv > sma(obv, 20))
    lr = np.log(c); rv = pd.Series(lr).diff().rolling(20).std().to_numpy()
    rp = pct_rank(rv, 500); C["vol_pct>0.5"] = B(rp > 0.5); C["vol_pct<0.5"] = B(rp < 0.5); C["vol_pct<0.8"] = B(rp < 0.8)
    m20 = sma(c, 20); sd20 = pd.Series(c).rolling(20).std().to_numpy()
    bw = (4 * sd20) / m20; C["bb_squeeze"] = B(pct_rank(bw, 500) < 0.2)
    C["close>bb_up"] = B(c > m20 + 2 * sd20); C["close<bb_lo"] = B(c < m20 - 2 * sd20)
    at = pine_atr(df, 20); C["close>keltner_up"] = B(c > ema(c, 20) + 2 * at)
    vz = (v - sma(v, 50)) / pd.Series(v).rolling(50).std().to_numpy(); C["volz>0"] = B(vz > 0)
    tb = pd.Series(df.tb_base.to_numpy() / np.where(v > 0, v, np.nan)).rolling(6).mean().to_numpy()
    C["takerbuy>0.5"] = B(tb > 0.5)
    return {prefix + kk2: vv for kk2, vv in C.items()}


def map_daily(df_base, cond_daily, daily_index):
    """Daily condition decided at the daily close -> usable from the base bar that closes at that instant."""
    step = df_base.index[1] - df_base.index[0]
    out = {}
    for k, arr in cond_daily.items():
        s = pd.Series(arr.astype(float), daily_index + pd.Timedelta("1D") - step)
        out[k] = s.reindex(df_base.index, method="ffill").fillna(0.0).to_numpy().astype(bool)
    return out


def invert(df):
    """Mirror a bar series (price -> 1/price, high<->low) so long-side indicator rules become short-side rules."""
    x = df.copy()
    x["open"], x["close"] = 1 / df.open, 1 / df.close
    x["high"], x["low"] = 1 / df.low, 1 / df.high
    x["tb_base"] = df.volume - df.tb_base
    return x


def library(tf="4h", side="long", df=None, d=None):
    """df/d may be injected (live signal generator); otherwise the research bars are used."""
    df = bars(tf) if df is None else df
    src = df if side == "long" else invert(df)
    L = conditions_for(src, "")
    if tf != "1d":
        d = bars("1d") if d is None else d
        Ld = conditions_for(d if side == "long" else invert(d), "D:")
        keep = [k for k in Ld if any(s in k for s in ("sma", "ema", "supertrend", "macd", "cloud", "roc", "donch", "psar", "rsi14>50", "adx>20"))]
        L.update(map_daily(df, {k: Ld[k] for k in keep}, d.index))
    # funding / sentiment / calendar (known at bar close; funding lagged one bar, F&G usable from D+1)
    f = funding_series()
    fr = f.reindex(df.index.union(f.index)).ffill().reindex(df.index).shift(1).to_numpy()
    fz = pd.Series(fr); fz = ((fz - fz.rolling(500, min_periods=50).mean()) / fz.rolling(500, min_periods=50).std()).to_numpy()
    g = fng(); gd = g.copy(); gd.index = gd.index + pd.Timedelta("1D")
    gv = gd.reindex(df.index.union(gd.index)).ffill().reindex(df.index).to_numpy()
    if side == "long":
        L["fund<0.01%"] = np.nan_to_num(fr < 0.0001).astype(bool)
        L["fund<0.03%"] = np.nan_to_num(fr < 0.0003).astype(bool)
        L["fund_z<0"] = np.nan_to_num(fz < 0).astype(bool)
        L["fng<25"] = np.nan_to_num(gv < 25).astype(bool); L["fng<50"] = np.nan_to_num(gv < 50).astype(bool)
        L["fng>50"] = np.nan_to_num(gv > 50).astype(bool); L["fng<75"] = np.nan_to_num(gv < 75, nan=1).astype(bool)
    else:   # crowded longs / greed favour shorts; shorts RECEIVE positive funding
        L["fund>0.01%"] = np.nan_to_num(fr > 0.0001).astype(bool)
        L["fund>-0.01%"] = np.nan_to_num(fr > -0.0001).astype(bool)
        L["fund_z>0"] = np.nan_to_num(fz > 0).astype(bool)
        L["fng>75"] = np.nan_to_num(gv > 75).astype(bool); L["fng>50"] = np.nan_to_num(gv > 50).astype(bool)
        L["fng<50"] = np.nan_to_num(gv < 50).astype(bool); L["fng>25"] = np.nan_to_num(gv > 25, nan=1).astype(bool)
    nxt = df.index + (df.index[1] - df.index[0])
    L["turn_of_month"] = np.asarray((nxt.day <= 4) | (nxt.day > nxt.days_in_month - 4))
    L["weekday"] = np.asarray(nxt.weekday < 5)
    return df, L


CATEGORY = {  # used to build curated triples: one trend x one timing x one filter
    "trend": ["sma", "ema", "macd", "supertrend", "donch", "di+", "adx>20", "cloud", "tenkan", "psar", "hma", "kama", "roc", "heikin"],
    "timing": ["rsi", "stoch", "cci", "mfi", "obv", "bb", "keltner", "volz", "takerbuy"],
    "filter": ["vol_pct", "squeeze", "fund", "fng", "turn_of_month", "weekday", "adx>25"],
}


def cat_of(name):
    n = name.replace("D:", "")
    for cat in ("filter", "timing", "trend"):
        if any(n.startswith(p) or p in n for p in CATEGORY[cat]):
            return cat
    return "trend"


def enumerate_combos(L, max_triples=6000, seed=1):
    names = list(L)
    combos = [(a,) for a in names]
    combos += list(itertools.combinations(names, 2))
    tr = [n for n in names if cat_of(n) == "trend"]; ti = [n for n in names if cat_of(n) == "timing"]
    fi = [n for n in names if cat_of(n) == "filter"]
    trip = list(itertools.product(tr, ti, fi)) + list(itertools.combinations(tr, 2))
    trip = [t for t in trip if len(t) == 3] + [(a, b, f_) for (a, b) in itertools.combinations(tr, 2) for f_ in fi]
    rng = np.random.default_rng(seed)
    if len(trip) > max_triples:
        trip = [trip[i] for i in rng.choice(len(trip), max_triples, replace=False)]
    return combos + trip
