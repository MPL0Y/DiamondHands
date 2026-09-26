"""ML strategies: LightGBM direction classifier and meta-labeling, with leakage-safe training.

Every 6 months (aligned with the outer walk-forward windows) a model is trained on a rolling window of
data strictly before the refit date. Hyperparameters are chosen by purged + embargoed k-fold CV inside
that training window. The model then predicts the next 6 months. So every prediction is out-of-sample
by construction, and the outer walk-forward only picks thresholds.

Feature audit (all known at bar t's close):
- log returns over k bars, realised vol, RSI, distance to SMAs, volume z-score and taker-buy ratio: price/volume data up to close t
- funding: last print strictly before bar t's open (shift 1)
- fear & greed: the value stamped day D is used only from D+1 00:00 UTC
- hour / weekday of the NEXT bar (calendar, known in advance)
Label: sign(open[t+1+H] / open[t+1] - 1) — the return actually capturable after the next-open fill.
Purge: training rows whose label window [t+1, t+1+H] reaches past the refit date (or into a CV
validation fold, plus an embargo of H bars) are dropped.
"""
import hashlib, json
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import PROC
from src.data import funding_series, fng
from src.strategies import rsi, sma, REGISTRY, _latch, nz

CACHE = PROC / "ml_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def features(df):
    c, v = df.close, df.volume
    lr = np.log(c)
    X = pd.DataFrame(index=df.index)
    for k in (1, 3, 6, 12, 24, 72, 168):
        X[f"r{k}"] = lr.diff(k)
    for k in (24, 168):
        X[f"vol{k}"] = lr.diff().rolling(k).std()
    X["vol_ratio"] = X["vol24"] / X["vol168"]
    X["rsi14"] = rsi(c.to_numpy(), 14)
    X["rsi2"] = rsi(c.to_numpy(), 2)
    for k in (50, 200, 800):
        X[f"dma{k}"] = lr - np.log(pd.Series(sma(c.to_numpy(), k), index=df.index))
    X["vz"] = (v - v.rolling(168).mean()) / v.rolling(168).std()
    X["tbr"] = (df.tb_base / v.replace(0, np.nan)).rolling(24).mean()
    X["range"] = np.log(df.high / df.low).rolling(24).mean()
    f = funding_series()
    fr = f.reindex(df.index.union(f.index)).ffill().reindex(df.index).shift(1)
    X["fund"] = fr
    X["fund_z"] = (fr - fr.rolling(24 * 21, min_periods=50).mean()) / fr.rolling(24 * 21, min_periods=50).std()
    g = fng()
    gd = g.copy(); gd.index = gd.index + pd.Timedelta("1D")        # usable from D+1
    X["fng"] = gd.reindex(df.index.union(gd.index)).ffill().reindex(df.index)
    nxt = df.index + (df.index[1] - df.index[0])
    X["hour"] = nxt.hour
    X["wday"] = nxt.weekday
    return X.replace([np.inf, -np.inf], np.nan)


def purged_cv_score(X, y, t_idx, H, params, k=4):
    n = len(X)
    folds = np.array_split(np.arange(n), k)
    losses = []
    for f in folds:
        a, b = f[0], f[-1]
        tr = np.ones(n, bool)
        tr[max(0, a - H - 1): min(n, b + 2 * H + 1)] = False   # purge overlap + embargo H
        if tr.sum() < 200 or len(f) < 50:
            continue
        m = lgb.LGBMClassifier(**params, verbose=-1)
        m.fit(X[tr], y[tr])
        p = np.clip(m.predict_proba(X[f])[:, 1], 1e-4, 1 - 1e-4)
        yy = y[f]
        losses.append(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
    return np.mean(losses) if losses else np.inf


HP = [dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7),
      dict(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=400, subsample=0.8, subsample_freq=1, colsample_bytree=0.7),
      dict(n_estimators=100, learning_rate=0.05, num_leaves=4, min_child_samples=800, subsample=0.8, subsample_freq=1, colsample_bytree=0.5)]


def oos_proba(df, H, train_years=3, meta_primary=None, tag=""):
    """Walk-forward OOS probabilities. meta_primary: array (+1/0) — if given, train only on bars where the
    primary is long and predict P(primary trade profitable over H)."""
    key = hashlib.md5(json.dumps([str(df.index[0]), str(df.index[-1]), len(df), H, train_years, tag]).encode()).hexdigest()[:12]
    fp = CACHE / f"p_{key}.npy"
    if fp.exists():
        return np.load(fp)
    X = features(df)
    o = df.open.to_numpy()
    fwd = np.full(len(df), np.nan)
    fwd[: len(df) - H - 1] = o[1 + H:] / o[1:len(df) - H] - 1
    y = (fwd > 0).astype(float)
    p = np.full(len(df), np.nan)
    idx = df.index
    refits = pd.date_range(pd.Timestamp("2018-01-01", tz="UTC"), idx[-1], freq="6MS")
    for r0, r1 in zip(refits, list(refits[1:]) + [idx[-1] + pd.Timedelta("1s")]):
        i_end = idx.searchsorted(r0)
        i_start = idx.searchsorted(r0 - pd.DateOffset(years=train_years))
        last_ok = i_end - H - 2                         # label must be fully known before r0
        rows = np.arange(i_start, max(i_start, last_ok))
        if meta_primary is not None:
            rows = rows[meta_primary[rows] > 0]
        Xa = X.iloc[rows].to_numpy()
        ya = y[rows]
        ok = ~np.isnan(fwd[rows])
        Xa, ya = Xa[ok], ya[ok]
        if len(ya) < (500 if (idx[1] - idx[0]) >= pd.Timedelta('1D') else 1500):
            continue
        best = min(HP, key=lambda hp: purged_cv_score(Xa, ya, rows, H, hp))
        m = lgb.LGBMClassifier(**best, verbose=-1)
        m.fit(Xa, ya)
        j0, j1 = i_end, idx.searchsorted(r1)
        p[j0:j1] = m.predict_proba(X.iloc[j0:j1].to_numpy())[:, 1]
    np.save(fp, p)
    return p


def ml_dir(df, H=24, thr=0.55, side="long", hyst=0.02):
    p = oos_proba(df, int(H))
    p = np.nan_to_num(p, nan=0.5)
    el, xl = p > thr, p < thr - hyst
    es, xs = p < 1 - thr, p > 1 - thr + hyst
    if side == "long":
        es = np.zeros_like(el)
    return {"target": _latch(el, xl, es, xs)}


def ml_meta(df, H=24, thr=0.55, primary="ma", fast=20, slow=100):
    c = df.close.to_numpy()
    if primary == "ma":
        prim = nz(sma(c, fast) > sma(c, slow)).astype(float)
    else:  # supertrend
        prim = REGISTRY["supertrend"](df, period=fast, mult=slow / 10.0)["target"]
    p = oos_proba(df, int(H), meta_primary=prim, tag=f"meta{primary}{fast}{slow}")
    p = np.nan_to_num(p, nan=0.0)
    return {"target": prim * (p > thr)}


REGISTRY["ml_dir"] = ml_dir
REGISTRY["ml_meta"] = ml_meta
