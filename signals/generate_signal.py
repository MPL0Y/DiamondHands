#!/usr/bin/env python
"""Live signal generator for the ranked finalist E0449. SIGNALS ONLY: never places orders, never asks for API keys.

E0449 = ensemble (inverse-vol weighted, quantised to 0.25, x2 leverage) of three causal sub-signals on BTC:
  C1 funding_level 1h : long while the last known 8h funding rate < `hi` (current window: 0.001)
  C2 vol_regime 4h    : long while SMA(60) > SMA(300) on 4h closes AND 20-bar realised vol percentile (over
                        2190 bars) > 0.5 ('trend_highvol' mode, current window)
  C3 ml_meta 4h       : C2's MA primary (SMA60 > SMA300) filtered by a LightGBM meta-model: P(primary trade
                        profitable over 18 bars) > 0.45; model refit every Jan-1/Jul-1 on the prior 3 years
                        with purged/embargoed CV (identical code path: src/ml.py)
Target leverage = 2 x round(sum_i w_i*C_i / 0.25) * 0.25,  w_i ~ 1/vol of component i's hourly returns (trailing
60 days, lagged). Parameters are the frozen walk-forward procedure's choices for the current window
(signals/current_selections.json). Re-run `python -m src.lockbox`-style selection only at the next 6-month window.

Data (public, no auth): Binance market-data mirror data-api.binance.vision (spot klines incl. taker-buy volume),
Binance monthly funding files (data.binance.vision) + OKX public funding for the current month, alternative.me
Fear & Greed, Delta Exchange India ticker for the mark price used for sizing.

Usage:  python signals/generate_signal.py [--capital 10000] [--paper]
"""
import argparse, io, json, math, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.strategies as S          # noqa: E402
import src.ml as ML                 # noqa: E402
import src.data as SD               # noqa: E402
from src.venues import DELTA_INDIA as V   # noqa: E402

INR_PER_USD_DELTA = 85.0
SEL = json.loads((ROOT / "signals" / "current_selections.json").read_text())
LEV = 2.0


def http(url, tries=4, raw=False):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.content if raw else r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def klines(interval, days):
    step = {"1h": 3600, "4h": 14400, "1d": 86400}[interval] * 1000
    end = int(time.time() * 1000)
    start = end - days * 86400 * 1000
    rows = []
    while start < end:
        js = http(f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval={interval}&startTime={start}&limit=1000")
        if not js:
            break
        rows += js
        start = js[-1][0] + step
    df = pd.DataFrame(rows).drop_duplicates(0)
    df.index = pd.to_datetime(df[0].astype("int64"), unit="ms", utc=True)
    out = pd.DataFrame({"open": df[1].astype(float), "high": df[2].astype(float), "low": df[3].astype(float),
                        "close": df[4].astype(float), "volume": df[5].astype(float), "tb_base": df[9].astype(float)})
    now = pd.Timestamp.now(tz="UTC")
    return out[out.index + pd.Timedelta(milliseconds=step) <= now]          # closed bars only


def funding_history(months=40):
    parts = []
    m = (pd.Timestamp.now(tz="UTC").replace(day=1) - pd.DateOffset(months=months)).tz_localize(None)
    while m < pd.Timestamp.now().replace(day=1):
        u = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{m:%Y-%m}.zip"
        try:
            z = zipfile.ZipFile(io.BytesIO(http(u, raw=True)))
            d = pd.read_csv(z.open(z.namelist()[0]))
            parts.append(pd.Series(d.last_funding_rate.values, pd.to_datetime(d.calc_time, unit="ms", utc=True).dt.floor("1h")))
        except Exception:
            pass
        m += pd.DateOffset(months=1)
    js = http("https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=100")["data"]
    okx = pd.Series({pd.Timestamp(int(r["fundingTime"]), unit="ms", tz="UTC"): float(r["realizedRate"] or r["fundingRate"]) for r in js})
    f = pd.concat(parts) if parts else pd.Series(dtype=float)
    f = pd.concat([f, okx[okx.index > (f.index.max() if len(f) else okx.index.min() - pd.Timedelta("1s"))]]).sort_index()
    return f[~f.index.duplicated()].rename("rate")


def fear_greed():
    d = pd.DataFrame(http("https://api.alternative.me/fng/?limit=0&format=json")["data"])
    d.index = pd.to_datetime(d["timestamp"].astype(int), unit="s", utc=True)
    return d["value"].astype(float).sort_index()


def ml_meta_live(df4, H, fast, slow, thr):
    """Refit the meta-model exactly as src/ml.oos_proba does for the current 6-month window and score the last bar."""
    import lightgbm as lgb
    c = df4.close.to_numpy()
    prim = np.nan_to_num(S.sma(c, fast) > S.sma(c, slow)).astype(float)
    X = ML.features(df4)
    o = df4.open.to_numpy()
    fwd = np.full(len(df4), np.nan)
    fwd[: len(df4) - H - 1] = o[1 + H:] / o[1:len(df4) - H] - 1
    y = (fwd > 0).astype(float)
    now = df4.index[-1]
    refit = pd.Timestamp(year=now.year, month=1 if now.month < 7 else 7, day=1, tz="UTC")
    idx = df4.index
    i_end = idx.searchsorted(refit)
    i_start = idx.searchsorted(refit - pd.DateOffset(years=3))
    rows = np.arange(i_start, max(i_start, i_end - H - 2))
    rows = rows[prim[rows] > 0]
    ok = ~np.isnan(fwd[rows])
    Xa, ya = X.iloc[rows].to_numpy()[ok], y[rows][ok]
    best = min(ML.HP, key=lambda hp: ML.purged_cv_score(Xa, ya, rows, H, hp))
    m = lgb.LGBMClassifier(**best, verbose=-1).fit(Xa, ya)
    p = m.predict_proba(X.iloc[[-1]].to_numpy())[:, 1][0]
    return prim * 0 + np.r_[np.zeros(len(prim) - 1), prim[-1] * (p > thr)], float(p), bool(prim[-1])


def to_1h(series4, idx1):
    """Map a 4h target (decided at the 4h close) onto 1h bars without look-ahead (same rule as src/ensemble.py)."""
    s = pd.Series(series4.values, series4.index + pd.Timedelta("4h") - pd.Timedelta("1h"))
    return s.reindex(idx1, method="ffill").fillna(0.0).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=10_000.0, help="account equity in ₹ (Delta India wallet)")
    ap.add_argument("--paper", action="store_true", help="append signal + hypothetical fill to signals/paper_log.csv")
    a = ap.parse_args()

    fund = funding_history()
    fg = fear_greed()
    SD.funding_series = lambda: fund
    S.funding_series = lambda: fund
    ML.funding_series = lambda: fund
    ML.fng = lambda: fg

    df1 = klines("1h", 90)
    df4 = klines("4h", 3 * 365 + 400)
    p1, p2, p3 = SEL["E0301"]["params"], SEL["E0292"]["params"], SEL["E0420"]["params"]
    c1 = S.REGISTRY["funding_level"](df1, **p1)["target"]
    c2_4 = pd.Series(S.REGISTRY["vol_regime"](df4, **p2)["target"], df4.index)
    c3_last, prob, prim = ml_meta_live(df4, int(p3["H"]), int(p3["fast"]), int(p3["slow"]), float(p3["thr"]))
    c3_4 = pd.Series(np.nan, df4.index); c3_4.iloc[-1] = c3_last[-1]
    # history of C3 for the vol weights: use the primary (conservative proxy) for past bars
    c3_4 = c3_4.fillna(pd.Series(np.nan_to_num(S.sma(df4.close.to_numpy(), p3["fast"]) > S.sma(df4.close.to_numpy(), p3["slow"])).astype(float), df4.index))
    c2 = to_1h(c2_4, df1.index)
    c3 = to_1h(c3_4, df1.index)
    T = np.vstack([c1, c2, c3])
    r1 = df1.close.pct_change().fillna(0.0).to_numpy()
    R = T[:, :-1] * r1[1:]                                    # component 1x hourly returns (cost-free proxy)
    n = 24 * 60
    vol = np.array([np.std(row[-n:]) for row in R])
    w = np.where(vol > 0, 1 / vol, 0.0)
    w = w / w.sum() if w.sum() > 0 else np.full(3, 1 / 3)
    if SEL.get("E0449_weighting", "invvol") == "equal":
        w = np.full(3, 1 / 3)
    comb = float((w * T[:, -1]).sum())
    tgt = round(comb / 0.25) * 0.25 * LEV
    prev_comb = float((w * T[:, -2]).sum())
    prev = round(prev_comb / 0.25) * 0.25 * LEV

    tick = http("https://api.india.delta.exchange/v2/tickers/BTCUSD")["result"]
    px = float(tick["mark_price"])
    cap_usd = a.capital / INR_PER_USD_DELTA
    lots = math.floor(abs(tgt) * cap_usd / px / V.lot_btc + 1e-9)
    qty = lots * V.lot_btc * (1 if tgt >= 0 else -1)
    if tgt > 0 and prev <= 0:
        action = "BUY"
    elif tgt == 0 and prev > 0:
        action = "CLOSE"
    elif tgt > prev:
        action = "BUY (increase)"
    elif tgt < prev:
        action = "SELL (reduce)"
    else:
        action = "HOLD"
    liq = (qty * px - cap_usd) / (qty - V.mmr * abs(qty)) if qty else None
    last_bar = df1.index[-1]
    out = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "strategy": "E0449 ensemble (funding-level 1h + vol-regime 4h + ML-meta 4h), x2, Delta India BTCUSD perp",
        "last_closed_1h_bar_utc": str(last_bar), "btc_close": float(df1.close.iloc[-1]),
        "components_now": {"funding_level": float(c1[-1]), "vol_regime": float(c2[-1]), "ml_meta": float(c3[-1]),
                           "ml_prob": round(prob, 3), "ma_primary_long": prim, "weights": [round(float(x), 3) for x in w]},
        "action": action, "target_leverage": tgt, "previous_target_leverage": prev,
        "position_btc": qty, "contracts_0.001BTC": lots, "position_notional_inr": round(abs(qty) * px * INR_PER_USD_DELTA),
        "capital_inr": a.capital, "mark_price_usd": px,
        "stop_loss": "none in the tested rules (exit when the target changes); liquidation below",
        "take_profit": "none in the tested rules",
        "approx_liquidation_price_usd": round(liq, 1) if liq and liq > 0 else "n/a",
        "execute": "market order now (= open of the bar after the last closed bar)" if action != "HOLD" else "no order",
        "next_check_utc": str(last_bar + pd.Timedelta("2h") + pd.Timedelta(seconds=30)),
        "note": ("target below one 0.001 BTC contract at this capital: stay flat" if tgt != 0 and lots == 0 else ""),
        "warning": "Lockbox (2025-09-26→2026-09-26) result: -24.5%. Paper-trade only; see reports/final_report.md.",
    }
    for k, v in out.items():
        print(f"{k:>30}: {v}")
    if a.paper:
        log = ROOT / "signals" / "paper_log.csv"
        row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in out.items()}
        trade = action not in ("HOLD",)
        side = np.sign(tgt - prev)
        row["hypothetical_fill_price_usd"] = px * (1 + V.slippage * side) if trade else ""
        row["hypothetical_fee_usd"] = abs(qty) * px * V.taker if trade else 0.0
        prev_rows = pd.read_csv(log) if log.exists() else pd.DataFrame()
        if len(prev_rows):
            last = prev_rows.iloc[-1]
            row["mtm_since_last_usd"] = float(last.get("position_btc", 0.0)) * (px - float(last.get("mark_price_usd", px)))
        pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
        print(f"paper log appended -> {log}")


if __name__ == "__main__":
    main()
