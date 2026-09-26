#!/usr/bin/env python
"""Live signal generator — SIGNALS ONLY. Never places orders, never asks for API keys.

Pulls the latest public data, recomputes the frozen finalist strategy (signals/finalist.json) on the
LAST CLOSED bar, and prints the action, size in ₹ and BTC, leverage, stop-loss, take-profit and next
check time.
  python signals/generate_signal.py [--capital 10000] [--paper]

Data sources (public, no auth):
- Delta Exchange India  /v2/history/candles  BTCUSD perp candles (the venue the strategy trades)
- OKX /api/v5/public/funding-rate-history    BTC-USDT-SWAP 8h funding (proxy for the Binance funding
  the strategy was researched on; Binance fapi is geo-blocked in some regions)
- Delta /v2/tickers/BTCUSD                   mark price (sizing reference)
--paper appends the signal and a hypothetical next-open fill to signals/paper_log.csv; on each run
it also marks earlier hypothetical fills to the latest price.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.strategies as S          # noqa: E402
import src.ml                       # noqa: F401,E402  (registers ML strategies)
from src.venues import VENUES        # noqa: E402

TF_SEC = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
DELTA = "https://api.india.delta.exchange"
DELTA_RES = {"15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
INR_PER_USD_DELTA = 85.0            # Delta India fixed conversion


def http_json(url, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def delta_candles(tf, n_bars):
    step = TF_SEC[tf]
    end = int(time.time())
    out = []
    start = end - n_bars * step
    while start < end:
        chunk_end = min(end, start + 1900 * step)       # API returns <= 2000 candles per call
        js = http_json(f"{DELTA}/v2/history/candles?resolution={DELTA_RES[tf]}&symbol=BTCUSD&start={start}&end={chunk_end}")
        out += js.get("result", [])
        start = chunk_end
    df = pd.DataFrame(out).drop_duplicates("time").sort_values("time")
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    # drop the still-forming bar: keep bars whose close time <= now
    df = df[df.index + pd.Timedelta(seconds=step) <= pd.Timestamp.now(tz="UTC")]
    df["tb_base"] = df["volume"] / 2.0   # taker-buy split unavailable from Delta; only ML features use it
    return df


def okx_funding(days=90):
    rows, after = [], ""
    for _ in range(12):
        js = http_json(f"https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP&limit=100{after}")
        d = js.get("data", [])
        if not d:
            break
        rows += d
        after = f"&after={d[-1]['fundingTime']}"
        if len(rows) >= days * 3:
            break
    f = pd.Series({pd.Timestamp(int(r["fundingTime"]), unit="ms", tz="UTC"): float(r["realizedRate"] or r["fundingRate"]) for r in rows})
    return f.sort_index().rename("rate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=10_000.0, help="account equity in ₹")
    ap.add_argument("--paper", action="store_true", help="append to signals/paper_log.csv")
    ap.add_argument("--config", default=str(ROOT / "signals" / "finalist.json"))
    a = ap.parse_args()
    fz = json.loads(Path(a.config).read_text())
    tf, strat, params, lev = fz["tf"], fz["strategy"], fz["params"], float(fz["leverage"])
    venue = VENUES[fz.get("venue", "delta_india_perp")]
    df = delta_candles(tf, fz.get("history_bars", 3000))
    fund = okx_funding()
    S.funding_series = lambda: fund          # inject live funding for funding-based signals
    sig = S.REGISTRY[strat](df, **params)
    base = float(sig["target"][-1])
    target_lev = base * lev
    last_close_t = df.index[-1] + pd.Timedelta(seconds=TF_SEC[tf])
    prev = float(sig["target"][-2]) * lev if len(df) > 1 else 0.0
    tick = http_json(f"{DELTA}/v2/tickers/BTCUSD")["result"]
    px = float(tick["mark_price"])
    cap_usd = a.capital / INR_PER_USD_DELTA
    lots = math.floor(abs(target_lev) * cap_usd / px / venue.lot_btc + 1e-9)
    qty = math.copysign(lots * venue.lot_btc, target_lev) if lots else 0.0
    if qty == 0 and target_lev != 0:
        note = f"target {target_lev:+.2f}x is below one {venue.lot_btc} BTC contract at this capital — stay flat"
    else:
        note = ""
    if target_lev > 0 and prev <= 0:
        action = "BUY"
    elif target_lev < 0 and prev >= 0:
        action = "SELL"
    elif target_lev == 0 and prev != 0:
        action = "CLOSE"
    else:
        action = "HOLD"
    stop = sig.get("stop")
    sl = None
    if stop is not None and qty != 0 and float(stop[-1]) > 0:
        sl = px * (1 - float(stop[-1]) * np.sign(qty))
    liq = None
    if qty != 0:
        liq = (qty * px - cap_usd) / (qty - venue.mmr * abs(qty))
    nxt = last_close_t + pd.Timedelta(seconds=TF_SEC[tf])
    out = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "strategy": f"{fz['id']} {strat} {tf} {json.dumps(params)}",
        "last_closed_bar_utc": str(df.index[-1]), "last_close": float(df.close.iloc[-1]),
        "action": action, "target_leverage": round(target_lev, 3), "position_btc": qty,
        "position_inr_notional": round(abs(qty) * px * INR_PER_USD_DELTA, 0),
        "margin_inr": round(a.capital, 0), "mark_price_usd": px,
        "stop_loss": round(sl, 1) if sl else "none (exit on signal change)",
        "take_profit": "none (exit on signal change)",
        "approx_liquidation_price": round(liq, 1) if liq and liq > 0 else "n/a",
        "execute_at": f"market order at the open of the bar starting {last_close_t} UTC (i.e. now)",
        "next_check_utc": str(nxt + pd.Timedelta(seconds=30)), "note": note,
    }
    for k, v in out.items():
        print(f"{k:>26}: {v}")
    if a.paper:
        log = ROOT / "signals" / "paper_log.csv"
        row = dict(out)
        row["hypothetical_fill_price"] = px * (1 + venue.slippage * np.sign(qty - 0)) if action in ("BUY", "SELL") else (
            px * (1 - venue.slippage * np.sign(prev)) if action == "CLOSE" else np.nan)
        row["fee_usd"] = abs(qty) * px * venue.taker if action in ("BUY", "SELL", "CLOSE") else 0.0
        pd.DataFrame([row]).to_csv(log, mode="a", header=not log.exists(), index=False)
        print(f"paper log appended -> {log}")


if __name__ == "__main__":
    main()
