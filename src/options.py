"""Weekly BTC option-selling structures on REAL Deribit trade prints (history.deribit.com).

Entry: Friday 08:00–10:00 UTC, right after the weekly expiry. Sell next Friday's option(s) whose trade-implied
|delta| is closest to the target. The entry premium is the MEDIAN traded price of that instrument in the
window, cut by a 5% haircut (selling at the bid, not the print), in USD at the trade's index price.
Settlement: next Friday 08:00 UTC at the Binance/perp price of the 08:00 bar open (Deribit settles on a
30-min index TWAP; the difference is noise). Fees: Delta Exchange India options fee 0.03% of underlying
notional, capped at 3.5% of premium, plus 18% GST, on entry and on exercise.
Risk: the short is marked daily at intrinsic value PLUS the entry premium (a crude time-value/vega
buffer). If equity at the day's worst price (1d low/high, built from 1-minute data) falls below 0,
the account is liquidated and the whole margin is lost.
Structures: put (cash-secured-like short put), call (short call), strangle (short put + short call).
Deribit weeklies with enough prints exist from mid-2019. OOS here therefore starts 2019-07 (earlier windows are flat).
"""
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import requests
from numba import njit
from scipy.stats import norm

from src.config import RAW, DEV_END, START_CAPITAL_INR
from src.data import bars

OPT = RAW / "deribit"
OPT.mkdir(parents=True, exist_ok=True)
FIRST = pd.Timestamp("2019-06-07", tz="UTC")


def fetch_window(t0, t1):
    out, s = [], t0
    for _ in range(50):
        for k in range(5):
            try:
                r = requests.get("https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time",
                                 params=dict(currency="BTC", kind="option", start_timestamp=s, end_timestamp=t1,
                                             count=1000, sorting="asc"), timeout=30).json()["result"]
                break
            except Exception:
                time.sleep(2 ** (k + 1))
        tr = r["trades"]
        out += tr
        if not r.get("has_more") or not tr:
            break
        s = tr[-1]["timestamp"] + 1
    return out


def download():
    f = FIRST
    while f + timedelta(days=7) + timedelta(hours=8) < DEV_END:   # expiry must be before the lockbox
        fp = OPT / f"{f:%Y%m%d}.parquet"
        if not fp.exists():
            t0 = int((f + pd.Timedelta("8h")).timestamp() * 1000)
            tr = fetch_window(t0, t0 + 2 * 3600 * 1000)
            nxt = (f + pd.Timedelta("7D")).strftime("%-d%b%y").upper()
            df = pd.DataFrame(tr)
            if len(df):
                df = df[df.instrument_name.str.startswith(f"BTC-{nxt}-")]
                df = df[["timestamp", "instrument_name", "price", "iv", "index_price", "amount", "direction"]]
            df.to_parquet(fp)
        f += timedelta(days=7)


def week_table():
    """Per entry Friday: all next-week instruments with median price (USD), strike, type and delta."""
    rows = []
    for fp in sorted(OPT.glob("*.parquet")):
        d = pd.read_parquet(fp)
        if not len(d):
            continue
        f = pd.Timestamp(fp.stem, tz="UTC")
        g = d.groupby("instrument_name").agg(price=("price", "median"), iv=("iv", "median"), S=("index_price", "median"),
                                             n=("price", "size"))
        g = g[g.n >= 1].reset_index()
        g["K"] = g.instrument_name.str.split("-").str[2].astype(float)
        g["type"] = g.instrument_name.str[-1]
        T = (7 * 24 - 1) / (365 * 24)
        sig = g.iv / 100
        d1 = (np.log(g.S / g.K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
        g["delta"] = np.where(g.type == "C", norm.cdf(d1), norm.cdf(d1) - 1)
        g["prem_usd"] = g.price * g.S * 0.95            # sell at bid: 5% haircut on the print
        g["entry"] = f
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def pick(wt, target, typ):
    x = wt[wt.type == typ]
    x = x.assign(dd=(x.delta.abs() - target).abs())
    return x.loc[x.groupby("entry").dd.idxmin()].set_index("entry")


@njit(cache=True)
def _opt(o, h, l, c, month_id, fx, is_entry, pK, pP, cK, cP, lev, mode, e0_inr, fee_rate, cap, lot):
    n = len(o)
    eq = np.zeros(n)
    pos = np.zeros(n)
    nm = month_id[-1] - month_id[0] + 1
    m_pnl = np.full(nm, np.nan)
    tr = np.zeros((n + 1, 4))
    ntr = 0
    E = e0_inr / fx[0]
    Em0 = E
    q = 0.0
    Kp = 0.0
    Kc = 0.0
    prem = 0.0
    buf = 0.0
    e_entry = 0.0
    i_entry = 0
    nliq = 0
    fees = 0.0
    dead = False
    m0 = month_id[0]
    for t in range(n):
        if t > 0 and month_id[t] != month_id[t - 1] and mode == 1:
            # withdraw: account resets; an open short keeps running on the reset account's margin
            m_pnl[month_id[t - 1] - m0] = ((E if not dead else 0.0) - Em0) * fx[t]
            E = e0_inr / fx[t]
            Em0 = E
            dead = False
        if dead:
            eq[t] = 0.0
            continue
        # settlement at this bar's open (entry days are expiry days: 08:00 bar on 1h, or daily open approx)
        if q > 0 and is_entry[t] != 0:
            pay = q * (max(Kp - o[t], 0.0) if Kp > 0 else 0.0) + q * (max(o[t] - Kc, 0.0) if Kc > 0 else 0.0)
            if pay > 0:
                f = min(fee_rate * q * o[t], cap * prem) * 1.18
                pay += f
                fees += f
            E -= pay
            if ntr < tr.shape[0]:
                tr[ntr, 0] = i_entry; tr[ntr, 1] = t; tr[ntr, 2] = E / e_entry - 1.0; tr[ntr, 3] = -1.0
                ntr += 1
            q = 0.0
            if E <= 0:
                dead = True
                eq[t] = 0.0
                continue
        # new entry
        if is_entry[t] != 0 and lev[t] > 0 and (pK[t] > 0 or cK[t] > 0) and E > 0:
            qq = np.floor(lev[t] * E / o[t] / lot + 1e-9) * lot
            if qq > 0:
                q = qq
                Kp = pK[t]
                Kc = cK[t]
                prem = q * (pP[t] + cP[t])
                f = min(fee_rate * q * o[t], cap * prem) * 1.18 * ((pK[t] > 0) + (cK[t] > 0))
                E += prem - f
                fees += f
                buf = prem
                e_entry = E - prem + f
                i_entry = t
        if q > 0:
            worst_l = q * (max(Kp - l[t], 0.0) if Kp > 0 else 0.0)
            worst_h = q * (max(h[t] - Kc, 0.0) if Kc > 0 else 0.0)
            if E - max(worst_l, worst_h) - buf <= 0:
                nliq += 1
                E = 0.0
                q = 0.0
                dead = True
                if ntr < tr.shape[0]:
                    tr[ntr, 0] = i_entry; tr[ntr, 1] = t; tr[ntr, 2] = -1.0; tr[ntr, 3] = -1.0
                    ntr += 1
                eq[t] = 0.0
                continue
            liab = q * (max(Kp - c[t], 0.0) if Kp > 0 else 0.0) + q * (max(c[t] - Kc, 0.0) if Kc > 0 else 0.0)
            eq[t] = E - liab
            pos[t] = -q
        else:
            eq[t] = E
    if mode == 1:
        m_pnl[month_id[n - 1] - m0] = ((eq[n - 1] if not dead else 0.0) - Em0) * fx[n - 1]
    return eq, pos, m_pnl, np.zeros(nm), tr[:ntr], nliq, fees, 0.0, 0.0


def opt_sim(df, arr, venue, mode, lotfree=False, delay=0, trail=0.0, band=0.0):
    k = venue.taker / 0.00059 if venue.taker > 0 else 1.0   # stress multiplier also scales option fees/haircut
    a = {kk: np.asarray(v, float) for kk, v in arr.items()}
    if delay:   # entering one bar late means the Friday prints no longer apply: the week is skipped
        a["is_entry"] = np.concatenate([np.zeros(delay), a["is_entry"][:-delay]])
        for kk in ("pK", "pP", "cK", "cP"):
            a[kk] = np.concatenate([np.zeros(delay), a[kk][:-delay]])
        a["pP"] = a["pP"] * 0.9; a["cP"] = a["cP"] * 0.9     # stale quotes: extra 10% haircut
    pP = a["pP"] * (1 - 0.05 * (k - 1)); cP = a["cP"] * (1 - 0.05 * (k - 1))
    out = _opt(df.open.to_numpy(), df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy(), df.month_id.to_numpy(),
               df.fx.to_numpy(), a["is_entry"], a["pK"], pP, a["cK"], cP, a["lev"], mode, START_CAPITAL_INR,
               0.0003 * k, 0.035, 1e-9 if lotfree else venue.lot_btc)
    eq, pos, m_pnl, m_liq, trades, nliq, fees, fundp, turn = out
    return {"eq": pd.Series(eq, df.index), "pos": pos, "m_pnl": m_pnl, "m_liq": m_liq, "trades": trades,
            "nliq": nliq, "fees": fees, "funding": 0.0, "turnover": 0.0}


_WT = None


def opt_build(df, cfg, params_list):
    """df must be 1h bars: entries/settlements happen at the 08:00 UTC bar open on Fridays."""
    global _WT
    if _WT is None:
        _WT = week_table()
    wt = _WT
    idx = df.index
    out = []
    for p in params_list:
        n = len(df)
        A = {k: np.zeros(n) for k in ("is_entry", "pK", "pP", "cK", "cP")}
        A["lev"] = np.zeros(n)
        st = p["structure"]
        tgt = float(p["delta"])
        puts = pick(wt, tgt, "P") if st in ("put", "strangle") else None
        calls = pick(wt, tgt, "C") if st in ("call", "strangle") else None
        entries = sorted(set(wt.entry))
        # the entry decision uses quotes from 08:00-10:00; execute at the 10:00 bar open (no look-ahead);
        # settlement one week later at the 08:00 bar open
        for f in entries:
            i_set = idx.searchsorted(f + pd.Timedelta("8h"))
            if i_set < n and idx[i_set] == f + pd.Timedelta("8h"):
                A["is_entry"][i_set] = 1        # settles the previous week's short
        for f in entries:
            i = idx.searchsorted(f + pd.Timedelta("10h"))
            if i >= n or idx[i] != f + pd.Timedelta("10h"):
                continue
            A["is_entry"][i] = 1
            if puts is not None and f in puts.index:
                A["pK"][i], A["pP"][i] = puts.loc[f, "K"], puts.loc[f, "prem_usd"]
            if calls is not None and f in calls.index:
                A["cK"][i], A["cP"][i] = calls.loc[f, "K"], calls.loc[f, "prem_usd"]
            A["lev"][i] = float(p.get("lev", 1.0))
        out.append({"params": p, "arrays": A})
    return out


def _register():
    from src.custom import CUSTOM
    CUSTOM["options"] = {"build": opt_build, "sim": opt_sim}


_register()

if __name__ == "__main__":
    download()
    print(week_table().groupby("entry").size().describe())
