"""Custom simulators for strategies that are not 'single target position' strategies.

1) Cash-and-carry funding harvest: long spot (CoinDCX, fractional BTC) + short Delta India perp.
2) Grid trading on the perp with limit orders (maker fees). Fills are pessimistic: a level fills only
   if price trades THROUGH it, and a level cannot fill and round-trip in the same bar. If a bar
   crosses both buy and sell levels, only the inventory-increasing buys are filled.

Both take per-bar parameter arrays so walk-forward stitching works like it does for target strategies.
They return the same dict shape as backtest.sim().
"""
import numpy as np
import pandas as pd
from numba import njit

from src.config import START_CAPITAL_INR
from src.venues import COINDCX_SPOT
from src.data import funding_series


# ============================================================ carry
@njit(cache=True)
def _carry(po, ph, pl, pc, sp_prev, sp_c, fund, active, levp, month_id, fx, lot, mmr, fee_p, slip, fee_s,
           transfer_usd, mode, e0_inr):
    n = len(po)
    eq = np.zeros(n)
    pos = np.zeros(n)
    nm = month_id[-1] - month_id[0] + 1
    m_pnl = np.full(nm, np.nan)
    tr = np.zeros((n + 1, 4))
    ntr = 0
    E = e0_inr / fx[0]
    Em0 = E
    u = 0.0          # BTC units: +u spot, -u perp
    S = 0.0          # spot leg value reference (units * spot price)
    M = 0.0          # perp margin equity
    sref = 0.0
    pref = 0.0
    fees = 0.0
    fpaid = 0.0
    nliq = 0
    tr_e0 = 0.0
    tr_i0 = 0
    m0 = month_id[0]
    want_prev = 0.0
    for t in range(n):
        s_open = sp_prev[t]
        # mark legs to open
        if u > 0:
            S = u * s_open
            M += -u * (po[t] - pref)
            pref = po[t]
            sref = s_open
        E = (S + M) if u > 0 else E
        reset = False
        if t > 0 and month_id[t] != month_id[t - 1]:
            if mode == 1:
                m_pnl[month_id[t - 1] - m0] = (E - Em0) * fx[t]
                reset = True
        want = active[t - 1] if t > 0 else 0.0
        L = levp[t - 1] if t > 0 else 1.0
        # rebalance triggers: open/close, month reset, perp margin ratio drifted out of [0.5, 2] x target
        rebal = False
        if want > 0 and u == 0:
            rebal = True
        elif want <= 0 and u > 0:
            rebal = True
        elif u > 0 and (reset or M / (u * po[t]) < 0.5 / L or M / (u * po[t]) > 2.0 / L):
            rebal = True
        if u > 0 and rebal:
            # close both legs
            c = u * s_open * fee_s + u * s_open * slip + u * po[t] * (fee_p + slip)
            E = S + M - c
            fees += c
            if ntr < tr.shape[0]:
                tr[ntr, 0] = tr_i0; tr[ntr, 1] = t; tr[ntr, 2] = E / tr_e0 - 1.0; tr[ntr, 3] = 1.0
                ntr += 1
            u = 0.0; S = 0.0; M = 0.0
        if reset:
            E = e0_inr / fx[t]
            Em0 = E
        if want > 0 and u == 0 and E > 0:
            N = E / (1.0 + 1.0 / L)
            uu = np.floor(N / po[t] / lot + 1e-9) * lot
            if uu > 0:
                c = uu * s_open * (fee_s + slip) + uu * po[t] * (fee_p + slip) + transfer_usd
                tr_e0 = E
                tr_i0 = t
                u = uu
                S = u * s_open
                M = E - S - c
                fees += c
                pref = po[t]
                sref = s_open
        # funding received by the short
        if u > 0:
            fr = u * po[t] * fund[t]
            M += fr
            fpaid -= fr
            # perp liquidation on the bar high
            liq_px = (M + u * po[t]) / (u * (1.0 + mmr))
            if ph[t] >= liq_px:
                nliq += 1
                # margin lost; spot leg survives and is sold at close
                c = u * sp_c[t] * (fee_s + slip)
                E = u * sp_c[t] - c
                fees += c
                if ntr < tr.shape[0]:
                    tr[ntr, 0] = tr_i0; tr[ntr, 1] = t; tr[ntr, 2] = E / tr_e0 - 1.0; tr[ntr, 3] = 1.0
                    ntr += 1
                u = 0.0; S = 0.0; M = 0.0
                eq[t] = E
                continue
            # mark to close
            S = u * sp_c[t]
            M += -u * (pc[t] - pref)
            pref = pc[t]
            E = S + M
            pos[t] = u
        eq[t] = E
    if mode == 1:
        m_pnl[month_id[n - 1] - m0] = (eq[n - 1] - Em0) * fx[n - 1]
    return eq, pos, m_pnl, np.zeros(nm), tr[:ntr], nliq, fees, fpaid, 0.0


def carry_sim(df, arr, venue, mode, lotfree=False, delay=0, trail=0.0, band=0.0):
    act = np.asarray(arr["active"], float)
    lv = np.asarray(arr["levp"], float)
    if delay:
        act = np.concatenate([np.zeros(delay), act[:-delay]])
        lv = np.concatenate([np.ones(delay), lv[:-delay]])
    spc = df.spot_close.to_numpy()
    sp_prev = np.concatenate([[df.open.iloc[0]], spc[:-1]])
    k = venue.taker / 0.00059 if venue.taker > 0 else 1.0   # stress multiplier propagates to spot leg
    out = _carry(df.open.to_numpy(), df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy(), sp_prev, spc,
                 df.fund.to_numpy(), act, lv, df.month_id.to_numpy(), df.fx.to_numpy(),
                 1e-9 if lotfree else venue.lot_btc, venue.mmr, venue.taker, venue.slippage,
                 COINDCX_SPOT.taker * k, 0.0 if lotfree else 1.0, mode, START_CAPITAL_INR)
    eq, pos, m_pnl, m_liq, trades, nliq, fees, fundp, turn = out
    return {"eq": pd.Series(eq, df.index), "pos": pos, "m_pnl": m_pnl, "m_liq": m_liq, "trades": trades,
            "nliq": nliq, "fees": fees, "funding": fundp, "turnover": turn}


def carry_build(df, cfg, params_list):
    f = funding_series()
    known = f.reindex(df.index.union(f.index)).ffill().reindex(df.index).shift(1)
    out = []
    for p in params_list:
        w = int(p.get("window", 21)) * 3  # in 8h prints ~ days*3
        # trailing mean of the last w prints, known at each bar
        prints = f.rolling(w, min_periods=3).mean()
        avg = prints.reindex(df.index.union(prints.index)).ffill().reindex(df.index).shift(1)
        on = (avg > p.get("enter", 0.0)).astype(float).to_numpy()
        out.append({"params": p, "arrays": {"active": np.nan_to_num(on), "levp": np.full(len(df), float(p.get("levp", 3.0)))}})
    return out


# ============================================================ grid
@njit(cache=True)
def _grid(o, h, l, c, fund, active, gsp, nlev, lev, month_id, fx, lot, mmr, fee_m, fee_t, slip, mode, e0_inr):
    n = len(o)
    eq = np.zeros(n)
    pos = np.zeros(n)
    nm = month_id[-1] - month_id[0] + 1
    m_pnl = np.full(nm, np.nan)
    tr = np.zeros((n // 2 + 1, 4))
    ntr = 0
    E = e0_inr / fx[0]
    Em0 = E
    q = 0.0
    inv = 0          # grid levels filled (0..k)
    C = 0.0          # grid anchor
    u = 0.0          # units per level
    P = o[0]
    fees = 0.0
    fpaid = 0.0
    nliq = 0
    dead = False
    m0 = month_id[0]
    on = False
    cost_basis = np.zeros(64)
    for t in range(n):
        if q != 0.0:
            E += q * (o[t] - P)
        P = o[t]
        if t > 0 and month_id[t] != month_id[t - 1] and mode == 1:
            # withdraw: flatten, reset equity, re-anchor
            if q != 0.0:
                cc = abs(q) * o[t] * (fee_t + slip)
                E -= cc
                fees += cc
                q = 0.0
                inv = 0
            m_pnl[month_id[t - 1] - m0] = (E - Em0) * fx[t]
            E = e0_inr / fx[t]
            Em0 = E
            dead = False
            on = False
        if dead:
            continue
        a = active[t - 1] if t > 0 else 0.0
        g = gsp[t - 1] if t > 0 else 0.01
        k = int(nlev[t - 1]) if t > 0 else 5
        if k > 60:
            k = 60
        L = lev[t - 1] if t > 0 else 1.0
        # switch off -> flatten at open
        if (a <= 0 or not on) and q != 0.0 and a <= 0:
            cc = abs(q) * o[t] * (fee_t + slip)
            E -= cc
            fees += cc
            q = 0.0
            inv = 0
            on = False
        if a > 0 and not on and E > 0:
            C = o[t]
            u = np.floor(L * E / (k * C) / lot + 1e-9) * lot
            on = u > 0
            inv = 0
        if on:
            # funding on inventory
            if q != 0.0:
                fc = q * o[t] * fund[t]
                E -= fc
                fpaid += fc
            buy_px = C * (1.0 - g) ** (inv + 1)
            sell_px = C * (1.0 - g) ** (inv - 1) if inv > 0 else 0.0
            bottom = C * (1.0 - g) ** (k + 1)
            did_buy = False
            # buys (inventory increasing) first — pessimistic
            while inv < k and l[t] < buy_px:
                fill = min(buy_px, o[t])
                cc = u * fill * fee_m
                E -= cc + u * (fill - o[t])  # mark relative to open
                fees += cc
                q += u
                cost_basis[inv] = fill
                inv += 1
                did_buy = True
                buy_px = C * (1.0 - g) ** (inv + 1)
            # stop-out: price trades through one level below the full grid
            if inv >= k and l[t] < bottom:
                ex = min(bottom, o[t]) * (1.0 - slip)
                e_prev = E + q * (o[t] - o[t])
                rl = u * (ex * inv - cost_basis[:inv].sum())
                E += q * (ex - o[t])
                cc = abs(q) * ex * fee_t
                E -= cc
                fees += cc
                if ntr < tr.shape[0]:
                    tr[ntr, 0] = t; tr[ntr, 1] = t; tr[ntr, 2] = rl / max(e_prev, 1e-9); tr[ntr, 3] = 1.0
                    ntr += 1
                q = 0.0
                inv = 0
                on = False          # re-anchor next bar
                P = o[t]
                pos[t] = 0.0
                # mark rest of bar flat
                eq[t] = E
                if E <= 0:
                    dead = True
                    eq[t] = 0.0
                continue
            if not did_buy:
                while inv > 0 and h[t] > sell_px:
                    fill = max(sell_px, o[t])
                    cc = u * fill * fee_m
                    E += u * (fill - o[t]) - cc
                    fees += cc
                    q -= u
                    inv -= 1
                    if ntr < tr.shape[0]:
                        tr[ntr, 0] = t; tr[ntr, 1] = t; tr[ntr, 2] = u * (fill - cost_basis[inv]) / max(E, 1e-9); tr[ntr, 3] = 1.0
                        ntr += 1
                    # booked P&L relative to open is now realised; position mark continues from open for remaining q
                    E -= 0.0
                    sell_px = C * (1.0 - g) ** (inv - 1) if inv > 0 else 0.0
                # trail anchor up when flat and price rose one spacing
                if inv == 0 and c[t] > C * (1.0 + g):
                    C = c[t]
                    u = np.floor(L * E / (k * C) / lot + 1e-9) * lot
                    if u <= 0:
                        on = False
            # liquidation check on the low (worst) for the long inventory
            if q > 0:
                liq_px = (q * o[t] - E) / (q * (1.0 - mmr))
                if l[t] <= liq_px:
                    nliq += 1
                    E = 0.0
                    q = 0.0
                    inv = 0
                    dead = True
                    eq[t] = 0.0
                    continue
        pos[t] = q
        if q != 0.0:
            E += q * (c[t] - P)
        P = c[t]
        eq[t] = E
    if mode == 1:
        m_pnl[month_id[n - 1] - m0] = (eq[n - 1] - Em0) * fx[n - 1]
    return eq, pos, m_pnl, np.zeros(nm), tr[:ntr], nliq, fees, fpaid, 0.0


def grid_sim(df, arr, venue, mode, lotfree=False, delay=0, trail=0.0, band=0.0):
    ks = ["active", "gsp", "nlev", "lev"]
    a = [np.asarray(arr[k], float) for k in ks]
    if delay:
        a = [np.concatenate([np.full(delay, x[0] if i else 0.0), x[:-delay]]) for i, x in enumerate(a)]
    out = _grid(df.open.to_numpy(), df.high.to_numpy(), df.low.to_numpy(), df.close.to_numpy(),
                df.fund.to_numpy() if venue.funding else np.zeros(len(df)), a[0], a[1], a[2], a[3],
                df.month_id.to_numpy(), df.fx.to_numpy(), 1e-9 if lotfree else venue.lot_btc, venue.mmr,
                venue.maker, venue.taker, venue.slippage, mode, START_CAPITAL_INR)
    eq, pos, m_pnl, m_liq, trades, nliq, fees, fundp, turn = out
    return {"eq": pd.Series(eq, df.index), "pos": pos, "m_pnl": m_pnl, "m_liq": m_liq, "trades": trades,
            "nliq": nliq, "fees": fees, "funding": fundp, "turnover": turn}


def grid_build(df, cfg, params_list):
    out = []
    c = df.close
    for p in params_list:
        n = len(df)
        filt = p.get("filter", "none")
        if filt == "none":
            act = np.ones(n)
        else:
            # daily SMA filter computed on completed daily closes only
            d = c.resample("1D").last()
            sm = d.rolling(int(p.get("sma_days", 100))).mean()
            above = (d > sm).astype(float).shift(1)   # known at next day's start
            act = np.nan_to_num(above.reindex(df.index, method="ffill").to_numpy())
            if filt == "below":
                act = 1.0 - act
        out.append({"params": p, "arrays": {"active": act, "gsp": np.full(n, float(p["spacing"])),
                                            "nlev": np.full(n, float(p["levels"])), "lev": np.full(n, float(p.get("lev", 1.0)))}})
    return out


CUSTOM = {
    "carry": {"build": carry_build, "sim": carry_sim},
    "grid": {"build": grid_build, "sim": grid_sim},
}


def _register_ensemble():
    from src.ensemble import ens_build, ens_sim
    CUSTOM["ensemble"] = {"build": ens_build, "sim": ens_sim}


_register_ensemble()
