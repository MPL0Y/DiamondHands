"""Fast array engine (numba) — used for search.

Execution model (identical contract to engine_event.py, which is independently written):
- tgt[t]  : signed target leverage (multiple of current equity) decided on the CLOSE of bar t.
            Executed at the OPEN of bar t+1 with slippage + taker fee. NaN = keep previous target.
- Rebalancing happens only when the target value changes (or at a withdraw-mode monthly reset,
  or when |actual exposure - target| > band*|target| if band > 0).
- Size is rounded toward zero to whole lots (lot_btc); below one lot the position is flat.
- stop[t]/tp[t]: fractional distances attached to a NEW position opened at t+1 (0 = none).
  Stop = stop-market: fills at stop*(1-slip) (long) or at the open if the bar gapped through it.
  TP = limit: fills only if price trades THROUGH the level (high > tp for longs), maker fee, no slip.
  If stop and TP could both trigger in one bar, the stop is assumed to hit first (pessimistic).
  After a stop/TP exit, the strategy stays flat until its target value changes.
- trail: trailing-stop distance; the trail level uses highs/lows of COMPLETED bars only.
- Funding: fund[t] = sum of 8h funding rates stamped inside [open_t, open_t+1), charged on q*open_t.
- Liquidation: cross margin — whole equity is margin. If equity at the bar's worst price falls
  below mmr*notional, the account loses everything (equity -> 0). Bar high/low are built from
  1-minute data, so a threshold test on them is exact.
- mode 0 = compounding (one account); mode 1 = withdraw (equity reset to e0_inr each month).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _lots(x, lot):
    n = np.floor(abs(x) / lot + 1e-9)
    return np.sign(x) * n * lot


@njit(cache=True)
def simulate(o, h, l, c, fund, tgt, stop, tp, trail, month_id, fx, lot, mmr, fee_t, fee_m, slip,
             mode, e0_inr, band):
    n = len(o)
    eq = np.zeros(n)            # equity (USD) marked at close
    pos = np.zeros(n)           # units held during bar
    nm = month_id[-1] - month_id[0] + 1
    m_pnl = np.full(nm, np.nan)  # withdraw-mode INR P&L per month
    m_liq = np.zeros(nm)
    # trades: entry bar, exit bar, return on entry equity, side
    tr = np.zeros((n // 1 + 1, 4))
    ntr = 0
    E = e0_inr / fx[0]
    q = 0.0
    cur = 0.0          # current target
    last_exec = 0.0    # target at last rebalance
    stopped = False
    stopped_tgt = 0.0
    sl_px = 0.0
    tp_px = 0.0
    tr_px = 0.0        # trailing stop level
    ext = 0.0          # best completed-bar extreme since entry
    tr_e0 = 0.0
    tr_i0 = 0
    nliq = 0
    fees_paid = 0.0
    fund_paid = 0.0
    turnover = 0.0
    m0 = month_id[0]
    dead = False
    P = o[0]
    for t in range(n):
        # ---- mark to open
        if q != 0.0:
            E += q * (o[t] - P)
        P = o[t]
        force = False
        # ---- month boundary
        if t > 0 and month_id[t] != month_id[t - 1]:
            k = month_id[t - 1] - m0
            if mode == 1:
                m_pnl[k] = E * fx[t] - e0_inr
                E = e0_inr / fx[t]
                dead = False
                force = True
        if dead:
            eq[t] = 0.0
            continue
        # ---- liquidation at open (gap)
        if q != 0.0 and E <= mmr * abs(q) * o[t]:
            nliq += 1
            m_liq[month_id[t] - m0] += 1
            E = 0.0
            if ntr < tr.shape[0]:
                tr[ntr, 0] = tr_i0; tr[ntr, 1] = t; tr[ntr, 2] = -1.0; tr[ntr, 3] = np.sign(q)
                ntr += 1
            q = 0.0
            dead = True
            eq[t] = 0.0
            continue
        # ---- new target from previous close
        if t > 0 and not np.isnan(tgt[t - 1]):
            cur = tgt[t - 1]
        d = cur
        want_trade = False
        if stopped:
            if abs(d - stopped_tgt) > 1e-12:
                stopped = False
        if not stopped:
            if abs(d - last_exec) > 1e-12 or force:
                want_trade = True
            elif band > 0 and d != 0.0 and E > 0:
                expo = q * o[t] / E
                if abs(expo - d) > band * abs(d):
                    want_trade = True
        if want_trade and E > 0:
            newq = _lots(d * E / o[t], lot)
            dq = newq - q
            if dq != 0.0:
                px = o[t] * (1.0 + slip * np.sign(dq))
                cost = abs(dq) * o[t] * slip + abs(dq) * px * fee_t
                # trade bookkeeping (close / flip)
                if q != 0.0 and (newq == 0.0 or np.sign(newq) != np.sign(q)):
                    # closing leg cost share
                    frac = abs(q) / abs(dq)
                    E_after_close = E - cost * frac
                    if ntr < tr.shape[0]:
                        tr[ntr, 0] = tr_i0; tr[ntr, 1] = t
                        tr[ntr, 2] = E_after_close / tr_e0 - 1.0; tr[ntr, 3] = np.sign(q)
                        ntr += 1
                    if newq != 0.0:
                        tr_e0 = E_after_close
                        tr_i0 = t
                elif q == 0.0 and newq != 0.0:
                    tr_e0 = E
                    tr_i0 = t
                E -= cost
                fees_paid += cost
                turnover += abs(dq) * o[t]
                entering = (q == 0.0) or (np.sign(newq) != np.sign(q))
                q = newq
                if entering and q != 0.0:
                    s = stop[t - 1] if t > 0 else 0.0
                    g = tp[t - 1] if t > 0 else 0.0
                    sl_px = o[t] * (1.0 - s * np.sign(q)) if s > 0 else 0.0
                    tp_px = o[t] * (1.0 + g * np.sign(q)) if g > 0 else 0.0
                    ext = o[t]
                    tr_px = o[t] * (1.0 - trail * np.sign(q)) if trail > 0 else 0.0
            last_exec = d
        # ---- funding
        if q != 0.0 and fund[t] != 0.0:
            fc = q * o[t] * fund[t]
            E -= fc
            fund_paid += fc
        # ---- intrabar: stops, liquidation, take-profit
        exit_px = 0.0
        exit_fee = 0.0
        exited = False
        if q != 0.0:
            side = np.sign(q)
            worst = l[t] if side > 0 else h[t]
            best = h[t] if side > 0 else l[t]
            # effective stop = tighter of fixed stop and trailing stop
            stp = 0.0
            if sl_px > 0 and tr_px > 0:
                stp = max(sl_px, tr_px) if side > 0 else min(sl_px, tr_px)
            elif sl_px > 0:
                stp = sl_px
            elif tr_px > 0:
                stp = tr_px
            liq_px = 0.0
            # solve E + q*(p - o) = mmr*|q|*p
            if side > 0:
                liq_px = (q * o[t] - E) / (q - mmr * q)
            else:
                liq_px = (q * o[t] - E) / (q + mmr * q)
            stop_hit = stp > 0 and ((side > 0 and worst <= stp) or (side < 0 and worst >= stp))
            liq_hit = (side > 0 and worst <= liq_px) or (side < 0 and worst >= liq_px)
            stop_first = stop_hit and ((side > 0 and stp > liq_px) or (side < 0 and stp < liq_px))
            if stop_hit and (stop_first or not liq_hit):
                if (side > 0 and o[t] <= stp) or (side < 0 and o[t] >= stp):
                    base = o[t]
                else:
                    base = stp
                exit_px = base * (1.0 - slip * side)
                exit_fee = fee_t
                exited = True
            elif liq_hit:
                nliq += 1
                m_liq[month_id[t] - m0] += 1
                E = 0.0
                if ntr < tr.shape[0]:
                    tr[ntr, 0] = tr_i0; tr[ntr, 1] = t; tr[ntr, 2] = -1.0; tr[ntr, 3] = side
                    ntr += 1
                q = 0.0
                dead = True
                pos[t] = 0.0
                eq[t] = 0.0
                continue
            elif tp_px > 0 and ((side > 0 and best > tp_px) or (side < 0 and best < tp_px)):
                base = o[t] if ((side > 0 and o[t] > tp_px) or (side < 0 and o[t] < tp_px)) else tp_px
                exit_px = base
                exit_fee = fee_m
                exited = True
            if exited:
                pos[t] = q
                E += q * (exit_px - o[t])
                cst = abs(q) * exit_px * exit_fee
                E -= cst
                fees_paid += cst
                if exit_fee == fee_t:
                    fees_paid += abs(q) * base * slip
                turnover += abs(q) * exit_px
                if ntr < tr.shape[0]:
                    tr[ntr, 0] = tr_i0; tr[ntr, 1] = t; tr[ntr, 2] = E / tr_e0 - 1.0; tr[ntr, 3] = side
                    ntr += 1
                q = 0.0
                stopped = True
                stopped_tgt = cur
                last_exec = cur
                P = c[t]
                eq[t] = E
                if E <= 0:
                    dead = True
                    eq[t] = 0.0
                continue
        pos[t] = q
        # ---- mark to close
        if q != 0.0:
            E += q * (c[t] - P)
            # update trailing stop with the completed bar
            if trail > 0:
                if q > 0:
                    ext = max(ext, h[t])
                    tr_px = max(tr_px, ext * (1.0 - trail))
                else:
                    ext = min(ext, l[t])
                    tr_px = min(tr_px, ext * (1.0 + trail))
        P = c[t]
        eq[t] = E
    # final month
    if mode == 1:
        k = month_id[n - 1] - m0
        m_pnl[k] = (eq[n - 1] if not dead else 0.0) * fx[n - 1] - e0_inr
    if q != 0.0 and ntr < tr.shape[0]:
        tr[ntr, 0] = tr_i0; tr[ntr, 1] = n - 1; tr[ntr, 2] = E / tr_e0 - 1.0; tr[ntr, 3] = np.sign(q)
        ntr += 1
    return eq, pos, m_pnl, m_liq, tr[:ntr], nliq, fees_paid, fund_paid, turnover
