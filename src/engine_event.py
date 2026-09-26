"""Event-driven verification engine — written independently of engine_fast.py.

Objects: Account (cash/margin/liquidation), Broker (order queue, fills), Strategy
(TargetFollower: turns a per-bar target-leverage intent into orders). The loop steps bar by bar
and emits events: BAR_OPEN -> fills of queued market orders -> funding -> intrabar resting-order
checks (stop / liquidation / limit TP) -> BAR_CLOSE -> strategy decision -> queue orders.

It can run on the strategy's own bars OR on 1-minute bars (`sub_bars`). In 1m mode the target
decided on a TF bar's close is acted on at the next 1m open, which is the same instant as the next
TF open. Resting stops, TPs and liquidation are then checked minute by minute, which resolves
intrabar ordering exactly. It also re-checks liquidations through the crash windows.
"""
from dataclasses import dataclass, field
import math
import numpy as np
import pandas as pd


@dataclass
class Fill:
    t: pd.Timestamp
    qty: float
    price: float
    fee: float
    kind: str


@dataclass
class Account:
    equity: float                 # marked equity in USD
    qty: float = 0.0
    mark: float = 0.0
    dead: bool = False
    fills: list = field(default_factory=list)
    fees: float = 0.0
    funding: float = 0.0
    liquidations: int = 0

    def revalue(self, price):
        self.equity += self.qty * (price - self.mark)
        self.mark = price

    def maintenance_breached(self, price, mmr):
        eq = self.equity + self.qty * (price - self.mark)
        return self.qty != 0 and eq <= mmr * abs(self.qty) * price

    def liquidation_price(self, mmr):
        if self.qty == 0:
            return None
        # equity + qty*(p-mark) = mmr*|qty|*p
        return (self.qty * self.mark - self.equity) / (self.qty - mmr * abs(self.qty))


class Broker:
    def __init__(self, venue):
        self.v = venue

    def round_lots(self, qty):
        n = math.floor(abs(qty) / self.v.lot_btc + 1e-9)
        return math.copysign(n * self.v.lot_btc, qty) if n >= self.v.min_lots else 0.0

    def market(self, acct, t, target_qty, ref_price):
        dq = target_qty - acct.qty
        if dq == 0:
            return None
        side = 1 if dq > 0 else -1
        px = ref_price * (1 + self.v.slippage * side)
        fee = abs(dq) * px * self.v.taker
        acct.revalue(ref_price)
        acct.equity -= abs(dq) * (px - ref_price) * side + fee
        acct.qty = target_qty
        acct.fees += fee
        f = Fill(t, dq, px, fee, "market")
        acct.fills.append(f)
        return f

    def close_at(self, acct, t, price, kind, maker=False):
        q = acct.qty
        side = 1 if q > 0 else -1
        px = price if maker else price * (1 - self.v.slippage * side)
        fee = abs(q) * px * (self.v.maker if maker else self.v.taker)
        acct.revalue(price)
        acct.equity += q * (px - price) - fee
        acct.qty = 0.0
        acct.fees += fee
        acct.fills.append(Fill(t, -q, px, fee, kind))


class TargetFollower:
    """Holds strategy state: current target, stop/TP brackets, stopped-out latch."""

    def __init__(self, trail):
        self.target = 0.0
        self.executed = 0.0
        self.latched = None     # target value at the time of a stop/TP exit
        self.trail = trail
        self.stop = self.tp = self.trail_lvl = None
        self.extreme = None

    def arm(self, entry, side, stop_pct, tp_pct):
        self.stop = entry * (1 - stop_pct * side) if stop_pct > 0 else None
        self.tp = entry * (1 + tp_pct * side) if tp_pct > 0 else None
        self.trail_lvl = entry * (1 - self.trail * side) if self.trail > 0 else None
        self.extreme = entry

    def effective_stop(self, side):
        lv = [x for x in (self.stop, self.trail_lvl) if x is not None]
        if not lv:
            return None
        return max(lv) if side > 0 else min(lv)


def run_event(bars, targets, venue, mode="compound", e0_inr=10_000.0, fx=None, fund=None,
              stops=None, tps=None, trail=0.0, band=0.0, sub_bars=None):
    """bars: DataFrame(open,high,low,close) at strategy TF. targets: Series aligned to bars (decided at close).
    fund: Series of funding sums per bar. fx: USDINR per bar. sub_bars: optional 1m DataFrame for intrabar replay.
    Returns dict with equity Series (at TF closes), monthly INR P&L (withdraw), trades list."""
    idx = bars.index
    tgt = targets.reindex(idx)
    stops = (stops if stops is not None else pd.Series(0.0, idx)).reindex(idx).fillna(0.0)
    tps = (tps if tps is not None else pd.Series(0.0, idx)).reindex(idx).fillna(0.0)
    fx = fx.reindex(idx)
    fund = (fund if fund is not None else pd.Series(0.0, idx)).reindex(idx).fillna(0.0)
    brk = Broker(venue)
    acct = Account(equity=e0_inr / fx.iloc[0], mark=bars.open.iloc[0])
    strat = TargetFollower(trail)
    eq_close, months, trades = [], {}, []
    trade_open = None
    month_start_eq = None

    step = idx[1] - idx[0]
    if sub_bars is not None:
        sb = sub_bars.loc[(sub_bars.index >= idx[0]) & (sub_bars.index < idx[-1] + step)]
        grp = np.searchsorted(idx.values, sb.index.values, side="right") - 1
        sub_o, sub_h, sub_l, sub_c = (sb[k].to_numpy() for k in ("open", "high", "low", "close"))
        bounds = np.searchsorted(grp, np.arange(len(idx) + 1))

    def record_trade_close(t_i, eq_after):
        nonlocal trade_open
        if trade_open is not None:
            i0, e0, side = trade_open
            trades.append((i0, t_i, eq_after / e0 - 1.0, side))
            trade_open = None

    O, H, L, C = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    for i, t in enumerate(idx):
        # ---------- BAR_OPEN
        acct.revalue(O[i])
        force = False
        if i > 0 and t.month != idx[i - 1].month:
            key = idx[i - 1].strftime("%Y-%m")
            if mode == "withdraw":
                months[key] = acct.equity * fx.iloc[i] - e0_inr if not acct.dead else -e0_inr
                acct.equity = e0_inr / fx.iloc[i]
                acct.dead = False
                force = True
        if acct.dead:
            eq_close.append(0.0)
            continue
        if acct.maintenance_breached(O[i], venue.mmr):
            acct.liquidations += 1
            record_trade_close(i, 0.0)
            acct.equity, acct.qty, acct.dead = 0.0, 0.0, True
            eq_close.append(0.0)
            continue
        # ---------- strategy intent from previous close
        if i > 0 and not np.isnan(tgt.iloc[i - 1]):
            strat.target = float(tgt.iloc[i - 1])
        if strat.latched is not None and abs(strat.target - strat.latched) > 1e-12:
            strat.latched = None
        rebalance = False
        if strat.latched is None:
            if abs(strat.target - strat.executed) > 1e-12 or force:
                rebalance = True
            elif band > 0 and strat.target != 0 and acct.equity > 0:
                if abs(acct.qty * O[i] / acct.equity - strat.target) > band * abs(strat.target):
                    rebalance = True
        if rebalance and acct.equity > 0:
            want = brk.round_lots(strat.target * acct.equity / O[i])
            prev_q = acct.qty
            if want != prev_q:
                opening = want != 0 and (prev_q == 0 or np.sign(want) != np.sign(prev_q))
                if prev_q != 0 and (want == 0 or np.sign(want) != np.sign(prev_q)):
                    brk.market(acct, t, 0.0, O[i])          # close leg
                    record_trade_close(i, acct.equity)
                e_entry = acct.equity
                if want != acct.qty:
                    brk.market(acct, t, want, O[i])
                if opening:
                    trade_open = (i, e_entry, np.sign(want))
                    strat.arm(O[i], np.sign(want), stops.iloc[i - 1] if i > 0 else 0.0,
                              tps.iloc[i - 1] if i > 0 else 0.0)
            strat.executed = strat.target
        # ---------- funding
        if acct.qty != 0 and fund.iloc[i] != 0:
            c = acct.qty * O[i] * fund.iloc[i]
            acct.equity -= c
            acct.funding += c
        # ---------- intrabar resting orders
        if acct.qty != 0:
            side = 1 if acct.qty > 0 else -1
            if sub_bars is None:
                paths = [(O[i], H[i], L[i], C[i])]
            else:
                a, b = bounds[i], bounds[i + 1]
                paths = list(zip(sub_o[a:b], sub_h[a:b], sub_l[a:b], sub_c[a:b])) or [(O[i], H[i], L[i], C[i])]
            done = False
            for (po, ph, pl, pc) in paths:
                stp = strat.effective_stop(side)
                liq = acct.liquidation_price(venue.mmr)
                worst, best = (pl, ph) if side > 0 else (ph, pl)
                hit_stop = stp is not None and ((worst <= stp) if side > 0 else (worst >= stp))
                hit_liq = (worst <= liq) if side > 0 else (worst >= liq)
                stop_before_liq = hit_stop and ((stp > liq) if side > 0 else (stp < liq))
                if hit_stop and (stop_before_liq or not hit_liq):
                    gapped = (po <= stp) if side > 0 else (po >= stp)
                    brk.close_at(acct, t, po if gapped else stp, "stop")
                    done = True
                elif hit_liq:
                    acct.liquidations += 1
                    acct.equity, acct.qty, acct.dead = 0.0, 0.0, True
                    record_trade_close(i, 0.0)
                    done = True
                    break
                elif strat.tp is not None and ((best > strat.tp) if side > 0 else (best < strat.tp)):
                    thru_open = (po > strat.tp) if side > 0 else (po < strat.tp)
                    brk.close_at(acct, t, po if thru_open else strat.tp, "tp", maker=True)
                    done = True
                if done:
                    record_trade_close(i, acct.equity)
                    strat.latched = strat.target
                    strat.executed = strat.target
                    acct.mark = pc  # remainder of bar is flat
                    break
                if sub_bars is not None:
                    acct.revalue(pc)
                    if strat.trail > 0:
                        # 1m mode: trail updates on each completed minute
                        if side > 0:
                            strat.extreme = max(strat.extreme, ph)
                            strat.trail_lvl = max(strat.trail_lvl, strat.extreme * (1 - strat.trail))
                        else:
                            strat.extreme = min(strat.extreme, pl)
                            strat.trail_lvl = min(strat.trail_lvl, strat.extreme * (1 + strat.trail))
            if acct.dead:
                eq_close.append(0.0)
                continue
            if not done and sub_bars is None and strat.trail > 0:
                if side > 0:
                    strat.extreme = max(strat.extreme, H[i])
                    strat.trail_lvl = max(strat.trail_lvl, strat.extreme * (1 - strat.trail))
                else:
                    strat.extreme = min(strat.extreme, L[i])
                    strat.trail_lvl = min(strat.trail_lvl, strat.extreme * (1 + strat.trail))
        # ---------- BAR_CLOSE
        acct.revalue(C[i])
        if acct.equity <= 0:
            acct.dead = True
            eq_close.append(0.0)
            continue
        eq_close.append(acct.equity)
    if mode == "withdraw":
        months[idx[-1].strftime("%Y-%m")] = (eq_close[-1] * fx.iloc[-1] - e0_inr)
    if trade_open is not None:
        record_trade_close(len(idx) - 1, eq_close[-1])
    return {"equity": pd.Series(eq_close, idx), "months": pd.Series(months), "trades": trades,
            "liquidations": acct.liquidations, "fees": acct.fees, "funding": acct.funding}
