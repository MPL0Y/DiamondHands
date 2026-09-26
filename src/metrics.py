"""Performance metrics, INR conversion, Monte Carlo, DSR and PBO."""
import math
import numpy as np
import pandas as pd
from scipy import stats

from src.config import START_CAPITAL_INR, TARGET_MONTHLY_INR


def daily_returns(eq: pd.Series) -> pd.Series:
    d = eq.resample("1D").last().ffill()
    r = d.pct_change().fillna(0.0)
    r[d.shift(1) <= 0] = 0.0
    r[(d <= 0) & (d.shift(1) > 0)] = -1.0
    return r


def max_dd(eq):
    eq = np.asarray(eq, float)
    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, 1 - eq / peak, 0)
    return float(np.nanmax(dd)) if len(dd) else 0.0


def perf(eq_usd: pd.Series, fx: pd.Series, m_pnl_inr: np.ndarray, month_index, trades, lotfree_m_pnl=None):
    r = daily_returns(eq_usd)
    n_days = len(r)
    yrs = n_days / 365.25
    final = eq_usd.iloc[-1] / eq_usd.iloc[0] if eq_usd.iloc[0] > 0 else 0
    cagr = final ** (1 / yrs) - 1 if final > 0 else -1.0
    sd = r.std()
    sharpe = r.mean() / sd * math.sqrt(365) if sd > 0 else 0.0
    dn = r[r < 0].std()
    sortino = r.mean() / dn * math.sqrt(365) if dn and dn > 0 else 0.0
    mdd = max_dd(eq_usd.values)
    calmar = cagr / mdd if mdd > 0 else 0.0
    wm = pd.Series(m_pnl_inr, index=month_index).dropna()
    med = float(wm.median()) if len(wm) else float("nan")
    pct1l = float((wm >= TARGET_MONTHLY_INR).mean()) if len(wm) else 0.0
    # compounding mode, INR
    eq_inr = eq_usd * fx
    me = eq_inr.resample("ME").last()
    ms = pd.concat([pd.Series([eq_inr.iloc[0]], index=[eq_inr.index[0]]), me]).diff().dropna()
    first_1l = next((str(k.date())[:7] for k, v in ms.items() if v >= TARGET_MONTHLY_INR), None)
    after = ms[ms.index >= pd.Timestamp(first_1l + "-01", tz="UTC")] if first_1l else ms.iloc[0:0]
    consist = float((after >= TARGET_MONTHLY_INR).mean()) if len(after) else 0.0
    lf = pd.Series(lotfree_m_pnl if lotfree_m_pnl is not None else m_pnl_inr).dropna()
    med_ret = float(lf.median()) / START_CAPITAL_INR
    req_cap = TARGET_MONTHLY_INR / med_ret if med_ret > 0 else float("inf")
    tr = np.asarray(trades)
    return {
        "oos_cagr": cagr, "sharpe": sharpe, "sortino": sortino, "max_dd": mdd, "calmar": calmar,
        "median_month_inr": med, "mean_month_inr": float(wm.mean()) if len(wm) else float("nan"),
        "pct_months_1L": pct1l, "worst_month_inr": float(wm.min()) if len(wm) else float("nan"),
        "median_month_ret_lotfree": med_ret, "req_capital_inr": req_cap,
        "compound_first_1L_month": first_1l, "compound_consistency_after": consist,
        "final_equity_inr": float(eq_inr.iloc[-1]), "n_trades": int(len(tr)),
        "win_rate": float((tr[:, 2] > 0).mean()) if len(tr) else 0.0,
        "n_days": n_days, "skew": float(stats.skew(r)), "kurt": float(stats.kurtosis(r, fisher=False)),
    }


def monte_carlo(r_daily: np.ndarray, trade_rets: np.ndarray, runs=5000, horizon=365, block=10, seed=7):
    """Stationary block bootstrap of daily returns (12-month paths) + trade-order reshuffle."""
    rng = np.random.default_rng(seed)
    r = np.asarray(r_daily, float)
    n = len(r)
    ruin = dd50 = 0
    terminal = np.empty(runs)
    mdds = np.empty(runs)
    for k in range(runs):
        idx = np.empty(horizon, dtype=np.int64)
        i = 0
        while i < horizon:
            s = rng.integers(0, n)
            L = min(rng.geometric(1 / block), horizon - i)
            idx[i:i + L] = (s + np.arange(L)) % n
            i += L
        path = np.cumprod(1 + r[idx])
        terminal[k] = path[-1]
        pk = np.maximum.accumulate(np.concatenate([[1.0], path]))
        mdds[k] = np.max(1 - np.concatenate([[1.0], path]) / pk)
        ruin += path.min() < 0.10
        dd50 += mdds[k] >= 0.5
    out = {"mc_p_ruin": ruin / runs, "mc_p_dd50": dd50 / runs, "mc_p5_terminal": float(np.percentile(terminal, 5)),
           "mc_median_terminal": float(np.median(terminal)), "mc_p95_dd": float(np.percentile(mdds, 95))}
    tr = np.asarray(trade_rets, float)
    if len(tr) > 5:
        tmdd = np.empty(runs)
        tr = np.clip(tr, -1, None)
        for k in range(runs):
            p = np.cumprod(1 + rng.permutation(tr))
            pk = np.maximum.accumulate(np.concatenate([[1.0], p]))
            tmdd[k] = np.max(1 - np.concatenate([[1.0], p]) / pk)
        out["tr_p_dd50"] = float((tmdd >= 0.5).mean())
        out["tr_p95_dd"] = float(np.percentile(tmdd, 95))
    return out


def deflated_sharpe(sr_daily, n_obs, skew, kurt, n_trials, var_sr_trials):
    """Bailey & López de Prado (2014). sr_daily: non-annualised SR of the selected strategy."""
    if n_trials <= 1 or var_sr_trials <= 0:
        sr0 = 0.0
    else:
        g = 0.5772156649
        z1 = stats.norm.ppf(1 - 1 / n_trials)
        z2 = stats.norm.ppf(1 - 1 / (n_trials * math.e))
        sr0 = math.sqrt(var_sr_trials) * ((1 - g) * z1 + g * z2)
    den = math.sqrt(max(1e-12, 1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily ** 2))
    return float(stats.norm.cdf((sr_daily - sr0) * math.sqrt(n_obs - 1) / den)), sr0


def pbo_cscv(R: np.ndarray, S=16, max_combos=4000, seed=3):
    """Probability of Backtest Overfitting via CSCV. R: T x N matrix of per-period returns of N configs."""
    from itertools import combinations
    T, N = R.shape
    if N < 2:
        return float("nan")
    blocks = np.array_split(np.arange(T), S)
    combos = list(combinations(range(S), S // 2))
    rng = np.random.default_rng(seed)
    if len(combos) > max_combos:
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    bm = np.array([R[b].mean(0) for b in blocks])
    bs = np.array([R[b].std(0) for b in blocks])
    bn = np.array([len(b) for b in blocks])
    logits = []
    for c in combos:
        ins = np.zeros(S, bool); ins[list(c)] = True
        def sr(mask):
            w = bn[mask][:, None]
            mu = (bm[mask] * w).sum(0) / w.sum()
            var = ((bs[mask] ** 2 + bm[mask] ** 2) * w).sum(0) / w.sum() - mu ** 2
            return mu / np.sqrt(np.maximum(var, 1e-18))
        si, so = sr(ins), sr(~ins)
        best = np.argmax(si)
        rank = stats.rankdata(so)[best] / (N + 1)
        logits.append(math.log(rank / (1 - rank)))
    return float((np.array(logits) <= 0).mean())
