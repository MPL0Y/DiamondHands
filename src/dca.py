"""DCA benchmark: buy ₹10,000 of BTC on CoinDCX spot (0.59% incl. GST + 0.05% slippage) at each month's first
daily open, 2018-01 → 2025-09 (dev window). Reports XIRR and the monthly ₹ change in portfolio value net of contributions."""
import numpy as np, pandas as pd
from scipy.optimize import brentq
from src.data import bars
from src.venues import COINDCX_SPOT as V

def main():
    d = bars("1d"); d = d[d.index >= "2018-01-01"]
    firsts = d.groupby(d.month_id).head(1)
    btc = 0.0; flows = []; vals = []
    for t, r in d.iterrows():
        if t in firsts.index:
            usd = 10_000 / r.fx
            btc += usd * (1 - V.taker) / (r.open * (1 + V.slippage))
            flows.append((t, -10_000.0))
        vals.append(btc * r.close * r.fx)
    v = pd.Series(vals, d.index)
    final = v.iloc[-1] * (1 - V.taker - V.slippage)
    flows.append((d.index[-1], final))
    yrs = np.array([(t - flows[0][0]).days / 365.25 for t, _ in flows]); cf = np.array([c for _, c in flows])
    irr = brentq(lambda r: (cf / (1 + r) ** yrs).sum(), -0.99, 10)
    me = v.resample("ME").last(); contrib = pd.Series(10_000.0, me.index)
    mp = me.diff().fillna(me.iloc[0]) - contrib
    out = {"months": len(me), "invested_inr": 10_000 * len(me), "final_value_inr": final, "xirr": irr,
           "median_month_change_ex_contrib_inr": float(mp.median()), "worst_month_inr": float(mp.min()),
           "max_dd_of_value": float((1 - v / v.cummax()).max())}
    print(out)
    return out

if __name__ == "__main__":
    main()
