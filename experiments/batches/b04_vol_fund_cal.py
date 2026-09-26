from src.experiment import run, next_id
M = {"1d": 1, "4h": 6, "1h": 24, "15m": 96}
HV = "Volatility: range expansion starts moves (Crabel/Williams volatility breakout). Trend works better in low-vol regimes (vol-managed portfolios, Moreira & Muir 2017)."
HF = "Funding: extreme perp funding marks crowded positioning, which then reverses; paying high carry erodes long returns (Schmeling et al. 2023 crypto carry)."
HC = "Calendar: BTC shows intraday and weekday seasonality (e.g. US-session and weekend effects; Baur et al. 2019; Aharon & Qadan 2019)."
ex = []
for tf in ["1h", "4h", "1d", "15m"]:
    m = M[tf]
    for side in ["long", "both"]:
        ex.append(("volatility", HV, dict(strategy="vol_breakout", tf=tf, grid={"n": [14, 30], "k": [1.0, 1.5, 2.0, 2.5], "hold": [max(1, m // 4), m, 3 * m]}, fixed={"side": side})))
for tf in ["1h", "4h", "1d"]:
    m = M[tf]
    for mode in ["trend_lowvol", "trend_highvol", "lowvol_long"]:
        ex.append(("volatility", HV, dict(strategy="vol_regime", tf=tf, grid={"fast": [10 * m, 20 * m], "slow": [50 * m, 100 * m], "vol_q": [0.5, 0.7, 0.85]},
                                          fixed={"mode": mode, "vol_n": 20 * max(1, m // 6), "q_window": 365 * m})))
for tf in ["1h", "4h", "1d"]:
    m = M[tf]
    for side in ["long", "short", "both"]:
        ex.append(("funding", HF, dict(strategy="funding_contra", tf=tf, grid={"window": [7, 21, 60], "z_in": [1.5, 2.0, 2.5, 3.0], "hold": [m, 3 * m, 7 * m]}, fixed={"side": side})))
    ex.append(("funding", HF, dict(strategy="funding_level", tf=tf, grid={"hi": [0.0001, 0.0002, 0.0004, 0.001], "fast": [10 * m, 20 * m], "slow": [50 * m, 100 * m]}, fixed={"base": "trend"})))
    ex.append(("funding", HF, dict(strategy="funding_level", tf=tf, grid={"hi": [0.00005, 0.0001, 0.0002, 0.0004, 0.001], "lo": [-1, 0.0]}, fixed={"base": "none"})))
hour_sets = [tuple(range(0, 8)), tuple(range(8, 16)), tuple(range(16, 24)), tuple(range(13, 21)), tuple(range(21, 24)) + tuple(range(0, 3)), tuple(range(0, 24, 1))]
for side in ["long", "short"]:
    ex.append(("calendar", HC, dict(strategy="calendar", tf="1h", grid={"hours": hour_sets}, fixed={"side": side})))
for side in ["long", "short"]:
    ex.append(("calendar", HC, dict(strategy="calendar", tf="1d", grid={"weekdays": [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (5, 6), (0, 1, 2, 3, 4), (0, 1), (3, 4)]}, fixed={"side": side})))
for h in hour_sets[:5]:
    ex.append(("calendar", HC, dict(strategy="calendar", tf="1h", grid={"weekdays": [(0, 1, 2, 3, 4), (5, 6), (0, 1, 2, 3, 4, 5, 6)]}, fixed={"side": "long", "hours": h})))
for tf in ["1h", "4h", "1d"]:
    m = M[tf]
    ex.append(("calendar", HC, dict(strategy="weekend_flat", tf=tf, grid={"fast": [5 * m, 10 * m, 20 * m], "slow": [50 * m, 100 * m, 200 * m]})))
for tf in ["4h", "15m"]:
    for side in ["long", "short"]:
        ex.append(("calendar", HC, dict(strategy="calendar", tf=tf, grid={"hours": [(0,), (4,), (8,), (12,), (16,), (20,)] if tf == "4h" else hour_sets}, fixed={"side": side})))
for fam, h, e in ex:
    run({"id": next_id(), "family": fam, "lev": 1.0, "venue": "delta_india_perp", "hypothesis": h, **e})
