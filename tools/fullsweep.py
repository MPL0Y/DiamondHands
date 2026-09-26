import sys, pandas as pd, numpy as np
from src.quick_eval import quick
ids = sys.argv[1].split(","); mults = tuple(float(x) for x in sys.argv[2].split(","))
with open("reports/lockbox_access.log","a") as f: f.write(f"{pd.Timestamp.now(tz='UTC')} FULL-PERIOD sweep {ids}\n")
rows=[]
for cid in ids:
    f=quick(cid, mults, period="full"); l=quick(cid, mults, period="lockbox")
    for m in mults:
        a=f[(f.mult==m)&(~f.lotfree)].iloc[0]; b=f[(f.mult==m)&(f.lotfree)].iloc[0]; c=l[(l.mult==m)&(~l.lotfree)].iloc[0]; d=l[(l.mult==m)&(l.lotfree)].iloc[0]
        rows.append(dict(id=cid,mult=m,full_cagr=a.cagr,full_dd=a.dd,full_sharpe=a.sharpe,full_final_inr=a.final_inr,lf_cagr=b.cagr,lf_dd=b.dd,last_yr=c.ret,last_yr_lf=d.ret,last_dd=c.dd,liq=a.liq))
t=pd.DataFrame(rows)
old = pd.read_csv("reports/combo_full_period.csv")
pd.concat([old[~old.id.isin(ids)], t]).to_csv("reports/combo_full_period.csv", index=False)
pd.set_option("display.width",220)
ok=t[t.full_dd<=0.5].sort_values("full_cagr",ascending=False)
print(ok.groupby("id").head(2).round(3).to_string(index=False))
