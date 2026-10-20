"""Train and export the grass-curing greenness model (drop-in for the analytic
grassCuring heuristic). In-scope fuels only: grass (grl/grm/grh -> CEH lc 4,7) and
mixed-heath-high-herb (mhh -> lc 10).

Target: per-cell-normalized cured fraction (robust min-max of EVI2 per cell,
greenest->0%, brownest->100%). Model outputs curing (0..100) DIRECTLY.
Features (3): [ddur, ddur_change, vpdmax-7mean], monotonic [0,0,+1].
Single universal grass-curing curve trained on PURE GRASS (lc 4 improved, 7 acid)
so evergreen-confounded heather-grassland does not bias it; applied to grl/grm/grh
and to mhh alike (mhh's small grass fraction makes curing marginal there).
Outputs model_curing.onnx (+ copy to the app static/wasm) and a validation figure."""
import numpy as np, pandas as pd, shutil
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, root_mean_squared_error as rmse
from skl2onnx import to_onnx

FEATS=["ddur","ddur_change","vpdmax-7mean"]; MONO=[0,0,+1]
OUT="model_curing.onnx"; APP="/Users/tadas/repos/fire-in-site/static/wasm/model_curing.onnx"

d=pd.read_parquet("data/phenology_training_dataset_features.parquet"); d=d.loc[:,~d.columns.duplicated()]
d["date"]=pd.to_datetime(d.date); d["cell"]=d.latind.astype(str)+"_"+d.lonind.astype(str); d["mon"]=d.date.dt.month
d=d[d.lc.isin([4,7])].copy(); d=d[d.EVI2>0].dropna(subset=["ddur","ddur_change","vpdmax-7mean"]).reset_index(drop=True)
def norm(x): p5,p95=np.percentile(x,[5,95]); return np.clip(100*(p95-x)/(p95-p5),0,100)
d["curing"]=d.groupby("cell").EVI2.transform(norm)
X=d[FEATS].astype("float32").values; y=d.curing.values.astype("float32"); g=d.cell.values

# ---- leave-one-cell-out (grouped 5-fold) validation ----
pred=np.full(len(d),np.nan)
for tri,tei in GroupKFold(5).split(X,y,g):
    m=HistGradientBoostingRegressor(monotonic_cst=MONO,learning_rate=0.1,min_samples_leaf=30,
        max_features=1.0,loss="absolute_error",random_state=0).fit(X[tri],y[tri])
    pred[tei]=m.predict(X[tei])
d["pred"]=pred; jja=d.mon.isin([6,7,8])
print(f"leave-cell-out  n={len(d)}  cells={d.cell.nunique()}")
print(f"  ALL   RMSE {rmse(y,pred):.1f}  annR2 {r2_score(y,pred):.3f}  JJAR2 {r2_score(y[jja],pred[jja]):.3f}")
for lc,lbl in [(4,'grass-improved'),(7,'grass-acid')]:
    s=d[d.lc==lc]; print(f"  lc{lc:2d} {lbl:14s} annR2 {r2_score(s.curing,s.pred):.3f}  (n={len(s)})")

# ---- fit final model on ALL data and export ONNX ----
final=HistGradientBoostingRegressor(monotonic_cst=MONO,learning_rate=0.1,min_samples_leaf=30,
    max_features=1.0,loss="absolute_error",random_state=0).fit(X,y)
onx=to_onnx(final,X[:1]); open(OUT,"wb").write(onx.SerializeToString())
try: shutil.copy(OUT,APP); print(f"exported {OUT} -> {APP}")
except Exception as e: print(f"exported {OUT} (app copy skipped: {e})")

# ---- validation figure: held-out time series, predicted curing vs raw EVI2 ----
# Southern grass cell (SE England) where the 2022 drought forced a strong, real
# curing event. Predicted curing (left axis) vs RAW satellite EVI2 greenness (right
# axis, inverted) so that browning (low EVI2) sits high, aligning with high curing.
fig,a2=plt.subplots(figsize=(15,6.5))
cell="67331_1787"   # lat 51.04, improved grassland, South-east
z=d[d.cell==cell].sort_values("date")
RED=(0.75,0.15,0.15); GRN=(0.13,0.50,0.13)
a2.axvspan(pd.Timestamp("2022-07-01"),pd.Timestamp("2022-08-31"),color="orange",alpha=0.10)
l1,=a2.plot(z.date,z.pred,color=RED,lw=2.4,label="Predicted curing (model)")
a2.set_ylim(0,105); a2.set_ylabel("Predicted cured fraction (%)",color=RED,fontsize=15)
a2.tick_params(axis="y",labelcolor=RED,labelsize=13); a2.tick_params(axis="x",labelsize=13)
a2.set_xlim(pd.Timestamp("2021-01-01"),pd.Timestamp("2023-06-30"))
ax2=a2.twinx()
ax2.plot(z.date,z.EVI2/10000,color=GRN,lw=0.8,alpha=0.4)
l2=ax2.scatter(z.date,z.EVI2/10000,s=16,color=GRN,label="Observed EVI2 (greenness)")
ax2.set_ylabel("EVI2 greenness — inverted",color=GRN,fontsize=15); ax2.invert_yaxis()
ax2.tick_params(axis="y",labelcolor=GRN,labelsize=13)
a2.legend(handles=[l1,l2],loc="upper right",fontsize=13)
a2.set_title("Held-out grass cell — predicted curing vs satellite greenness",fontsize=15)
fig.tight_layout(); fig.savefig("figures/curing_greenness_validation.png",dpi=300,bbox_inches="tight")
print("saved figures/curing_greenness_validation.png")
