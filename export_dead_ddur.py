"""Train + export the deployed dFMC model.
- 'month' replaced by continuous daylength ddur + ddur_change (Cooper 1969):
  smoother, higher-skill, hemisphere-generalisable.
- Terrain (slope/aspect/elevation) DROPPED: GTI is already tilt/azimuth-adjusted
  so it carries the slope/aspect drying signal, and per-site-constant terrain
  descriptors leaked site identity and hurt leave-one-site-out generalisation.
- 5th fuel category 'Litter' added (from the 'Litter' field data): previously the
  app's litter fuels borrowed the Bracken prediction (biased ~10% too dry, R2<0);
  a dedicated category cuts litter RMSE 15.8->11.6 and R2 -0.12->0.40 while
  leaving the four vegetation categories' predictions essentially unchanged.

Feature order (MUST match the app's input array + deadFuelsCategories order):
  vpd_20..24, gti_20..24, ddur, ddur_change, Bracken, Gorse, Heather, Moor, Litter
  (17 features)
Output: dead FMC (%).
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from skl2onnx import to_onnx
import onnxruntime as rt

OUT="model_onehot_dead.onnx"

def daylight_seconds(lat, date):
    n=pd.Timestamp(date).dayofyear
    phi=np.radians(lat); decl=0.4092797*np.sin(2*np.pi*(284+n)/365)
    cosH=np.clip(-np.tan(phi)*np.tan(decl),-1,1)
    return (24*np.arccos(cosH)/np.pi)*3600

fe=pd.read_parquet("data/features_All_1.parquet"); fe=fe.loc[:,~fe.columns.duplicated()]
fe["date"]=pd.to_datetime(fe.date)
# 7 dead-vegetation fuel types + forest litter (own category)
dead=[x for x in fe.fuel_type.unique() if "dead" in x]
d=fe[fe.fuel_type.isin(dead+["Litter"])].copy()
d=d[(d["fmc_%"]>0)&(d["fmc_%"]<60)].reset_index(drop=True)
# category one-hot (order must match app deadFuelsCategories): Bracken, Gorse,
# Heather, Moor, Litter. Litter is the 'Litter' fuel_type; the rest by name.
d["Litter"]=(d.fuel_type=="Litter").astype("float32")
for f in ["Bracken","Gorse","Heather","Moor"]:
    d[f]=((d.fuel_type.str.contains(f)) & (d.fuel_type!="Litter")).astype("float32")

# site -> latitude (features_All_1 has no coords) and astronomical daylength
site_lat=pd.read_parquet("data/training_dataset_features_full.parquet")
site_lat=site_lat.loc[:,~site_lat.columns.duplicated()].groupby("site")["latitude"].median()
d["lat"]=d.site.map(site_lat)
d=d.dropna(subset=["lat"]).reset_index(drop=True)
d["ddur"]=[daylight_seconds(la,dt) for la,dt in zip(d.lat,d.date)]
d["ddur_change"]=[daylight_seconds(la,dt-pd.Timedelta(days=1)) for la,dt in zip(d.lat,d.date)]-d["ddur"]

FEATURES=([f"vpd_2{i}" for i in range(5)]+[f"gti_2{i}" for i in range(5)]
          +["ddur","ddur_change","Bracken","Gorse","Heather","Moor","Litter"])
MONO=[-1]*10+[0,0,0,0,0,0,0]       # VPD & GTI decreasing; season/fuel unconstrained
assert len(FEATURES)==len(MONO)==17

d=d.dropna(subset=FEATURES+["fmc_%"]).reset_index(drop=True)
X=d[FEATURES].astype("float32").values
y=d["fmc_%"].values.astype("float32")
print(f"training samples: {len(d)}   features ({len(FEATURES)}): {FEATURES}")

model=HistGradientBoostingRegressor(monotonic_cst=MONO, max_depth=7, learning_rate=0.1,
                                    min_samples_leaf=10, max_features=0.9, loss="squared_error")
model.fit(X,y)
from sklearn.metrics import root_mean_squared_error,r2_score
print(f"in-sample: RMSE={root_mean_squared_error(y,model.predict(X)):.2f}  R2={r2_score(y,model.predict(X)):.3f}")

onx=to_onnx(model, X[:1])
with open(OUT,"wb") as f: f.write(onx.SerializeToString())
sess=rt.InferenceSession(OUT,providers=["CPUExecutionProvider"])
inn=sess.get_inputs()[0].name; out=sess.get_outputs()[0].name
diff=np.max(np.abs(sess.run(None,{inn:X.astype(np.float32)})[0].ravel()-model.predict(X)))
print(f"ONNX input='{inn}' shape={sess.get_inputs()[0].shape}  output='{out}'  ONNX-vs-sklearn max diff {diff:.4f}")
print(f"saved {OUT}")
