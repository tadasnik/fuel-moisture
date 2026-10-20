"""Train + export the minimal LFMC model to ONNX for FireInSite.
Features (fixed order): vpdmax-7mean, ddur, ddur_change, then 7 live-fuel one-hots.
Output: live fuel moisture content (%).

VPD is the 7-day trailing mean of daily-max VPD (not same-day). Leave-one-
location-out validation showed this matches or slightly beats same-day VPD on
skill (RMSE 32.76 vs 33.05; July 31.95 vs 34.01) while producing a temporally
smooth LFMC series consistent with the slow autocorrelation of live fuels.
The app must feed slot 0 the same 7-day trailing mean at inference time."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from skl2onnx import to_onnx
import onnxruntime as rt

DATA = "data/training_dataset_features_full.parquet"
OUT = "lfmc_min.onnx"

FUELS = ["Bracken leaves", "Bracken stem", "Gorse canopy", "Gorse stem",
         "Heather canopy", "Heather stem", "Moor grass"]
FUEL_MAP = {
    "Bracken live leaves": "Bracken leaves", "Bracken live stem": "Bracken stem",
    "Gorse live canopy": "Gorse canopy", "Gorse live stem": "Gorse stem",
    "Heather live canopy": "Heather canopy", "Heather live stem": "Heather stem",
    "Moor grass live": "Moor grass",
}
BASE = ["vpdmax-7mean", "ddur", "ddur_change"]
MONO = [-1, 0, 0] + [0] * len(FUELS)  # VPD decreasing; season & fuel unconstrained
FEATURES = BASE + [f"fuel_{f}" for f in FUELS]

d = pd.read_parquet(DATA)
d = d.loc[:, ~d.columns.duplicated()].copy()
d = d[d.fuel_type.isin(FUEL_MAP)].copy()
d["fuel"] = d.fuel_type.map(FUEL_MAP)
d = d[(d["fmc_%"] > 30) & (d["fmc_%"] < 300)].dropna(subset=["fmc_%"] + BASE).reset_index(drop=True)
for f in FUELS:
    d[f"fuel_{f}"] = (d["fuel"] == f).astype("float32")

X = d[FEATURES].astype("float32").values
y = d["fmc_%"].values.astype("float32")
print(f"training samples: {len(d)}   features ({len(FEATURES)}): {FEATURES}")

model = HistGradientBoostingRegressor(
    loss="squared_error", learning_rate=0.1, min_samples_leaf=10,
    max_features=1.0, monotonic_cst=MONO,
)
model.fit(X, y)
print(f"in-sample: RMSE={root_mean_squared_error(y, model.predict(X)):.2f}  R2={r2_score(y, model.predict(X)):.3f}")

# export
onx = to_onnx(model, X[:1])
with open(OUT, "wb") as f:
    f.write(onx.SerializeToString())

# verify ONNX == sklearn
sess = rt.InferenceSession(OUT, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
pred_onnx = sess.run(None, {in_name: X.astype(np.float32)})[0].ravel()
pred_skl = model.predict(X)
print(f"ONNX input='{in_name}' shape={sess.get_inputs()[0].shape}  output='{out_name}'")
print(f"ONNX vs sklearn max abs diff: {np.max(np.abs(pred_onnx - pred_skl)):.4f}")

# example predictions per fuel at a mid-summer dry spell (7-day mean VPD)
print("\nexample predictions (vpdmax-7mean=2.5 kPa, ddur=57000s ~15.8h, ddur_change=-50):")
for i, f in enumerate(FUELS):
    row = [2.5, 57000, -50] + [1.0 if j == i else 0.0 for j in range(len(FUELS))]
    p = sess.run(None, {in_name: np.array([row], dtype=np.float32)})[0].ravel()[0]
    print(f"  {f:16s} -> {p:5.0f}% FMC")
print(f"\nsaved: {OUT}")
