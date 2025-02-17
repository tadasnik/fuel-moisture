import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from process_moisture_data import proc_fuel_moisture_UK
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.utils.fixes import parse_version
from skl2onnx import convert_sklearn
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dfr = proc_fuel_moisture_UK()
fe = pd.read_parquet("data/features_All_1.parquet")

prec_cols = [col for col in fe.columns if "prec" in col]
fe["prec_sum"] = fe[prec_cols].sum(axis=1)

fe["month"] = fe.date.dt.month
fe["doy"] = fe.date.dt.dayofyear
fe["hour"] = fe.date.dt.hour

vpss = pd.read_parquet("data/weather_features_nelson.parquet")
fe = fe.merge(vpss[["site", "date", "nelson"]], on=["site", "date"], how="left")
dead = [x for x in fe.fuel_type.unique() if "dead" in x]
# dead = [x for x in fe.fuel_type.unique() if "dead" in x]
fed = fe[fe.fuel_type.isin(dead)].copy()

fuels = ["Bracken", "Gorse", "Heather", "Moor"]
fed["fuel"] = 0
for nr, fuel in enumerate(fuels):
    fed[fuel] = 0
    fed.loc[fed["fuel_type"].str.contains(fuel), fuel] = 1

fuel_type_column = fed[["fuel"]]
encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
fuel_encoded = encoder.fit_transform(fuel_type_column)
fed = fed.join(fuel_encoded)
features_dict = {
    "vpd_20": {"type": "float32", "monotonic": -1},
    "vpd_21": {"type": "float32", "monotonic": -1},
    "vpd_22": {"type": "float32", "monotonic": -1},
    "vpd_23": {"type": "float32", "monotonic": -1},
    "vpd_24": {"type": "float32", "monotonic": -1},
    "gti_20": {"type": "float32", "monotonic": -1},
    "gti_21": {"type": "float32", "monotonic": -1},
    "gti_22": {"type": "float32", "monotonic": -1},
    "gti_23": {"type": "float32", "monotonic": -1},
    "gti_24": {"type": "float32", "monotonic": -1},
    "slope": {"type": "float32", "monotonic": 0},
    "aspect": {"type": "float32", "monotonic": 0},
    "elevation": {"type": "float32", "monotonic": 0},
    "month": {"type": "float32", "monotonic": 0},
    "Bracken": {"type": "float32", "monotonic": 0},
    "Gorse": {"type": "float32", "monotonic": 0},
    "Heather": {"type": "float32", "monotonic": 0},
    "Moor": {"type": "float32", "monotonic": 0},
    # "fuel": {"type": "float32", "monotonic": 0},
}

fed = fed[(fed["fmc_%"] < 60) & (fed["fmc_%"] > 0)]
fed_features = (
    fed[features_dict.keys()]
    .copy()
    .astype({k: v["type"] for k, v in features_dict.items()})
)
X_train, X_test, y_train, y_test = train_test_split(
    fed_features,
    fed["fmc_%"],
    test_size=1 / 6,
)
# categorical_transformer = Pipeline(
#     [("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))]
# )
# preprocessor = ColumnTransformer(
#     transformers=[("cat", categorical_transformer, ["fuel"])], remainder="passthrough"
# )
params_mono = {
    "max_depth": 7,
    "learning_rate": 0.1,
    "min_samples_leaf": 10,
    "max_features": 0.9,
    "loss": "squared_error",
}

# model = ensemble.HistGradientBoostingRegressor(
#     **params_mono,
# )

model = ensemble.HistGradientBoostingRegressor(
    monotonic_cst=[features_dict[k]["monotonic"] for k in features_dict.keys()],
    **params_mono,
)

# pipe = Pipeline([("preprocess", preprocessor), ("hgbr", model)])
# pipe.fit(X_train, y_train)
model.fit(X_train, y_train)
onx = to_onnx(model, X_train[:1].values.astype(np.float32))
with open("model_onehot_dead.onnx", "wb") as f:
    f.write(onx.SerializeToString())


sess = rt.InferenceSession("model_onehot_dead.onnx", providers=["CPUExecutionProvider"])
pred_ort = sess.run(None, {"X": X_test.values.astype(np.float32)})[0]

pred_skl = model.predict(X_test.values.astype(np.float32))

print("Onnx Runtime prediction:\n", pred_ort[:5])
print("Sklearn rediction:\n", pred_skl[:5])
plt.scatter(pred_ort, pred_skl)
plt.show()
