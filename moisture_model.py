from operator import mod
import sys
import scipy
from sklearn.metrics.pairwise import rbf_kernel
import lightgbm as lgb
from sklearn.inspection import PartialDependenceDisplay

# modify this path to match your environment
sys.path.append("/Users/tadas/repos/nelson-python/")

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from nelson_moisture import nelson_fuel_moisture
from process_moisture_data import proc_fuel_moisture_UK
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.utils.fixes import parse_version
from skl2onnx import to_onnx

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
# axes.scatter(dfr_live.date, dfr_live["fmc_%"],
#     c = dfr_live['2m_relative_humidity_15h_%'])
# plt.show()
#
# dfr = proc_fuel_moisture_UK()
# live_mask = [x for x in dfr.fuel_type.unique() if "live" in x]
# dfr_live = dfr[dfr.fuel_type.isin(live_mask)]
# dfr_live = dfr_live[dfr_live.month == 7]


dfr = proc_fuel_moisture_UK()
fe = pd.read_parquet("data/features_All_1.parquet")

prec_cols = [col for col in fe.columns if "prec" in col]
# fe = fe.drop(prec_cols, axis=1)
# prec_sum = fe[prec_cols].sum(axis=1)

# vpd_cols = [col for col in fe.columns if "vpd" in col]
# vpd_mean = fe[vpd_cols].mean(axis=1)
# fe = dfr.copy()

fe["month"] = fe.date.dt.month
fe["doy"] = fe.date.dt.dayofyear
fe["hour"] = fe.date.dt.hour

# fe["week"] = fe.date.dt.isocalendar().week
vpss = pd.read_parquet("data/weather_features_nelson.parquet")
fe = fe.merge(vpss[["site", "date", "nelson"]], on=["site", "date"], how="left")
dead = [x for x in fe.fuel_type.unique() if "dead" in x]
# dead = [x for x in fe.fuel_type.unique() if "dead" in x]
fed = fe[fe.fuel_type.isin(dead)].copy()

fuels = ["Bracken", "Gorse", "Heather", "Moor"]
fed["fuel"] = 0
for nr, fuel in enumerate(fuels):
    fed.loc[fed["fuel_type"].str.contains(fuel), "fuel"] = nr

# fuel_type_column = fed[["fuel"]]
# encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
# fuel_encoded = encoder.fit_transform(fuel_type_column)
# fed = fed.join(fuel_encoded)

fed = fed[(fed["fmc_%"] < 40) & (fed["fmc_%"] > 0)]
fed_features = fed[
    [
        "vpd_20",
        "vpd_21",
        "vpd_22",
        "vpd_23",
        "vpd_24",
        "gti_20",
        "gti_21",
        "gti_22",
        "gti_23",
        "gti_24",
        "slope",
        "aspect",
        "elevation",
        "month",
        "fuel",
    ]
]
X_train, X_test, y_train, y_test = train_test_split(
    fed_features,
    # fed.drop(
    #     ["site", "lonind", "latind", "date", "fmc_%", "fuel_type", "fuel", "nelson"],
    #     axis=1,
    # ),
    # fed.drop(
    #     [
    #         "site",
    #         "lonind",
    #         "latind",
    #         "date",
    #         "longitude",
    #         "latitude",
    #         "climate_region_of_uk",
    #         "lcm_land_cover",
    #         "soil_type",
    #         "igbp_land_cover_id",
    #         "igbp_land_cover",
    #         "outlier_removed",
    #         "year",
    #         "week",
    #         "fmc_%",
    #         "fuel_type",
    #         "species_name",
    #         "nelson",
    #     ],
    #     axis=1,
    # ).fillna(0),
    fed["fmc_%"],
    test_size=1 / 6,
)

params = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "loss": "squared_error",
}
params_mono = {
    "max_depth": 7,
    "learning_rate": 0.1,
    "min_samples_leaf": 10,
    "max_features": 0.9,
    "loss": "squared_error",
}
# model = ensemble.GradientBoostingRegressor(**params)
# model = ensemble.HistGradientBoostingRegressor(**params)
model_mono = ensemble.HistGradientBoostingRegressor(
    monotonic_cst=[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
    categorical_features=["fuel"],
    **params_mono,
)

# scaler = StandardScaler().fit(X_train)

# X_train.values = scaler.transform(X_train)
# X_test.values = scaler.transform(X_test)
lw = 2
# model.fit(X_train, y_train)
model_mono.fit(X_train, y_train)
fig, ax = plt.subplots()
features = [4, 13]
disp = PartialDependenceDisplay.from_estimator(
    model_mono,
    X_train,
    features=features,
    line_kw={"linewidth": 4, "label": "unconstrained", "color": "tab:blue"},
    ax=ax,
)
PartialDependenceDisplay.from_estimator(
    model_mono,
    X_train,
    features=features,
    line_kw={"linewidth": 4, "label": "constrained", "color": "tab:orange"},
    ax=disp.axes_,
)
for f_idx in (0, 1):
    disp.axes_[0, f_idx].plot(
        X_train.values[:, features[f_idx]],
        y_train,
        "o",
        alpha=0.3,
        zorder=-1,
        color="tab:green",
    )
    disp.axes_[0, f_idx].set_ylim(-5, 40)

plt.legend()
fig.suptitle("Monotonic constraints effect on partial dependences")
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), sharey=True)
axes[0].scatter(
    y_train,
    model_mono.predict(X_train),
)
axes[0].scatter(
    y_test,
    model_mono.predict(X_test),
    c="r",
)
axes[0].set_ylim(0, 40)
axes[0].set_xlim(0, 40)

mse_m = mean_squared_error(y_test, model_mono.predict(X_test))
axes[0].title.set_text(
    f"sklearn FMC R2: {np.round(model_mono.score(X_test, y_test), 2)} MSE: {mse_m}"
)
axes[1].scatter(y_train, model_mono.predict(X_train), label="mono")
axes[1].scatter(y_test, model_mono.predict(X_test), label="mono test")
mse_mono = mean_squared_error(y_test, model_mono.predict(X_test))
axes[1].title.set_text(
    f"sklearn FMC R2: {np.round(model_mono.score(X_test, y_test), 2)} MSE: {mse_mono}"
)
fig.text(0.5, 0.04, "Observed fmc", ha="center", va="center")
fig.text(0.06, 0.5, "Predicted fmc", ha="center", va="center", rotation="vertical")
# expr = skompile(model.predict)
# sql = expr.to("sqlalchemy/sqlite")
# fig.suptitle("SVR vs Nelson's FMC", fontsize=14)
plt.show()
# onx = to_onnx(model_mono, X_train[:1].values)
# with open("mono_model.onnx", "wb") as f:
#     f.write(onx.SerializeToString())

"""
# onx = to_onnx(model, X[:1])
# with open("gbm_model.onnx", "wb") as f:
#     f.write(onx.SerializeToString())
#
fed_features_gbm = fed[
    [
        "vpd_21",
        "vpd_22",
        "vpd_23",
        "vpd_24",
        "gti_21",
        "gti_22",
        "gti_23",
        "gti_24",
        "fuel",
    ]
].copy()
X_train, X_test, y_train, y_test = train_test_split(
    fed_features_gbm,
    fed["fmc_%"],
    test_size=1 / 4,
)
params_l = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 16,
    "max_depth": 4,
    "learning_rate": 0.1,
    "importance_type": "gain",
    "feature_fraction": 0.8,
    # "mc": [-1, -1, -1, -1, -1, -1, -1, -1, 0],
}

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=-1)
test_data = lgb.Dataset(
    X_test, label=y_test, reference=train_data, categorical_feature=-1
)
num_round = 100
bst = lgb.train(params_l, train_data, num_round, valid_sets=[test_data])
# X = lgb.Dataset(X_train, y_train)
# parameters = {"n_estimators": 100, "max_depth": 4, "random_state": 43}
# bm = lgb.train(params_l, train_set=X)
ypred = bst.predict(X_test)


slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_test, ypred)

mse_n = mean_squared_error(y_test, ypred)
# axes[0].set_ylim(0, 50)
# axes[0].set_xlim(0, 50)
# axes[1].scatter(fed["fmc_%"], fed.nelson, label="Nelson model")

axes[1].scatter(y_train, bst.predict(X_train), label="gbm")
axes[1].scatter(y_test, ypred, label="gbm")
axes[1].title.set_text(f"Nelson's FMC R2: {np.round(r_value**2, 2)} MSE: {mse_n}")


fig.text(0.5, 0.04, "Observed fmc", ha="center", va="center")
fig.text(0.06, 0.5, "Predicted fmc", ha="center", va="center", rotation="vertical")
# expr = skompile(model.predict)
# sql = expr.to("sqlalchemy/sqlite")
# fig.suptitle("SVR vs Nelson's FMC", fontsize=14)
plt.show()
disp = PartialDependenceDisplay.from_estimator(
    model,
    X_test,
    features=[15, 16],
    feature_names=(
        "First feature",
        "Second feature",
    ),
    line_kw={"linewidth": 4, "label": "unconstrained", "color": "tab:blue"},
)
"""
