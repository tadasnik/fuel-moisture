import sys
import scipy
from sklearn.metrics.pairwise import rbf_kernel

# modify this path to match your environment
sys.path.append("/Users/tadas/repos/nelson-python/")

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, SGDRegressor
import matplotlib.pyplot as plt
from nelson_moisture import nelson_fuel_moisture
from process_moisture_data import proc_fuel_moisture_UK
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.utils.fixes import parse_version
from skompiler import skompile

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
# axes.scatter(dfr_live.date, dfr_live["fmc_%"],
#     c = dfr_live['2m_relative_humidity_15h_%'])
# plt.show()
#
# dfr = proc_fuel_moisture_UK()
# live_mask = [x for x in dfr.fuel_type.unique() if "live" in x]
# dfr_live = dfr[dfr.fuel_type.isin(live_mask)]
# dfr_live = dfr_live[dfr_live.month == 7]


# dfr = proc_fuel_moisture_UK()
fe = pd.read_parquet("data/features_All.parquet")
# prec_cols = [col for col in fe.columns if "prec" in col]
# prec_sum = fe[prec_cols].sum(axis=1)

# vpd_cols = [col for col in fe.columns if "vpd" in col]
# vpd_mean = fe[vpd_cols].mean(axis=1)
# fe = dfr.copy()

fe["month"] = fe.date.dt.month
# fe["week"] = fe.date.dt.isocalendar().week
vpss = pd.read_parquet("data/weather_features_nelson.parquet")
fe = fe.merge(vpss[["site", "date", "nelson"]], on=["site", "date"], how="left")
dead = [x for x in fe.fuel_type.unique() if "dead" in x]
# dead = [x for x in fe.fuel_type.unique() if "dead" in x]
fed = fe[fe.fuel_type.isin(dead)]

fuel_type_column = fed[["fuel_type"]]
encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
fuel_encoded = encoder.fit_transform(fuel_type_column)
fed = fed.join(fuel_encoded)

fed = fed[(fed["fmc_%"] < 60) & (fed["fmc_%"] > 0)]
X_train, X_test, y_train, y_test = train_test_split(
    fed.drop(
        ["site", "lonind", "latind", "date", "fmc_%", "fuel_type", "nelson"], axis=1
    ),
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
    test_size=1 / 4,
)

params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}
model = ensemble.GradientBoostingRegressor(**params)

# model = SVR(kernel="rbf", C=1, gamma=0.001, epsilon=0.1)
# model = SVR(kernel="linear", C=10, gamma=0.1)

# pipe = make_pipeline(StandardScaler(), model)
# pipe.fit(X_train, y_train)
# print(pipe.score(X_test, y_test))


# svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
svrs = [model]  # , svr_lin]
kernel_label = ["RBF", "Linear"]
model_color = ["m", "c", "g"]

scaler = StandardScaler().fit(X_train)
X = scaler.transform(X_train)
y = y_train.values
X_t = scaler.transform(X_test)
yt = y_test.values
lw = 2
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), sharey=True)
# for ix, svr in enumerate(svrs):
axes[0].scatter(
    y,
    model.fit(X, y).predict(X),
)
axes[0].scatter(
    yt,
    model.predict(X_t),
    c="r",
)

# axes[ix].scatter(
#     X[svr.support_],
#     y[svr.support_],
#     facecolor="none",
#     edgecolor=model_color[ix],
#     s=50,
#     label="{} support vectors".format(kernel_label[ix]),
# )
# axes[0].legend(
#     loc="upper center",
#     bbox_to_anchor=(0.5, 1.1),
#     ncol=1,
#     fancybox=True,
#     shadow=True,
# )
mse_m = mean_squared_error(yt, model.predict(X_t))
axes[0].title.set_text(f"SVR FMC R2: {np.round(model.score(X_t, yt), 2)} MSE: {mse_m}")

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
    fed["fmc_%"], fed.nelson
)

mse_n = mean_squared_error(fed["fmc_%"], fed.nelson)
# axes[0].set_ylim(0, 50)
# axes[0].set_xlim(0, 50)
# axes[1].scatter(fed["fmc_%"], fed.nelson, label="Nelson model")
axes[1].scatter(fed["fmc_%"], fed.nelson, label="Nelson model")
axes[1].title.set_text(f"Nelson's FMC R2: {np.round(r_value**2, 2)} MSE: {mse_n}")

# axes[1].set_ylim(0, 50)
# axes[1].set_xlim(0, 50)

fig.text(0.5, 0.04, "Observed fmc", ha="center", va="center")
fig.text(0.06, 0.5, "Predicted fmc", ha="center", va="center", rotation="vertical")
# expr = skompile(model.predict)
# sql = expr.to("sqlalchemy/sqlite")
# fig.suptitle("SVR vs Nelson's FMC", fontsize=14)
plt.show()

import lightgbm as lgb

params_l = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
}
train_data = lgb.Dataset(X, label=y)
test_data = lgb.Dataset(X_t, label=yt, reference=train_data)
num_round = 100
bst = lgb.train(params_l, train_data, num_round, valid_sets=[test_data])
