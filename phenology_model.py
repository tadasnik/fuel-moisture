import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import alpha, linregress

import onnxruntime as rt
from skl2onnx import to_onnx
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay


def plot_predicted_per_year(dfr):
    sns.lineplot(dfr)


def plot_predicted_vs_obs_year_fuel(dfr, model, year, fuel):
    # model = clone(model)
    test = dfr[(dfr["year"] == year) & (dfr["lc"] == fuel)].copy()
    train = dfr[dfr["year"] != year].copy()
    model.train(train)
    test["preds"] = model.predict(test)
    train["preds"] = model.predict(train)
    sl, inter, pearsr, pv, stde = linregress(test["EVI2"], test["preds"])
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    sns.lineplot(y="preds", x="date", data=test[test.lc == fuel], ax=axe, c="blue")
    sns.lineplot(y="EVI2", x="date", data=test[test.lc == fuel], ax=axe, c="red")
    # sns.scatterplot(
    #     y="EVI2", x="clim", data=test[test.fuel_type == fuel], ax=axe, c="orange"
    # )
    # sns.lineplot(
    #     y="EVI2",
    #     x="date",
    #     data=train[train.lc == fuel],
    #     alpha=0.3,
    #     zorder=10,
    #     ax=axe,
    # )
    axe.set_title(f"R2: {pearsr**2:.2f}, PV: {pv:.2e} Size: {test.shape[0]}")
    plt.show()


def partial_dep(model):
    X_train, X_test, y_train, y_test = train_test_split(
        model.features,
        model.y_features,
        test_size=1 / 4,
    )
    features_info = {
        # features of interest
        "features": [e for e in model.phen_features_dict.keys()],
        # type of partial dependence plot
        "kind": "average",
    }
    common_params = {
        "subsample": 50,
        "n_jobs": 2,
        "grid_resolution": 20,
        "random_state": 0,
    }
    print("Computing partial dependence plots...")
    tic = time.time()
    _, ax = plt.subplots(
        ncols=4,
        nrows=np.ceil(len(model.phen_features_dict.keys()) / 4).astype(int),
        figsize=(9, 8),
        constrained_layout=True,
    )
    display = PartialDependenceDisplay.from_estimator(
        model.model,
        X_train,
        **features_info,
        ax=ax,
        **common_params,
    )
    print(f"done in {time.time() - tic:.3f}s")
    _ = display.figure_.suptitle(
        (
            "Partial dependence of the number of bike rentals\n"
            "for the bike rental dataset with a gradient boosting"
        ),
        fontsize=16,
    )
    plt.show()


def plot_predicted_vs_obs_year_fuel_region(dfr, model, year, fuel, region):
    # model = clone(model)
    test = dfr[
        (dfr["year"] == year) & (dfr["lc"] == fuel) & (dfr.region == region)
    ].copy()
    train = dfr[dfr["year"] != year].copy()
    # train = dfr.copy()
    model.train(train)
    test["preds"] = model.predict(test)
    train["preds"] = model.predict(train)
    sl, inter, pearsr, pv, stde = linregress(test["EVI2"], test["preds"])
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    sns.lineplot(
        y="preds",
        x="date",
        data=test[test.lc == fuel],
        ax=axe,
        c="blue",
        errorbar=("pi"),
        label="Predicted",
    )
    sns.lineplot(
        y="EVI2",
        x="date",
        data=test[test.lc == fuel],
        ax=axe,
        c="red",
        label="Observed",
    )
    axe.set_title(f"{region}, {year}, {fuel}, R2: {pearsr**2:.2f}")
    axe.set_ylabel("EVI2")
    plt.savefig(
        f"figures/evi2_pred_{region}_{year}_{fuel}.png", dpi=300, bbox_inches="tight"
    )
    # sns.lineplot(y="EVI2", x="date", data=test[test.lc == fuel], ax=axe, c="red")
    # sns.scatterplot(
    #     y="EVI2", x="clim", data=test[test.fuel_type == fuel], ax=axe, c="orange"
    # )
    # sns.lineplot(
    #     y="EVI2",
    #     x="date",
    #     data=train[train.lc == fuel],
    #     alpha=0.3,
    #     zorder=10,
    #     ax=axe,
    # )
    # axe.set_title(f"R2: {pearsr**2:.2f}, PV: {pv:.2e} Size: {test.shape[0]}")
    plt.show()


def plot_training_vs_testing(model, X_train, X_test, y_train, y_test):
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    axe.scatter(
        y_train,
        model.predict(X_train),
    )
    axe.scatter(
        y_test,
        model.predict(X_test),
        c="r",
    )
    mse_m = root_mean_squared_error(y_test, model.predict(X_test))
    axe.title.set_text(
        f"sklearn FMC R2: {np.round(model.score(X_test, y_test), 2)} MSE: {mse_m}"
    )
    plt.show()


def training_dataset():
    # Read features dataset
    fe = pd.read_parquet("data/phenology_training_dataset_features.parquet")
    fe["year"] = fe.date.dt.year
    fe["doy"] = fe.date.dt.dayofyear
    return fe


def validation_per_year(model, dfr):
    results = []
    summary = []
    for year in dfr["year"].unique():
        test = dfr[dfr["year"] == year].copy()
        train = dfr[dfr["year"] != year].copy()
        model.train(train)
        test["preds"] = model.predict(test)
        train["preds"] = model.predict(train)
        results.append(test)
        sl, inter, pearsr, pv, stde = linregress(test["EVI2"], test["preds"])
        rms = root_mean_squared_error(test["EVI2"], test["preds"])

        summary.append(
            {
                "year": year,
                "r2": pearsr**2,
                "rmse": rms,
                "pv": pv,
            }
        )
    sum_res = pd.DataFrame(summary)
    res = pd.concat(results, ignore_index=True)
    return res, sum_res


class PhenologyModel:
    def __init__(self):
        self.fuels_live = [3, 4, 7, 9, 10]
        self.y_column = "EVI2"
        self.fuels_cat_column = "lc"
        # Dictionary of feature columns, their dtypes and monotonicity constraints
        self.phen_features_dict = {
            # "tmean-7mean": {"type": "float32", "monotonic": 0},
            "tmean-15mean": {"type": "float32", "monotonic": 0},
            # "tmean-15m": {"type": "float32", "monotonic": 0},
            # "tmean-26m": {"type": "float32", "monotonic": 0},
            # "tmax-3m": {"type": "float32", "monotonic": 0},
            # "tmax-7max": {"type": "float32", "monotonic": 0},
            # "tmax-15max": {"type": "float32", "monotonic": 0},
            # "tmax-26m": {"type": "float32", "monotonic": 0},
            # "sdur-3m": {"type": "float32", "monotonic": 0},
            # "sdur-7m": {"type": "float32", "monotonic": 0},
            # "sdur-15m": {"type": "float32", "monotonic": 0},
            # "sri-3m": {"type": "float32", "monotonic": 0},
            "sri-15mean": {"type": "float32", "monotonic": 0},
            # "sri-15m": {"type": "float32", "monotonic": 0},
            # "sri-26m": {"type": "float32", "monotonic": 0},
            # "prec-3s": {"type": "float32", "monotonic": 0},
            # "prec-15sum": {"type": "float32", "monotonic": 0},
            # "prec-15s": {"type": "float32", "monotonic": 0},
            # "doy": {"type": "float32", "monotonic": 0},
            # "vpdmax-3max": {"type": "float32", "monotonic": 0},
            # "vpdmax": {"type": "float32", "monotonic": 0},
            # "vpdmax-1": {"type": "float32", "monotonic": 0},
            # "vpdmax-3max": {"type": "float32", "monotonic": 0},
            "vpdmax-7max": {"type": "float32", "monotonic": -1},
            "vpdmax-15mean": {"type": "float32", "monotonic": -1},
            # "vpdmax-26m": {"type": "float32", "monotonic": 0},
            # "doy": {"type": "float32", "monotonic": 0},
            "smm28": {"type": "float32", "monotonic": 0},
            "smm100": {"type": "float32", "monotonic": 0},
            "ddur": {"type": "float32", "monotonic": 0},
            "ddur_change": {"type": "float32", "monotonic": 0},
        }
        # for day in range(1, 15):
        #     self.phen_features_dict[f"tmean-{day}"] = {
        #         "type": "float32",
        #         "monotonic": 0,
        #     }
        #     self.phen_features_dict[f"prec-{day}"] = {
        #         "type": "float32",
        #         "monotonic": 0,
        #     }
        #     self.phen_features_dict[f"sdur-{day}"] = {
        #         "type": "float32",
        #         "monotonic": 0,
        #     }

        self.model_params = {
            "learning_rate": 0.1,
            "min_samples_leaf": 30,
            "max_features": 1.0,
            "loss": "absolute_error",
        }
        self.model = ensemble.HistGradientBoostingRegressor(
            monotonic_cst=[
                self.phen_features_dict[k]["monotonic"]
                for k in self.phen_features_dict.keys()
            ],
            **self.model_params,
        )

    def train(self, dfr: pd.DataFrame):
        features = dfr[self.phen_features_dict.keys()]
        self.model.fit(features.values.astype(np.float32), dfr[self.y_column].values)

    def predict(self, dfr: pd.DataFrame) -> pd.DataFrame:
        features = dfr[self.phen_features_dict.keys()]
        lfmc = self.model.predict(features.values.astype(np.float32))
        return lfmc

    def prepare_training_dataset(self):
        dfr = training_dataset()
        dfr = dfr.dropna(subset=["sri", "ddur"])
        dfr = dfr[dfr.smm100 > 0.01].copy()
        dfr = dfr[dfr.EVI2 > 0].copy()
        dfr = self.encode_fuel_features(dfr)
        return dfr

    def encode_fuel_features(self, dfr):
        """Encode fuel features using OneHotEncoder"""
        encoder = OneHotEncoder(
            sparse_output=False,
        ).set_output(transform="pandas")
        fuel_encoded = encoder.fit_transform(dfr[[self.fuels_cat_column]])
        for fuel in self.fuels_live:
            if self.fuels_cat_column + "_" + str(fuel) not in fuel_encoded.columns:
                fuel_encoded[self.fuels_cat_column + "_" + str(fuel)] = 0.0
        dfr = dfr.join(fuel_encoded)
        # add types and constrains for OneHotEncoder fuel categories/columns
        for fuel_name in fuel_encoded.columns:
            self.phen_features_dict[fuel_name] = {"type": "float32", "monotonic": 0}
        return dfr

    def prepare_features_and_types(self):
        """Read and prepare features dataset and dictionary
        with feature types and monotonic constrain indicators for fitting"""
        dfr = self.prepare_training_dataset()
        # Dictionary of feature columns, their dtypes and monotonicity constraints
        # for day in range(1, 15):
        #     features_dict[f"smm7-{day}"] = {"type": "float32", "monotonic": 1}
        # for day in range(1, 15):
        #     features_dict[f"smm28-{day}"] = {"type": "float32", "monotonic": 1}
        # for day in range(1, 15):
        #     features_dict[f"smm100-{day}"] = {"type": "float32", "monotonic": 1}

        # Select feature columns and cast them to the correct data types
        #
        features = (
            dfr[self.phen_features_dict.keys()]
            .copy()
            .astype({k: v["type"] for k, v in self.phen_features_dict.items()})
        )
        return features, dfr[self.y_column]

    def validation_train_model(self, drf: pd.DataFrame):
        X_train, X_test, y_train, y_test = train_test_split(
            dfr[self.phen_features_dict.keys()],
            dfr[self.y_column],
            test_size=1 / 4,
        )
        self.model.fit(X_train, y_train)
        plot_training_vs_testing(self.model, X_train, X_test, y_train, y_test)

    def train_model(self, dfr: pd.DataFrame):
        self.model.fit(dfr[self.phen_features_dict.keys()], dfr[self.y_column])

    def save_model(self, file_name: str):
        onx = to_onnx(self.model, self.features[:1].values.astype(np.float32))
        with open(file_name, "wb") as f:
            f.write(onx.SerializeToString())

    def predict_evi2_live_moisture(self, dfr: pd.DataFrame) -> pd.DataFrame:
        """Predict EVI2 live moisture using the model"""
        fuels_live_to_phenology = {
            "Bracken live leaves": 7,
            "Bracken live stem": 7,
            "Gorse live canopy": 10,
            "Gorse live stem": 10,
            "Heather live canopy": 9,
            "Heather live stem": 9,
            "Moor grass live": 3,
        }
        dfr["lc"] = dfr.copy()["fuel_type"].map(fuels_live_to_phenology)
        dfr = self.encode_fuel_features(dfr)
        features = dfr[self.phen_features_dict.keys()]
        lfmc = self.model.predict(features.values.astype(np.float32))
        dfr["EVI2"] = lfmc
        return dfr


if __name__ == "__main__":
    model = PhenologyModel()
    # model.validation_train_model()
    # model.train_model()
    dfr = model.prepare_training_dataset()
    # model.validation_train_model()

    # model.train_model()
    # model.save_model("phenology_model.onnx")

    # plot_predicted_vs_obs_year_fuel(dfr, model, 2018, 9)
    # re, preds = model.validation_per_location(group_cols=["site"])
    # re, preds = model.validation_per_location_per_fuel(
    #     dfr=dfr, fuel="Heather live canopy", group_cols=["site"]
    # )
    # res, preds = validation_per_year(model, dfr)
    # plot_predicted_per_year(preds)
