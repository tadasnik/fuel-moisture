import sys

# modify this path to match your environment
sys.path.append("/Users/tadas/repos/nelson-fuel-moisture-python/")
from typing import List

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

from process_moisture_data import proc_fuel_moisture_UK
from nelson_moisture import nelson_fuel_moisture

clim_moist = {
    "Heather": {
        1: 90,
        2: 90,
        3: 75,
        4: 75,
        5: 110,
        6: 110,
        7: 110,
        8: 110,
        9: 90,
        10: 90,
        11: 90,
        12: 90,
    },
    "Moor": {
        1: 85,
        2: 85,
        3: 85,
        4: 85,
        5: 135,
        6: 135,
        7: 135,
        8: 115,
        9: 115,
        10: 115,
        11: 85,
        12: 85,
    },
    "Gorse": {
        1: 140,
        2: 140,
        3: 140,
        4: 95,
        5: 140,
        6: 190,
        7: 190,
        8: 140,
        9: 140,
        10: 140,
        11: 140,
        12: 140,
    },
    "Bracken": {
        1: 85,
        2: 85,
        3: 85,
        4: 85,
        5: 135,
        6: 135,
        7: 135,
        8: 115,
        9: 115,
        10: 115,
        11: 85,
        12: 85,
    },
}


def climatology(dfr: pd.DataFrame) -> pd.DataFrame:
    clims = []
    for nr, row in dfr.iterrows():
        clims.append(clim_moist[row["fuel"]][row["month"]])

    dfr["clim"] = clims
    return dfr


def r2_rmse(g, meas_col="fmc_%", pred_col="prediction"):
    sl, inter, pearsr, pv, stde = linregress(g[meas_col], g[pred_col])
    rmse = np.sqrt(root_mean_squared_error(g[meas_col], g[pred_col]))
    return pd.Series({"r2": pearsr**2, "rmse": rmse, "pv": pv})


def calculate_nelson_moisture(dfr):
    pass


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
    # axes[0].set_ylim(0, 40)
    # axes[0].set_xlim(0, 40)

    mse_m = root_mean_squared_error(y_test, model.predict(X_test))
    axe.title.set_text(
        f"sklearn FMC R2: {np.round(model.score(X_test, y_test), 2)} MSE: {mse_m}"
    )
    # axes[1].scatter(y_train, model.predict(X_train), label="mono")
    # axes[1].scatter(y_test, model.predict(X_test), label="mono test")
    # mse_mono = mean_squared_error(y_test, model.predict(X_test))
    # axes[1].title.set_text(
    #     f"sklearn FMC R2: {np.round(model.score(X_test, y_test), 2)} MSE: {mse_mono}"
    # )
    # fig.text(0.5, 0.04, "Observed fmc", ha="center", va="center")
    # fig.text(0.06, 0.5, "Predicted fmc", ha="center", va="center", rotation="vertical")
    plt.show()


def plot_predicted_vs_obs_site_fuel(dfr, site, model, fuel):
    # model = clone(model)
    test = dfr[(dfr["site"] == site) & (dfr["fuel"] == fuel)].copy()
    train = dfr[dfr["site"] != site].copy()
    model.train(train)
    test["preds"] = model.predict(test)
    train["preds"] = model.predict(train)
    sl, inter, pearsr, pv, stde = linregress(test["fmc_%"], test["preds"])
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    # sns.scatterplot(y='preds', x='fmc_%', hue='fuel_type', data=test[test.fuel == fuel], ax=axe)
    sns.scatterplot(
        y="preds", x="fmc_%", hue="fuel_type", data=test[test.fuel == fuel], ax=axe
    )
    sns.scatterplot(
        y="preds",
        x="fmc_%",
        hue="fuel_type",
        data=train[train.fuel == fuel],
        alpha=0.3,
        ax=axe,
    )

    # axe.scatter(
    #     train[train.fuel == fuel]["fmc_%"],
    #     preds_train[train.fuel == fuel],
    #     label="Train",
    #     alpha=0.5,
    # )
    # axe.scatter(test["fmc_%"], preds_test, label="Train", c="r", alpha=0.5)
    axe.set_title(f"R2: {pearsr**2:.2f} PV: {pv:.2e} Size: {test.shape[0]}")
    plt.show()


def plot_predicted_vs_obs(dfr, model):
    preds = model.predict(dfr)
    sl, inter, pearsr, pv, stde = linregress(dfr["fmc_%"], preds)
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    axe.scatter(dfr["fmc_%"], preds)
    axe.set_title(f"R2: {pearsr**2:.2f} PV: {pv:.2e}")
    plt.show()


def training_dataset():
    # Read features dataset
    fe = pd.read_parquet("data/training_dataset_features.parquet")
    fe["month"] = fe.date.dt.month
    fe["doy"] = fe.date.dt.dayofyear
    fe["hour"] = fe.date.dt.hour

    # Read features required by Nelson model
    # weather_features_nelson = pd.read_parquet("data/weather_features_nelson.parquet")
    # fe = fe.merge(
    #     weather_features_nelson[["site", "date", "nelson"]],
    #     on=["site", "date"],
    #     how="left",
    # )
    fe["fuel_cat"] = "other"
    for cat in ["live", "dead"]:
        fe.loc[fe["fuel_type"].str.contains(cat), "fuel_cat"] = cat

    fe.loc[fe["fuel_type"] == "Litter", "fuel_cat"] = "dead"
    # live = [x for x in fe.fuel_type.unique() if "live" in x]
    # Litter here is live fuel

    # Only live fuels
    # fed = fe[fe.fuel_type.isin(live)].copy()

    return fe[fe.fuel_cat == "live"].copy()


class FuelMoistureModel:
    def __init__(self):
        self.fuels_live = [
            "Bracken",
            "Gorse",
            "Heather live canopy",
            "Heather live stem",
            "Moor",
        ]
        self.y_column = "fmc_%"
        self.fuels_cat_column = "fuel"
        self.model_params = {
            "max_depth": 7,
            "learning_rate": 0.1,
            "min_samples_leaf": 10,
            "max_features": 0.9,
            "loss": "squared_error",
        }
        # Dictionary of feature columns, their dtypes and monotonicity constraints
        self.live_features_dict = {
            # "vpd": {"type": "float32", "monotonic": -1},
            # "vpd-1": {"type": "float32", "monotonic": -1},
            # "vpd-2": {"type": "float32", "monotonic": -1},
            # "vpd-3": {"type": "float32", "monotonic": -1},
            # "vpd-4": {"type": "float32", "monotonic": -1},
            # "gti": {"type": "float32", "monotonic": -1},
            # "gti-1": {"type": "float32", "monotonic": -1},
            # "gti-2": {"type": "float32", "monotonic": -1},
            # "gti-3": {"type": "float32", "monotonic": -1},
            # "gti-4": {"type": "float32", "monotonic": -1},
            # "smm7": {"type": "float32", "monotonic": 1},
            "smm28": {"type": "float32", "monotonic": 0},
            "smm100": {"type": "float32", "monotonic": 0},
            "slope": {"type": "float32", "monotonic": 0},
            "aspect": {"type": "float32", "monotonic": 0},
            "elevation": {"type": "float32", "monotonic": 0},
            # "month": {"type": "float32", "monotonic": 0},
            "ddur": {"type": "float32", "monotonic": 0},
            # "sdur": {"type": "float32", "monotonic": 0},
        }
        for day in range(1, 7):
            self.live_features_dict[f"vpd-{day}d"] = {
                "type": "float32",
                "monotonic": -1,
            }
        (
            self.features,
            self.y_features,
        ) = self.prepare_features_and_types()
        self.model = ensemble.HistGradientBoostingRegressor(
            monotonic_cst=[
                self.live_features_dict[k]["monotonic"]
                for k in self.live_features_dict.keys()
            ],
            **self.model_params,
        )

    def encode_fuel_features(self, dfr):
        """Encode fuel features using OneHotEncoder"""
        dfr["fuel"] = "Other"
        for fuel in self.fuels_live:
            dfr.loc[dfr["fuel_type"].str.contains(fuel), "fuel"] = fuel
        fuel_type_column = dfr[[self.fuels_cat_column]]
        encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        fuel_encoded = encoder.fit_transform(fuel_type_column)
        dfr = dfr.join(fuel_encoded)
        # add types and constrains for OneHotEncoder fuel categories/columns
        for fuel_name in fuel_encoded.columns:
            self.live_features_dict[fuel_name] = {"type": "float32", "monotonic": 0}
        return dfr

    def prepare_training_dataset(self):
        dfr = training_dataset()
        dfr = self.encode_fuel_features(dfr)
        # TODO set negative fmc_% values to zero instead?
        dfr = dfr[(dfr["fmc_%"] < 300) & (dfr["fmc_%"] > 0)]
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
            dfr[self.live_features_dict.keys()]
            .copy()
            .astype({k: v["type"] for k, v in self.live_features_dict.items()})
        )
        return features, dfr[self.y_column]

    def train_model(self):
        # X_train, X_test, y_train, y_test = train_test_split(
        #     self.features,
        #     self.y_features,
        #     test_size=1 / 4,
        # )
        self.model.fit(self.features.values, self.y_features)

    def validation_train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.y_features,
            test_size=1 / 4,
        )
        self.model.fit(X_train, y_train)
        plot_training_vs_testing(self.model, X_train, X_test, y_train, y_test)

    def validation_per_location(self, group_cols: List[str] = ["lonind", "latind"]):
        """
        Perform spatial cross-validation using unique (lonind, latind) groups.

        Parameters:
            group_cols (list): Columns used for grouping (default ['lonind', 'latind']).

        Returns:
            results (pd.DataFrame): Per-group scores.
            all_predictions (pd.DataFrame): DataFrame with true/predicted values for each group.
        """
        # fets = pd.read_parquet("data/weather_site_features.parquet")
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
            dfr[self.live_features_dict.keys()]
            .copy()
            .astype({k: v["type"] for k, v in self.live_features_dict.items()})
        )

        results = []
        predictions = []

        grouped = dfr.groupby(group_cols)
        print("features", self.live_features_dict.keys())
        for group_key, val_df in grouped:
            print("proc group", group_key)
            train_df = dfr.loc[~dfr.index.isin(val_df.index)]
            X_train = train_df[self.live_features_dict.keys()]
            y_train = train_df[self.y_column]
            X_val = val_df[self.live_features_dict.keys()]
            y_val = val_df[self.y_column]
            model_copy = clone(self.model)
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_val)
            pred_df = val_df.copy()
            pred_df["prediction"] = y_pred
            pred_df = climatology(pred_df)
            for i, col in enumerate(group_cols):
                pred_df[col] = group_key[i]
            predictions.append(pred_df)
            sl, inter, pearsr, pv, stde = linregress(y_val, y_pred)
            rms = root_mean_squared_error(y_val, y_pred)
            slc, inter2c, pearsrc, pvc, stdec = linregress(y_val, pred_df["clim"])
            rmsc = root_mean_squared_error(y_val, pred_df["clim"])
            group_result = {
                "group": group_key,
                "r2": pearsr**2,
                "rmse": rms,
                "r2c": pearsrc**2,
                "rmsec": rmsc,
                "pv": pv,
                "pvc": pvc,
                "size": X_val.shape[0],
            }
            results.append(group_result)
        return pd.DataFrame(results), pd.concat(predictions)

    def save_model(self, file_name: str):
        onx = to_onnx(self.model, self.features[:1].values.astype(np.float32))
        with open(file_name, "wb") as f:
            f.write(onx.SerializeToString())

    def test_saved_model(self, file_name: str):
        sess = rt.InferenceSession(file_name, providers=["CPUExecutionProvider"])
        pred_ort = sess.run(None, {"X": self.features.values.astype(np.float32)})[0]
        pred_skl = self.model.predict(self.features.values.astype(np.float32))
        print("Onnx Runtime prediction:\n", pred_ort[:5])
        print("Sklearn rediction:\n", pred_skl[:5])
        plt.scatter(pred_ort, pred_skl)
        plt.show()

    def train(self, dfr: pd.DataFrame):
        features = dfr[self.live_features_dict.keys()]
        self.model.fit(features.values.astype(np.float32), dfr[self.y_column].values)

    def predict(self, dfr: pd.DataFrame) -> pd.DataFrame:
        features = dfr[self.live_features_dict.keys()]
        lfmc = self.model.predict(features.values.astype(np.float32))
        return lfmc


if __name__ == "__main__":
    model = FuelMoistureModel()
    # model.validation_train_model()
    # model.train_model()
    dfr = model.prepare_training_dataset()

    # plot_predicted_vs_obs(dfr, model)
    # re, preds = model.validation_per_location()
    # model.save_model("live_full_ddur.onnx")
# model.train_model()
# model.save_model("model_onehot_dead.onnx")
# preds.groupby('fuel').apply(r2_rmse).reset_index()
