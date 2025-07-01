import sys

# modify this path to match your environment
sys.path.append("/Users/tadas/repos/nelson-fuel-moisture-python/")
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


import onnxruntime as rt
from skl2onnx import to_onnx
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_error

from process_moisture_data import proc_fuel_moisture_UK
from nelson_moisture import nelson_fuel_moisture


def calculate_nelson_moisture(dfr):
    pass


def plot_training_vs_testing(model, X_train, X_test, y_train, y_test):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7), sharey=True)
    axes[0].scatter(
        y_train,
        model.predict(X_train),
    )
    axes[0].scatter(
        y_test,
        model.predict(X_test),
        c="r",
    )
    # axes[0].set_ylim(0, 40)
    # axes[0].set_xlim(0, 40)

    mse_m = mean_squared_error(y_test, model.predict(X_test))
    axes[0].title.set_text(
        f"sklearn FMC R2: {np.round(model.score(X_test, y_test), 2)} MSE: {mse_m}"
    )
    axes[1].scatter(y_train, model.predict(X_train), label="mono")
    axes[1].scatter(y_test, model.predict(X_test), label="mono test")
    mse_mono = mean_squared_error(y_test, model.predict(X_test))
    axes[1].title.set_text(
        f"sklearn FMC R2: {np.round(model.score(X_test, y_test), 2)} MSE: {mse_mono}"
    )
    fig.text(0.5, 0.04, "Observed fmc", ha="center", va="center")
    fig.text(0.06, 0.5, "Predicted fmc", ha="center", va="center", rotation="vertical")
    plt.show()


def training_dataset(fuels: List[str]):
    # Read UK fuel moisture dataset (Birmingham)
    dfr = proc_fuel_moisture_UK()
    # Read features dataset
    fe = pd.read_parquet("data/training_dataset_features.parquet")
    fe["month"] = fe.date.dt.month
    fe["doy"] = fe.date.dt.dayofyear
    fe["hour"] = fe.date.dt.hour

    # Read features required by Nelson model
    weather_features_nelson = pd.read_parquet("data/weather_features_nelson.parquet")
    fe = fe.merge(
        weather_features_nelson[["site", "date", "nelson"]],
        on=["site", "date"],
        how="left",
    )
    dead = [x for x in fe.fuel_type.unique() if "dead" in x]
    # Litter here is dead fuel
    dead.extend(["Litter"])

    # Only dead fuels
    fed = fe[fe.fuel_type.isin(dead)].copy()

    return fed


class DeadFuelMoistureModel:
    def __init__(self):
        self.fuels = ["Bracken", "Gorse", "Heather", "Moor", "Litter"]
        self.y_column = "fmc_%"
        self.fuels_cat_column = "fuel"
        self.model_params = {
            "max_depth": 7,
            "learning_rate": 0.1,
            "min_samples_leaf": 10,
            "max_features": 0.9,
            "loss": "squared_error",
        }
        self.features, self.y_features, self.feature_types = (
            self.prepare_features_and_types()
        )
        self.model = ensemble.HistGradientBoostingRegressor(
            monotonic_cst=[
                self.feature_types[k]["monotonic"] for k in self.feature_types.keys()
            ],
            **self.model_params,
        )

    def prepare_features_and_types(self):
        """Read and prepare features dataset and dictionary
        with feature types and monotonic constrain indicators for fitting"""
        dfr = training_dataset(self.fuels)
        dfr["fuel"] = "Other"
        for fuel in self.fuels:
            dfr.loc[dfr["fuel_type"].str.contains(fuel), "fuel"] = fuel
        fuel_type_column = dfr[[self.fuels_cat_column]]
        encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        fuel_encoded = encoder.fit_transform(fuel_type_column)
        dfr = dfr.join(fuel_encoded)

        # TODO set negative fmc_% values to zero instead?
        dfr = dfr[(dfr["fmc_%"] < 60) & (dfr["fmc_%"] > 0)]

        # Dictionary of feature columns, their dtypes and monotonicity constraints
        features_dict = {
            "vpd": {"type": "float32", "monotonic": -1},
            "vpd-1": {"type": "float32", "monotonic": -1},
            "vpd-2": {"type": "float32", "monotonic": -1},
            "vpd-3": {"type": "float32", "monotonic": -1},
            "vpd-4": {"type": "float32", "monotonic": -1},
            "gti": {"type": "float32", "monotonic": -1},
            "gti-1": {"type": "float32", "monotonic": -1},
            "gti-2": {"type": "float32", "monotonic": -1},
            "gti-3": {"type": "float32", "monotonic": -1},
            "gti-4": {"type": "float32", "monotonic": -1},
            # "smm7": {"type": "float32", "monotonic": 1},
            # "smm28": {"type": "float32", "monotonic": 1},
            # "smm100": {"type": "float32", "monotonic": 1},
            # "slope": {"type": "float32", "monotonic": 0},
            # "aspect": {"type": "float32", "monotonic": 0},
            # "elevation": {"type": "float32", "monotonic": 0},
            # "month": {"type": "float32", "monotonic": 0},
        }

        # add types and constrains for OneHotEncoder fuel categories/columns
        for fuel_name in fuel_encoded.columns:
            features_dict[fuel_name] = {"type": "float32", "monotonic": 0}

        # Select feature columns and cast them to the correct data types
        features = (
            dfr[features_dict.keys()]
            .copy()
            .astype({k: v["type"] for k, v in features_dict.items()})
        )

        return features, dfr[self.y_column], features_dict

    def encode_fuel_features(self, dfr):
        """Encode fuel features using OneHotEncoder"""
        dfr["fuel"] = "Other"
        for fuel in self.fuels:
            dfr.loc[dfr["fuel_type"].str.contains(fuel), "fuel"] = fuel
        fuel_type_column = dfr[[self.fuels_cat_column]]
        encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        fuel_encoded = encoder.fit_transform(fuel_type_column)
        dfr = dfr.join(fuel_encoded)
        # add types and constrains for OneHotEncoder fuel categories/columns
        # for fuel_name in fuel_encoded.columns:
        #     self.feature_types[fuel_name] = {"type": "float32", "monotonic": 0}
        return dfr

    def train_model_full(self):
        # X_train, X_test, y_train, y_test = train_test_split(
        #     self.features,
        #     self.y_features,
        #     test_size=1 / 4,
        # )
        self.model.fit(self.features.values, self.y_features)

    def prepare_training_dataset(self):
        dfr = training_dataset(self.fuels)
        dfr = self.encode_fuel_features(dfr)
        # TODO set negative fmc_% values to zero instead?
        dfr = dfr[(dfr["fmc_%"] < 60) & (dfr["fmc_%"] > 0)]
        return dfr

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.y_features,
            test_size=1 / 4,
        )
        self.model.fit(X_train, y_train)
        plot_training_vs_testing(self.model, X_train, X_test, y_train, y_test)

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
        # features = (
        #     dfr[self.live_features_dict.keys()]
        #     .copy()
        #     .astype({k: v["type"] for k, v in self.live_features_dict.items()})
        # )

        results = []
        predictions = []

        grouped = dfr.groupby(group_cols)
        print("features", self.feature_types.keys())
        for group_key, val_df in grouped:
            print("proc group", group_key)
            train_df = dfr.loc[~dfr.index.isin(val_df.index)]

            X_train = train_df[self.feature_types.keys()]
            y_train = train_df[self.y_column]
            X_val = val_df[self.feature_types.keys()]
            y_val = val_df[self.y_column]

            model_copy = clone(self.model)
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_val)

            pred_df = val_df.copy()
            pred_df["prediction"] = y_pred

            for i, col in enumerate(group_cols):
                pred_df[col] = group_key[i]
            predictions.append(pred_df)
            sl, inter, pearsr, pv, stde = linregress(y_val, y_pred)
            rms = root_mean_squared_error(y_val, y_pred)
            slc, inter2c, pearsrc, pvc, stdec = linregress(y_val, pred_df["nelson"])
            rmsc = root_mean_squared_error(y_val, pred_df["nelson"])
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


if __name__ == "__main__":
    model = DeadFuelMoistureModel()

    re, preds = model.validation_per_location()
    # model.train_model()
    # model.save_model("dead_full.onnx")
    # model.test_saved_model("model_onehot_dead.onnx")
