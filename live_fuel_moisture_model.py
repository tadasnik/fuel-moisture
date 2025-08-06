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

from phenology_model import PhenologyModel

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


def climatology_actual(dfr: pd.DataFrame) -> pd.DataFrame:
    cli = dfr.groupby(["month", "fuel_type"])["fmc_%"].mean().reset_index()
    cli.rename(columns={"fmc_%": "clim"}, inplace=True)
    dfr = dfr.merge(cli, on=["month", "fuel_type"], how="left")
    return dfr


def r2_rmse(g, meas_col="fmc_%", pred_col="clim"):
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


def plot_predicted_vs_obs_test_case(train, test, model, fuel):
    # model = clone(model)
    model.train(train)
    test["preds"] = model.predict(test)
    train["preds"] = model.predict(train)
    rms = root_mean_squared_error(test["fmc_%"], test["preds"])
    rmc = root_mean_squared_error(test["fmc_%"], test["clim"])
    sl, inter, pearsr, pv, stde = linregress(test["fmc_%"], test["preds"])
    slc, inter2c, pearsrc, pvc, stdec = linregress(test["fmc_%"], test["clim"])
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    # sns.scatterplot(y='preds', x='fmc_%', hue='fuel_type', data=test[test.fuel == fuel], ax=axe)
    sns.scatterplot(y="fmc_%", x="preds", data=test, hue="fuel_type", ax=axe)
    sns.scatterplot(y="fmc_%", x="clim", data=test, hue="fuel_type", ax=axe, marker="x")

    axe.set_ylim(50, 160)
    axe.set_xlim(50, 160)

    # axe.scatter(
    #     train[train.fuel == fuel]["fmc_%"],
    #     preds_train[train.fuel == fuel],
    #     label="Train",
    #     alpha=0.5,
    # )
    # axe.scatter(test["fmc_%"], preds_test, label="Train", c="r", alpha=0.5)
    axe.set_title(
        f"R2: {pearsr**2:.2f}, R2C: {pearsrc**2:.2f}, PV: {pv:.2e} RMS: {rms}, RMSC: {rmc} Size: {test.shape[0]}"
    )
    plt.show()


def plot_predicted_vs_obs_site_fuel(dfr, site, model, fuel):
    # model = clone(model)
    test = dfr[(dfr["site"] == site) & (dfr["fuel_type"] == fuel)].copy()
    train = dfr[dfr["site"] != site].copy()
    model.train(train)
    test["preds"] = model.predict(test)
    train["preds"] = model.predict(train)
    sl, inter, pearsr, pv, stde = linregress(test["fmc_%"], test["preds"])
    slc, inter2c, pearsrc, pvc, stdec = linregress(test["fmc_%"], test["clim"])
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    # sns.scatterplot(y='preds', x='fmc_%', hue='fuel_type', data=test[test.fuel == fuel], ax=axe)
    sns.scatterplot(
        y="fmc_%", x="preds", data=test[test.fuel_type == fuel], ax=axe, c="blue"
    )
    sns.scatterplot(
        y="fmc_%", x="clim", data=test[test.fuel_type == fuel], ax=axe, c="orange"
    )

    sns.scatterplot(
        y="preds",
        x="fmc_%",
        data=train[train.fuel_type == fuel],
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
    axe.set_title(
        f"R2: {pearsr**2:.2f}, R2C: {pearsrc**2:.2f}, PV: {pv:.2e} Size: {test.shape[0]}"
    )
    plt.show()


def plot_predicted_vs_obs(dfr, model):
    preds = model.predict(dfr)
    sl, inter, pearsr, pv, stde = linregress(dfr["fmc_%"], preds)
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    axe.scatter(dfr["fmc_%"], preds)
    axe.set_title(f"R2: {pearsr**2:.2f} PV: {pv:.2e}")
    plt.show()


def testing_dataset_dorset_surrey():
    test = pd.read_parquet("data/training_dataset_features_dorset_surrey.parquet")
    live = test[test["Live/dead"] == "live"].copy()
    live["Component"] = live["Component"].str.replace("tips", "canopy", regex=False)
    live["Component"] = live["Component"].str.replace("stems", "stem", regex=False)
    live["Plant"] = live["Plant"].str.replace("Erica", "Heather", regex=False)
    live["Plant"] = live["Plant"].str.replace("Calluna", "Heather", regex=False)
    live["fuel_type"] = live["Plant"] + " live " + live["Component"]
    return live


def testing_dataset_uob_2025():
    uob = pd.read_parquet("data/training_dataset_features_uob_2025.parquet")
    uob = uob[uob["fuel"].str.contains("Calluna")]
    uob["fuel"] = uob["fuel"].replace(
        {"Calluna canopy": "Heather live canopy", "Calluna stems": "Heather live stem"},
    )
    uob = uob.rename({"fuel": "fuel_type"}, axis=1)
    return uob


def training_dataset():
    # Read features dataset
    # fe = pd.read_parquet("data/training_dataset_features_evi.parquet")
    fe = pd.read_parquet("data/training_dataset_features.parquet")
    fe["month"] = fe.date.dt.month
    fe["doy"] = fe.date.dt.dayofyear
    fe["hour"] = fe.date.dt.hour
    fe["fuel_cat"] = "other"
    for cat in ["live", "dead"]:
        fe.loc[fe["fuel_type"].str.contains(cat), "fuel_cat"] = cat
    fe.loc[fe["fuel_type"] == "Litter", "fuel_cat"] = "dead"
    return fe[fe.fuel_cat == "live"].copy()


class FuelMoistureModel:
    def __init__(self):
        self.fuels_live = [
            "Bracken live leaves",
            "Bracken live stem",
            "Gorse live canopy",
            "Gorse live stem",
            "Heather live canopy",
            "Heather live stem",
            "Moor grass live",
        ]
        self.y_column = "fmc_%"
        self.fuels_cat_column = "fuel_type"
        self.model_params = {
            "learning_rate": 0.1,
            "min_samples_leaf": 10,
            "max_features": 1.0,
            "loss": "squared_error",
        }
        # Dictionary of feature columns, their dtypes and monotonicity constraints
        self.live_features_dict = {
            "vpdmax-10max": {"type": "float32", "monotonic": -1},
            "vpdmax-15mean": {"type": "float32", "monotonic": -1},
            "EVI2": {"type": "float32", "monotonic": 0},
            # "smm28": {"type": "float32", "monotonic": 0},
            "smm100": {"type": "float32", "monotonic": 0},
            # "slope": {"type": "float32", "monotonic": 0},
            # "aspect": {"type": "float32", "monotonic": 0},
            # "elevation": {"type": "float32", "monotonic": 0},
            "doy": {"type": "float32", "monotonic": 0},
            # "ddur": {"type": "float32", "monotonic": 0},
            # "ddur_change": {"type": "float32", "monotonic": 0},
            # "sdur": {"type": "float32", "monotonic": 0},
        }
        # add types and constrains for OneHotEncoder fuel categories/columns
        for fuel_name in self.fuels_live:
            self.live_features_dict[self.fuels_cat_column + "_" + fuel_name] = {
                "type": "float32",
                "monotonic": 0,
            }

        self.model = ensemble.HistGradientBoostingRegressor(
            monotonic_cst=[
                self.live_features_dict[k]["monotonic"]
                for k in self.live_features_dict.keys()
            ],
            **self.model_params,
        )

    def encode_fuel_features(self, dfr):
        """Encode fuel features using OneHotEncoder"""
        encoder = OneHotEncoder(
            sparse_output=False,
        ).set_output(transform="pandas")
        fuel_encoded = encoder.fit_transform(dfr[[self.fuels_cat_column]])
        for fuel in self.fuels_live:
            if self.fuels_cat_column + "_" + fuel not in fuel_encoded.columns:
                fuel_encoded[self.fuels_cat_column + "_" + fuel] = 0.0

        try:
            dfr = dfr.join(fuel_encoded)
        except ValueError as e:
            print("fuel_type encoded columns exist")
            return dfr
        # add types and constrains for OneHotEncoder fuel categories/columns
        for fuel_name in fuel_encoded.columns:
            self.live_features_dict[fuel_name] = {"type": "float32", "monotonic": 0}
        return dfr

    def encode_fuel_features_old(self, dfr):
        """Encode fuel features using OneHotEncoder"""
        dfr["fuel"] = "Other"
        for fuel in self.fuels_live:
            dfr.loc[dfr["fuel_type"].str.contains(fuel), "fuel"] = fuel
        print("Unique fuel types:", dfr["fuel"].unique())
        fuel_type_column = dfr[[self.fuels_cat_column]]
        encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        fuel_encoded = encoder.fit_transform(fuel_type_column)
        dfr = dfr.join(fuel_encoded)
        # add types and constrains for OneHotEncoder fuel categories/columns
        for fuel_name in fuel_encoded.columns:
            self.live_features_dict[fuel_name] = {"type": "float32", "monotonic": 0}
        # chesk if all columns are present
        for fuel in self.fuels_live:
            if fuel not in dfr.columns:
                dfr["fuel_" + fuel] = 0.0
        return dfr

    def prepare_training_dataset(self):
        dfr = training_dataset()
        # dfr = pd.read_parquet("data/live_training_dataset_evi.parquet")
        dfr = self.encode_fuel_features(dfr)
        # TODO set negative fmc_% values to zero instead?
        dfr = dfr[(dfr["fmc_%"] < 300) & (dfr["fmc_%"] > 0)]
        dfr = climatology_actual(dfr)
        return dfr

    def prepare_test_dataset(self):
        training_dataset = self.prepare_training_dataset()
        test = testing_dataset_uob_2025()
        test2 = testing_dataset_dorset_surrey()
        test = pd.concat([test, test2], ignore_index=True)
        test = self.encode_fuel_features(test)
        cli = (
            training_dataset.groupby(["month", "fuel_type"])["fmc_%"]
            .mean()
            .reset_index()
        )
        cli.rename(columns={"fmc_%": "clim"}, inplace=True)
        test = test.merge(cli, on=["month", "fuel_type"], how="left")
        return test

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

    def validation_per_location_per_fuel(
        self, dfr, fuel, group_cols: List[str] = ["lonind", "latind"]
    ):
        """
        Perform spatial cross-validation using unique (lonind, latind) groups.

        Parameters:
            group_cols (list): Columns used for grouping (default ['lonind', 'latind']).

        Returns:
            results (pd.DataFrame): Per-group scores.
            all_predictions (pd.DataFrame): DataFrame with true/predicted values for each group.
        """
        results = []
        predictions = []

        grouped = dfr.groupby(group_cols)
        print("features", self.live_features_dict.keys())
        for group_key, val_df in grouped:
            if val_df[val_df.fuel_type == fuel].shape[0] < 20:
                continue  # Skip groups with too few samples
            else:
                print(
                    "proc group",
                    group_key,
                    "size",
                    val_df[val_df.fuel_type == fuel].shape,
                )
            pred_df = val_df[val_df.fuel_type == fuel].copy()
            train_df = dfr.loc[~dfr.index.isin(val_df.index)]
            X_train = train_df[self.live_features_dict.keys()]
            y_train = train_df[self.y_column]
            X_val = pred_df[self.live_features_dict.keys()]
            y_val = pred_df[self.y_column]
            model_copy = clone(self.model)
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_val)
            pred_df["prediction"] = y_pred
            # pred_df = climatology(pred_df)
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
        print("features", self.live_features_dict.keys())
        for group_key, val_df in grouped:
            print("proc group", group_key, "size", val_df.shape)
            if val_df.shape[0] < 30:
                continue  # Skip groups with too few samples
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
            # pred_df = climatology(pred_df)
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
    # ph_model = PhenologyModel()

    model = FuelMoistureModel()
    model.validation_train_model()
    dfr = model.prepare_training_dataset()
    # dfr = ph_model.predict_evi2_live_moisture(dfr)
    # dfr.to_parquet("data/live_training_dataset_evi.parquet")

    # test = model.prepare_test_dataset()
    # sess = rt.InferenceSession("phenology_model.onnx", providers=["CPUExecutionProvider"])
    # pred_ort = sess.run(None, {"X": model.features.values.astype(np.float32)})[0]

    # model.train_model()

    # plot_predicted_vs_obs(dfr, model)
    # re, preds = model.validation_per_location(group_cols=["site"])
    # re, preds = model.validation_per_location_per_fuel(
    #     dfr=dfr, fuel="Heather live canopy", group_cols=["site"]
    # )
    # model.save_model("live_full.onnx")
# model.train_model()
# model.save_model("model_onehot_dead.onnx")
# preds.groupby('fuel').apply(r2_rmse).reset_index()
