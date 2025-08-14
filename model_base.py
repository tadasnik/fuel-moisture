from typing import List

import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
import onnxruntime as rt
from skl2onnx import to_onnx
from sklearn.base import clone
from scipy.stats import linregress
from sklearn.metrics import root_mean_squared_error, r2_score

from validation_figures import plot_training_vs_testing


def climatology_test(
    dfr: pd.DataFrame, test: pd.DataFrame, prediction_column: str, fuels_cat_column: str
) -> pd.DataFrame:
    """returns climatological monthly average fmc prediction
    for the test dataset, based on values in dfr"""
    cli = (
        dfr.groupby(["month", fuels_cat_column])[prediction_column].mean().reset_index()
    )
    cli.rename(columns={prediction_column: "clim"}, inplace=True)
    test = test.merge(cli, on=["month", fuels_cat_column], how="left")
    return test


class BaseModel:
    def __init__(
        self,
        y_column="fmc_%",
        fuels_cat_column="fuel_type",
        fuel_names=None,
        model_params=None,
        base_features=None,
        pickled_model_fname=None,
    ):
        self.y_column = y_column
        self.fuels_cat_column = fuels_cat_column

        self.fuel_names = fuel_names or [
            "Bracken live leaves",
            "Bracken live stem",
            "Gorse live canopy",
            "Gorse live stem",
            "Heather live canopy",
            "Heather live stem",
            "Moor grass live",
        ]

        self.model_params = model_params or {
            "learning_rate": 0.1,
            "min_samples_leaf": 10,
            "max_features": 1.0,
            "quantile": 0.5,
            "loss": "absolute_error",
        }

        self.features_dict = base_features or {}

        for fuel_name in self.fuel_names:
            col_name = f"{self.fuels_cat_column}_{fuel_name}"
            self.features_dict[col_name] = {"type": "float32", "monotonic": 0}

        monotonic_constraints = [
            self.features_dict[k]["monotonic"] for k in self.features_dict.keys()
        ]
        #
        self.feature_columns = list(self.features_dict.keys())

        if pickled_model_fname:
            self.model = rt.InferenceSession(
                pickled_model_fname, providers=["CPUExecutionProvider"]
            )
        else:
            self.model = ensemble.HistGradientBoostingRegressor(
                monotonic_cst=monotonic_constraints,
                **self.model_params,
            )

    def prepare_training_dataset(self, fname: str):
        dfr = pd.read_parquet(fname)
        dfr = self.encode_fuel_features(dfr)
        return dfr

    def encode_fuel_features(self, dfr):
        """Encode fuel features using OneHotEncoder"""
        encoder = OneHotEncoder(
            sparse_output=False,
        ).set_output(transform="pandas")
        fuel_encoded = encoder.fit_transform(dfr[[self.fuels_cat_column]])
        for fuel in self.fuel_names:
            if self.fuels_cat_column + "_" + str(fuel) not in fuel_encoded.columns:
                fuel_encoded[self.fuels_cat_column + "_" + str(fuel)] = 0.0
        try:
            dfr = dfr.join(fuel_encoded)
        except ValueError as e:
            print("fuel_type encoded columns exist", e)
            return dfr
        # add types and constrains for OneHotEncoder fuel categories/columns
        # for fuel_name in fuel_encoded.columns:
        #     self.features_dict[fuel_name] = {"type": "float32", "monotonic": 0}
        return dfr

    def validation_train_model(self, dfr: pd.DataFrame):
        X_train, X_test, y_train, y_test = train_test_split(
            dfr[self.feature_columns],
            dfr[self.y_column],
            test_size=0.25,
        )
        self.model.fit(X_train, y_train)
        plot_training_vs_testing(self.model, X_train, X_test, y_train, y_test)

    def train_model(self, dfr: pd.DataFrame):
        self.model.fit(dfr[self.feature_columns], dfr[self.y_column])

    def predict(self, dfr: pd.DataFrame) -> pd.Series:
        if type(self.model) is rt.InferenceSession:
            return self.model.run(
                None, {"X": dfr[self.feature_columns].values.astype(np.float32)}
            )[0]
        else:
            return self.model.predict(dfr[self.feature_columns])

    def save_model(self, file_name: str, sample_df: pd.DataFrame = None):
        if sample_df is None:
            raise ValueError("You must provide sample_df for ONNX export.")
        onx = to_onnx(
            self.model,
            sample_df[self.feature_columns].iloc[:1].values.astype(np.float32),
        )
        with open(file_name, "wb") as f:
            f.write(onx.SerializeToString())

    def validation_per_fuel_location(
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
        print("features", self.features_dict.keys())
        for group_key, val_df in grouped:
            if val_df[val_df[self.fuels_cat_column] == fuel].shape[0] < 20:
                continue  # Skip groups with too few samples
            else:
                print(
                    "proc group",
                    group_key,
                    "size",
                    val_df[val_df[self.fuels_cat_column] == fuel].shape,
                )
            pred_df = val_df[val_df[self.fuels_cat_column] == fuel].copy()
            train_df = dfr.loc[~dfr.index.isin(val_df.index)]
            X_train = train_df[self.features_dict.keys()]
            y_train = train_df[self.y_column]
            X_val = pred_df[self.features_dict.keys()]
            y_val = pred_df[self.y_column]
            model_copy = clone(self.model)
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_val)
            pred_df["prediction"] = y_pred
            pred_df = climatology_test(
                train_df, pred_df, self.y_column, self.fuels_cat_column
            )
            for i, col in enumerate(group_cols):
                pred_df[col] = group_key[i]
            predictions.append(pred_df)
            sl, inter, pearsr, pv, stde = linregress(y_val, y_pred)
            rms = root_mean_squared_error(y_val, y_pred)
            slc, inter2c, pearsrc, pvc, stdec = linregress(
                pred_df[self.y_column], pred_df["clim"]
            )

            pred_df = pred_df.dropna()
            rmsc = root_mean_squared_error(pred_df[self.y_column], pred_df["clim"])
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


class PhenologyModel(BaseModel):
    def __init__(self, pickled_model_fname=None):
        base_features = {
            "vpdmax-7max": {"type": "float32", "monotonic": -1},
            "vpdmax-15mean": {"type": "float32", "monotonic": -1},
            "tmean-15mean": {"type": "float32", "monotonic": 0},
            "sri-15mean": {"type": "float32", "monotonic": 0},
            "smm28": {"type": "float32", "monotonic": 0},
            "smm100": {"type": "float32", "monotonic": 0},
            "ddur": {"type": "float32", "monotonic": 0},
            "ddur_change": {"type": "float32", "monotonic": 0},
        }

        fuel_names = [3, 4, 7, 9, 10]
        super().__init__(
            fuel_names=fuel_names,
            y_column="ph",
            fuels_cat_column="lc",
            model_params=None,
            base_features=base_features,
            pickled_model_fname=pickled_model_fname,
        )

    def transform_evi2_to_phenology(self, dfr: pd.DataFrame) -> pd.Series:
        """Map EVI2 values to phenological phase; [0, 1]."""
        dfrs = []
        grouped = dfr.groupby(["latind", "lonind", "lc"])
        qt = QuantileTransformer(n_quantiles=100, random_state=42)
        for group_key, group in grouped:
            qt = QuantileTransformer(n_quantiles=100, random_state=42)
            ph = qt.fit_transform(group.EVI2.values.reshape(-1, 1))
            dfrs.append(group.assign(ph=ph.flatten()))
        return pd.concat(dfrs)

    def prepare_training_dataset(self, fname: str):
        dfr = super().prepare_training_dataset(fname)
        # Add phenology specific features
        dfr = dfr[dfr.smm100 > 0.01].copy()
        dfr["month"] = dfr["date"].dt.month
        # dfr = dfr[dfr[self.y_column] > 0].copy()
        return dfr


class LiveFuelMoistureModel(BaseModel):
    def __init__(self, pickled_model_fname=None, phenology_model=None):
        base_features = {
            # "vpdmax-7max": {"type": "float32", "monotonic": -1},
            # "vpdmax-15max": {"type": "float32", "monotonic": -1},
            "vpd": {"type": "float32", "monotonic": 0},
            # "vpdmax-3mean": {"type": "float32", "monotonic": -1},
            # "vpdmax-7mean": {"type": "float32", "monotonic": -1},
            "vpdmax-15mean": {"type": "float32", "monotonic": -1},
            "ph": {"type": "float32", "monotonic": 0},
            # "ph": {"type": "float32", "monotonic": 0},
            # "smm28": {"type": "float32", "monotonic": 0},
            "smm28-15mean": {"type": "float32", "monotonic": 1},
            # "smm28": {"type": "float32", "monotonic": 1},
            # "smm100": {"type": "float32", "monotonic": 0},
            "smm100-15mean": {"type": "float32", "monotonic": 1},
            # "ddur": {"type": "float32", "monotonic": 0},
            "ddur_change": {"type": "float32", "monotonic": 0},
        }

        super().__init__(
            fuel_names=None,
            y_column="fmc_%",
            fuels_cat_column="fuel_type",
            model_params=None,
            base_features=base_features,
            pickled_model_fname=pickled_model_fname,
        )
        if phenology_model:
            self.ph_model = PhenologyModel(pickled_model_fname=phenology_model)
        else:
            self.ph_model = PhenologyModel()

    def prepare_training_dataset(self, fname: str):
        dfr = super().prepare_training_dataset(fname)
        dfr["fuel_cat"] = "other"
        for cat in ["live", "dead"]:
            dfr.loc[dfr["fuel_type"].str.contains(cat), "fuel_cat"] = cat
        dfr.loc[dfr["fuel_type"] == "Litter", "fuel_cat"] = "dead"
        dfr = dfr[(dfr["fmc_%"] < 300) & (dfr["fmc_%"] > 0)].copy()
        if self.ph_model.y_column not in dfr.columns:
            dfr = self.predict_phenology(dfr)
        dfr = dfr[dfr[self.ph_model.y_column] > 0].copy()
        return dfr[dfr.fuel_cat == "live"].copy()

    def add_phenology_lc(self, dfr: pd.DataFrame):
        """A mapping between fuel type and land cover categories for
        phenology predictions. Adds a column 'lc' to the DataFrame."""
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
        return dfr

    def predict_phenology(self, dfr: pd.DataFrame) -> pd.DataFrame:
        """Predict phenology EVI2 values using the PhenologyModel."""
        dfr = self.add_phenology_lc(dfr)
        dfr = self.ph_model.encode_fuel_features(dfr)
        dfr[self.ph_model.y_column] = self.ph_model.predict(dfr)
        return dfr

    def predict_phenology_fuel_moisture(self, dfr: pd.DataFrame):
        """
        This wrapper method is used to predict EVI2 and then
        predicts fuel moisture.
        """
        dfr = self.predict_phenology(dfr)
        dfr = self.encode_fuel_features(dfr)
        dfr["pred"] = self.predict(dfr)
        return dfr

    def validation_per_fuel_location(
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
        print("features", self.features_dict.keys())
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
            X_train = train_df[self.features_dict.keys()]
            y_train = train_df[self.y_column]
            X_val = pred_df[self.features_dict.keys()]
            y_val = pred_df[self.y_column]
            model_copy = clone(self.model)
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_val)
            pred_df["prediction"] = y_pred
            pred_df = climatology_test(
                train_df, pred_df, self.y_column, self.fuels_cat_column
            )
            for i, col in enumerate(group_cols):
                pred_df[col] = group_key[i]
            predictions.append(pred_df)
            sl, inter, pearsr, pv, stde = linregress(y_val, y_pred)
            rms = root_mean_squared_error(y_val, y_pred)
            slc, inter2c, pearsrc, pvc, stdec = linregress(
                pred_df[self.y_column], pred_df["clim"]
            )

            pred_df = pred_df.dropna()
            rmsc = root_mean_squared_error(pred_df[self.y_column], pred_df["clim"])
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
    # ph_model = PhenologyModel(pickled_model_fname="ph_model.onnx")
    ph_model = PhenologyModel()
    dfr = ph_model.prepare_training_dataset(
        fname="data/phenology_training_dataset_features.parquet"
    )
    dfr = ph_model.transform_evi2_to_phenology(dfr)
    res, df = ph_model.validation_per_fuel_location(dfr, 4)
    # ph_model.validation_train_model(dfr)
    # ph_model.train_model(dfr)
    # ph_model.save_model("ph_model_q.onnx", dfr)
    # Example usage
    # lfmc_model = LiveFuelMoistureModel(phenology_model="ph_model_q.onnx")
    # dfrl = lfmc_model.prepare_training_dataset(
    #     fname="data/training_dataset_features_full_sm.parquet"
    # )
    # lfmc_model.train_model(dfrl)
    # lfmc_model.save_model("lfmc_model_no_evi.onnx", dfrl)
    # lfmc_model.validation_train_model(dfrl)
    # res = lfmc_model.validation_per_location(group_cols=["site"])
    #
    # res, df = lfmc_model.validation_per_fuel_location(dfrl, "Heather live stem")
    # gres, gdf = lfmc_model.validation_per_fuel_location(dfrl, "Gorse live canopy")
    # bres, bdf = lfmc_model.validation_per_fuel_location(dfrl, "Moor grass live")
# lfmc = lfmc_model.predict(dfrl)

# print(fuel_model.feature_columns)
# print(fuel_model.model_params)
#
# phenology_model = PhenologyModel()
# print(phenology_model.phen_features_dict)
#
# live_fuel_model = LiveFuelMoistureModel()
# print(live_fuel_model.feature_columns)
