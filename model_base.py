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
from sklearn.metrics import (
    mean_absolute_percentage_error,
    r2_score,
)
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
# from validation_figures import plot_training_vs_testing


def validation_nos(
    dfr,
    group_cols: List[str] = ["lonind", "latind"],
    y_column="fmc_%",
    prediction_column="pred",
    base_column="clim",
):
    res = []
    grouped = dfr.groupby(group_cols)
    for group_key, val_df in grouped:
        r2 = r2_score(val_df[y_column], val_df[prediction_column])
        rms = mean_absolute_percentage_error(
            val_df[y_column], val_df[prediction_column]
        )
        # _, _, prc, _, _ = linregress(val_df[y_column], val_df[base_column])
        r2c = r2_score(val_df[y_column], val_df[base_column])
        rmsc = mean_absolute_percentage_error(val_df[y_column], val_df[base_column])
        group_result = {
            "group": group_key,
            "r2": r2,
            "rmse": rms,
            "r2c": r2c,
            "rmsec": rmsc,
            "size": val_df.shape[0],
        }
        res.append(group_result)
    return pd.DataFrame(res)


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
    test = test.fillna(90)
    return test


class BaseModel:
    def __init__(
        self,
        y_column="fmc_%",
        fuels_cat_column="fuel",
        fuel_names=None,
        model_params=None,
        base_features=None,
        pickled_model_fname=None,
    ):
        self.y_column = y_column
        self.fuels_cat_column = fuels_cat_column

        self.fuel_names = fuel_names or [
            "Bracken leaves",
            "Bracken stem",
            "Gorse canopy",
            "Gorse stem",
            "Heather canopy",
            "Heather stem",
            "Moor grass",
            "Surface",
        ]

        self.model_params = model_params or {
            "learning_rate": 0.1,
            "min_samples_leaf": 10,
            "max_features": 1.0,
            "loss": "quantile",
            "quantile": 0.5,
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
        fuel_map = {
            "Bracken dead stem": "Bracken stem",
            "Bracken live stem": "Bracken stem",
            "Bracken dead leaves": "Bracken leaves",
            "Bracken live leaves": "Bracken leaves",
            "Gorse live canopy": "Gorse canopy",
            "Gorse dead canopy": "Gorse canopy",
            "Gorse live stem": "Gorse stem",
            "Gorse dead stem": "Gorse stem",
            "Heather live canopy": "Heather canopy",
            "Heather dead canopy": "Heather canopy",
            "Heather live stem": "Heather stem",
            "Heather dead stem": "Heather stem",
            "Moor grass live": "Moor grass",
            "Moor grass dead": "Moor grass",
            "Moss": "Surface",
            "Organic layer": "Surface",
            "Twigs": "Surface",
            "Litter": "Surface",
        }
        fmc_cat_map = {
            "Bracken dead stem": "dead",
            "Bracken live stem": "live",
            "Bracken dead leaves": "dead",
            "Bracken live leaves": "live",
            "Gorse live canopy": "live",
            "Gorse dead canopy": "dead",
            "Gorse live stem": "live",
            "Gorse dead stem": "dead",
            "Heather live canopy": "live",
            "Heather dead canopy": "dead",
            "Heather live stem": "live",
            "Heather dead stem": "dead",
            "Moor grass live": "live",
            "Moor grass dead": "dead",
            "Moss": "None",
            "Organic layer": "None",
            "Twigs": "dead",
            "Litter": "dead",
        }

        dfr["fuel"] = dfr["fuel_type"].map(fuel_map)
        dfr["fmc_cat"] = dfr["fuel_type"].map(fmc_cat_map)
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

    def validation_per_location(
        self, dfr, group_cols: List[str] = ["lonind", "latind"]
    ):
        """
        Perform spatial cross-validation using unique (lonind, latind) groups.

        Parameters:
            dfr (pd.DataFrame): DataFrame containing features and target variable.
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
            pred_df = val_df.copy()
            train_df = dfr.loc[~dfr.index.isin(val_df.index)]
            X_train = train_df[self.features_dict.keys()]
            y_train = train_df[self.y_column]
            X_val = pred_df[self.features_dict.keys()]
            y_val = pred_df[self.y_column]
            model_copy = clone(self.model)
            model_copy.fit(X_train, y_train)
            y_pred = model_copy.predict(X_val)
            pred_df["pred"] = y_pred
            pred_df = climatology_test(
                train_df, pred_df, self.y_column, self.fuels_cat_column
            )
            for i, col in enumerate(group_cols):
                pred_df[col] = group_key[i]
            predictions.append(pred_df)
            sl, inter, pearsr, pv, stde = linregress(y_val, y_pred)
            rms = mean_absolute_percentage_error(y_val, y_pred)
            slc, inter2c, pearsrc, pvc, stdec = linregress(
                pred_df[self.y_column], pred_df["clim"]
            )

            # pred_df = pred_df.dropna()
            rmsc = mean_absolute_percentage_error(
                pred_df[self.y_column], pred_df["clim"]
            )
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
            if val_df[val_df[self.fuels_cat_column] == fuel].shape[0] < 5:
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
            rms = mean_absolute_percentage_error(y_val, y_pred)
            slc, inter2c, pearsrc, pvc, stdec = linregress(
                pred_df[self.y_column], pred_df["clim"]
            )

            # pred_df = pred_df.dropna()
            rmsc = mean_absolute_percentage_error(
                pred_df[self.y_column], pred_df["clim"]
            )
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
    def __init__(self, y_column=None, pickled_model_fname=None):
        base_features = {
            "vpdmax-7max": {"type": "float32", "monotonic": -1},
            "vpdmax-15mean": {"type": "float32", "monotonic": -1},
            "tmean-15mean": {"type": "float32", "monotonic": 0},
            "sri-15mean": {"type": "float32", "monotonic": 0},
            "smm28": {"type": "float32", "monotonic": 1},
            "smm100": {"type": "float32", "monotonic": 1},
            "ddur": {"type": "float32", "monotonic": 0},
            "ddur_change": {"type": "float32", "monotonic": 0},
        }

        fuel_names = [3, 4, 7, 9, 10]
        super().__init__(
            fuel_names=fuel_names,
            y_column=y_column,
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
            group = group.assign(ph=ph.flatten())
            endog = group.ph.values.reshape(-1, 1)
            exog = sm.add_constant(
                (group.date.astype(int) - group.date.astype(int).min())
                / 604800000000000
            )
            rols = RollingOLS(endog, exog, window=3)
            rres = rols.fit()
            params = rres.params.copy()
            group = group.assign(phs=params.date.values)
            dfrs.append(group)
        return pd.concat(dfrs)

    def prepare_training_dataset(self, fname: str):
        dfr = super().prepare_training_dataset(fname)
        # Add phenology specific features
        dfr = dfr[dfr.smm100 > 0.0].copy()
        dfr["month"] = dfr["date"].dt.month
        # dfr = dfr[dfr[self.y_column] > 0].copy()
        return dfr


class LiveFuelMoistureModel(BaseModel):
    def __init__(
        self,
        pickled_model_fname=None,
        phenology_ph_model=None,
        phenology_phs_model=None,
    ):
        base_features = {
            "vpdmax-7max": {"type": "float32", "monotonic": -1},
            # "vpdmax-15max": {"type": "float32", "monotonic": -1},
            # "vpd": {"type": "float32", "monotonic": 0},
            # "vpdmax-3mean": {"type": "float32", "monotonic": -1},
            # "vpdmax-7mean": {"type": "float32", "monotonic": -1},
            "vpdmax-15mean": {"type": "float32", "monotonic": -1},
            # "ph": {"type": "float32", "monotonic": 0},
            # "phs": {"type": "float32", "monotonic": 0},
            # "sri-15mean": {"type": "float32", "monotonic": -1},
            # "smm28": {"type": "float32", "monotonic": 0},
            # "smm28-15mean": {"type": "float32", "monotonic": 1},
            # "smm28": {"type": "float32", "monotonic": 0},
            "smm100": {"type": "float32", "monotonic": 1},
            # "prec-15sum": {"type": "float32", "monotonic": 0},
            # "smm100-15mean": {"type": "float32", "monotonic": 1},
            "ddur": {"type": "float32", "monotonic": 0},
            "ddur_change": {"type": "float32", "monotonic": 0},
        }

        super().__init__(
            base_features=base_features,
            pickled_model_fname=pickled_model_fname,
        )
        if phenology_ph_model:
            self.ph_model = PhenologyModel(
                y_column="ph", pickled_model_fname=phenology_ph_model
            )
        else:
            self.ph_model = PhenologyModel(y_column="ph")

        if phenology_phs_model:
            self.phs_model = PhenologyModel(
                y_column="phs", pickled_model_fname=phenology_phs_model
            )
        else:
            self.phs_model = PhenologyModel(y_column="phs")

    def prepare_training_dataset(self, fname: str):
        dfr = super().prepare_training_dataset(fname)
        dfr = dfr[
            (dfr["fmc_%"] < 300) & (dfr["fmc_%"] > 30) & (dfr.fmc_cat == "live")
        ].copy()
        if self.ph_model.y_column not in dfr.columns:
            dfr = self.predict_phenology(dfr)
        # dfr = dfr[dfr[self.phs_model.y_column] < 0].copy()
        return dfr

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
            "Moor grass live": 7,
        }
        dfr["lc"] = dfr.copy()["fuel_type"].map(fuels_live_to_phenology)
        return dfr

    def predict_phenology(self, dfr: pd.DataFrame) -> pd.DataFrame:
        """Predict phenology EVI2 values using the PhenologyModel."""
        dfr = self.add_phenology_lc(dfr)
        dfr = self.ph_model.encode_fuel_features(dfr)
        dfr[self.ph_model.y_column] = self.ph_model.predict(dfr)
        dfr[self.phs_model.y_column] = self.phs_model.predict(dfr)
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


class DeadFuelMoistureModel(BaseModel):
    def __init__(
        self,
        pickled_model_fname=None,
    ):
        base_features = {
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
            "vpdmax-15mean": {"type": "float32", "monotonic": -1},
            "smm100": {"type": "float32", "monotonic": 1},
            "ddur": {"type": "float32", "monotonic": 0},
            "ddur_change": {"type": "float32", "monotonic": 0},
        }

        super().__init__(
            base_features=base_features,
            pickled_model_fname=pickled_model_fname,
        )

    def prepare_training_dataset(self, fname: str):
        dfr = super().prepare_training_dataset(fname)
        dfr["fuel_cat"] = "other"
        for cat in ["live", "dead"]:
            dfr.loc[dfr["fuel_type"].str.contains(cat), "fuel_cat"] = cat
        dfr.loc[dfr["fuel_type"] == "Litter", "fuel_cat"] = "dead"
        dfr = dfr[
            (dfr["fmc_%"] < 60) & (dfr["fmc_%"] > 0) & (dfr.fuel_cat == "dead")
        ].copy()
        return dfr


def plot_training_vs_testing(df, fuels):
    torange = (1.0, 0.4980392156862745, 0.054901960784313725)
    tblue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    tgreen = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    tpurple = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    tpink = (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
    tbrown = (0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
    tchaki = (0.7372549019607844, 0.7411764705882353, 0.13333333333333333)
    tgrey = (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)
    tred = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    telectric = (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)
    COLOR = (0.2, 0.2, 0.2)

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), sharey=True)
    markers = {
        "leaves": "o",
        "canopy": "o",
        "stem": "^",
        "grass": "o",
        "Surface": "*",
    }
    colors = {
        "Heather canopy": tpink,
        "Heather stem": tpink,
        "Gorse canopy": tgreen,
        "Gorse stem": tgreen,
        "Bracken leaves": torange,
        "Bracken stem": torange,
        "Moor grass": tbrown,
        "Surface": tchaki,
    }

    for nr, fuel in enumerate(fuels):
        dfs = df[df.fuel == fuel]
        mark = markers.get(fuel.split()[-1], "*")
        axe.scatter(
            dfs["fmc_%"],
            dfs["pred"],
            marker=mark,
            alpha=0.8,
            color=colors.get(fuel, tred),
            label=fuel,
        )
    axe.plot([0, 62], [0, 62], color=tgrey, linestyle="--", linewidth=1)
    axe.set_ylim(0, 62)
    axe.set_xlim(0, 62)
    axe.legend()
    plt.show()
    # sns.scatterplot(
    #     x="fmc_%",
    #     y="clim",
    #     hue="fuel_type",
    #     data=res,
    #     alpha=0.3,
    # )
    # plt.savefig(f"figures/site_validation_lfmc_phen.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # ph_model = PhenologyModel(pickled_model_fname="ph_model.onnx")
    # ph_model = PhenologyModel(y_column="phs")
    # dfr = ph_model.prepare_training_dataset(
    #     fname="data/phenology_training_dataset_features.parquet"
    # )
    # dfr = ph_model.transform_evi2_to_phenology(dfr)
    #
    lfmc_model = LiveFuelMoistureModel(
        phenology_ph_model="ph_model_q.onnx", phenology_phs_model="ph_model_q_phs.onnx"
    )
    dfrl = lfmc_model.prepare_training_dataset(
        fname="data/training_dataset_features_full.parquet"
    )
    # lfmc_model.train_model(dfrl)
    res, df = lfmc_model.validation_per_location(dfrl)
    rr = validation_nos(df, group_cols=["fuel"])
    #
    # uob = pd.read_parquet("data/training_dataset_features_uob_2025.parquet")
    # oub = uob.reset_index(drop=True)
    # uob = uob[(uob.fuel == "Calluna canopy")].copy()
    # uob["fuel_type"] = "Heather live canopy"
    # dfrl_feats = pd.concat([dfrl, uob], ignore_index=True)
    #
    # dfmc_model = DeadFuelMoistureModel()
    # dfr = dfmc_model.prepare_training_dataset(
    #     fname="data/training_dataset_features_full.parquet"
    # )
    # dfr["lfmc"] = lfmc_model.predict(dfr)
    # dfmc_model.train_model(dfr)
    # res, df = dfmc_model.validation_per_location(dfr)
    # rr = validation_nos(df, group_cols=["fuel"])
    # dfr = pd.read_parquet('data/phenology_training_dataset_features_phs.parquet')
    #
    # res, df = ph_model.validation_per_fuel_location(dfr.dropna().copy(), 10)
    # ph_model.validation_train_model(dfr)
    # ph_model.train_model(dfr.dropna().copy())
    # ph_model.save_model("ph_model_q_phs.onnx", dfr.dropna().copy())
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
#   (Bracken leaves,)  0.420618  0.317287  0.179614  0.479077   296
# 1    (Bracken stem,)  0.395924  0.317670  0.225460  0.598477   262
# 2    (Gorse canopy,)  0.427986  0.284281 -0.354888  0.651894   174
# 3      (Gorse stem,)  0.425882  0.292105 -0.087038  0.760251   179
# 4  (Heather canopy,)  0.556521  0.263892  0.288182  0.553898   347
# 5    (Heather stem,)  0.540547  0.265305  0.291759  0.502822   361
# 6      (Moor grass,)  0.507567  0.549805  0.173272  1.233366   162
# 7         (Surface,)  0.357291  0.416063  0.064073  0.769700   393
# dfs = []
# for fuel in [
#     "Heather live canopy",
#     "Heather live stem",
#     "Gorse live canopy",
#     "Gorse live stem",
#     "Moor grass live",
#     "Bracken live leaves",
#     "Bracken live stem",
# ]:
#     print(f"Validating {fuel}")
#     res, df = lfmc_model.validation_per_fuel_location(dfrl, fuel)
#     dfs.append(df)
#     print((res.rmse - res.rmsec).sum())
#     print(res)
# res, df = lfmc_model.validation_per_fuel_location(dfrl, "Heather live canopy")
# gres, gdf = lfmc_model.validation_per_fuel_location(dfrl, "Gorse live canopy")
# bres, bdf = lfmc_model.validation_per_fuel_location(dfrl, "Moor grass live")


# print(fuel_model.feature_columns)
# print(fuel_model.model_params)
#
# phenology_model = PhenologyModel()
# print(phenology_model.phen_features_dict)
#
# live_fuel_model = LiveFuelMoistureModel()
# print(live_fuel_model.feature_columns)
