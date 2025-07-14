import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from skl2onnx import to_onnx


class BaseModel:
    def __init__(
        self,
        y_column="fmc_%",
        fuels_cat_column="fuel_type",
        fuel_names=None,
        model_params=None,
        base_features=None,
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
            "loss": "squared_error",
        }

        self.features_dict = base_features or {}

        for fuel_name in self.fuel_names:
            col_name = f"{self.fuels_cat_column}_{fuel_name}"
            self.features_dict[col_name] = {"type": "float32", "monotonic": 0}

        monotonic_constraints = [
            self.features_dict[k]["monotonic"] for k in self.features_dict.keys()
        ]

        self.feature_columns = list(self.features_dict.keys())

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
        for fuel_name in fuel_encoded.columns:
            self.features_dict[fuel_name] = {"type": "float32", "monotonic": 0}
        return dfr

    def validation_train_model(self, dfr: pd.DataFrame):
        X_train, X_test, y_train, y_test = train_test_split(
            dfr[self.feature_columns],
            dfr[self.y_column],
            test_size=0.25,
        )
        self.model.fit(X_train, y_train)

        # Assume this function is available from elsewhere in your codebase
        # plot_training_vs_testing(self.model, X_train, X_test, y_train, y_test)

    def train_model(self, dfr: pd.DataFrame):
        self.model.fit(dfr[self.feature_columns], dfr[self.y_column])

    def save_model(self, file_name: str, sample_df: pd.DataFrame = None):
        if sample_df is None:
            raise ValueError("You must provide sample_df for ONNX export.")
        onx = to_onnx(
            self.model,
            sample_df[self.feature_columns].iloc[:1].astype(np.float32),
        )
        with open(file_name, "wb") as f:
            f.write(onx.SerializeToString())


class PhenologyModel(BaseModel):
    def __init__(self):
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
            y_column="EVI2",
            fuels_cat_column="lc",
            model_params=None,
            base_features=base_features,
        )

    def prepare_training_dataset(self, fname: str):
        dfr = super().prepare_training_dataset(fname)
        # Add phenology specific features
        dfr = dfr[dfr.smm100 > 0.01].copy()
        dfr = dfr[dfr.EVI2 > 0].copy()
        return dfr


class LiveFuelMoistureModel(BaseModel):
    def __init__(self):
        base_features = {
            "vpdmax-10max": {"type": "float32", "monotonic": -1},
            "vpdmax-15mean": {"type": "float32", "monotonic": -1},
            "EVI2": {"type": "float32", "monotonic": 0},
            "smm100": {"type": "float32", "monotonic": 0},
            "doy": {"type": "float32", "monotonic": 0},
        }

        super().__init__(
            fuel_names=None,
            y_column="fmc_%",
            fuels_cat_column="fuel_type",
            model_params=None,
            base_features=base_features,
        )

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

    def prepare_training_dataset(self, fname: str):
        dfr = pd.read_parquet(fname)
        dfr = self.encode_fuel_features(dfr)
        dfr = dfr[(dfr["fmc_%"] < 300) & (dfr["fmc_%"] > 0)]
        # Calculate monthly climatology
        cli = dfr.groupby(["month", "fuel_type"])["fmc_%"].mean().reset_index()
        cli.rename(columns={"fmc_%": "clim"}, inplace=True)
        dfr = dfr.merge(cli, on=["month", "fuel_type"], how="left")
        return dfr


if __name__ == "__main__":
    ph_odel = PhenologyModel()
    dfr = ph_odel.prepare_training_dataset(
        fname="data/phenology_training_dataset_features.parquet"
    )
    # Example usage
    # fuel_model = FuelMoistureModel(
    #     fuel_names=["Example Fuel"],
    #     y_column="fmc_%",
    #     fuels_cat_column="fuel_type",
    # )
    # print(fuel_model.feature_columns)
    # print(fuel_model.model_params)
    #
    # phenology_model = PhenologyModel()
    # print(phenology_model.phen_features_dict)
    #
    # live_fuel_model = LiveFuelMoistureModel()
    # print(live_fuel_model.feature_columns)
