import sys

sys.path.append("../nelson-fuel-moisture-python/")

import requests
import io

import nfdrs4py
import pandas as pd
from nfdrs4_moisture import compute_nfdrs4
from NG_FWI import hFWI
from open_meteo import fetch_hourly
from process_moisture_data import get_elevation_slope_aspect
from nelson_moisture import nelson_fuel_moisture
from model_base import DeadFuelMoistureModel
from process_moisture_data import prepare_weather_features


def fetch_meteo_data(
    lat: float, lon: float, start_date: str, end_date: str
) -> pd.DataFrame:
    terrain = get_elevation_slope_aspect(lat, lon)
    meteo_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "wind_speed_10m",
        # "wind_direction_10m",
        # "wind_gusts_10m",
        "vapour_pressure_deficit",
        "shortwave_radiation",
        "global_tilted_irradiance",
    ]
    url = "https://archive-api.open-meteo.com/v1/archive"
    # opt_params = {
    # "wind_speed_unit": "mph",
    # "temperature_unit": "fahrenheit",
    # "precipitation_unit": "inch",
    # }
    opt_params = {
        "tilt": terrain["slope"],
        "azimuth": terrain["aspect"] - 180,
    }

    dfr = fetch_hourly(
        url,
        lat,
        lon,
        start_date,
        end_date,
        meteo_vars,
        opt_params,
    )
    return dfr


def compute_simple_nelson(dfr):
    mcs = []
    prev_moist = None
    for nr, row in dfr.iterrows():
        sim_fmc = nelson_fuel_moisture(
            prev_moist=prev_moist,
            temp=row.temperature_2m,
            sol_rad=row.global_tilted_irradiance,
            rel_hum=row.relative_humidity_2m,
            rainfall=row.precipitation,
        )
        prev_moist = sim_fmc
        mcs.append(sim_fmc)
    return mcs


def prepare_lagged_features_dead(dfr: pd.DataFrame) -> pd.DataFrame:
    """prepare weather features (results) for training dataset (dfr). Return training
    dataset with weather features merget and also hourly time series dataset"""
    dfr.rename(
        {
            "vapour_pressure_deficit": "vpd",
            "global_tilted_irradiance": "gti",
        },
        axis=1,
        inplace=True,
    )
    for hour in range(1, 6):
        dfr[f"vpd-{hour}"] = dfr["vpd"].shift(hour)
        dfr[f"gti-{hour}"] = dfr["gti"].shift(hour)
    return dfr


def compute_fwi_system(dfr, lon, lat):
    dfwi = dfr.copy()
    fwi_columns = [
        "id",
        "lat",
        "long",
        "timezone",
        "yr",
        "mon",
        "day",
        "hr",
        "temp",
        "rh",
        "ws",
        "prec",
        "grass_fuel_load",
        "timestamp",
        "date",
    ]
    dfwi = dfwi.rename(
        {
            "temperature_2m": "temp",
            "relative_humidity_2m": "rh",
            "wind_speed_10m": "ws",
            "precipitation": "prec",
            "utc_offset": "timezone",
            "date": "timestamp",
        },
        axis=1,
    )
    dfwi["yr"] = dfwi.timestamp.dt.year
    dfwi["mon"] = dfwi.timestamp.dt.month
    dfwi["day"] = dfwi.timestamp.dt.day
    dfwi["hr"] = dfwi.timestamp.dt.hour
    dfwi["grass_fuel_load"] = 0.1
    dfwi["id"] = 1
    dfwi["long"] = lon
    dfwi["lat"] = lat
    dfwi["date"] = dfwi["timestamp"].dt.date
    # utc_offset from seconds to hours
    dfwi["timezone"] = (dfwi["timezone"] / 3600).astype(int)
    fwi_hourly = hFWI(dfwi[fwi_columns])
    fwi_hourly = fwi_hourly.drop(
        [
            "date",
            "yr",
            "mon",
            "day",
            "hr",
            "id",
        ],
        axis=1,
    )
    fwi_hourly = fwi_hourly.rename({"timestamp": "date"}, axis=1)
    return fwi_hourly


def predict_dfmc_fireinsite(dfr):
    model = DeadFuelMoistureModel(pickled_model_fname="model_onehot_dead.onnx")
    fuel_columns = [x for x in model.features_dict.keys() if x.startswith("fuel_")]
    for fuel in fuel_columns:
        dfr_f = dfr.copy()
        dfr_f = model.add_fuel_columns(dfr_f)
        dfr_f[fuel] = 1
        dfr["dfmc" + fuel] = model.predict(dfr_f)
    return dfr


def gen_data_for_site(lat, lon, start_date, end_date):
    dfr = fetch_meteo_data(lat, lon, start_date, end_date)
    ffeats = prepare_lagged_features_dead(dfr.copy())
    ffeats = predict_dfmc_fireinsite(ffeats)
    dfr["simple_fmc"] = compute_simple_nelson(dfr)
    nfd = compute_nfdrs4(dfr)
    nfd_fmc_cols = [x for x in nfd.columns if x.startswith("dmc_")]
    nfd_fmc_cols += ["lmc_herb", "lmc_woody"]
    nfd.loc[:, nfd_fmc_cols] *= 100
    nfd_fmc_cols += ["date"]
    fuel_columns = [x for x in ffeats.columns if x.startswith("dfmcfuel_")]
    fuel_columns += ["date"]
    fwi_h = compute_fwi_system(dfr, lon, lat)
    res = dfr.merge(ffeats[fuel_columns], on="date", how="left")
    res = res.merge(nfd[nfd_fmc_cols], on="date", how="left")
    return res


def uk_weather_site(site="Cobham Common H15"):
    model = DeadFuelMoistureModel(pickled_model_fname="model_onehot_dead.onnx")
    dfr = model.prepare_training_dataset(
        fname="data/training_dataset_features_full.parquet"
    )
    site = "Cobham Common H15"
    results = pd.read_parquet("data/training_dataset_weather.parquet")
    resub = results[results.site == site].copy()
    fe, fets = prepare_weather_features(dfr[dfr.site == site], resub)


def uk_dataset(site="Cobham Common H15"):
    model = DeadFuelMoistureModel(pickled_model_fname="model_onehot_dead.onnx")
    dfr = model.prepare_training_dataset(
        fname="data/training_dataset_features_full.parquet"
    )
    dfrts = pd.read_parquet("data/training_dataset_features_full_time_series.parquet")
    fuel = "Heather canopy"  # Change this to the desired fuel type
    # fuel = "Gorse live canopy"  # Change this to the desired fuel type
    site = "Cobham Common H15"
    sub = dfrts[dfrts.site == site].copy()
    sub["fuel"] = fuel
    sub = model.add_fuel_columns(sub)
    sub["fuel" + fuel] = 1.0
    sub["dFMC"] = model.predict(sub)


if __name__ == "__main__":
    pass
    # lat = 35.27618889
    # lon = -112.0632472
    # start_date = "2018-05-23"
    # end_date = "2018-06-03"
    # dfr = fetch_meteo_data(lat, lon, start_date, end_date)
    # ffeats = prepare_lagged_features_dead(dfr.copy())
    # ffeats = perdict_dfmc_fireinsite(ffeats)
    # dfr['simple_fmc'] = compute_simple_nelson(dfr)
    # nfd = compute_nfdrs4(dfr)
    # dfr.merge
