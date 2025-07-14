import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import onnxruntime as rt

from dead_fuel_moisture_model import DeadFuelMoistureModel
from live_fuel_moisture_model import FuelMoistureModel

import geopandas as gpd
from shapely.geometry import Point


def make_map_locations(df):
    # Sample dataframe with lat/lon points

    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Load a world map and filter for UK
    url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
    world = gpd.read_file(url)
    uk = world[world["ADMIN"] == "United Kingdom"]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 10))
    uk.plot(ax=ax, color="lightgrey", edgecolor="black")
    gdf_points.plot(ax=ax, color="red", markersize=50)

    # Improve display
    ax.set_title("Points over UK map")
    ax.set_xlim(-10, 2)  # UK longitude bounds
    ax.set_ylim(49.5, 61)  # UK latitude bounds
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def add_dead_fuel_columns(dfr, model):
    fuel_columns = [x for x in model.feature_types.keys() if x.startswith("fuel_")]
    for fuel in fuel_columns:
        dfr[fuel] = 0.0
    return dfr


def add_fuel_columns(dfr, model):
    fuel_columns = [x for x in model.live_features_dict.keys() if x.startswith("fuel_")]
    for fuel in fuel_columns:
        dfr[fuel] = 0.0
    return dfr


def predict_site_dead_fuel_moisture(model, site, fuel):
    fets = pd.read_parquet("data/weather_site_features.parquet")

    fets["month"] = fets.date.dt.month
    fets["doy"] = fets.date.dt.dayofyear

    test = fets[fets.site == site].copy()

    test = add_dead_fuel_columns(test, model)
    test[fuel] = 1.0
    features = (
        test[model.feature_types.keys()]
        .copy()
        .astype({k: v["type"] for k, v in model.feature_types.items()})
    )

    sess = rt.InferenceSession("dead_full.onnx", providers=["CPUExecutionProvider"])
    pred_ort = sess.run(None, {"X": features.values.astype(np.float32)})[0]
    test["pred"] = pred_ort
    # Predict training dataset fuel moisture using the model
    dfr = model.prepare_training_dataset()
    features_dfr = (
        dfr[model.feature_types.keys()]
        .copy()
        .astype({k: v["type"] for k, v in model.feature_types.items()})
    )

    pred_ort_dfr = sess.run(None, {"X": features_dfr.values.astype(np.float32)})[0]
    dfr["pred"] = pred_ort_dfr

    return test, dfr


def predict_site_fuel_moisture(model, site, fuel):
    fets = pd.read_parquet("data/weather_site_features_evi.parquet")

    # fets["year"] = fets.date.dt.year
    # fets["month"] = fets.date.dt.month
    # fets["doy"] = fets.date.dt.dayofyear
    #
    test = fets[fets.site == site].copy()

    test = add_fuel_columns(test, model)
    test[fuel] = 1.0
    # features = (
    #     test[model.live_features_dict.keys()]
    #     .copy()
    #     .astype({k: v["type"] for k, v in model.live_features_dict.items()})
    # )

    # sess = rt.InferenceSession(
    #     "live_full_ddur.onnx", providers=["CPUExecutionProvider"]
    # )
    # pred_ort = sess.run(None, {"X": features.values.astype(np.float32)})[0]
    test["pred"] = model.predict(test)
    print(test.columns)
    # Predict training dataset fuel moisture using the model
    dfr = model.prepare_training_dataset()
    # features_dfr = (
    #     dfr[model.live_features_dict.keys()]
    #     .copy()
    #     .astype({k: v["type"] for k, v in model.live_features_dict.items()})
    # )
    #
    # pred_ort_dfr = sess.run(None, {"X": features_dfr.values.astype(np.float32)})[0]
    dfr["pred"] = model.predict(dfr)

    return test, dfr


def plot_predictions_for_site_fuel(site, fuel, var):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    preds, dfr = predict_site_fuel_moisture(model, site, fuel)
    sel = dfr[(dfr[fuel] == 1.0)].copy()
    selsite = dfr[(dfr[fuel] == 1.0) & (dfr.site == site)].copy()
    sns.lineplot(x="date", y="pred", data=preds, label="Prediction", color="0.5", ax=ax)
    # sns.scatterplot(x="date", y="pred", data=sel, color="green", label="Predicted FMC")
    sns.scatterplot(
        x="date", y="fmc_%", data=sel, color="orange", label="Measured FMC", ax=ax
    )
    sns.scatterplot(
        x="date", y="fmc_%", data=selsite, color="red", label="Measured FMC", ax=ax
    )

    sns.lineplot(x="date", y=var, data=preds, color="green", alpha=0.5, ax=ax2)
    plt.title(f"LFM for {site} - {fuel.replace('_', ' ').title()}")
    plt.legend()
    # plt.savefig(f"figures/{site}_live_{fuel}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_predictions_for_site_dead_fuel(site, fuel):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    preds, dfr = predict_site_dead_fuel_moisture(model, site, fuel)
    sel = dfr[(dfr.site == site) & (dfr[fuel] == 1.0)].copy()
    sns.lineplot(x="date", y="pred", data=preds, label="Prediction", color="0.5", ax=ax)
    # sns.scatterplot(x="date", y="pred", data=sel, color="green", label="Predicted FMC")
    sns.scatterplot(x="date", y="fmc_%", data=sel, color="red", label="Measured FMC")
    plt.title(
        f"Dead Fuel Moisture Predictions for {site} - {fuel.replace('_', ' ').title()}"
    )
    plt.xlabel(" ")
    plt.ylabel("Fuel Moisture Content (%)")
    plt.legend()
    plt.savefig(f"figures/{site}_dead_{fuel}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_r2_per_group():
    fig = plt.figure(figsize=(6, 4))
    re = pd.read_parquet("figures/dead_r2.parquet")
    ref = re[["group", "r2", "rmse"]].copy()
    ref["model"] = "FireInSite"
    ren = re[["group", "r2c", "rmsec"]].copy()
    ren = ren.rename(columns={"r2c": "r2", "rmsec": "rmse"})
    ren["model"] = "Simple Nelson"
    res = pd.concat([ref, ren]).reset_index()
    sns.catplot(data=res, x="model", y="r2", hue="index", kind="point", legend=False)
    plt.title(f"Dead FMC validation per site")
    plt.savefig(f"figures/r2_validation_dead.png", dpi=300, bbox_inches="tight")


def plot_july_dead():
    preds = pd.read_parquet("figures/dead_predictions.parquet")
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    sns.boxplot(
        x="year",
        y=preds["nelson"] - preds["fmc_%"],
        hue="fuel",
        data=preds[preds.month == 7],
        ax=ax,
        legend=False,
    )
    ax.set_ylim(50, -50)
    ax.set_title("Nelson Model Residuals")
    ax2 = fig.add_subplot(122)
    sns.boxplot(
        x="year",
        y=preds["prediction"] - preds["fmc_%"],
        hue="fuel",
        data=preds[preds.month == 7],
        ax=ax2,
    )
    ax2.set_ylim(50, -50)
    ax2.set_title("FireInSite Model Residuals")
    # plt.savefig(f"figures/july_validation_dead.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_phenology_weather():
    sampled = pd.read_parquet("data/phenology_sampled_locations.parquet")
    weather = pd.read_parquet("data/phenology_weather.parquet")
    lc = 4
    region = "South-west"
    ph = pd.read_parquet(
        f"/Users/tadas/modFire/fire_lc_ndvi/data/cehlc/gee_results/VNP13A1_{region}_{lc}_sample.parquet"
    )
    samp_sub = sampled[(sampled.lc == lc) & (sampled.Region == region)]
    weather_sub = weather[
        weather.lonind.isin(samp_sub.lonind) & weather.latind.isin(samp_sub.latind)
    ].copy()


if __name__ == "__main__":
    model = FuelMoistureModel()
    # model.validation_train_model()
    model.train_model()
    # model = DeadFuelMoistureModel()
    fuel = "fuel_type_Heather live canopy"  # Change this to the desired fuel type
    # site = "Cobham Common H15"
    site = "Ockham Common H15"
    # site = "Sugar Loaf H6"
    var = "EVI2"
    plot_predictions_for_site_fuel(site, fuel, var)
    # model_dead = DeadFuelMoistureModel()
    # model.train()
    # plot_predictions_for_site_dead_fuel(site, fuel)
    # plot_r2_per_group()
