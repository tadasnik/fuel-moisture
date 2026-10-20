from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import onnxruntime as rt

import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import linregress
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    r2_score,
)
from model_base import LiveFuelMoistureModel, DeadFuelMoistureModel, PhenologyModel

COLOR = "0.3"
plt.rcParams["font.family"] = "Fira Sans"
plt.rcParams["text.color"] = COLOR
plt.rcParams["axes.labelcolor"] = COLOR
plt.rcParams["axes.edgecolor"] = COLOR
plt.rcParams["xtick.color"] = COLOR
plt.rcParams["ytick.color"] = COLOR


def climatology_test(dfr: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    cli = dfr.groupby(["month", "fuel_type"])["fmc_%"].mean().reset_index()
    cli.rename(columns={"fmc_%": "clim"}, inplace=True)
    test = test.merge(cli, on=["month", "fuel_type"], how="left")
    return test


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


def add_fuel_columns(dfr, model):
    fuel_columns = [x for x in model.features_dict.keys() if x.startswith("fuel_")]
    for fuel in fuel_columns:
        dfr[fuel] = 0.0
    return dfr


def predict_site_dead_fuel_moisture(model, site, fuel):
    fets = pd.read_parquet("data/weather_site_features.parquet")

    fets["month"] = fets.date.dt.month
    fets["doy"] = fets.date.dt.dayofyear

    test = fets[fets.site == site].copy()

    test = add_fuel_columns(test, model)
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


def predict_site_fuel_moisture(dfr, dfrts, model, site, fuel):
    """Predict fuel moisture for a specific site and fuel type.
    Predictions are made using the LiveFuelMoistureModel and phenology model
    using the pickled pretrained model_base
    """
    # dfr = model.predict_phenology(dfr)
    dfrst = dfrts[dfrts.site == site].copy()
    dfrst["fuel"] = fuel
    # dfrst = model.predict_phenology(dfrst)
    dfrst = add_fuel_columns(dfrst, model)
    dfrst["fuel" + fuel] = 1.0
    dfrst["pred"] = model.predict(dfrst)
    return dfr, dfrst


def plot_predictions_for_site_dead_fuel(model, site, fuel):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    dfr
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
    # plt.savefig(f"figures/{site}_dead_{fuel}.png", dpi=300, bbox_inches="tight")
    plt.show()


def validation_nos_per_fuel(
    dfr,
    group_cols: List[str] = ["lonind", "latind"],
    y_column="fmc_%",
    prediction_column="pred",
    base_column="clim",
):
    res = []
    for fuel in dfr.fuel.unique():
        grouped = dfr[dfr.fuel == fuel].groupby(group_cols)
        for group_key, val_df in grouped:
            if val_df.shape[0] > 10:
                for model in [prediction_column, base_column]:
                    rms = mean_absolute_percentage_error(
                        val_df[y_column], val_df[model]
                    )
                    group_result = {
                        "group": group_key,
                        "mae": rms,
                        "fuel": fuel,
                        "model": model,
                        "size": val_df.shape[0],
                    }
                    res.append(group_result)
    return pd.DataFrame(res)


def plot_compare_score_per_group(df, col_val, col_bench):
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


def plot_training_vs_testing(model, X_train, X_test, y_train, y_test):
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    axe.scatter(
        y_train,
        model.predict(X_train),
        alpha=0.3,
    )
    axe.scatter(
        y_test,
        model.predict(X_test),
        c="r",
        alpha=0.3,
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


def plot_predicted_evi2_vs_obs_year_fuel(dfr, model, year, fuel):
    y_col = model.y_column
    # model = clone(model)
    test = dfr[(dfr["date"].dt.year == year) & (dfr["lc"] == fuel)].copy()
    train = dfr[dfr["date"].dt.year != year].copy()
    model.train_model(train)
    test["preds"] = model.predict(test)
    train["preds"] = model.predict(train)
    sl, inter, pearsr, pv, stde = linregress(test[y_col], test["preds"])
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


def get_rms_r2(dfr: pd.DataFrame, y_true: str, y_pred: str) -> tuple:
    """
    Calculate root mean squared error and R-squared value.
    """
    y_true = dfr[y_true].values
    y_pred = dfr[y_pred].values
    rms = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rms, r2


def dorset_surrey_and_uob_2025_results(fuel_type="Heather live canopy"):
    dfrl = model.prepare_training_dataset(
        fname="data/training_dataset_features_full.parquet"
    )
    uob = pd.read_parquet("data/training_dataset_features_uob_2025.parquet")
    ds = pd.read_parquet("data/training_dataset_features_dorset_surrey.parquet")
    if fuel_type == "Heather live canopy":
        uobs = uob[(uob.fuel == "Calluna canopy")].copy()
        dss = ds[
            (ds.Plant == "Calluna")
            & (ds.Component == "tips")
            & (ds["Live/dead"] == "live")
        ].copy()
    else:
        pass
    uobs["fuel_type"] = fuel_type
    dss["fuel_type"] = fuel_type
    uobs = model.predict_phenology(uobs)
    dss = model.predict_phenology(dss)
    uobs = add_fuel_columns(uobs, model)
    dss = add_fuel_columns(dss, model)
    uobs["fuel_type_" + fuel] = 1.0
    dss["fuel_type_" + fuel] = 1.0
    uobs["pred"] = model.predict(uobs)
    dss["pred"] = model.predict(dss)
    uobs = climatology_test(dfrl, uobs)
    dss = climatology_test(dfrl, dss)
    rms, r2 = get_rms_r2(
        pd.concat([uobs[["fmc_%", "pred"]], dss[["fmc_%", "pred"]]]), "fmc_%", "pred"
    )
    rmsc, r2c = get_rms_r2(
        pd.concat([uobs[["fmc_%", "clim"]], dss[["fmc_%", "clim"]]]), "fmc_%", "clim"
    )
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    sns.scatterplot(
        x="date", y="fmc_%", data=uobs, color="orange", label="Measured FMC", ax=ax
    )
    sns.scatterplot(
        x="date", y="fmc_%", data=dss, color="blue", label="Measured FMC", ax=ax
    )
    ax.set_title(f"R2: {r2:.2f}, R2C: {r2c:.2f} RMSE: {rms:.2f} - RMSC: {rmsc:.2f}")
    plt.show()

    oubt = pd.read_parquet("data/weather_site_features_uob_2025.parquet")


def plot_predictions_for_site_fuel(dfr, dfrts, model, site, fuel, var):
    dfr = dfr[dfr.fuel == fuel].copy()
    lonind = dfr[dfr.site == site].lonind.unique()[0]
    latind = dfr[dfr.site == site].latind.unique()[0]
    dfrtsub = dfrts[(dfrts.lonind == lonind) & (dfrts.latind == latind)].copy()
    dfrtnot = dfrts[(dfrts.lonind != lonind) & (dfrts.latind != latind)].copy()
    dfrsub = dfr[(dfr.lonind == lonind) & (dfr.latind == latind)].copy()
    dfrnot = dfr[(dfr.lonind != lonind) & (dfr.latind != latind)].copy()
    model.train_model(dfr[(dfr.lonind != lonind) & (dfr.latind != latind)])
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    ax2 = ax.twinx()
    dfrsub, dfrtsub = predict_site_fuel_moisture(dfrsub, dfrtsub, model, site, fuel)
    # dfrnot, dfrtnot = predict_site_fuel_moisture(dfrnot, dfrtnot, model, site, fuel)
    # sel = dfr[(dfr["fuel_type_" + fuel] == 1.0)].copy()
    # selsite = dfrsub[(dfrsub["fuel_type_" + fuel] == 1.0) & (dfr.site == site)].copy()
    sns.lineplot(
        x="date", y="pred", data=dfrtsub, label="Predicted lFMC", color="0.5", ax=ax
    )
    # dfrtnotmed = dfrtnot.groupby("date")[var].mean().reset_index()
    # sns.lineplot(
    #     x="date", y=var, data=dfrtnotmed, color="0.7", alpha=0.5, dashes=True, ax=ax2
    # )
    sns.scatterplot(
        x="date",
        y="fmc_%",
        data=dfrnot,
        color="0.8",
        label="Observed lFMC training",
        ax=ax,
    )
    sns.scatterplot(
        x="date", y="fmc_%", data=dfrsub, color="red", label="Observed lFMC test", ax=ax
    )
    # sns.lineplot(x="date", y=var, data=dfrtsub, color="green", alpha=0.5, ax=ax2)
    ax.set_ylabel("FMC (%)")
    # ax2.set_ylabel(var.replace("_", " ").title())
    # ax2.yaxis.label.set_color("green")
    # sns.lineplot(
    #     x="date", y="smm100-15mean", data=dfrtsub, color="green", alpha=0.5, ax=ax2
    # )
    plt.title(f"Validation of live FMC for {site} - {fuel.replace('_', ' ').title()}")
    plt.legend()
    # plt.savefig(
    #     f"figures/validation_{site}_live_{fuel}.png", dpi=300, bbox_inches="tight"
    # )
    plt.show()


def plot_predictions_for_fuel_all_sites_dorset(
    dfr: pd.DataFrame, dfrts: pd.DataFrame, model, fuel: str
):
    """Plot predictions for a specific fuel type across all sites.
    dfr: DataFrame with measured fuel moisture data and predictions.
    dfrts: DataFrame with time series data for predictions.
    """
    sites = dfr.site.unique()
    # dfrtsub = dfrts[dfrts.site == site].copy()
    # dfrsub = dfr[dfr.site == site].copy()
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    ax2 = ax.twinx()
    dfr["fuel_type"] = fuel
    dfr = model.predict_phenology_fuel_moisture(dfr)
    dfrts["fuel_type"] = fuel
    dfrts = model.predict_phenology_fuel_moisture(dfrts)
    for site in sites:
        if site != "Wareham":
            continue
        dfrsub = dfr[dfr.site == site].copy()
        dfrtsub = dfrts[dfrts.site == site].copy()
        sel = dfrsub[(dfrsub["fuel_type_" + fuel] == 1.0)].copy()
        selsite = dfrsub[
            (dfrsub["fuel_type_" + fuel] == 1.0) & (dfr.site == site)
        ].copy()
        sns.lineplot(x="date", y="pred", data=dfrtsub, label=site, ax=ax)
        # sns.scatterplot(x="date", y="pred", data=sel, color="green", label="Predicted FMC")
        sns.scatterplot(x="date", y="fmc_%", data=sel, label=site, ax=ax)
        sns.lineplot(x="date", y="smm100", data=dfrtsub, alpha=0.5, dashes=True, ax=ax2)
        sns.lineplot(
            x="date", y="vpdmax-15mean", data=dfrtsub, alpha=0.3, dashes=True, ax=ax2
        )

    # sns.scatterplot(
    #     x="date", y="fmc_%", data=selsite, color="red", label="Measured FMC", ax=ax
    # )
    # sns.lineplot(x="date", y="smm100", data=dfrtsub, color="green", alpha=0.5, ax=ax2)
    # plt.title(f"LFM for {site} - {fuel.replace('_', ' ').title()}")
    plt.legend()
    # plt.savefig(f"figures/new_{site}_live_{fuel}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_predictions_for_fuel_all_sites(
    dfr: pd.DataFrame, dfrts: pd.DataFrame, model, fuel: str
):
    """Plot predictions for a specific fuel type across all sites.
    dfr: DataFrame with measured fuel moisture data and predictions.
    dfrts: DataFrame with time series data for predictions.
    """
    sites = dfr.site.unique()
    # dfrtsub = dfrts[dfrts.site == site].copy()
    # dfrsub = dfr[dfr.site == site].copy()
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7), sharey=True)
    ax2 = ax.twinx()
    dfr["fuel_type"] = fuel
    dfr = model.predict_phenology_fuel_moisture(dfr)
    dfrts["fuel_type"] = fuel
    dfrts = model.predict_phenology_fuel_moisture(dfrts)
    sites_list = [
        # ["Horsey_Scotland"],
        # ["CringleMoor_England"],
        [
            "Kinver1_England",
            "Kinver4_England",
            "Kinver7_England",
            "Kinver10_England",
            "Kinver13_England",
            "Kinver16_England",
            "Kinver19_England",
        ],
        # ["ForsinainFarm_Scotland"],
    ]
    for site_l in sites_list:
        dfrsub = dfr[dfr.site.isin(site_l)].copy()
        dfrtsub = dfrts[dfrts.site == site_l[0]].copy()
        sel = dfrsub[(dfrsub["fuel_type_" + fuel] == 1.0)].copy()
        # selsite = dfrsub[
        #     (dfrsub["fuel_type_" + fuel] == 1.0) & (dfr.site.isin(site_l))
        # ].copy()
        sns.lineplot(x="date", y="pred", data=dfrtsub, label=site_l[0], ax=ax)
        # sns.scatterplot(x="date", y="pred", data=sel, color="green", label="Predicted FMC")
        sns.scatterplot(x="date", y="fmc_%", data=sel, label=site_l[0], ax=ax)
        sns.lineplot(
            x="date",
            y="smm100-15mean",
            data=dfrtsub,
            alpha=0.5,
            color="orange",
            dashes=True,
            ax=ax2,
        )
        # sns.lineplot(
        # sns.lineplot(

        # sns.lineplot(
        # x="date", y="prec-15sum", data=dfrtsub, alpha=0.5, dashes=True, ax=ax2
        # )
        # sns.lineplot(x="date", y="smm100", data=dfrtsub, alpha=0.5, dashes=True, ax=ax2)

    # sns.scatterplot(
    #     x="date", y="fmc_%", data=selsite, color="red", label="Measured FMC", ax=ax
    # )
    # sns.lineplot(x="date", y="smm100", data=dfrtsub, color="green", alpha=0.5, ax=ax2)
    # plt.title(f"LFM for {site} - {fuel.replace('_', ' ').title()}")
    plt.legend()
    # plt.savefig(f"figures/new_{site}_live_{fuel}.png", dpi=300, bbox_inches="tight")
    plt.show()


def validation_per_fuel_location(
    model, dfr, fuel, group_cols: List[str] = ["lonind", "latind"]
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
    print("features", model.features_dict.keys())
    for group_key, val_df in grouped:
        if val_df[val_df.fuel_type == fuel].shape[0] < 20:
            continue  # Skip groups with too few samples
        else:
            print(
                "proc group",
                group_key,
                "size",
                val_df[val_df[model.fuels_cat_column] == fuel].shape,
            )
        pred_df = val_df[val_df[model.fuels_cat_column] == fuel].copy()
        train_df = dfr.loc[~dfr.index.isin(val_df.index)]
        X_train = train_df[self.features_dict.keys()]
        y_train = train_df[self.y_column]
        X_val = pred_df[self.features_dict.keys()]
        y_val = pred_df[self.y_column]
        model_copy = clone(self.model)
        model_copy.fit(X_train, y_train)
        y_pred = model_copy.predict(X_val)
        pred_df["prediction"] = y_pred
        pred_df = climatology_test(train_df, pred_df)
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


def preds_vs_bench_(df, bench_col, pred_col, title):
    fig = plt.figure(figsize=(6, 4))
    res = validation_nos_per_fuel(df)
    sns.violinplot(
        data=res, x="fuel", y="mae", hue="model", split=True, gap=0.1, inner="quart"
    )
    sns.catplot(data=res, x="model", y="mae", hue="fuel", kind="point", legend=False)
    plt.title(f"Dead FMC validation per site")
    # plt.savefig(f"figures/r2_validation_dead.png", dpi=300, bbox_inches="tight")

    torange = (1.0, 0.4980392156862745, 0.054901960784313725)
    tblue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    tgreen = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    tpurple = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    tpink = (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
    tbrown = (0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
    tchaki = (0.7372549019607844, 0.7411764705882353, 0.13333333333333333)
    tgrey = (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)
    tred = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
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

    for nr, fuel in enumerate(df.fuel.unique()):
        dfs = df[df.fuel == fuel]
        mark = markers.get(fuel.split()[-1], "*")
        axe.scatter(
            dfs["fmc_%"],
            dfs["pred"],
            marker=mark,
            alpha=0.7,
            color=colors.get(fuel, tred),
            s=10,
            label=fuel,
        )
    axe.plot([0, 62], [0, 62], color=tgrey, linestyle="--", linewidth=1)
    axe.set_ylim(0, 62)
    axe.set_xlim(0, 62)
    axe.legend(markerscale=2)
    axe.set_xlabel("Observed Dead FMC (%)")
    axe.set_ylabel("Predicted Dead FMC (%)")
    plt.savefig(
        Path("figures", "dead_location_validation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def dfmc_location_validation():
    df = pd.read_parquet("data/tmp/dead_location_validation.parquet")
    torange = (1.0, 0.4980392156862745, 0.054901960784313725)
    tblue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    tgreen = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    tpurple = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    tpink = (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
    tbrown = (0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
    tchaki = (0.7372549019607844, 0.7411764705882353, 0.13333333333333333)
    tgrey = (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)
    tred = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
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

    for nr, fuel in enumerate(df.fuel.unique()):
        dfs = df[df.fuel == fuel]
        mark = markers.get(fuel.split()[-1], "*")
        axe.scatter(
            dfs["fmc_%"],
            dfs["pred"],
            marker=mark,
            alpha=0.7,
            color=colors.get(fuel, tred),
            s=10,
            label=fuel,
        )
    axe.plot([0, 62], [0, 62], color=tgrey, linestyle="--", linewidth=1)
    axe.set_ylim(0, 62)
    axe.set_xlim(0, 62)
    axe.legend(markerscale=2)
    axe.set_xlabel("Observed Dead FMC (%)")
    axe.set_ylabel("Predicted Dead FMC (%)")
    # plt.savefig(
    #     Path("figures", "dead_location_validation.png"),
    #     dpi=300,
    #     bbox_inches="tight",
    # )
    plt.show()


def lfmc_location_validation():
    df = pd.read_parquet("data/tmp/live_location_validation.parquet")
    torange = (1.0, 0.4980392156862745, 0.054901960784313725)
    tblue = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    tgreen = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    tpurple = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    tpink = (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)
    tbrown = (0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
    tchaki = (0.7372549019607844, 0.7411764705882353, 0.13333333333333333)
    tgrey = (0.4980392156862745, 0.4980392156862745, 0.4980392156862745)
    tred = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)

    fuels = [
        "Bracken leaves",
        "Bracken stem",
        "Gorse canopy",
        "Gorse stem",
        "Heather canopy",
        "Heather stem",
        "Moor grass",
    ]
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
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
            # alpha=0.7,
            color=colors.get(fuel, tred),
            s=10,
            label=fuel,
        )
    min_lim = 25
    max_lim = 310
    axe.plot(
        [min_lim, max_lim], [min_lim, max_lim], color=tgrey, linestyle="--", linewidth=1
    )
    axe.set_ylim(min_lim, max_lim)
    axe.set_xlim(min_lim, max_lim)
    axe.legend(markerscale=2)
    axe.set_xlabel("Observed Live FMC (%)")
    axe.set_ylabel("Predicted Live FMC (%)")
    plt.savefig(
        Path("figures", "live_location_validation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    pass
    # lfmc_location_validation()
    # model = LiveFuelMoistureModel(
    #     pickled_model_fname="lfmc_model_no_evi.onnx", phenology_model="ph_model.onnx"
    # )
    # model = LiveFuelMoistureModel(
    #     phenology_ph_model="ph_model_q.onnx", phenology_phs_model="ph_model_q_phs.onnx"
    # )
    # dfrl = model.prepare_training_dataset(
    #     fname="data/training_dataset_features_full.parquet"
    # )

    # ph_model = PhenologyModel(pickled_model_fname="ph_model.onnx")
    # ph_model = PhenologyModel()
    # dfr = ph_model.prepare_training_dataset(
    #     fname="data/phenology_training_dataset_features.parquet"
    # )
    #
    # uob = pd.read_parquet("data/training_dataset_features_uob_2025.parquet")
    # oub = uob.reset_index(drop=True)
    # ds = pd.read_parquet("data/training_dataset_features_dorset_surrey_sm.parquet")
    # dss = ds[
    #     (ds.Plant == "Calluna") & (ds.Component == "tips") & (ds["Live/dead"] == "live")
    # ].copy()
    #
    # uob = uob[(uob.fuel == "Calluna canopy")].copy()
    # uob["fuel"] = "Heather canopy"
    # uob["fuel_cat"] = "live"
    # uob = model.encode_fuel_features(uob)
    # # uob = model.predict_phenology_fuel_moisture(uob)
    # cols = list(
    #     model.feature_columns + ["site", "fuel", "lonind", "latind", "date", "fmc_%"]
    # )
    # dfrl_feats = pd.concat([dfrl[cols], uob[cols]], axis=0)
    # dfrl_feats = pd.concat([dfrl, uob], ignore_index=True)
    #
    # uobt = pd.read_parquet("data/weather_site_features_uob_2025.parquet")
    # uobt = uobt.reset_index(drop=True)
    #
    # dsst = pd.read_parquet("data/weather_site_features_dorset_surrey_sm.parquet")
    # dsst = dsst.reset_index(drop=True)
    # dfrl = model.prepare_training_dataset(
    #     fname="data/training_dataset_features_full_sm.parquet"
    # )
    # model.train_model(dfrl_feats)
    #
    # plot_predictions_for_fuel_all_sites(uob, uobt, model, "Heather live canopy")
    # plot_predictions_for_fuel_all_sites_dorset(dss, dsst, model, "Heather live canopy")

    # dfrts = pd.read_parquet("data/training_dataset_features_full_time_series.parquet")
    # plot_predicted_evi2_vs_obs_year_fuel(dfr, ph_model, 2022, 9)

    # model.validation_train_model()
    model = DeadFuelMoistureModel()  # pickled_model_fname="model_onehot_dead.onnx")
    dfr = model.prepare_training_dataset(
        fname="data/training_dataset_features_full.parquet"
    )
    dfrts = pd.read_parquet("data/training_dataset_features_full_time_series.parquet")
    fuel = "Heather canopy"  # Change this to the desired fuel type
    # fuel = "Gorse live canopy"  # Change this to the desired fuel type
    site = "Cobham Common H15"
    # site = "Ockham Common H15"
    # site = "Sugar Loaf H6"
    # site = "Thursley Common H14"
    # var = "ph"
    var = "prec-15sum"  # Change this to the desired variable
    plot_predictions_for_site_fuel(
        dfr, dfrts[dfrts.date < dfr.date.max()], model, site, fuel, var
    )
