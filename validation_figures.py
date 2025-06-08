import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import onnxruntime as rt
from dead_fuel_moisture_model import DeadFuelMoistureModel
from live_fuel_moisture_model import FuelMoistureModel


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
    fets = pd.read_parquet("data/weather_site_features.parquet")

    fets["month"] = fets.date.dt.month
    fets["doy"] = fets.date.dt.dayofyear

    test = fets[fets.site == site].copy()

    test = add_fuel_columns(test, model)
    test[fuel] = 1.0
    features = (
        test[model.live_features_dict.keys()]
        .copy()
        .astype({k: v["type"] for k, v in model.live_features_dict.items()})
    )

    sess = rt.InferenceSession(
        "live_full_ddur.onnx", providers=["CPUExecutionProvider"]
    )
    pred_ort = sess.run(None, {"X": features.values.astype(np.float32)})[0]
    test["pred"] = pred_ort
    # Predict training dataset fuel moisture using the model
    dfr = model.prepare_training_dataset()
    features_dfr = (
        dfr[model.live_features_dict.keys()]
        .copy()
        .astype({k: v["type"] for k, v in model.live_features_dict.items()})
    )

    pred_ort_dfr = sess.run(None, {"X": features_dfr.values.astype(np.float32)})[0]
    dfr["pred"] = pred_ort_dfr

    return test, dfr


def plot_predictions_for_site_fuel(site, fuel):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    preds, dfr = predict_site_fuel_moisture(model, site, fuel)
    sel = dfr[(dfr.site == site) & (dfr[fuel] == 1.0)].copy()
    sns.lineplot(x="date", y="pred", data=preds, label="Prediction", color="0.5", ax=ax)
    # sns.scatterplot(x="date", y="pred", data=sel, color="green", label="Predicted FMC")
    sns.scatterplot(x="date", y="fmc_%", data=sel, color="red", label="Measured FMC")
    plt.title(
        f"Live Fuel Moisture Predictions for {site} - {fuel.replace('_', ' ').title()}"
    )
    plt.xlabel(" ")
    plt.ylabel("Fuel Moisture Content (%)")
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


if __name__ == "__main__":
    model = FuelMoistureModel()
    # model = DeadFuelMoistureModel()
    fuel = "fuel_Heather"  # Change this to the desired fuel type
    site = "Cobham Common H15"
    plot_predictions_for_site_fuel(site, fuel)
    # plot_predictions_for_site_dead_fuel(site, fuel)
    # plot_r2_per_group()
