import time

import pandas as pd
from sklearn.model_selection import LearningCurveDisplay

from grids import ERA5LandGrid
from open_meteo import fetch_hourly, fetch_daily


def proc_fuel_moisture_meg():
    """Read and prepare observed fuel moisture data for UK"""
    dfr = pd.concat(
        pd.read_excel(
            "/Users/tadas/repos/fuel-moisture/data/Flammability Data 30.01.xlsx",
            sheet_name=None,
        ),
        ignore_index=True,
    )
    dfr = dfr.ffill()
    dfr[["latitude", "longitude"]] = (
        dfr["Coordinates"].str.split(", ", n=1, expand=True).astype("float")
    )
    dfr.loc[dfr.longitude > 0, "longitude"] = (
        dfr.loc[dfr.longitude > 0, "longitude"] * -1
    )
    dfr["date"] = pd.to_datetime(dfr.Date.astype(str) + " " + dfr.Time.astype(str))


def proc_fuel_moisture_UK():
    """Read and prepare observed fuel moisture data for UK"""
    dfr = pd.read_csv("./data/UK_fuel_moisture.csv")
    dfr.columns = dfr.columns.str.lower()
    dfr["date"] = pd.to_datetime(dfr["date"], dayfirst=True)
    # filling missing time of collection with 13:00
    dfr["time_collected"] = dfr["time_collected"].fillna(1300.0)
    # add collection time to date
    hours = pd.to_timedelta((dfr["time_collected"] * 0.01).astype(int), "h")
    dfr["date"] = dfr["date"] + hours
    dfr["date"] = (
        pd.to_datetime(dfr["date"]).dt.tz_localize("Europe/London").dt.tz_convert("UTC")
    )
    dfr["year"] = dfr["date"].dt.year
    dfr["week"] = dfr["date"].dt.isocalendar().week
    dfr["month"] = dfr["date"].dt.month
    er5g = ERA5LandGrid()
    xx, yy = er5g.find_point_xy(dfr["latitude"], dfr["longitude"])
    dfr["latind"] = yy.astype(int)
    dfr["lonind"] = xx.astype(int)
    return dfr


def get_weather_features(dfr: pd.DataFrame) -> pd.DataFrame:
    """Fetch historical weather for given observations in dfr which
    must have columns: site, latitude, longitude, latind, lonind and date"""
    # First fetch hourly weather variables per unique grid cell
    url = "https://archive-api.open-meteo.com/v1/archive"

    dfrg = (
        dfr.groupby(["site"])[
            ["longitude", "latitude", "latind", "lonind", "slope", "aspect"]
        ]
        .first()
        .reset_index()
    )
    hourly_variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "vapour_pressure_deficit",
        "global_tilted_irradiance",
    ]
    results = []
    for nr, row in dfrg.iterrows():
        print("Fetching data for row", nr, row)
        mindate = dfr.loc[
            (dfr.latind == row.latind) & (dfr.lonind == row.lonind), "date"
        ].min()
        maxdate = dfr.loc[
            (dfr.latind == row.latind) & (dfr.lonind == row.lonind), "date"
        ].max()
        start_date = (mindate - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        end_date = (maxdate + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        opt_params = {
            "tilt": row.slope,
            "azimuth": row.aspect - 180,
        }

        res = fetch_hourly(
            url,
            row.latitude,
            row.longitude,
            start_date,
            end_date,
            hourly_variables,
            opt_params,
        )
        time.sleep(1)
        res_daily = fetch_daily(
            url,
            row.latitude,
            row.longitude,
            start_date,
            end_date,
            [
                "soil_moisture_0_to_7cm_mean",
                "soil_moisture_7_to_28cm_mean",
                "soil_moisture_28_to_100cm_mean",
                "daylight_duration",
                "sunshine_duration",
            ],
        )
        res_comb = pd.concat([res, pd.DataFrame(row).T.reset_index(drop=True)], axis=1)
        res_comb = res_comb.ffill()
        res_comb["date_"] = res_comb.date.dt.floor("d")

        res_daily["date_"] = res_daily.date.dt.floor("d")
        res_comb = res_comb.merge(
            res_daily[
                [
                    "date_",
                    "soil_moisture_0_to_7cm_mean",
                    "soil_moisture_7_to_28cm_mean",
                    "soil_moisture_28_to_100cm_mean",
                    "daylight_duration",
                    "sunshine_duration",
                ]
            ],
            on=["date_"],
            how="left",
        )
        results.append(res_comb)
        time.sleep(10)
    results.to_parquet("data/weather_results.parquet", index=False)
    return pd.concat(results)


def prepare_weather_features() -> pd.DataFrame:
    # First generate columns with 5 hours of past vpd and gti values
    dfr = proc_fuel_moisture_UK()
    results = pd.read_parquet("data/weather_results.parquet")
    results.rename(
        {
            "vapour_pressure_deficit": "vpd",
            "soil_moisture_0_to_7cm_mean": "smm7",
            "soil_moisture_7_to_28cm_mean": "smm28",
            "soil_moisture_28_to_100cm_mean": "smm100",
            "global_tilted_irradiance": "gti",
            "daylight_duration": "ddur",
            "sunshine_duration": "sdur",
        },
        axis=1,
        inplace=True,
    )
    for hour in range(1, 6):
        results[f"vpd-{hour}"] = results["vpd"].shift(hour)
        results[f"gti-{hour}"] = results["gti"].shift(hour)
    # Then, generate columns with 24 hours of past soil moisture values
    # for hour in range(1, 25):
    #     results[f"smm7-{hour}"] = results["smm7"].shift(hour)
    #     results[f"smm28-{hour}"] = results["smm28"].shift(hour)
    #     results[f"smm100-{hour}"] = results["smm100"].shift(hour)

    daily_vpd = results.groupby(results.date_)["vpd"].mean().reset_index()
    # Step 2: Create a DataFrame of lagged daily values

    for i in range(1, 16):
        daily_vpd[f"vpd-{i}d"] = daily_vpd["vpd"].shift(i)

    daily_vpd.rename(columns={"vpd": "vpd-0d"}, inplace=True)

    results = results.merge(daily_vpd, on="date_", how="left")
    # Training dataset
    fe = dfr.merge(
        results.drop(
            ["slope", "aspect", "lonind", "latind", "longitude", "latitude"], axis=1
        ),
        on=["site", "date"],
        how="left",
    )
    fe.to_parquet("data/training_dataset_features.parquet")
    # Time Series features dataset

    fe_time_series = results.merge(
        dfr.groupby(["site"])["elevation"].first().reset_index(),
        on=["site"],
        how="left",
    )
    fe_time_series.to_parquet("data/weather_site_features.parquet")


def get_soil_features(dfr: pd.DataFrame) -> pd.DataFrame:
    """Fetch historical weather for given observations in dfr which
    must have columns: site, latitude, longitude, latind, lonind and date"""
    # First fetch hourly weather variables per unique grid cell
    url = "https://archive-api.open-meteo.com/v1/archive"

    dfrg = (
        dfr.groupby(["site"])[
            ["longitude", "latitude", "latind", "lonind", "slope", "aspect"]
        ]
        .first()
        .reset_index()
    )
    daily_variables = [
        "global_tilted_irradiance",
    ]
    results = []
    for nr, row in dfrg.iterrows():
        print("Fetching data for row", nr, row)
        mindate = dfr.loc[
            (dfr.latind == row.latind) & (dfr.lonind == row.lonind), "date"
        ].min()
        maxdate = dfr.loc[
            (dfr.latind == row.latind) & (dfr.lonind == row.lonind), "date"
        ].max()
        start_date = (mindate - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        end_date = (maxdate + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        opt_params = {
            "tilt": row.slope,
            "azimuth": row.aspect - 180,
        }

        res = fetch_hourly(
            url,
            row.latitude,
            row.longitude,
            start_date,
            end_date,
            hourly_variables,
            opt_params,
        )
        res_comb = pd.concat([res, pd.DataFrame(row).T.reset_index(drop=True)], axis=1)
        results.append(res_comb.ffill())

        time.sleep(10)
    return pd.concat(results)


def fetch_gti_hourly_hist(dfr):
    dfrg = (
        dfr.groupby(["site"])[["latitude", "longitude", "aspect", "slope"]]
        .first()
        .reset_index()
    )
    dfrs = []
    start_date = "2021-01-01"
    end_date = dfr.date.max().strftime("%Y-%m-%d")
    for nr, row in dfrg.iterrows():
        try:
            vpd_h = fetch_global_tilted_irradiance(
                row.latitude,
                row.longitude,
                start_date,
                end_date,
                row.slope,
                row.aspect - 180,
            )
            vpd_h["site"] = row.site
            dfrs.append(vpd_h)
        except:
            print(f"Failed to fetch data for row {nr}, {row}")
            break
    return pd.concat(dfrs)


def get_vpd_precipitation(dfr):
    er5g = ERA5LandGrid()
    xx, yy = er5g.find_point_xy(dfr["latitude"], dfr["longitude"])
    dfr["latind"] = yy.astype(int)
    dfr["lonind"] = xx.astype(int)
    dfrg = (
        dfr.groupby(["latind", "lonind"])[["latitude", "longitude"]]
        .median()
        .reset_index()
    )
    start_date = "2021-01-01"
    end_date = dfr.date.max().strftime("%Y-%m-%d")
    dfrs = []
    for nr, row in dfrg.iterrows():
        try:
            vpd_h = fetch_vpd_prec(
                row.latitude,
                row.longitude,
                start_date,
                end_date,
            )
            vpd_h["latind"] = row.latind.astype(int)
            vpd_h["lonind"] = row.lonind.astype(int)
            dfrs.append(vpd_h)
        except:
            print(f"Failed to fetch data for row {nr}, {row}")
            break
    return pd.concat(dfrs)


def get_features(fuel_type, days):
    """A very slow function to get features for fuel type"""
    dfr = proc_fuel_moisture_UK()
    vp = pd.read_parquet("data/vp.parquet")
    gti = pd.read_parquet("data/gti.parquet")

    # fuel_type = "Moor grass dead"
    # dfs = dfr[dfr.fuel_type == fuel_type]
    dfs = dfr.copy()
    dfsg = dfs[
        [
            "fmc_%",
            "site",
            "date",
            "lonind",
            "latind",
            "elevation",
            "slope",
            "aspect",
            "fuel_type",
        ]
    ].copy()
    dfsgs = []
    for nr, row in dfsg.iterrows():
        # find precipitation and vpd for the site and date
        start_date = row.date - pd.Timedelta(f"{days} day")
        end_date = row.date
        vps = vp[
            (vp.latind == row.latind)
            & (vp.lonind == row.lonind)
            & (vp.date >= start_date)
            & (vp.date <= end_date)
        ]
        gtis = gti[
            (gti.site == row.site) & (gti.date >= start_date) & (gti.date <= end_date)
        ]
        vpds = (
            vps["vapour_pressure_deficit"].to_list()
            + vps["precipitation"].to_list()
            + gtis["global_tilted_irradiance"].to_list()
        )
        column_names = (
            [f"vpd_{i}" for i in range(len(vps))]
            + [f"prec_{i}" for i in range(len(vps))]
            + [f"gti_{i}" for i in range(len(gtis))]
        )
        df = pd.DataFrame([vpds], columns=column_names)
        df = df.assign(**(row.to_frame().T.reset_index(drop=True)))

        dfsgs.append(df)
    fe = pd.concat(dfsgs)
    fe.to_parquet(f"data/features_{fuel_type}_{days}.parquet")
    return fe
    # fe.to_parquet(f"data/features_{fuel_type}.parquet")
    #


if __name__ == "__main__":
    # dfr = proc_fuel_moisture_UK()
    # results = pd.read_parquet("data/weather_results.parquet")
    # dfr = proc_fuel_moisture_UK()
    # results = get_weather_features(dfr)
    prepare_weather_features()
    # results.to_parquet("data/weather_features.parquet")
    # get_features("Moor grass dead", 7)
    # get_features("Moor grass dead", 14)
    # get_features("Moor grass dead", 30)
    # get_features("Moor grass dead", 60)
