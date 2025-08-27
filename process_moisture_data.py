import time
import math

import pandas as pd
from sklearn.model_selection import LearningCurveDisplay

from grids import ERA5LandGrid
from open_meteo import fetch_hourly, fetch_daily, get_elevation

from live_fuel_moisture_model import FuelMoistureModel


def offset_lat_lon(lat, lon, distance_m, bearing_deg):
    """Calculate destination point given distance and bearing (spherical)."""
    R = 6371000  # Earth radius in meters
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(distance_m / R)
        + math.cos(lat1) * math.sin(distance_m / R) * math.cos(bearing)
    )

    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(distance_m / R) * math.cos(lat1),
        math.cos(distance_m / R) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def calculate_slope_aspect(e_center, e_n, e_s, e_e, e_w, distance=90):
    """Calculate slope (degrees) and aspect (azimuth from north, degrees)."""
    # dz/dx = (e_e - e_w) / (2 * dx)
    # dz/dy = (e_s - e_n) / (2 * dy)  -> note: y axis points south
    dz_dx = (e_e - e_w) / (2 * distance)
    dz_dy = (e_s - e_n) / (2 * distance)

    slope_rad = math.atan(math.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = math.degrees(slope_rad)

    aspect_rad = math.atan2(dz_dy, -dz_dx)
    aspect_deg = math.degrees(aspect_rad)

    if aspect_deg < 0:
        aspect_deg += 360

    return slope_deg, aspect_deg


def get_elevation_slope_aspect(lat, lon):
    # Elevation at center point
    e_center = get_elevation(lat, lon)

    # Get surrounding points at 90m distance
    lat_n, lon_n = offset_lat_lon(lat, lon, 90, 0)
    lat_e, lon_e = offset_lat_lon(lat, lon, 90, 90)
    lat_s, lon_s = offset_lat_lon(lat, lon, 90, 180)
    lat_w, lon_w = offset_lat_lon(lat, lon, 90, 270)

    # Retrieve elevations
    e_n = get_elevation(lat_n, lon_n)
    e_e = get_elevation(lat_e, lon_e)
    e_s = get_elevation(lat_s, lon_s)
    e_w = get_elevation(lat_w, lon_w)

    # Calculate slope and aspect
    slope, aspect = calculate_slope_aspect(e_center, e_n, e_s, e_e, e_w)

    return {
        "elevation": e_center,
        "slope": slope,
        "aspect": aspect,
    }


def get_terrain(dfr):
    dfrg = (
        dfr.groupby(["site"])[["longitude", "latitude", "latind", "lonind"]]
        .first()
        .reset_index()
    )
    results = []
    for nr, row in dfrg.iterrows():
        terrain = get_elevation_slope_aspect(row.latitude, row.longitude)
        terrain["site"] = row.site
        results.append(terrain)
    df = pd.DataFrame(results)
    dfr = dfr.merge(df, on="site", how="left")
    return dfr


def proc_uob_2025():
    df = pd.read_excel("data/spring_2025_fuelmoisture_Uob.xlsx", sheet_name="Sheet1")
    # Integrate summer data from ghostplot
    dfs = pd.read_excel("data/kinver_ghostplot_fmc.xlsx")
    dfs = dfs.rename(
        {"Date": "date", "Time": "time", "Fuel moisture percent": "fuel moisture"},
        axis=1,
    )
    dfs["location"] = "Kinver1_England"
    dfs["lat"] = df[df["location"] == "Kinver1_England"].lat.values[0]
    dfs["lon"] = df[df["location"] == "Kinver1_England"].lon.values[0]
    dfs["fuel"] = ""
    dfs.loc[dfs.Fuel == "C", "fuel"] = "Calluna canopy"
    dfs.loc[dfs.Fuel == "S", "fuel"] = "Calluna stem"
    dfs.loc[dfs.Fuel == "O", "fuel"] = "Organic"
    df = pd.concat(
        [df, dfs[["location", "date", "time", "fuel", "fuel moisture", "lon", "lat"]]]
    )
    dt = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    df["date"] = dt.dt.tz_localize("Europe/London").dt.tz_convert("UTC")
    df["date"] = df["date"].dt.round("h")
    df = df.drop("time", axis=1)
    df = df.rename(
        {
            "fuel moisture": "fmc_%",
            "location": "site",
            "lon": "longitude",
            "lat": "latitude",
        },
        axis=1,
    )
    df["year"] = df["date"].dt.year
    df["week"] = df["date"].dt.isocalendar().week
    df["month"] = df["date"].dt.month
    er5g = ERA5LandGrid()
    xx, yy = er5g.find_point_xy(df["latitude"], df["longitude"])
    df["latind"] = yy.astype(int)
    df["lonind"] = xx.astype(int)
    df = get_terrain(df)
    get_weather_features(df, "data/weather_features_uob_2025.parquet")
    weather = pd.read_parquet("data/weather_features_uob_2025.parquet")
    fe, fe_time_series = prepare_weather_features(df, weather)
    fe.to_parquet("data/training_dataset_features_uob_2025.parquet")
    fe_time_series.to_parquet("data/weather_site_features_uob_2025.parquet")


def join_dorset_surrey_and_uob_2025(fuel_type="Heather_live_canopy"):
    uob = pd.read_parquet("data/training_dataset_features_uob_2025.parquet")
    ds = pd.read_parquet("data/training_dataset_features_dorset_surrey.parquet")
    if fuel_type == "Heather_live_canopy":
        uobs = uob[(uob.fuel == "Calluna canopy")]
        dss = ds[(ds.Plant == "Calluna") & (ds.Component == "tips")]
    oub_mapping = {
        "Calluna canopy": "Heather live canopy",
        "Calluna stem": "Heather live stem",
    }


def proc_dorset_surrey():
    """Read and prepare observed fuel moisture data for Dorset and Surrey from Claire"""
    df = pd.read_excel(
        "data/Dorset-Surrey_Fuel_moisture_April_25_mod.xlsx", sheet_name="Sheet1 (2)"
    )

    df = df.dropna(subset=["Time", "date"])
    df["date_clean"] = df["date"].str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True)

    # Combine date and time into a single string
    df["datetime_str"] = df["date_clean"] + " " + df["Time"].astype(str)

    # Step 3: Parse into datetime (assumes '25' means 2025; you can adjust that if needed)
    df["date"] = pd.to_datetime(
        df["datetime_str"], format="%d %B %y %H:%M:%S", utc=True
    )
    df["date"] = df["date"].dt.round("h")

    # Optional: Drop the intermediate columns if you don't need them
    df.drop(columns=["date_clean", "datetime_str"], inplace=True)
    df = df.rename(
        {
            "long": "longitude",
            "lat": "latitude",
            "Fuel moisture (%)": "fmc_%",
            "Site": "site",
        },
        axis=1,
    )
    df["year"] = df["date"].dt.year
    df["week"] = df["date"].dt.isocalendar().week
    df["month"] = df["date"].dt.month
    er5g = ERA5LandGrid()
    xx, yy = er5g.find_point_xy(df["latitude"], df["longitude"])
    df["latind"] = yy.astype(int)
    df["lonind"] = xx.astype(int)
    df = get_terrain(df)
    weather = get_weather_features(df, "data/weather_features_dorset_surrey.parquet")
    fe, fe_time_series = prepare_weather_features(df, weather)
    fe.to_parquet("data/training_dataset_features_dorset_surrey.parquet")
    fe_time_series.to_parquet("data/weather_site_features_dorset_surrey.parquet")
    return df


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


def sample_phenology_locations() -> pd.DataFrame:
    """Sample phenology locations from the CEH Land Cover dataset and store to file"""
    lcs = [3, 4, 7, 9, 10]
    res = []
    for lc in lcs:
        df = pd.read_parquet(
            f"/Users/tadas/modFire/fire_lc_ndvi/data/cehlc/LCD_2018_sampled_eroded_5_lc_{lc}.parquet"
        )
        samp = df.groupby("Region").sample(n=3).reset_index(names="fid")
        res.append(samp)
    results = pd.concat(res)
    er5g = ERA5LandGrid()
    xx, yy = er5g.find_point_xy(results["latitude"], results["longitude"])
    results["latind"] = yy.astype(int)
    results["lonind"] = xx.astype(int)
    resg = (
        results.groupby(["latind", "lonind"])[
            ["latitude", "longitude", "fid", "Region", "lc"]
        ]
        .first()
        .reset_index()
    )
    resg.to_parquet("data/phenology_sampled_locations.parquet")
    return resg


def get_phenology_features() -> pd.DataFrame:
    """Fetch daily weather features for given points in dfr which
    must have columns: site, latitude, longitude, latind, lonind and date"""
    dfr = pd.read_parquet("data/phenology_sampled_locations.parquet")
    try:
        completed = pd.read_parquet("data/phenology_weather.parquet")
    except FileNotFoundError:
        completed = pd.DataFrame()
    start_date = "2012-01-01"
    end_date = "2024-02-18"
    url = "https://archive-api.open-meteo.com/v1/archive"
    for nr, row in dfr.iterrows():
        if not completed.empty:
            comp_sub = completed[
                (completed.lonind == row.lonind) & (completed.latind == row.latind)
            ]
            print(f"Completed rows for {row.lonind}, {row.latind}: {len(comp_sub)}")
            if len(comp_sub) > 0:
                print("Skipping already completed row", nr, row)
                continue
        #     if (row.lonind in completed.lonind.values) & (
        #         row.latind in completed.latind.values
        #     ):
        #         print("yes")
        #         continue
        # print("Fetching data for row", nr, row)
        res_daily = fetch_daily(
            url,
            row.latitude,
            row.longitude,
            start_date,
            end_date,
            [
                "shortwave_radiation_sum",
                "temperature_2m_mean",
                # "temperature_2m_max",
                # "temperature_2m_min",
                "sunshine_duration",
                "precipitation_sum",
                "daylight_duration",
                "vapour_pressure_deficit_max",
                "soil_moisture_7_to_28cm_mean",
                "soil_moisture_28_to_100cm_mean",
            ],
        )
        res_daily["latind"] = row.latind
        res_daily["lonind"] = row.lonind
        time.sleep(180)  # Sleep to avoid rate limiting
        completed = pd.concat([completed, res_daily], ignore_index=True)
        completed.to_parquet("data/phenology_weather.parquet")
    return completed


def get_phenology_vpd() -> pd.DataFrame:
    """Fetch daily weather features for given points in dfr which
    must have columns: site, latitude, longitude, latind, lonind and date"""
    dfr = pd.read_parquet("data/phenology_sampled_locations.parquet")
    try:
        completed = pd.read_parquet("data/phenology_weather_daylight_vpd.parquet")
    except FileNotFoundError:
        completed = pd.DataFrame()
    start_date = "2012-01-01"
    end_date = "2024-02-18"
    url = "https://archive-api.open-meteo.com/v1/archive"
    for nr, row in dfr.iterrows():
        if not completed.empty:
            comp_sub = completed[
                (completed.lonind == row.lonind) & (completed.latind == row.latind)
            ]
            print(f"Completed rows for {row.lonind}, {row.latind}: {len(comp_sub)}")
            if len(comp_sub) > 0:
                print("Skipping already completed row", nr, row)
                continue
        print("Fetching data for row", nr, row)
        res_daily = fetch_daily(
            url,
            row.latitude,
            row.longitude,
            start_date,
            end_date,
            [
                "vapour_pressure_deficit_max",
            ],
        )
        res_daily["latind"] = row.latind
        res_daily["lonind"] = row.lonind
        time.sleep(120)  # Sleep to avoid rate limiting
        completed = pd.concat([completed, res_daily], ignore_index=True)
        completed.to_parquet("data/phenology_weather_daylight_vpd.parquet")
    return completed


def prepare_phenology_features():
    """merges weather record with evi data"""
    sampled = pd.read_parquet("data/phenology_sampled_locations.parquet")
    weather = pd.read_parquet("data/phenology_weather.parquet")
    weather = weather.drop("daylight_duration", axis=1)
    soil = pd.read_parquet("data/phenology_weather_daylight_soil.parquet")
    vpd = pd.read_parquet("data/phenology_weather_daylight_vpd.parquet")
    weather = weather.merge(soil, on=["latind", "lonind", "date"], how="left")
    weather = weather.merge(vpd, on=["latind", "lonind", "date"], how="left")

    weather.rename(
        {
            "shortwave_radiation_sum": "sri",
            "temperature_2m_mean": "tmean",
            "temperature_2m_max": "tmax",
            "temperature_2m_min": "tmin",
            # "sunshine_duration": "sdur",
            # "precipitation_sum": "prec",
            "soil_moisture_7_to_28cm_mean": "smm28",
            "soil_moisture_28_to_100cm_mean": "smm100",
            "daylight_duration": "ddur",
            "vapour_pressure_deficit_max": "vpdmax",
        },
        axis=1,
        inplace=True,
    )

    lcs = [3, 4, 7, 9, 10]
    regions = [
        "Northern Scotland",
        "Eastern Scotland",
        "Southern Scotland",
        "North-east",
        "North-west",
        "Northern Ireland",
        "South-west",
        "South-east",
    ]
    results = []
    for lc in lcs:
        points = pd.read_parquet(
            f"/Users/tadas/modFire/fire_lc_ndvi/data/cehlc/LCD_2018_sampled_eroded_5_lc_{lc}.parquet"
        )
        er5g = ERA5LandGrid()
        xx, yy = er5g.find_point_xy(points["latitude"], points["longitude"])
        points["latind"] = yy.astype(int)
        points["lonind"] = xx.astype(int)

        for region in regions:
            ph = pd.read_parquet(
                f"/Users/tadas/modFire/fire_lc_ndvi/data/cehlc/gee_results/VNP13A1_{region}_{lc}_sample.parquet"
            )
            samp_sub = sampled[(sampled.lc == lc) & (sampled.Region == region)]
            for nr, row in samp_sub.iterrows():
                points_sub = points[
                    (points.lonind == row.lonind) & (points.latind == row.latind)
                ]
                weather_sub = weather[
                    (weather.lonind == row.lonind) & (weather.latind == row.latind)
                ].copy()
                weather_sub = weather_sub.sort_values("date")
                weather_sub["date"] = pd.to_datetime(weather_sub["date"].dt.date)
                weather_features = shift_weather_features(weather_sub)
                ph_sub = ph[ph.fid.isin(points_sub.index)]
                ph_obs = ph_sub.groupby("date")[["EVI2", "NDVI"]].mean().reset_index()
                ph_obs = ph_obs.drop_duplicates("EVI2")

                df = ph_obs.merge(weather_features, on="date", how="left")
                df["lc"] = lc
                df["region"] = row.Region
                df["latind"] = row.fid
                df["longitude"] = row.longitude
                df["latitude"] = row.latitude
                results.append(df)
    df = pd.concat(results).reset_index(drop=True)
    df.to_parquet("data/phenology_training_dataset_features.parquet")
    return df


def shift_weather_features(dfr: pd.DataFrame) -> pd.DataFrame:
    dfrs = {}
    vars = ["sri", "tmean", "vpdmax", "ddur", "prec", "smm100", "smm28"]
    for day in range(1, 16):
        for var in vars:
            dfrs[f"{var}-{day}"] = dfr[var].shift(day).values
    # compute mean values
    shifted = pd.DataFrame.from_dict(dfrs, orient="columns")
    ress = {}
    for var in vars:
        for days in [3, 7, 15]:
            if var == "prec":
                ress[f"{var}-{days}sum"] = shifted[
                    [f"{var}-{i}" for i in range(1, days)]
                ].sum(axis=1)
            elif var in ["vpdmax", "tmax", "smm100", "smm28"]:
                ress[f"{var}-{days}max"] = shifted[
                    [f"{var}-{i}" for i in range(1, days)]
                ].max(axis=1)

                ress[f"{var}-{days}mean"] = shifted[
                    [f"{var}-{i}" for i in range(1, days)]
                ].mean(axis=1)
            else:
                ress[f"{var}-{days}mean"] = shifted[
                    [f"{var}-{i}" for i in range(1, days)]
                ].mean(axis=1)
    means = pd.DataFrame.from_dict(ress, orient="columns")
    df = pd.concat([dfr.reset_index(drop=True), means], axis=1)
    df["ddur_change"] = shifted["ddur-1"] - df["ddur"]
    return df


def get_weather_features(dfr: pd.DataFrame, completed_fname: str) -> pd.DataFrame:
    """Fetch historical weather for given observations in dfr which
    must have columns: site, latitude, longitude, latind, lonind and date"""
    try:
        completed = pd.read_parquet(completed_fname)
    except FileNotFoundError:
        completed = pd.DataFrame()
    # First fetch hourly weather variables per unique grid cell
    #
    url = "https://archive-api.open-meteo.com/v1/archive"
    dfrg = (
        dfr.groupby(["site"])[
            ["longitude", "latitude", "latind", "lonind", "slope", "aspect"]
        ]
        .first()
        .reset_index()
    )
    hourly_variables = [
        # "temperature_2m",
        # "relative_humidity_2m",
        # "precipitation",
        # "cloud_cover",
        # "wind_speed_10m",
        # "wind_direction_10m",
        # "wind_gusts_10m",
        "vapour_pressure_deficit",
        "global_tilted_irradiance",
    ]
    daily_variables = [
        "precipitation_sum",
        "temperature_2m_mean",
        "shortwave_radiation_sum",
        "daylight_duration",
        "vapour_pressure_deficit_max",
        "soil_moisture_7_to_28cm_mean",
        "soil_moisture_28_to_100cm_mean",
    ]
    results = []
    for nr, row in dfrg.iterrows():
        if not completed.empty:
            comp_sub = completed[(completed.site == row.site)]
            if len(comp_sub) > 0:
                continue
            else:
                print("Fetching data for ", row.site)
        mindate = dfr.loc[
            (dfr.latind == row.latind) & (dfr.lonind == row.lonind), "date"
        ].min()
        maxdate = dfr.loc[
            (dfr.latind == row.latind) & (dfr.lonind == row.lonind), "date"
        ].max()
        # Add a month before the first date to ensure we have enough data
        # start_date = (mindate - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        # start from January of the first year
        start_date = f"{mindate.year}-01-01"
        print("Start date:", start_date)
        # end_date = (maxdate + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = "2025-08-10"

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
        time.sleep(5)
        res_daily = fetch_daily(
            url,
            row.latitude,
            row.longitude,
            start_date,
            end_date,
            daily_variables,
        )
        res_comb = pd.concat([res, pd.DataFrame(row).T.reset_index(drop=True)], axis=1)
        res_comb = res_comb.ffill()
        res_comb["date_"] = res_comb.date.dt.floor("d")

        res_daily["date_"] = res_daily.date.dt.floor("d")
        res_comb = res_comb.merge(
            res_daily[["date_"] + daily_variables],
            on=["date_"],
            how="left",
        )
        # results.append(res_comb)
        time.sleep(7)
        # results.to_parquet("data/weather_results.parquet", index=False)
        completed = pd.concat([completed, res_comb], ignore_index=True)
        completed.to_parquet(completed_fname, index=False)
    # return pd.concat(results)


def shift_soil_moisture(dfr: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """helper method to shift soil moisture features and calc mean for 15days"""
    temp = []
    dfrs = {}
    vars = ["smm100", "smm28"]
    for site in results.site.unique():
        print(site)
        ressite = results[results.site == site].copy()
        daily = ressite.groupby(ressite.date_)[vars].first().reset_index()

        for day in range(1, 16):
            for var in vars:
                dfrs[f"{var}-{day}"] = daily[var].shift(day).values
        # compute mean values
        shifted = pd.DataFrame.from_dict(dfrs, orient="columns")

        ress = {}
        for var in vars:
            for days in [15]:
                ress[f"{var}-{days}mean"] = shifted[
                    [f"{var}-{i}" for i in range(1, days)]
                ].mean(axis=1)
        means = pd.DataFrame.from_dict(ress, orient="columns")
        means["date_"] = daily.date_
        ressite = ressite.merge(means, on="date_", how="left")
        temp.append(ressite)
    results = pd.concat(temp)

    fe = dfr.merge(
        results[["site", "date", "smm100-15mean", "smm28-15mean"]],
        on=["site", "date"],
        how="left",
    )
    return fe, results


def prepare_weather_features(dfr: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """prepare weather features (results) for training dataset (dfr). Return training
    dataset with weather features merget and also hourly time series dataset"""
    temp = []
    for site in results.site.unique():
        ressite = results[results.site == site].copy()
        ressite = ressite.sort_values("date")
        ressite.rename(
            {
                "vapour_pressure_deficit": "vpd",
                # "soil_moisture_0_to_7cm_mean": "smm7",
                "soil_moisture_7_to_28cm_mean": "smm28",
                "soil_moisture_28_to_100cm_mean": "smm100",
                "global_tilted_irradiance": "gti",
                "daylight_duration": "ddur",
                # "sunshine_duration": "sdur",
                "precipitation_sum": "prec",
                "temperature_2m_mean": "tmean",
                "shortwave_radiation_sum": "sri",
                "vapour_pressure_deficit_max": "vpdmax",
            },
            axis=1,
            inplace=True,
        )
        for hour in range(1, 6):
            ressite[f"vpd-{hour}"] = ressite["vpd"].shift(hour)
            ressite[f"gti-{hour}"] = ressite["gti"].shift(hour)
        daily_vars = [
            "tmean",
            "vpdmax",
            "ddur",
            # "smm7",
            "smm28",
            "smm100",
            "sri",
            # "sdur",
            "prec",
        ]
        daily_vpd = ressite.groupby(ressite.date_)[daily_vars].first().reset_index()
        # Step 2: Create a DataFrame of lagged daily values
        daily_vpd = shift_weather_features(daily_vpd)
        ressite = ressite.drop(daily_vars, axis=1).merge(
            daily_vpd, on="date_", how="left"
        )
        temp.append(ressite)
    results = pd.concat(temp)
    # Training dataset with features at observation time
    fe = dfr.merge(
        results.drop(
            ["slope", "aspect", "lonind", "latind", "longitude", "latitude"], axis=1
        ),
        on=["site", "date"],
        how="left",
    )
    # Time Series features dataset
    fe_time_series = results.merge(
        dfr.groupby(["site"])["elevation"].first().reset_index(),
        on=["site"],
        how="left",
    )
    return fe, fe_time_series


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
    dfsg = dfr[
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
    for _, row in dfsg.iterrows():
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
    # pass
    # dfr = pd.read_parquet("data/training_dataset_features_full.parquet")
    # dfrts = pd.read_parquet("data/training_dataset_features_full_time_series.parquet")
    # dfr, dfrts = shift_soil_moisture(dfr, dfrts)
    # dfr.to_parquet("data/training_dataset_features_full_sm.parquet", index=False)
    # dfrts.to_parquet(
    #     "data/training_dataset_features_full_time_series_sm.parquet", index=False
    # )
    # uob = pd.read_parquet("data/training_dataset_features_uob_2025.parquet")
    # uobt = pd.read_parquet("data/weather_site_features_uob_2025.parquet")
    # uob, uobt = shift_soil_moisture(uob, uobt)
    # uob.to_parquet("data/training_dataset_features_uob_2025_sm.parquet")
    # uobt.to_parquet("data/weather_site_features_uob_2025_sm.parquet")
    #
    # ds = pd.read_parquet("data/training_dataset_features_dorset_surrey.parquet")
    # dsst = pd.read_parquet("data/weather_site_features_dorset_surrey.parquet")
    # ds, dsst = shift_soil_moisture(ds, dsst)
    # ds.to_parquet("data/training_dataset_features_dorset_surrey_sm.parquet")
    # dsst.to_parquet("data/weather_site_features_dorset_surrey_sm.parquet")

    # pass
    proc_uob_2025()
    # proc_dorset_surrey()
    # dfr = proc_fuel_moisture_UK()
    # results = pd.read_parquet("data/weather_results.parquet")
    # dfr = proc_dorset_surrey()
    # dfr = proc_uob_2025()
    # results = get_weather_features(dfr, "data/training_dataset_weather.parquet")
    # model = FuelMoistureModel()
    # dfr = model.prepare_training_dataset()
    #
    # results = get_weather_features(dfr, "data/training_dataset_weather.parquet")
    # dfr = proc_fuel_moisture_UK()
    # results = pd.read_parquet("data/training_dataset_weather.parquet")
    # fe, fe_time = prepare_weather_features(dfr, results)
    # fe.to_parquet("data/training_dataset_features_full.parquet")
    # fe_time.to_parquet("data/training_dataset_features_full_time_series.parquet")
