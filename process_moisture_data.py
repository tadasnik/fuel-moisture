import pandas as pd
from grids import ERA5LandGrid
from open_meteo import fetch_global_tilted_irradiance, fetch_vpd_prec


def proc_fuel_moisture_UK():
    """Read and prepare observed fuel moisture data for UK"""
    dfr = pd.read_csv("/Users/tadas/modFire/ndmi/data/UK_fuel_moisture_dataset.csv")
    dfr.columns = dfr.columns.str.lower()
    dfr["date"] = pd.to_datetime(dfr["date"], dayfirst=True)
    # filling missing time of collection with 13:00
    dfr["time_collected"] = dfr["time_collected"].fillna(1300.0)
    # add collection time to date
    hours = pd.to_timedelta((dfr["time_collected"] * 0.01).astype(int), "h")
    dfr["date"] = dfr["date"] + hours
    dfr["year"] = dfr["date"].dt.year
    dfr["week"] = dfr["date"].dt.isocalendar().week
    dfr["month"] = dfr["date"].dt.month
    return dfr


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