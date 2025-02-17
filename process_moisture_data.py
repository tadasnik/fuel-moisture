import pandas as pd
from grids import ERA5LandGrid
from open_meteo import fetch_global_tilted_irradiance, fetch_vpd_prec

def proc_fuel_moisture_meg():
    """Read and prepare observed fuel moisture data for UK"""
    dfr = pd.concat(pd.read_excel('/home/tadas/Downloads/Flammability_1.xlsx', sheet_name=None), ignore_index=True)
    dfr = dfr.ffill()
    dfr[["latitude", "longitude"]] = dfr["Coordinates"].str.split(
        ", ", n=1, expand=True
    ).astype('float')
    dfr.loc[dfr.longitude > 0, 'longitude'] = dfr.loc[dfr.longitude > 0, 'longitude'] * -1 
    dfr['date'] = pd.to_datetime(dfr.Date.astype(str) + ' ' + dfr.Time.astype(str))
    dfr.


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
