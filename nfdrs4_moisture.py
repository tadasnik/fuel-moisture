import requests
import io

import nfdrs4py
import pandas as pd

# from open_meteo import fetch_hourly


def test_station_data():
    station_id = "20284"
    start_date = "2018-05-23T23:30:00Z"
    end_date = "2018-06-03T23:29:59Z"
    url = f"https://fems.fs2c.usda.gov/api/climatology/download-weather?stationIds={station_id}&startDate={start_date}&endDate={end_date}&dataset=observation&dataFormat=fw21&dataIncrement=hourly&stationtypes=RAWS(SATNFDRS)"
    request = requests.get(url)
    wx = pd.read_csv(io.BytesIO(request.content))
    wx["SnowFlag"] = wx["SnowFlag"].astype(bool)
    return wx


def test_model_data():
    lat = 35.27618889
    lon = -112.0632472
    start_date = "2018-05-23"
    end_date = "2018-06-03"

    columns = [
        "StationId",
        "DateTime",
        "Temperature(F)",
        "RelativeHumidity(%)",
        "Precipitation(in)",
        "WindSpeed(mph)",
        "WindAzimuth(degrees)",
        "GustSpeed(mph)",
        "GustAzimuth(degrees)",
        "SnowFlag",
        "SolarRadiation(W/m2)",
        "Tflag",
        "RHflag",
        "PCPflag",
        "WSflag",
        "WAflag",
        "SRflag",
        "GSflag",
        "GAflag",
    ]

    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        # "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "terrestrial_radiation",
    ]

    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        # "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
        "terrestrial_radiation",
    ]
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    opt_params = {
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
    }
    # dfr = dfr.rename(
    #     {
    #         "temperature_2m": "Temperature(F)",
    #         "relative_humidity_2m": "RelativeHumidity(%)",
    #         "precipitation": "Precipitation(in)",
    #         "terrestrial_radiation": "SolarRadiation(W/m2)",
    #         "wind_speed_10m": "WindSpeed(mph)",
    #     },
    #     axis=1,
    # )

    dfr = fetch_hourly(
        url,
        lat,
        lon,
        start_date,
        end_date,
        hourly_vars,
        opt_params,
    )
    return dfr


def compute_nfdrs4_test(dfr, lat=35, fuel_model="W", slope=5, average_ann_prec=30):
    slope_class = 0
    if slope < 20:
        slope_class = 1
    elif slope < 45:
        slope_class = 2
    else:
        slope_class = 3

    interface = nfdrs4py.NFDRS4py(
        Lat=lat,
        FuelModel=fuel_model,
        SlopeClass=slope_class,
        AvgAnnPrecip=average_ann_prec,
    )
    results = interface.process_df(dfr)
    return results


def compute_nfdrs4(dfr, lat=35, fuel_model="W", slope=5, average_ann_prec=30):
    slope_class = 0
    if slope < 20:
        slope_class = 1
    elif slope < 45:
        slope_class = 2
    else:
        slope_class = 3

    # C to F
    dfr["Temperature(F)"] = (dfr["temperature_2m"] * 9 / 5) + 32
    dfr["RelativeHumidity(%)"] = dfr.relative_humidity_2m
    # To inches
    dfr["Precipitation(in)"] = dfr.precipitation / 25.4
    dfr["SolarRadiation(W/m2)"] = dfr.shortwave_radiation
    # wind speed from km/h to miles/h
    dfr["WindSpeed(mph)"] = dfr.wind_speed_10m * 0.621371
    dfr["SnowFlag"] = False
    dfr["DateTime"] = dfr.date.dt.strftime("%Y-%m-%dT%H:%M:%S")

    interface = nfdrs4py.NFDRS4py(
        Lat=lat,
        FuelModel=fuel_model,
        SlopeClass=slope_class,
        AvgAnnPrecip=average_ann_prec,
    )
    results = interface.process_df(dfr)
    return results
