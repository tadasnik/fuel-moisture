import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry


def fetch_hourly(
    url: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    variables: list[str],
    opt_params: None | dict[str, float] = None,
) -> pd.DataFrame:
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": variables,
    }
    if opt_params is not None:
        params.update(opt_params)
    print(params)
    responses = openmeteo.weather_api(url, params=params)
    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for nr, variable in enumerate(variables):
        var_dfr = hourly.Variables(nr).ValuesAsNumpy()
        hourly_data[variable] = var_dfr
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe


# Setup the Open-Meteo API client with cache and retry on error
def fetch_global_tilted_irradiance(lat, lon, start_date, end_date, tilt, azimuth):
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"

    opt_params = {
        "tilt": tilt,
        "azimuth": azimuth,
    }
    dfr = fetch_hourly(
        url, lat, lon, start_date, end_date, ["global_tilted_irradiance"], opt_params
    )
    return dfr


# Setup the Open-Meteo API client with cache and retry on error
def fetch_vpd_prec(lat, lon, start_date, end_date):
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below

    url = "http://0.0.0.0:8080/v1/archive"
    dfr = fetch_hourly(
        url,
        lat,
        lon,
        start_date,
        end_date,
        ["vapour_pressure_deficit", "precipitation"],
    )
    return dfr


# Setup the Open-Meteo API client with cache and retry on error
def fetch_vpd_hist(lat, lon, start_date, end_date):
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "http://0.0.0.0:8080/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "vapour_pressure_deficit",
    }
    print(params)
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_vapour_pressure_deficit = hourly.Variables(0).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    hourly_data["vpd"] = hourly_vapour_pressure_deficit

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    return hourly_dataframe