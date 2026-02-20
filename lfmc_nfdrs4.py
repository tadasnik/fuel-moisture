import math
from typing import List

from itertools import combinations
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, stats, linregress
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_absolute_error, r2_score
from nfdrs4_live import make_param_ranges
from open_meteo import fetch_daily

from matplotlib.dates import DateFormatter
from model_base import LiveFuelMoistureModel


gsilim = {
    "TminLow": -2,  # Lower limit for minimum temperature (C)
    "TminUp": 10,  # Upper limit for minimum temperature (C)
    "DaylLow": 20000,  # Lower limit for daylength (seconds)
    "DaylUp": 40600,  # Upper limit for daylight (seconds)
    "VPDLow": 0.9,  # Lower limit for maximum VPD (kilopascals)
    "VPDUp": 5.0,  # Upper limit for maximum VPD (kilopascals)
    "PrcpRTLow": 0,  # Lower limit for total precipitaiton (mm)
    "PrcpRTUp": 50,  # Upper limit for total precipitaiton (mm)
    "PrcpRTPeriod": 28,  # Running total period for precipitaiton (days)
    "GSIPeriod": 28,  # Running average period for final GSI (days)
    "GUThresh": 0.2,  # Green-up threshold
    "LFMMax": 200,  # Maximum fuel moisture (%)
    "LFMMin": 60,  # Minimum fuel moisture (%)
    "Lat": 45,
}


def fetch_daily_meteo_data(
    lat: float, lon: float, start_date: str, end_date: str
) -> pd.DataFrame:
    meteo_vars = [
        "daylight_duration",
        "vapour_pressure_deficit_max",
        "temperature_2m_min",
        "precipitation_sum",
    ]
    url = "https://archive-api.open-meteo.com/v1/archive"
    dfr = fetch_daily(
        url,
        lat,
        lon,
        start_date,
        end_date,
        meteo_vars,
    )
    return dfr


def calc_ramp_index(values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    ramp_index = (values - min_val) / (max_val - min_val)
    ramp_index[values < min_val] = 0
    ramp_index[values > max_val] = 1
    return ramp_index


def calc_lfmc_from_gsi(
    gsi: np.ndarray, gu_thresh: float, lfmc_min: float, lfmc_max: float
) -> np.ndarray:
    m = (lfmc_max - lfmc_min) / (1 - gu_thresh)
    b = lfmc_max - m
    lfmc = (m * gsi) + b
    lfmc[lfmc < gu_thresh] = lfmc_min
    return lfmc


def calc_gsi(df: pd.DataFrame, gsilim, return_intermediate=False) -> pd.DataFrame:
    """Calculate live fuel moisture using the ramp functions with limits
    given in the gsilim dict"""
    # Min temp index
    temp_min_ind = calc_ramp_index(
        df["temperature_2m_min"].values, gsilim["TminLow"], gsilim["TminUp"]
    )
    # Vapor Pressure Deficit
    vpd_max_ind = 1 - (
        calc_ramp_index(
            df["vapour_pressure_deficit_max"].values, gsilim["VPDLow"], gsilim["VPDUp"]
        )
    )
    # Photoperiod / Daylength
    day_len_ind = calc_ramp_index(
        df["daylight_duration"].values, gsilim["DaylLow"], gsilim["DaylUp"]
    )
    # Running Total Precipitation
    prec_roll = (
        df["precipitation_sum"]
        .rolling(gsilim["PrcpRTPeriod"], min_periods=1)
        .sum()
        .values
    )
    prec_ind = calc_ramp_index(prec_roll, gsilim["PrcpRTLow"], gsilim["PrcpRTUp"])

    # Daily GSI for four ramp model
    i_gsi_pe = temp_min_ind * vpd_max_ind * day_len_ind * prec_ind
    # Smoothed GSI (running average over the GSIPeriod)
    gsi_pe = np.convolve(
        i_gsi_pe, np.ones(gsilim["GSIPeriod"]) / gsilim["GSIPeriod"], mode="same"
    )
    # Rescale the final GSI time series
    gsi_pe_rs = gsi_pe / gsi_pe.max()
    # GSI to LFMC
    lfmc = calc_lfmc_from_gsi(
        gsi_pe_rs, gsilim["GUThresh"], gsilim["LFMMin"], gsilim["LFMMax"]
    )

    if not return_intermediate:
        df["LFMC"] = lfmc
    else:
        intermediate = pd.DataFrame(
            {
                "TminInd": temp_min_ind,
                "VPDInd": vpd_max_ind,
                "DaylInd": day_len_ind,
                "PrcpInd": prec_ind,
                "iGSI_PE": i_gsi_pe,
                "GSI_PE": gsi_pe,
                "GSI_PE_RS": gsi_pe_rs,
                "LFMC": lfmc,
            }
        )
        df = pd.concat([df, intermediate], axis=1)
    return df


def make_param_ranges_interval(min_val, max_val, interv, min_sep):
    values = np.arange(min_val, max_val + interv, interv)
    combs = np.array(np.meshgrid(values, values)).T.reshape(-1, 2)
    mask = (combs[:, 0] + min_sep) < combs[:, 1]
    return combs[mask]


def make_param_ranges_number(min_val, max_val, num):
    """Returns DataFrame with "Lower" and "Upper" columns
    containing parameter values between min_val and
    max_val for model optimization."""
    values = np.linspace(min_val, max_val, num)
    combs = np.array(np.meshgrid(values, values)).T.reshape(-1, 2)
    mask = combs[:, 0] < combs[:, 1]
    return combs[mask]


def generate_parameter_list(n_iter):
    params_dict = {
        "min_temp": make_param_ranges_interval(
            gsilim["TminLow"], gsilim["TminUp"], 1, 2
        ),
        "max_vpd": make_param_ranges_interval(
            gsilim["VPDLow"], gsilim["VPDUp"], 0.1, 3
        ),
        "day_length": make_param_ranges_interval(
            gsilim["DaylLow"], gsilim["DaylUp"], 1000, 5000
        ),
        "prec_sum": make_param_ranges_interval(
            gsilim["PrcpRTLow"], gsilim["PrcpRTUp"], 5, 20
        ),
        "lfmc_lim": make_param_ranges_interval(30, 200, 20, 50),
        "gu_thresh": np.linspace(0, 0.8, 9),
        "prec_period": np.arange(7, 61),
        "gsi_period": np.arange(7, 61),
    }
    samples_dict = {}
    for key, values in params_dict.items():
        samples_dict[key] = np.arange(len(values))
    param_index_list = list(
        ParameterSampler(
            samples_dict, n_iter=n_iter, random_state=np.random.RandomState(0)
        )
    )
    params = []
    for item in param_index_list:
        params.append(
            {
                "TminLow": params_dict["min_temp"][item["min_temp"], 0],
                "TminUp": params_dict["min_temp"][item["min_temp"], 1],
                "VPDLow": params_dict["max_vpd"][item["max_vpd"], 0],
                "VPDUp": params_dict["max_vpd"][item["max_vpd"], 1],
                "DaylLow": params_dict["day_length"][item["day_length"], 0],
                "DaylUp": params_dict["day_length"][item["day_length"], 1],
                "PrcpRTLow": params_dict["prec_sum"][item["prec_sum"], 0],
                "PrcpRTUp": params_dict["prec_sum"][item["prec_sum"], 1],
                "LFMMin": params_dict["lfmc_lim"][item["lfmc_lim"], 0],
                "LFMMax": params_dict["lfmc_lim"][item["lfmc_lim"], 1],
                "PrcpRTPeriod": params_dict["prec_period"][item["prec_period"]],
                "GSIPeriod": params_dict["gsi_period"][item["gsi_period"]],
                "GUThresh": params_dict["gu_thresh"][item["gu_thresh"]],
            }
        )
    return params


def gsi_optimization(weather_inp, observed, n_iter):
    weather = weather_inp.merge(observed[["date", "fmc_%"]], on="date", how="left")
    param_list = generate_parameter_list(n_iter)
    best_r2 = 0
    best_params = []
    for params in param_list:
        weather = calc_gsi(weather, params, return_intermediate=False)
        s2, p2 = spearmanr(weather["fmc_%"], weather["LFMC"], nan_policy="omit")
        _, _, r_value, _, _ = linregress(
            weather.dropna()["fmc_%"], weather.dropna()["LFMC"]
        )
        r2 = r_value**2
        if r2 > best_r2:
            best_r2 = r2
            best_params = params
            print(r2, s2, best_params)


def GridSearchOptimizeGSILFM(
    weather,
    observed,
    label,
    maxSim=2,
    mySeed=123456,
    UseLFMMinMax=False,
    Herb=True,
    PInt=20,
    Lat=45,
):
    DEBUG = False

    # Create a range of smoothing / running precip period length ranges intervals
    smint = range(7, 60, 7)
    dfsmint = pd.DataFrame(data={"SMInt": smint})

    #### Get a Random Set of Parameters
    # Make the VPD ranges
    iVPDP = make_param_ranges(0.5, 9.0, 0.1)  # VPD ranges from 500 to 9000 Pascals
    # Make the Temperature ranges
    iTMinP = make_param_ranges(-5, 10, 1)  # MinT ranges from -5 to 10 deg C
    # Make the Daylength Ranges
    iDaylP = make_param_ranges(32400, 46800, 3600)  # Dayl ranges from 9 to 13 hours
    # Make the Prcp ranges
    iPrcpP = make_param_ranges(0, 50, 1)

    # Make the green-up threshold
    iThreshP = []
    for i in range(0, 81, 10):
        iThreshP.append(float(i / 100))
    iThreshP = pd.DataFrame(data={"iThresh": iThreshP})

    BestS = 0
    BestParams = {}

    for i in range(0, maxSim):

        rs = dfsmint.sample(n=1, random_state=mySeed + i)
        smint = rs.iloc[0].SMInt.astype(int)

        # VPD Params
        rs = iVPDP.sample(n=1, random_state=mySeed + i)
        iVPDMin = rs.iloc[0].Lower
        iVPDMax = rs.iloc[0].Upper

        # TMin Params
        rs = iTMinP.sample(n=1, random_state=mySeed + i)
        iTminMin = rs.iloc[0].Lower
        iTminMax = rs.iloc[0].Upper

        # Daylength Params
        rs = iDaylP.sample(n=1, random_state=mySeed + i)
        iDaylMin = rs.iloc[0].Lower
        iDaylMax = rs.iloc[0].Upper

        # RT Precip Params
        rs = iPrcpP.sample(n=1, random_state=mySeed + i)
        iPrcpMin = rs.iloc[0].Lower
        iPrcpMax = rs.iloc[0].Upper

        # Greenup Threshold
        rs = iThreshP.sample(n=1, random_state=mySeed + i)
        iThreshVal = rs.iloc[0].iThresh.astype(float)

        Params = [
            iTminMin,
            iTminMax,
            iVPDMin,
            iVPDMax,
            iDaylMin,
            iDaylMax,
            iPrcpMin,
            iPrcpMax,
            smint,
            iThreshVal,
            smint,
        ]

        S = MakeGSILFMCompareTable4Param(
            weather,
            observed,
            Params,
            label=label,
            UseLFMMinMax=UseLFMMinMax,
            Herb=Herb,
            Lat=Lat,
        )

        if S[1] > BestS:
            BestS = S[1]
            BestParams = Params
            print(S[1], S[3], BestS, BestParams)

        # if i % PInt == 0:
        #     print(S[1], BestS, BestParams)
    return BestParams


if __name__ == "__main__":
    lfmc_model = LiveFuelMoistureModel()
    dfrl = lfmc_model.prepare_training_dataset(
        fname="data/training_dataset_features_full.parquet"
    )
    dfrl = dfrl.rename({"date": "datetime"}, axis=1)
    dfrl["date"] = dfrl.datetime.dt.date
    lonind = 1764
    latind = 1412
    sub_hc = dfrl[
        (dfrl.lonind == lonind)
        & (dfrl.latind == latind)
        & (dfrl.fuel == "Heather canopy")
    ].copy()
    start_date = (sub_hc.datetime.min() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
    end_date = (sub_hc.datetime.max()).strftime("%Y-%m-%d")
    wx = fetch_daily_meteo_data(
        sub_hc.latitude.iloc[0], sub_hc.longitude.iloc[0], start_date, end_date
    )
    wx["date"] = wx.datetime.dt.date
