import math
from typing import List

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, stats, linregress
from sklearn.metrics import mean_absolute_error, r2_score
from open_meteo import fetch_daily

from matplotlib.dates import DateFormatter
from model_base import LiveFuelMoistureModel


gsilim = {
    "TminLow": -2,  # Lower limit for minimum temperature (C)
    "TminUp": 6,  # Upper limit for minimum temperature (C)
    "DaylLow": 32000,  # Lower limit for daylength (seconds)
    "DaylUp": 39600,  # Upper limit for daylight (seconds)
    "VPDLow": 0.9,  # Lower limit for maximum VPD (kilopascals)
    "VPDUp": 4.0,  # Upper limit for maximum VPD (kilopascals)
    "PrcpRTLow": 0,  # Lower limit for total precipitaiton (mm)
    "PrcpRTUp": 50,  # Upper limit for total precipitaiton (mm)
    "PrcpRTPeriod": 28,  # Running total period for precipitaiton (days)
    "GSIPeriod": 28,  # Running average period for final GSI (days)
    "GUThresh": 0.2,  # Green-up threshold
    "LFMMax": 200,  # Maximum fuel moisture (%)
    "LFMMin": 60,  # Minimum fuel moisture (%)
    "Lat": 45,
}


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
