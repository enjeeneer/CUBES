"""Defining constants for use in the cubes package"""

import pandas as pd

EPLUS_PATH = "/usr/local/EnergyPlus-9-5-0/"

DATABASE = [1, 2, 3, 4, 5, 6, 7, 8]

# Path will need changed when we get a data folder in construct
raw_geometry_data = pd.read_excel("exp/jack/Data/AmBIENCe_Geometry_Constructions.xlsx")
raw_system_data = pd.read_excel("exp/jack/Data/AmBIENCe_Energy_Systems.xlsx")


def clean_ambience_system_data(sy_dt):
    sy_dt.columns = sy_dt.iloc[0, :]
    sy_dt = sy_dt.iloc[1:, :]
    sy_dt = sy_dt.drop([1793, 1794])
    sy_dt = sy_dt.reset_index(drop=True)
    return sy_dt


def filter_geometry_data_by_boiler(gm_dt, sy_dt):
    # print(sy_dt.columns)
    sy_dt = sy_dt[sy_dt["HEATING SYSTEM 1 TECHNOLOGY"].str.contains("boiler")]
    sy_dt = pd.merge(sy_dt, gm_dt, left_index=True, right_index=True)
    gm_dt = pd.DataFrame(sy_dt.iloc[:, 34:])
    return gm_dt


clean_system_data = clean_ambience_system_data(raw_system_data)

filtered_geometry_data = filter_geometry_data_by_boiler(
    raw_geometry_data, clean_system_data
)
