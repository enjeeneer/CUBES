"""Defining constants for use in the cubes package"""

import pandas as pd
import material as mat
import re


EPLUS_PATH = "/usr/local/EnergyPlus-9-5-0/"

# Path will need changed when we get a data folder in construct
raw_geometry_data = pd.read_excel("exp/jack/Data/AmBIENCe_Geometry_Constructions.xlsx")
raw_system_data = pd.read_excel("exp/jack/Data/AmBIENCe_Energy_Systems.xlsx")

# Path for this needs to be properly defined in either location
materials_data = pd.read_excel(
    "/workspaces/elizabeth-homes/src/cubes/data/materials/Materials_extended.xlsx"
)

MATERIALS = {}

for i, row in materials_data.iterrows():
    MATERIALS[row.Material] = mat.Material(
        row.Material,
        row.Density,
        row.Specific_Heat_Capacity,
        row.Thermal_Conductivity,
        row.Roughness,
        row.Thermal_Absorptance,
        row.Solar_Absorptance,
        row.Visual_Absorptance,
    )

WINDOW_GLASS_MATERIAL_NAMES = ["CLEAR 3MM", "LoE CLEAR 3MM"]
WINDOW_GLASS_MATERIALS = {}

with open(
    EPLUS_PATH + "DataSets/WindowGlassMaterials.idf", "r", encoding="utf-8"
) as file:
    searchlines = file.readlines()
    for i, line in enumerate(searchlines):
        if any(w in line for w in WINDOW_GLASS_MATERIAL_NAMES):
            values = []
            for data in searchlines[i : i + 14]:
                values.append(re.split("; |, ", data)[0].strip())
            WINDOW_GLASS_MATERIALS[values[0]] = mat.WindowMaterialGlazing(*values)


def get_window_gap_width(window_description):
    if "Single" in window_description:
        return 0
    else:
        first_part = window_description.split("mm")[0]
        return float(first_part.split()[-1])


# filter for the housing stock database
filter_limit_to = {"HEATING SYSTEM 1 TECHNOLOGY": "boiler"}
filter_exclude = {}


def clean_ambience_system_data(sy_dt):
    sy_dt.columns = sy_dt.iloc[0, :]
    sy_dt = sy_dt.iloc[1:, :]
    sy_dt = sy_dt.drop([1793, 1794])
    sy_dt = sy_dt.reset_index(drop=True)
    return sy_dt


def filter_geometry_data(gm_dt, sy_dt):
    # print(sy_dt.columns)
    for key, value in filter_limit_to.items():
        sy_dt = sy_dt[sy_dt[key].str.contains(value)]

    for key, value in filter_exclude.items():
        sy_dt = sy_dt.drop(sy_dt[key].str.contains(value).index)

    sy_dt = pd.merge(sy_dt, gm_dt, left_index=True, right_index=True)
    gm_dt = pd.DataFrame(sy_dt.iloc[:, 34:])
    return gm_dt


clean_system_data = clean_ambience_system_data(raw_system_data)

filtered_geometry_data = filter_geometry_data(raw_geometry_data, clean_system_data)
