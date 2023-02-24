"""Defining constants for use in the cubes package"""

import pandas as pd
from cubes.construct import material as mat
import re
import os

from cubes.constants import EPLUS_PATH

package_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Path will need changed when we get a data folder in construct
raw_geometry_data = pd.read_excel(
    package_directory + "/data/housing_stock/AmBIENCe_Geometry_Constructions.xlsx"
)
raw_system_data = pd.read_excel(
    package_directory + "/data/housing_stock/AmBIENCe_Energy_Systems.xlsx"
)

# Path for this needs to be properly defined in either location
materials_data = pd.read_csv(
    package_directory + "/data/materials/Materials_extended.csv"
)

# Path for this needs to be properly defined in either location
uk_materials_data = pd.read_excel("../../../exp/jack/Data/UK_Data/UK_Materials.xlsx")

materials_data = pd.concat([materials_data, uk_materials_data])

energy_systems_map = pd.read_excel("../../../exp/jack/Data/Map_EnergySystems.xlsx")
energy_systems_map = energy_systems_map.fillna("")

gb_ambience = pd.read_excel("/workspaces/CUBES/exp/jack/Data/UK_Data/GB_Ambience.xlsx")

map_gb_constructions = pd.read_excel(
    "/workspaces/CUBES/exp/jack/Data/UK_Data/TABULA_to_UWE.xlsx",
    sheet_name="UWE_Constructions",
)

raw_geometry_data = pd.concat([raw_geometry_data, gb_ambience]).reset_index(drop=True)

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
            WINDOW_GLASS_MATERIALS[values[0]] = mat.WindowMaterialGlazing(
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
                values[8],
                values[9],
                values[10],
                values[11],
                values[12],
                values[13],
            )


def get_window_gap_width(window_description):
    if "Single" in window_description:
        return 0
    else:
        first_part = window_description.split("mm")[0]
        return float(first_part.split()[-1])


simple_glazing_data = pd.read_csv(
    package_directory + "/data/materials/Window_materials_simple.csv"
)

SIMPLE_GLAZINGS = {}

for i, row in simple_glazing_data.iterrows():
    SIMPLE_GLAZINGS[row.Name] = mat.WindowMaterialSimpleGlazing(
        row.Name,
        row.U_Factor,
        row.SHGC,
        row.Visible_Transmittance,
    )


# filter for the housing stock database
filter_limit_to = {}
filter_exclude = {"REFERENCE BUILDING COUNTRY CODE": "CY"}


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
        gm_dt = gm_dt[gm_dt[key] != value]

    sy_dt = pd.merge(sy_dt, gm_dt, left_index=True, right_index=True)
    gm_dt = pd.DataFrame(sy_dt.iloc[:, 34:])
    return gm_dt


clean_system_data = clean_ambience_system_data(raw_system_data)

filtered_geometry_data = filter_geometry_data(raw_geometry_data, clean_system_data)


def get_schedule(name):
    with open(
        package_directory + "/data/schedules/" + name + ".sch", "r", encoding="utf-8"
    ) as file2:
        schedule_str = file2.read()
    return schedule_str
