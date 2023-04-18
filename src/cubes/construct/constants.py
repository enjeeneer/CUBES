"""Defining constants for use in the cubes package"""

import pandas as pd
from cubes.construct import material as mat
import re
from cubes.constants import EPLUS_PATH, package_directory

# Path will need changed when we get a data folder in construct
raw_geometry_data = pd.read_excel(
    package_directory + "/data/housing_stock/ambience-geometry-eu.xlsx"
)
raw_system_data = pd.read_excel(
    package_directory + "/data/housing_stock/ambience-energy-eu.xlsx"
)

# Read in materials data
materials_data = pd.read_csv(
    package_directory + "/data/materials/Materials_extended.csv"
)

uk_materials_data = pd.read_csv(package_directory + "/data/materials/UK_Materials.csv")

materials_data = pd.concat([materials_data, uk_materials_data])

# Read in no-mass materials data
no_mass_materials_data = pd.read_csv(
    package_directory + "/data/materials/Nomass_materials.csv"
)

uk_no_mass_materials_data = pd.read_csv(
    package_directory + "/data/materials/Nomass_materials_UK.csv"
)

no_mass_materials_data = pd.concat([no_mass_materials_data, uk_no_mass_materials_data])


energy_systems_map = pd.read_excel(
    package_directory + "/data/housing_stock/energy-system-schema.xlsx"
)
energy_systems_map = energy_systems_map.fillna("")

gb_ambience = pd.read_excel(
    package_directory + "/data/housing_stock/ambience-geometry-gb.xlsx"
)

map_gb_constructions = pd.read_csv(
    package_directory + "/data/housing_stock/map_gb_uwe_constructions_to_tabula.csv",
)

raw_geometry_data = pd.concat([raw_geometry_data, gb_ambience]).reset_index(drop=True)

MATERIALS = {}


def decomment(csvfile):
    for row_i in csvfile:
        raw = row_i.split("#")[0].strip()
        if raw:
            yield raw


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

for i, row in no_mass_materials_data.iterrows():
    MATERIALS[row.Material] = mat.NoMassMaterial(
        row.Material,
        row.Roughness,
        row.Thermal_Resistance,
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
