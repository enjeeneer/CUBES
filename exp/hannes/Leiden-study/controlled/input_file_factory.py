"""make an input file"""

from cubes.construct.schedules import OccupancyScheduler
from cubes.constants import package_directory
from cubes.construct.buildingconfig_options import ResWindowOpeningModel

import pandas as pd
import json
import os
import random

cwd_path = os.getcwd()
print(cwd_path)


occupancy_scheduler = OccupancyScheduler(
    year=2022,
    sample_length="week",
    weekday_init_state_df=pd.read_parquet(
        package_directory + "/data/" "occupants/weekday_occupancy_init_states.parquet"
    ),
    weekend_init_state_df=pd.read_parquet(
        package_directory + "/data/" "occupants/weekend_occupancy_init_states.parquet"
    ),
    weekday_transition_matrix_df=pd.read_parquet(
        package_directory + "/data/" "occupants/weekday_occupancy_transition.parquet"
    ),
    weekend_transition_matrix_df=pd.read_parquet(
        package_directory + "/data/" "occupants/weekend_occupancy_transition.parquet"
    ),
)

sleep_time_range = {
    "start": {"hour": 22, "minute": 00},
    "stop": {"hour": 8, "minute": 00},
}


def presample():

    living_schedule, sleeping_schedule = occupancy_scheduler.sample(2, sleep_time_range)
    data = {}
    data["year"] = random.randrange(2017, 2023)
    data["occupant_schedule_living"] = living_schedule
    data["occupant_schedule_bedroom"] = sleeping_schedule
    data["natural_ventilation_model"] = random.choice(list(ResWindowOpeningModel)).value
    data["natural_ventilation_rate_open_windows"] = random.random() * 4 + 1
    return data


def get_input_file_with_schedules_etc(
    base_file_path, output_file_path, control_vent, fixed_year=None
):
    living_schedule, sleeping_schedule = occupancy_scheduler.sample(2, sleep_time_range)
    if fixed_year:
        year = fixed_year
    else:
        year = random.randrange(2017, 2023)
    weather_file_name = "Cambridgeshire_CC_" + str(year) + ".epw"
    grid_file_name = "grid_carbon_GB_" + str(year) + ".csv"
    with open(base_file_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["year"] = year
        data["occupant_schedule_living"] = living_schedule
        data["occupant_schedule_bedroom"] = sleeping_schedule
        if not control_vent:
            data["natural_ventilation_model"] = random.choice(
                list(ResWindowOpeningModel)
            ).value
        else:
            data["natural_ventilation_method"] = "air changes per hour"
        # data["natural_ventilation_rate_open_windows"] = random.random() * 4 + 1
        data["natural_ventilation_rate_open_windows"] = 2
        data["weather_file_name"] = weather_file_name
        data["grid_carbon_intensity_file_name"] = grid_file_name

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_input_file_with_schedules_etc_presampled(
    presampled, base_file_path, output_file_path
):
    weather_file_name = "Cambridgeshire_CC_" + str(presampled["year"]) + ".epw"
    grid_file_name = "grid_carbon_GB_" + str(presampled["year"]) + ".csv"
    with open(base_file_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["occupant_schedule_living"] = presampled["occupant_schedule_living"]
        data["occupant_schedule_bedroom"] = presampled["occupant_schedule_bedroom"]
        data["natural_ventilation_model"] = data["natural_ventilation_model"]

        data["natural_ventilation_rate_open_windows"] = random.random() * 4 + 1
        data["weather_file_name"] = weather_file_name
        data["grid_carbon_intensity_file_name"] = grid_file_name

    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
