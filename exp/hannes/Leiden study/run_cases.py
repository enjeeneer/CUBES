"""script to run the base cases for the Leiden paper"""
from cubes.construct.schedules import OccupancyScheduler
from cubes.constants import package_directory
from cubes.construct.buildingconfig_options import ResWindowOpeningModel
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.package.simple_simulation import prepare_simulation
from cubes.package import envconfig
from cubes.construct.core import materials_evaluator, windows_evaluator


import pandas as pd
import json
import os
import random
import numpy as np

cwd_path = os.getcwd()
print(cwd_path)

materials = materials_evaluator()
windows = windows_evaluator()

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


def get_input_file_with_schedules_etc(base_file_path, output_file_path):
    living_schedule, sleeping_schedule = occupancy_scheduler.sample(2, sleep_time_range)
    year = random.randrange(2017, 2023)
    weather_file_name = "Cambridgeshire_CC_" + str(year) + ".epw"
    grid_file_name = "grid_carbon_GB_" + str(year) + ".csv"
    with open(base_file_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["occupant_schedule_living"] = living_schedule
        data["occupant_schedule_bedroom"] = sleeping_schedule
        data["natural_ventilation_model"] = random.choice(
            list(ResWindowOpeningModel)
        ).value
        data["natural_ventilation_rate_open_windows"] = random.random() * 4 + 1
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


reps = 3
cases = np.arange(15)
# cases = [14]
run_name = "debug_runs"
# run_name = "runs_15_6_23"

sampled = presample()

for case in cases:
    input_file_path = "base_input/case" + str(case) + ".json"
    for r in range(reps):
        case_path = cwd_path + "/" + run_name + "/case_" + str(case) + "/rep_" + str(r)
        complete_input_file_path = case_path + "/input.json"
        if not os.path.exists(case_path):
            os.makedirs(case_path)
        get_input_file_with_schedules_etc_presampled(
            sampled, input_file_path, complete_input_file_path
        )
        # get_input_file_with_schedules_etc(input_file_path, complete_input_file_path)

        BC = load_building_config(complete_input_file_path)
        EC = envconfig.EnvConfig(
            observe_zone_temperature=True,
            observe_electricity_demand=True,
            observe_outside_temperature=True,
        )
        building = Building(BC, materials, windows)
        building.build()
        idf = building.get_idf()

        idf, weather_file = prepare_simulation(idf, BC, EC)
        idf.save(case_path + "/input.idf")
        print(weather_file)
        idf.run(
            expandobjects=False,
            readvars=True,
            weather=weather_file,
            output_directory=case_path + "/output/",
        )
