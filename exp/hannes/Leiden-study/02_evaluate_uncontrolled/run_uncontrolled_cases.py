"""script to run the uncontrolled cases for the Leiden paper"""
from cubes.construct.buildingconfig import load_building_config
from cubes.construct.building import Building
from cubes.package.simple_simulation import prepare_simulation
from cubes.package import envconfig
from cubes.construct.core import materials_evaluator, windows_evaluator

import os
import numpy as np

cwd_path = os.getcwd()
print(cwd_path)

materials = materials_evaluator()
windows = windows_evaluator()

cases = np.arange(3, 15)
years = np.arange(2017, 2023)
reps_per_year = 5
run_name = "evaluation_uncontrolled"


for case in cases:
    for y in years:
        for r in range(reps_per_year):
            case_path = (
                cwd_path
                + "/"
                + run_name
                + "/case_"
                + str(case)
                + "/year_"
                + str(y)
                + "/rep_"
                + str(r)
            )
            complete_input_file_path = (
                "../01_evaluate_input/evaluation"
                "/case_"
                + str(case)
                + "/year_"
                + str(y)
                + "/rep_"
                + str(r)
                + "/input_uc.json"
            )
            if not os.path.exists(case_path):
                os.makedirs(case_path)

            BC = load_building_config(complete_input_file_path)

            EC = envconfig.EnvConfig(
                observe_zone_temperature=True,
                observe_electricity_demand=True,
                observe_outside_temperature=True,
                observe_fuel_demand=True,
                observe_zone_co2=True,
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
