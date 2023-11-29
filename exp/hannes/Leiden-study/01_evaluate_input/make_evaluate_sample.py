# pylint: disable=R1703

"""module for creating input files for leiden study."""
import os
import numpy as np
import json
import argparse
from input_file_factory import presample, get_input_file_with_schedules_etc_presampled

cwd_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--always_occupied", type=str, default="False")
parser.add_argument("--deterministic_occupancy", type=str, default="True")
args = parser.parse_args()

if args.always_occupied == "True":
    always_occupied = True
else:
    always_occupied = False
if args.deterministic_occupancy == "True":
    deterministic_occupancy = True
else:
    deterministic_occupancy = False

assert not (always_occupied and deterministic_occupancy)
cases = np.arange(20)
# cases = [5]
# years = np.arange(2017,2023)
years = [2022]
reps_per_year = 20

if always_occupied:
    run_name = "evaluation_always_occupied"
elif deterministic_occupancy:
    run_name = "evaluation_deterministic_occupancy"
else:
    run_name = "evaluation_new"

for r in range(reps_per_year):
    presampled = presample(
        always_occupied=always_occupied,
        deterministic_occupancy=deterministic_occupancy,
    )
    for i_case in cases:
        input_file_path = "../00_base_input/case" + str(i_case) + ".json"
        for y in years:
            presampled["year"] = int(y)

            case_path = (
                cwd_path
                + "/"
                + run_name
                + "/case_"
                + str(i_case)
                + "/year_"
                + str(y)
                + "/rep_"
                + str(r)
            )
            complete_input_file_path_uc = case_path + "/input_uc.json"
            complete_input_file_path_c = case_path + "/input_c.json"

            if not os.path.exists(case_path):
                os.makedirs(case_path)

            get_input_file_with_schedules_etc_presampled(
                presampled, input_file_path, complete_input_file_path_uc
            )

            with open(complete_input_file_path_uc, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["natural_ventilation_method"] = "air changes per hour"
                data["natural_ventilation_rate_open_windows"] = 2

            with open(complete_input_file_path_c, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
