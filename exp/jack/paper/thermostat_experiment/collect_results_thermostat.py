# pylint: disable=all
import os
import pandas as pd
import numpy as np

control_options = {
    "all_zones_controlled": ["eco", 0],
    "six_zones_controlled": ["eco", 6],
    "four_zones_controlled": ["eco", 4],
    "2_zones_controlled": ["eco", 2],
    "1_zone_controlled": ["eco", 1],
    "manual_control": ["manual", 0],
    "no_control": ["eco", 0],
}


reps = np.arange(5)
cases = np.arange(5)

for case in cases:
    data = {}
    for rep in reps:
        for k, i in control_options.items():
            if rep == 0:
                dir_name = f"Eplus-env-2024-05-09_{k}_rep{rep}_zone_{i[-1]}_{i[0]}_2022_case_{case}_rep_{rep}-res1"
            else:
                dir_name = f"Eplus-env-2024-05-10_{k}_rep{rep}_zone_{i[-1]}_{i[0]}_2022_case_{case}_rep_{rep}-res1"
            file_name = "/progress.csv"
            path = dir_name + file_name

            # Check if the directory exists
            if os.path.exists(dir_name):
                df = pd.read_csv(path, index_col=0)
                emissions = df["cumulative_emissions"]
                data[f"{k}_rep_{rep}"] = emissions
            else:
                print(f"Directory '{dir_name}' does not exist. Skipping...")

        # Check if data is not empty before creating DataFrame and saving to CSV
        if data:
            out_data = pd.DataFrame(data)
            out_data.to_csv(
                f"exp/jack/paper/thermostat/output/final_results_{case}.csv"
            )
        else:
            print(f"No data for case '{case}'. Skipping CSV creation.")
