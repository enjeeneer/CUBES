# pylint: disable=all
import os
import pandas as pd
import numpy as np

control_options = {
    "all_zones_controlled": ["eco", 0],
    "6_zones_controlled": ["eco", 6],
    "4_zones_controlled": ["eco", 4],
    "2_zones_controlled": ["eco", 2],
    "1_zone_controlled": ["eco", 1],
    "manual_control": ["manual", 0],
    "no_control": ["eco", 0],
}
reps = np.arange(5)

data = {}
for k, i in control_options.items():
    for rep in reps:
        dir_name = (
            f"Eplus-env-2024-05-07_{k}_zone_{i[-1]}_{i[0]}_2022_case_0_rep_{rep}-res1"
        )
        file_name = "/progress.csv"
        path = dir_name + file_name

        # Check if the directory exists
        if os.path.exists(dir_name):
            df = pd.read_csv(path, index_col=0)
            emissions = df["cumulative_emissions"]
            data[k] = emissions
        else:
            print(f"Directory '{dir_name}' does not exist. Skipping...")

out_data = pd.DataFrame(data)
out_data.to_csv("final_results.csv")
