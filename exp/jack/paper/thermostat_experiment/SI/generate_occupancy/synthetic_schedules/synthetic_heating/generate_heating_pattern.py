import os
import pandas as pd
import numpy as np

occupancy_dir = "/workspaces/CUBES/exp/jack/paper/thermostat_experiment/SI/generate_occupancy/synthetic_schedules/synthetic_occupancy"
output_dir = "/workspaces/CUBES/exp/jack/paper/thermostat_experiment/SI/generate_occupancy/synthetic_schedules/synthetic_heating"

rooms = [
    "backroom",
    "bathroom",
    "front_room",
    "hall_downstairs",
    "bedroom_2",
    "kitchen",
    "bedroom_1",
    "bedroom_3",
    "hall_upstairs",
]


def apply_heating_rule(series, time_mask):
    heating = np.zeros_like(series)
    min_duration = 5

    i = 0
    while i < len(series):
        if time_mask[i]:
            heating[i] = 0
        elif series[i] > 0:
            heating[i : i + min_duration] = 1
            i += min_duration - 1
        elif heating[i - 1] == 1 and sum(series[i - min_duration : i]) > 0:
            heating[i] = 1
        else:
            heating[i] = 0
        i += 1

    return heating


# Get the list of occupancy files in the directory
files = [f for f in os.listdir(occupancy_dir) if f.endswith(".sch")]

# Loop through each file dynamically
for file in files:
    print(f"Processing file {file}")
    synthetic_schedule_path = os.path.join(occupancy_dir, file)
    df = pd.read_csv(synthetic_schedule_path, parse_dates=["UTC_Time"])
    df_copy = df.copy()
    time_mask = (df_copy["UTC_Time"].dt.hour >= 23) | (df_copy["UTC_Time"].dt.hour < 6)

    for room in rooms:
        df_copy[room] = apply_heating_rule(df[room].values, time_mask)

    # Construct output file name based on the original file name
    heating_schedule_path = os.path.join(output_dir, file)
    print(f"Saving file {heating_schedule_path}")
    df_copy.to_csv(heating_schedule_path, index=False)
