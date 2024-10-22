# pylint: disable=all
import os
import pandas as pd
import gc

dir_path = "/home/jjjl4/rds/hpc-work/CUBES/"
out_dir = "/home/jjjl4/rds/hpc-work/CUBES/processed_monitors/"
today_str = "higher_setback_check"

# Create output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

utc_col = pd.read_csv("/home/jjjl4/rds/hpc-work/CUBES/cubes/data/occupants/rep_0.sch")[
    "UTC_Time"
]

# Define columns to keep
columns = [
    "Environmental Impact Total CO2 Emissions Carbon Equivalent Mass(Site)",
    "Zone Operative Temperature(hall_downstairs)",
    "Zone Operative Temperature(front_room)",
    "Zone Operative Temperature(kitchen)",
    "Zone Operative Temperature(backroom)",
    "Zone Operative Temperature(bedroom_3)",
    "Zone Operative Temperature(bedroom_1)",
    "Zone Operative Temperature(hall_upstairs)",
    "Zone Operative Temperature(bathroom)",
    "Zone Operative Temperature(bedroom_2)",
    "Zone Air Temperature(hall_downstairs)",
    "Zone Air Temperature(front_room)",
    "Zone Air Temperature(kitchen)",
    "Zone Air Temperature(backroom)",
    "Zone Air Temperature(bedroom_3)",
    "Zone Air Temperature(bedroom_1)",
    "Zone Air Temperature(hall_upstairs)",
    "Zone Air Temperature(bathroom)",
    "Zone Air Temperature(bedroom_2)",
    "Zone People Occupant Count(hall_downstairs)",
    "Zone People Occupant Count(front_room)",
    "Zone People Occupant Count(kitchen)",
    "Zone People Occupant Count(backroom)",
    "Zone People Occupant Count(bedroom_3)",
    "Zone People Occupant Count(bedroom_1)",
    "Zone People Occupant Count(hall_upstairs)",
    "Zone People Occupant Count(bathroom)",
    "Zone People Occupant Count(bedroom_2)",
]

# Function to process and save monitor files
def process_and_save_monitor(file_name, output_file, columns_to_keep, utc_index):
    monitor = pd.read_csv(file_name, usecols=columns_to_keep)
    monitor = monitor.iloc[1:]
    monitor.index = utc_index
    monitor.index = pd.to_datetime(monitor.index)
    monitor.to_csv(output_file, index=True)
    del monitor
    gc.collect()


# Find and process directories that match today's date
for root, dirs, files in os.walk(dir_path):
    for dir_name in dirs:
        if today_str in dir_name and dir_name.startswith("Eplus-env"):
            monitor_file = os.path.join(
                root, dir_name, "Eplus-env-sub_run1", "monitor.csv"
            )
            if os.path.isfile(monitor_file):
                output_file = os.path.join(out_dir, f"{dir_name}_condensed.csv")
                process_and_save_monitor(monitor_file, output_file, columns, utc_col)
