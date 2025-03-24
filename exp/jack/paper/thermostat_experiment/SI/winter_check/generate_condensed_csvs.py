import pandas as pd
import gc
from tqdm import tqdm  # Import tqdm for progress bar

dir_path = "/workspaces/CUBES/exp/jack/paper/thermostat_experiment/SI/winter_check/monitor_files/"
out_dir = "/workspaces/CUBES/exp/jack/paper/thermostat_experiment/SI/winter_check/condensed_monitor_files/"
utc_col = pd.read_csv("/workspaces/CUBES/cubes/data/occupants/rep_0.sch")["UTC_Time"]  # Load UTC column once

# Define the columns to keep
columns = ["Environmental Impact Total CO2 Emissions Carbon Equivalent Mass(Site)",
           'Zone Operative Temperature(hall_downstairs)', 'Zone Operative Temperature(front_room)',
           'Zone Operative Temperature(kitchen)', 'Zone Operative Temperature(backroom)',
           'Zone Operative Temperature(bedroom_3)', 'Zone Operative Temperature(bedroom_1)',
           'Zone Operative Temperature(hall_upstairs)', 'Zone Operative Temperature(bathroom)',
           'Zone Operative Temperature(bedroom_2)', 'Zone Air Temperature(hall_downstairs)',
           'Zone Air Temperature(front_room)', 'Zone Air Temperature(kitchen)', 'Zone Air Temperature(backroom)',
           'Zone Air Temperature(bedroom_3)', 'Zone Air Temperature(bedroom_1)', 'Zone Air Temperature(hall_upstairs)',
           'Zone Air Temperature(bathroom)', 'Zone Air Temperature(bedroom_2)', 'Zone People Occupant Count(hall_downstairs)',
           'Zone People Occupant Count(front_room)', 'Zone People Occupant Count(kitchen)', 'Zone People Occupant Count(backroom)',
           'Zone People Occupant Count(bedroom_3)', 'Zone People Occupant Count(bedroom_1)',
           'Zone People Occupant Count(hall_upstairs)', 'Zone People Occupant Count(bathroom)',
           'Zone People Occupant Count(bedroom_2)']

# Function to process each monitor file
def process_and_save_monitor(file_name, output_file, columns_to_keep, utc_index):
    monitor = pd.read_csv(file_name, usecols=columns_to_keep)  # Load specific columns
    monitor = monitor.iloc[1:]  # Remove the first row (assuming it’s the header)
    monitor.index = utc_index  # Assign the UTC time as index
    monitor.index = pd.to_datetime(monitor.index)  # Convert the index to datetime
    monitor.to_csv(output_file, index=True)  # Save the processed DataFrame
    del monitor  # Clear the DataFrame from memory
    gc.collect()  # Run garbage collection to free memory

# List of input file names and corresponding output file names
monitor_files = [
    ("Eplus-env-10-18_thermostat_rbc_eco_year_2023_case_0_rep_H28_zone_0_inactive_threshold_30_comforttemp_20.0_setbacktemp_16_pattern_twice_timesteps_60_holiday_check-res1/Eplus-env-sub_run1/monitor.csv", "zone0_monitor_condensed.csv"),
    ("Eplus-env-10-18_thermostat_rbc_eco_year_2023_case_0_rep_H28_zone_9_inactive_threshold_30_comforttemp_20.0_setbacktemp_16_pattern_twice_timesteps_60_holiday_check-res1/Eplus-env-sub_run1/monitor.csv", "zone9_20_16_monitor_condensed.csv"),
    ("Eplus-env-10-18_thermostat_rbc_eco_year_2023_case_0_rep_H28_zone_9_inactive_threshold_30_comforttemp_21.0_setbacktemp_15_pattern_twice_timesteps_60_setback_check-res1/Eplus-env-sub_run1/monitor.csv", "zone9_21_15_monitor_condensed.csv"),
    ("Eplus-env-10-21_thermostat_rbc_eco_year_2023_case_0_rep_H28_zone_9_inactive_threshold_30_comforttemp_20.5_setbacktemp_16_pattern_twice_timesteps_60_setback_check-res1/Eplus-env-sub_run1/monitor.csv", "zone9_20.5_16_monitor_condensed.csv"),
    ("Eplus-env-10-21_thermostat_rbc_eco_year_2023_case_0_rep_H28_zone_9_inactive_threshold_30_comforttemp_21.0_setbacktemp_16_pattern_twice_timesteps_60_setback_check-res1/Eplus-env-sub_run1/monitor.csv", "zone9_21_16_monitor_condensed.csv")
]

# Process each monitor file one by one with tqdm progress bar
for file_name, output_file in tqdm(monitor_files, desc="Processing monitors"):
    process_and_save_monitor(dir_path + file_name, out_dir + output_file, columns, utc_col)
