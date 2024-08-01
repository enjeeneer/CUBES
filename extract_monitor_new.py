# pylint: disable=all
import os
import subprocess
import pandas as pd
import numpy as np


def find_dirs_with_name(root_dir, part1, part2, part3):
    if not os.path.isdir(root_dir):
        raise ValueError(
            f"The directory {root_dir} does not exist or is not a directory."
        )
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if part1 in dirpath and part2 in dirpath and part3 in filenames:
            matching_dirs.append(dirpath)
    return matching_dirs


def flatten_and_sample(final):
    flattened = final.values.flatten()
    flattened = flattened[~np.isnan(flattened)]
    sample_fraction = 0.1
    num_samples = int(len(flattened) * sample_fraction)
    sampled_data = np.random.choice(flattened, size=num_samples, replace=False)
    return sampled_data


def collect_metadata(sampled_data, filename):
    metadata = {
        "filename": filename,
        "year": filename.split("year_")[-1].split("_")[0],
        "case": filename.split("case_")[-1].split("_")[0],
        "rep": filename.split("rep_")[-1].split("_")[0],
        "zones_controlled": filename.split("zone_")[-1].split("_")[0],
        "onoffseed": filename.split("onoffseed_")[-1].split("_")[0],
        "tempseed": filename.split("tempseed_")[-1].split("_")[0],
        "comforttemp": filename.split("comforttemp_")[-1].split("_")[0],
        "setbacktemp": filename.split("setbacktemp_")[-1].split("_")[0],
        "temp_data": sampled_data,
    }
    return metadata


def filter_df(df_csv, file_name):
    zones = [
        "hall_downstairs",
        "front_room",
        "kitchen",
        "backroom",
        "bedroom_3",
        "bedroom_1",
        "hall_upstairs",
        "bathroom",
        "bedroom_2",
    ]
    df_filtered = df_csv[(df_csv["hour"] > 7) & (df_csv["hour"] < 23)]
    air_temp_data, opr_temp_data = {}, {}

    for zone in zones:
        df_zone = df_filtered.filter(like=zone)
        df_zone = df_zone[df_zone.iloc[:, -1] > 0]

        if f"Zone Air Temperature({zone})" in df_zone.columns:
            air_temp_data[zone] = df_zone[f"Zone Air Temperature({zone})"]
        if f"Zone Air Temperature({zone})" in df_zone.columns:
            opr_temp_data[zone] = df_zone[f"Zone Operative Temperature({zone})"]

    final_air, final_opr = pd.DataFrame(air_temp_data), pd.DataFrame(opr_temp_data)
    sampled_air, sampled_opr = flatten_and_sample(final_air), flatten_and_sample(
        final_opr
    )
    entry_air, entry_opr = collect_metadata(sampled_air, file_name), collect_metadata(
        sampled_opr, file_name
    )

    return entry_air, entry_opr


def calculate_temperature_stats(temp_list):
    total_time = len(temp_list)

    below_15 = sum(temp < 15 for temp in temp_list)
    below_16 = sum(temp < 16 for temp in temp_list)
    below_17 = sum(temp < 17 for temp in temp_list)
    below_18 = sum(temp < 18 for temp in temp_list)
    above_25 = sum(temp > 25 for temp in temp_list)

    within_15_25 = sum(15 <= temp <= 25 for temp in temp_list)
    within_16_25 = sum(16 <= temp <= 25 for temp in temp_list)
    within_17_25 = sum(17 <= temp <= 25 for temp in temp_list)
    within_18_25 = sum(18 <= temp <= 25 for temp in temp_list)

    percent_below_15 = (below_15 / total_time) * 100
    percent_below_16 = (below_16 / total_time) * 100
    percent_below_17 = (below_17 / total_time) * 100
    percent_below_18 = (below_18 / total_time) * 100
    percent_above_25 = (above_25 / total_time) * 100

    percent_within_15_25 = (within_15_25 / total_time) * 100
    percent_within_16_25 = (within_16_25 / total_time) * 100
    percent_within_17_25 = (within_17_25 / total_time) * 100
    percent_within_18_25 = (within_18_25 / total_time) * 100

    return {
        "Percent below 15": percent_below_15,
        "Percent below 16": percent_below_16,
        "Percent below 17": percent_below_17,
        "Percent below 18": percent_below_18,
        "Percent above 25": percent_above_25,
        "Percent within 15-25": percent_within_15_25,
        "Percent within 16-25": percent_within_16_25,
        "Percent within 17-25": percent_within_17_25,
        "Percent within 18-25": percent_within_18_25,
    }


def copy_and_move_file(directories, file_name, destination):
    os.makedirs(destination, exist_ok=True)
    rows_air, rows_opr = [], []

    for directory in directories:
        src_file = os.path.join(directory, file_name)
        if os.path.exists(src_file):
            df = pd.read_csv(
                src_file,
                usecols=list(range(5)) + list(range(15, 33)) + list(range(52, 60)),
            )
            entry_air, entry_opr = filter_df(df, directory)
            rows_air.append(entry_air)
            rows_opr.append(entry_opr)
            print(f"Copied {src_file}")
        else:
            print(f"{src_file} does not exist, skipping {directory}")

    df_air, df_opr = pd.DataFrame(rows_air), pd.DataFrame(rows_opr)

    # Applying the function to each row
    air_temperature_stats = df_air["temp_data"].apply(calculate_temperature_stats)
    opr_temperature_stats = df_opr["temp_data"].apply(calculate_temperature_stats)

    air_stats_df = df_air.iloc[:, :-1]
    opr_stats_df = df_air.iloc[:, :-1]

    # Creating new columns in the dataframe
    for key in air_temperature_stats.iloc[0].keys():
        air_stats_df[key] = air_temperature_stats.apply(lambda x: x[key])

    # Creating new columns in the dataframe
    for key in opr_temperature_stats.iloc[0].keys():
        opr_stats_df[key] = opr_temperature_stats.apply(lambda x: x[key])

    df_air_file_path = os.path.join(
        destination, "Zone_Air_temperature_distribution.pkl"
    )
    df_opr_file_path = os.path.join(
        destination, "Zone_Operative_temperature_distribution.pkl"
    )

    df_air_stats_path = os.path.join(destination, "Zone_Air_temperature_summary.pkl")
    df_opr_stats_path = os.path.join(
        destination, "Zone_Operative_temperature_summary.pkl"
    )

    df_air.to_pickle(df_air_file_path)
    df_opr.to_pickle(df_opr_file_path)

    air_stats_df.to_pickle(df_air_stats_path)
    opr_stats_df.to_pickle(df_opr_stats_path)

    return [df_air_file_path, df_opr_file_path, df_air_stats_path, df_opr_stats_path]


def get_git_repo_root():
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
        ).strip()
        return repo_root
    except subprocess.CalledProcessError as e:
        print(f"Error finding Git repo root: {e}")
        return None


def git_commit(commit_message, repo_dir, files_to_commit):
    try:
        os.chdir(repo_dir)
        for file in files_to_commit:
            subprocess.run(["git", "add", file], check=True)
        # subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print("Changes committed to git.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


# Example usage
search_name = "Eplus-env-sub_run1"
required_dir = "final_runs_v3"
file_name = "monitor.csv"
root_directory = "."
destination_directory = "/home/jjjl4/rds/hpc-work/CUBES/exp/jack/paper/thermostat_experiment/Eplus_files/final_runs_v3/monitor"
commit_message = "Copied monitor.csv files to final_runs_v3"

matching_directories = find_dirs_with_name(
    root_directory, search_name, required_dir, file_name
)
copied_files = copy_and_move_file(
    matching_directories, file_name, destination_directory
)
repo_directory = get_git_repo_root()

if repo_directory and copied_files:
    git_commit(commit_message, repo_directory, copied_files)
else:
    if not repo_directory:
        print("Could not find the Git repository root. Skipping Git commit.")
    if not copied_files:
        print("No files were copied. Skipping Git commit.")
