# pylint: disable=all
import os
import pandas as pd
import shutil
import sys
import numpy as np
import time
import glob


def delete_files_and_dirs(base_dir, search_string):
    """
    Delete specific files and directories within a directory that matches a search string.

    Args:
        base_dir (str): The base directory where the search should begin.
        search_string (str): The string to search for in directory names.

    Returns:
        None
    """
    dir_name = find_dir(search_string, base_dir)
    if not dir_name:
        print(f"No directory found matching {search_string}")
        return
    for dir_path in glob.iglob(os.path.join(base_dir, dir_name)):
        if os.path.isdir(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if (file_name == "progress.csv" and root == dir_path) or (
                        file_name == "monitor.csv"
                        and os.path.basename(root) == "Eplus-env-sub_run1"
                    ):
                        continue
                    print(f"Deleting file: {file_path}")
                    os.remove(file_path)
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if dir_name == "Eplus-env-sub_run1":
                        continue
                    print(f"Deleting directory: {dir_path}")
                    shutil.rmtree(dir_path)


def get_unique_directory(base_dir, run_name):
    """
    Create a unique directory based on the run name and current timestamp.

    Args:
        base_dir (str): The base directory where the unique directory should be created.
        run_name (str): The name of the run to include in the directory name.

    Returns:
        str: The path to the unique directory.
    """
    timestamp = int(time.time())
    unique_dir = os.path.join(base_dir, f"{run_name}_{timestamp}")
    os.makedirs(unique_dir, exist_ok=True)
    return unique_dir


def flatten_and_sample(final):
    """
    Flatten a DataFrame and randomly sample a fraction of its values.

    Args:
        final (pd.DataFrame): The DataFrame to be flattened and sampled.

    Returns:
        np.ndarray: A numpy array containing the sampled values.
    """
    flattened = final.values.flatten()
    flattened = pd.to_numeric(flattened, errors="coerce")
    flattened = flattened[~np.isnan(flattened)]
    sample_fraction = 0.1
    num_samples = int(len(flattened) * sample_fraction)
    sampled_data = np.random.choice(flattened, size=num_samples, replace=False)
    return sampled_data


def collect_metadata(sampled_data, filename):
    """
    Collect metadata from a filename and combine it with sampled data.

    Args:
        sampled_data (np.ndarray): The sampled temperature data.
        filename (str): The filename containing metadata.

    Returns:
        dict: A dictionary containing metadata and sampled data.
    """
    return {
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


def filter_df(df_csv, file_name):
    """
    Filter and process a DataFrame to extract air and operative temperature data for specific zones.

    Args:
        df_csv (pd.DataFrame): The original DataFrame containing temperature data.
        file_name (str): The filename used to extract metadata.

    Returns:
        tuple: Two DataFrames, one for air temperature and one for operative temperature, each with metadata.
    """
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
        print(df_zone.columns)
        df_zone = df_zone[df_zone[f"Zone People Occupant Count({zone})"] > 0]
        air_temp_data[zone] = df_zone[f"Zone Air Temperature({zone})"]
        opr_temp_data[zone] = df_zone[f"Zone Operative Temperature({zone})"]

    final_air, final_opr = pd.DataFrame(air_temp_data), pd.DataFrame(opr_temp_data)
    sampled_air, sampled_opr = flatten_and_sample(final_air), flatten_and_sample(
        final_opr
    )
    entry_air, entry_opr = pd.DataFrame(
        collect_metadata(sampled_air, file_name)
    ), pd.DataFrame(collect_metadata(sampled_opr, file_name))

    return entry_air, entry_opr


def calculate_and_package_stats(df):
    """
    Calculate temperature statistics for specific ranges and package them with metadata.

    Args:
        df (pd.DataFrame): The DataFrame containing temperature data and metadata.

    Returns:
        pd.Series: A Series containing metadata and temperature statistics.
    """
    temp_data = df["temp_data"]
    total_time = len(temp_data)

    # Initialize a dictionary to store the percentage of time at each temperature range
    percentages = {}

    # Define the temperature bins and calculate percentages
    bins = list(range(15, 26))  # from 15 to 25
    labels = [f"Percent at {i}" for i in range(15, 25)] + ["Percent at 25 or above"]

    # Calculate percentage for each bin
    temp_counts, _ = np.histogram(temp_data, bins=bins + [float("inf")])
    for i, label in enumerate(labels):
        percentages[label] = (temp_counts[i] / total_time) * 100

    # Calculate the percentage of time the temperature is 15 or below
    below_15 = sum(t <= 15 for t in temp_data)
    percentages["Percent at 15 or below"] = (below_15 / total_time) * 100

    # Add the computed percentages to the DataFrame
    percentages_df = pd.DataFrame(percentages, index=[0])

    # Extract the metadata (excluding the temp_data column)
    metadata = df.iloc[0, :].drop("temp_data")

    # Combine metadata with the calculated statistics
    combined_df = pd.concat([metadata, percentages_df.iloc[0]])

    return combined_df


def get_monitor_data(directory, destination, run_name, file_name="monitor.csv"):
    """
    Process a monitor CSV file containing air and operative temperature data, calculate statistics, and store results.

    Args:
        directory (str): The directory containing the monitor CSV file.
        destination (str): The directory where the output files will be saved.
        run_name (str): The specific run identifier for the case.
        file_name (str, optional): The name of the monitor CSV file. Defaults to "monitor.csv".

    Returns:
        list: Paths to the saved distribution and summary files.
    """
    destination = os.path.join(destination, "monitor")
    os.makedirs(destination, exist_ok=True)
    data_file = os.path.join(directory, "Eplus-env-sub_run1", file_name)

    if not os.path.exists(data_file):
        print(f"{data_file} does not exist, skipping {directory}")
        return

    print(f"Processing {directory}")
    try:
        df = pd.read_csv(
            data_file,
        )
        if df.shape[0] < 52561:
            print(f"Rerun analysis of {directory}")
            return
    except pd.errors.EmptyDataError:
        print(f"{data_file} is empty, skipping {directory}")
        return

    df_air, df_opr = filter_df(df, directory)  # Using your provided filter function

    df_air_stats = calculate_and_package_stats(df_air)
    df_opr_stats = calculate_and_package_stats(df_opr)

    # Extract the 'temp_data' column as a list and store it in a single entry in the dataframe
    if "temp_data" in df_air.columns:
        temp_data_list_air = df_air["temp_data"].tolist()
        df_air = df_air.iloc[0, :].copy()
        df_air["temp_data"] = temp_data_list_air

    if "temp_data" in df_opr.columns:
        temp_data_list_opr = df_opr["temp_data"].tolist()
        df_opr = df_opr.iloc[0, :].copy()
        df_opr["temp_data"] = temp_data_list_opr

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

    # Concatenate distribution data correctly
    if os.path.exists(df_air_file_path):
        existing_df_air = pd.read_pickle(df_air_file_path)
        df_air = pd.concat([existing_df_air, df_air.to_frame().T], ignore_index=True)
    else:
        df_air = df_air.to_frame().T
    df_air.to_pickle(df_air_file_path)

    if os.path.exists(df_opr_file_path):
        existing_df_opr = pd.read_pickle(df_opr_file_path)
        df_opr = pd.concat([existing_df_opr, df_opr.to_frame().T], ignore_index=True)
    else:
        df_opr = df_opr.to_frame().T
    df_opr.to_pickle(df_opr_file_path)

    # Ensure that df_air_stats and df_opr_stats have the correct column structure
    if os.path.exists(df_air_stats_path):
        existing_air_stats_df = pd.read_pickle(df_air_stats_path)
        df_air_stats = (
            df_air_stats.to_frame().T
            if isinstance(df_air_stats, pd.Series)
            else df_air_stats
        )
        df_air_stats.columns = (
            existing_air_stats_df.columns
        )  # Ensure consistent columns
        df_air_stats = pd.concat(
            [existing_air_stats_df, df_air_stats], ignore_index=True
        )
    else:
        df_air_stats = (
            df_air_stats.to_frame().T
            if isinstance(df_air_stats, pd.Series)
            else df_air_stats
        )
    df_air_stats.to_pickle(df_air_stats_path)

    if os.path.exists(df_opr_stats_path):
        existing_opr_stats_df = pd.read_pickle(df_opr_stats_path)
        df_opr_stats = (
            df_opr_stats.to_frame().T
            if isinstance(df_opr_stats, pd.Series)
            else df_opr_stats
        )
        df_opr_stats.columns = (
            existing_opr_stats_df.columns
        )  # Ensure consistent columns
        df_opr_stats = pd.concat(
            [existing_opr_stats_df, df_opr_stats], ignore_index=True
        )
    else:
        df_opr_stats = (
            df_opr_stats.to_frame().T
            if isinstance(df_opr_stats, pd.Series)
            else df_opr_stats
        )
    df_opr_stats.to_pickle(df_opr_stats_path)

    return [df_air_file_path, df_opr_file_path, df_air_stats_path, df_opr_stats_path]


def get_progress_data(directory, destination, run_name):
    """
    Process the progress CSV file, append metadata, and save the results.

    Args:
        directory (str): The directory containing the progress CSV file.
        destination (str): The directory where the output files will be saved.
        run_name (str): The specific run identifier for the case.

    Returns:
        str: The path to the saved progress data file.
    """
    destination = os.path.join(destination, "progress")
    os.makedirs(destination, exist_ok=True)

    final_results_path = os.path.join(destination, "progress.csv")

    data_file = os.path.join(directory, "progress.csv")
    df = pd.read_csv(data_file)

    filename = os.path.basename(directory)
    filename_split = filename.split("thermostat_rbc_eco_case_0_")[-1]

    df["date"] = filename.split("Eplus-env-")[-1].split("_thermostat_rbc")[0]
    df["year"] = filename_split.split("year_")[-1].split("_")[0]
    df["case"] = filename_split.split("case_")[-1].split("_")[0]
    df["rep"] = filename_split.split("rep_")[-1].split("_")[0]
    df["zones_controlled"] = filename_split.split("zone_")[-1].split("_")[0]
    df["onoffseed"] = filename_split.split("onoffseed_")[-1].split("_")[0]
    df["tempseed"] = filename_split.split("tempseed_")[-1].split("_")[0]
    df["comforttemp"] = filename_split.split("comforttemp_")[-1].split("_")[0]
    df["setbacktemp"] = filename_split.split("setbacktemp_")[-1].split("_")[0]
    df["filename"] = filename

    if not os.path.exists(final_results_path):
        pd.DataFrame(columns=df.columns).to_csv(final_results_path, index=False)

    results_df = pd.read_csv(final_results_path)
    results_df = pd.concat([results_df, df])
    results_df.to_csv(final_results_path, index=False)

    return final_results_path


def delete_directory(directory_path):
    """
    Delete the specified directory and all its contents.

    Args:
        directory_path (str): The path of the directory to delete.

    Returns:
        None
    """
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
        print(f"Deleted directory: {directory_path}")
    else:
        print(f"Directory does not exist: {directory_path}")


def find_dir(search_string, base_dir):
    matched_dirs = glob.glob(os.path.join(base_dir, f"*{search_string}*"))
    return matched_dirs[0] if matched_dirs else None


if __name__ == "__main__":
    base_directory = "."
    if len(sys.argv) != 3:
        print("Usage: python cleanup.py <search_string> <run_name>")
        sys.exit(1)

    search_string = sys.argv[1]
    run_name = sys.argv[2]
    unique_directory = get_unique_directory(
        "/home/jjjl4/rds/hpc-work/CUBES/exp/jack/paper/thermostat_experiment/Eplus_files/",
        run_name,
    )
    commit_message = f"Copied {run_name} progress.csv files"

    files = []

    dir = find_dir(search_string, base_directory)
    delete_files_and_dirs(base_directory, search_string)
    file_to_add = get_progress_data(dir, unique_directory, run_name)
    files.append(file_to_add)
    files.extend(get_monitor_data(dir, unique_directory, run_name))

    # Delete the directory after data extraction
    delete_directory(dir)
