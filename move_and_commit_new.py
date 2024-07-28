# pylint: disable=all
import os
import shutil
import subprocess
import pandas as pd


def find_dirs_with_name(root_dir, part1):
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if part1 in dirpath:
            matching_dirs.append(dirpath)
    return matching_dirs


def copy_and_move_progress_csv(directories, destination):
    # List to store all dataframes
    dfs = []

    final_results_dir = destination
    os.makedirs(
        final_results_dir, exist_ok=True
    )  # Create the final_results_v0 directory if it doesn't exist

    copied_files = []
    for directory in directories:
        source_file = os.path.join(directory, "progress.csv")
        if os.path.exists(source_file):
            print(directory)

            df = pd.read_csv(source_file)

            # Extract filename from file_path
            filename = os.path.basename(directory)

            filename_split = filename.split("thermostat_rbc_eco_case_0_")[-1]

            df["year"] = filename_split.split("year_")[-1].split("_")[0]
            df["case"] = filename_split.split("case_")[-1].split("_")[0]
            df["rep"] = filename_split.split("rep_")[-1].split("_")[0]
            df["zones_controlled"] = filename_split.split("zone_")[-1].split("_")[0]
            df["onoffseed"] = filename_split.split("onoffseed_")[-1].split("_")[0]
            df["tempseed"] = filename_split.split("tempseed_")[-1].split("_")[0]
            df["comforttemp"] = filename_split.split("comforttemp_")[-1].split("_")[0]
            df["setbacktemp"] = filename_split.split("setbacktemp_")[-1].split("_")[0]
            df["filename"] = filename

            # Append DataFrame to dfs list
            dfs.append(df)

            final_dest_file = os.path.join(
                final_results_dir, os.path.basename(directory) + "_progress.csv"
            )
            shutil.copy(source_file, final_dest_file)
            copied_files.append(final_dest_file)
            print(f"Copied {source_file} to {final_dest_file}")
        else:
            print(f"{source_file} does not exist, skipping {directory}")

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    results_condensed = combined_df[
        [
            "year",
            "case",
            "rep",
            "zones_controlled",
            "onoffseed",
            "tempseed",
            "cost",
            "cumulative_emissions",
            "comfort_violation (%)",
            "mean_comfort_violation",
            "std_comfort_violation",
        ]
    ]

    results_condensed = results_condensed.apply(pd.to_numeric, errors="coerce")

    # Define building_cost_df
    building_cost = {
        "case": [0, 1, 2, 3, 4, 5, 10],
        "building_cost": [0, 3800, 16000, 51000, 71000, 28000, 48000],
    }
    building_cost_df = pd.DataFrame(building_cost)

    # Define sensing_cost_df
    sensing_cost = {
        "zones_controlled": [0, 1, 4, 9],
        "sensing_cost": [0, 60, 240, 520],
        "control_names": [
            "Manual Control",
            "One Zone Controlled",
            "Partial Control",
            "Full Control",
        ],
    }
    sensing_cost_df = pd.DataFrame(sensing_cost)

    # Define sensing_cost_df
    case_names = {
        "case": [0, 1, 2, 3, 4, 5, 10],
        "retrofit_name": [
            "Baseline",
            "Fabric - Low cost",
            "Fabric - Shallow",
            "Fabric - Deep",
            "Fabric - Passivhaus",
            "System - ASHP",
            "System - ASHP+PV+Batt",
        ],
    }
    case_names_df = pd.DataFrame(case_names)

    ## Merge building_cost_df with master_df
    merged_df = pd.merge(results_condensed, building_cost_df, on="case", how="left")

    ## Merge sensing_cost_df with merged_df
    merged_df = pd.merge(merged_df, sensing_cost_df, on="zones_controlled", how="left")

    merged_df = pd.merge(merged_df, case_names_df, on="case", how="left")

    # Optionally, rename columns if needed
    merged_df.rename(columns={"building_cost_x": "building_cost"}, inplace=True)

    merged_df["installation_cost"] = (
        merged_df["building_cost"] + merged_df["sensing_cost"]
    )

    merged_df["name"] = merged_df["retrofit_name"] + " " + merged_df["control_names"]

    merged_df["cumulative_emissions"] = merged_df["cumulative_emissions"] / 1000
    merged_df["cost"] = merged_df["cost"] / 100

    df_2022 = merged_df[merged_df["year"] == 2022]
    df_2023 = merged_df[merged_df["year"] == 2023]

    df_2022 = df_2022.reset_index()
    df_2023 = df_2023.reset_index()

    df_2022 = (
        df_2022.groupby(
            ["case", "zones_controlled", "name", "retrofit_name", "installation_cost"]
        )
        .agg(
            mean_emissions=("cumulative_emissions", "mean"),
            std_emissions=("cumulative_emissions", "std"),
            mean_cost=("cost", "mean"),
            std_cost=("cost", "std"),
            mean_comfort=("comfort_violation (%)", "mean"),
            std_comfort=("comfort_violation (%)", "std"),
        )
        .reset_index()
    )

    df_2023 = (
        df_2023.groupby(
            ["case", "zones_controlled", "name", "retrofit_name", "installation_cost"]
        )
        .agg(
            mean_emissions=("cumulative_emissions", "mean"),
            std_emissions=("cumulative_emissions", "std"),
            mean_cost=("cost", "mean"),
            std_cost=("cost", "std"),
            mean_comfort=("comfort_violation (%)", "mean"),
            std_comfort=("comfort_violation (%)", "std"),
        )
        .reset_index()
    )

    df_2022["bill_saving"] = -1 * (
        df_2022["mean_cost"].loc[:] - df_2022["mean_cost"].iloc[0]
    )

    df_2022["payback"] = (
        df_2022["installation_cost"].loc[:] / df_2022["bill_saving"].loc[:]
    )

    df_2023["bill_saving"] = -1 * (df_2023["mean_cost"] - df_2023["mean_cost"].iloc[0])

    df_2023["payback"] = df_2023["installation_cost"] / df_2023["bill_saving"]

    df_2022["relative_emissions"] = (
        1 - (df_2022["mean_emissions"] / df_2022["mean_emissions"].iloc[0])
    ) * -100
    df_2022["relative_emissions_std"] = (
        df_2022["std_emissions"] / df_2022["mean_emissions"].loc[0]
    ) * 100

    df_2022["relative_cost"] = (
        (df_2022["mean_cost"] - df_2022["mean_cost"].iloc[0])
        / df_2022["mean_cost"].iloc[0]
    ) * 100
    df_2022["relative_cost_std"] = (
        df_2022["std_cost"] / df_2022["mean_cost"].loc[0]
    ) * 100

    df_2023["relative_cost"] = (
        (df_2023["mean_cost"] - df_2023["mean_cost"].iloc[0])
        / df_2023["mean_cost"].iloc[0]
    ) * 100
    df_2023["relative_cost_std"] = (
        df_2023["std_cost"] / df_2023["mean_cost"].loc[0]
    ) * 100

    df_2023["relative_emissions"] = (
        1 - (df_2023["mean_emissions"] / df_2023["mean_emissions"].iloc[0])
    ) * -100
    df_2023["relative_emissions_std"] = (
        df_2023["std_emissions"] / df_2023["mean_emissions"].loc[0]
    ) * 100

    # Assuming master_df is your DataFrame
    # Group by specified columns and calculate both mean and standard deviation
    combined = (
        merged_df.groupby(
            ["case", "zones_controlled", "name", "retrofit_name", "installation_cost"]
        )
        .agg(
            mean_emissions=("cumulative_emissions", "mean"),
            std_emissions=("cumulative_emissions", "std"),
            mean_cost=("cost", "mean"),
            std_cost=("cost", "std"),
            mean_comfort=("comfort_violation (%)", "mean"),
            std_comfort=("comfort_violation (%)", "std"),
        )
        .reset_index()
    )

    combined["bill_saving"] = -1 * (
        combined["mean_cost"] - combined["mean_cost"].iloc[0]
    )

    combined["payback"] = combined["installation_cost"] / combined["bill_saving"]

    combined["relative_emissions"] = (
        1 - (combined["mean_emissions"] / combined["mean_emissions"].iloc[0])
    ) * -100
    combined["relative_emissions_std"] = (
        combined["std_emissions"] / combined["mean_emissions"].loc[0]
    ) * 100

    combined["relative_cost"] = (
        (combined["mean_cost"] - combined["mean_cost"].iloc[0])
        / combined["mean_cost"].iloc[0]
    ) * 100
    combined["relative_cost_std"] = (
        combined["std_cost"] / combined["mean_cost"].loc[0]
    ) * 100

    merged_df_file_path = final_results_dir + "all_results.csv"
    combined_file_path = final_results_dir + "combined_results.csv"
    df_2022_file_path = final_results_dir + "2022_results.csv"
    df_2023_file_path = final_results_dir + "2023_results.csv"

    merged_df.to_csv(merged_df_file_path)
    combined.to_csv(combined_file_path)
    df_2022.to_csv(df_2022_file_path)
    df_2023.to_csv(df_2023_file_path)

    copied_files = [
        merged_df_file_path,
        combined_file_path,
        df_2022_file_path,
        df_2023_file_path,
    ]

    return copied_files


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
        # Navigate to the repository directory
        os.chdir(repo_dir)

        # Add specific files to staging
        for file in files_to_commit:
            subprocess.run(["git", "add", file], check=True)

        # Commit changes
        # subprocess.run(["git", "commit", "-m", commit_message], check=True)

        print("Changes committed to git.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


# Start searching from the current directory
root_directory = "."
search_part1 = "final_runs_v2"
destination_directory = "/home/jjjl4/rds/hpc-work/CUBES/exp/jack/paper/thermostat_experiment/Eplus_files/final_runs_v2/progress/"
commit_message = "Copied progress.csv files to final_runs_v0"

# Find matching directories
matching_directories = find_dirs_with_name(root_directory, search_part1)

# Copy and move progress.csv files to the destination
copied_files = copy_and_move_progress_csv(matching_directories, destination_directory)


# Find the root of the Git repository
repo_directory = get_git_repo_root()

if repo_directory and copied_files:
    # Commit the copy to git
    git_commit(commit_message, repo_directory, copied_files)
else:
    if not repo_directory:
        print("Could not find the Git repository root. Skipping Git commit.")
    if not copied_files:
        print("No files were copied. Skipping Git commit.")
