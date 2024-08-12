# pylint: disable=all
import os
import shutil
import pandas as pd


def combine_files_of_type(
    base_dir, run_name, file_suffix, output_filename, output_dir, output_format="csv"
):
    """
    Combine all result files of a specific type (determined by file_suffix) from different experiment runs
    that match the specified run_name into a single output file (CSV or pickle).

    Args:
        base_dir (str): The base directory where the unique directories are located.
        run_name (str): The specific run name to filter directories.
        file_suffix (str): The suffix of the file type to combine (e.g., "_summary.pkl", "_distribution.pkl").
        output_filename (str): The name of the output file for combined results of this type.
        output_dir (str): The directory where the combined file will be saved.
        output_format (str): The format of the output file, either "csv" or "pickle". Defaults to "csv".

    Returns:
        str: The path to the combined results file.
    """
    all_files = []

    # Traverse the base directory to find all files with the specific suffix and run_name in their directory path
    for root, dirs, files in os.walk(base_dir):
        if run_name in root:  # Filter directories based on the run_name
            for file in files:
                if file.endswith(file_suffix):  # Match files based on suffix
                    all_files.append(os.path.join(root, file))

    if not all_files:
        print(f"No {file_suffix} files found in directories matching '{run_name}'.")
        return None

    # Combine all found files into a single DataFrame
    combined_df = pd.concat(
        [
            pd.read_pickle(f) if f.endswith(".pkl") else pd.read_csv(f)
            for f in all_files
        ],
        ignore_index=True,
    )

    # Determine the output file path
    combined_file_path = os.path.join(output_dir, output_filename)

    # Save the combined DataFrame in the specified format
    if output_format == "csv":
        combined_df.to_csv(combined_file_path, index=False)
    elif output_format == "pickle":
        combined_df.to_pickle(combined_file_path)
    else:
        raise ValueError(
            f"Unsupported output format: {output_format}. Use 'csv' or 'pickle'."
        )

    print(f"Combined {file_suffix} results saved to: {combined_file_path}")
    return combined_file_path


def delete_run_directories(base_dir, run_name, exclude_dir):
    """
    Delete all directories matching the specified run_name, excluding the destination directory.

    Args:
        base_dir (str): The base directory where the unique directories are located.
        run_name (str): The specific run name to filter directories.
        exclude_dir (str): The directory to exclude from deletion.

    Returns:
        None
    """
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Check if the directory matches the run_name and is not the exclude_dir
            if run_name in dir_name and dir_path != exclude_dir:
                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")


if __name__ == "__main__":
    # Base directory where all unique experiment directories are located
    base_directory = "/home/jjjl4/rds/hpc-work/CUBES/exp/jack/paper/thermostat_experiment/Eplus_files/"

    # The specific run name to filter directories (e.g., "final_runs_v4")
    run_name = "final_runs_v4"

    # Destination directory to store the combined files
    destination_directory = os.path.join(base_directory, f"{run_name}_combined_results")

    # Ensure the destination directory exists
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Combine different types of files and save them directly into the destination directory
    combine_files_of_type(
        base_directory,
        run_name,
        "progress.csv",
        "combined_progress.csv",
        destination_directory,
        output_format="csv",
    )
    combine_files_of_type(
        base_directory,
        run_name,
        "Zone_Air_temperature_summary.pkl",
        "combined_air_temp_summary.pkl",
        destination_directory,
        output_format="pickle",
    )
    combine_files_of_type(
        base_directory,
        run_name,
        "Zone_Air_temperature_distribution.pkl",
        "combined_air_temp_distribution.pkl",
        destination_directory,
        output_format="pickle",
    )
    combine_files_of_type(
        base_directory,
        run_name,
        "Zone_Operative_temperature_summary.pkl",
        "combined_opr_temp_summary.csv",
        destination_directory,
        output_format="csv",
    )
    combine_files_of_type(
        base_directory,
        run_name,
        "Zone_Operative_temperature_distribution.pkl",
        "combined_opr_temp_distribution.pkl",
        destination_directory,
        output_format="pickle",
    )

    # Delete the original run directories, excluding the final storage directory
    delete_run_directories(base_directory, run_name, destination_directory)
