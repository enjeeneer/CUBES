# pylint: disable=all
import os
import subprocess
import pandas as pd


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


def copy_and_move_file(directories, file_name, destination):
    os.makedirs(destination, exist_ok=True)
    copied_files = []
    for directory in directories:
        src_file = os.path.join(directory, file_name)
        if os.path.exists(src_file):
            parts = directory.split(os.sep)
            first_dir_name = parts[1] if len(parts) > 1 else parts[0]
            dest_file = os.path.join(destination, first_dir_name + "_monitor.csv")
            df = pd.read_csv(
                src_file,
                usecols=list(range(5)) + list(range(15, 33)) + list(range(52, 60)),
            )
            df.to_csv(dest_file, index=False)
            copied_files.append(dest_file)
            print(f"Copied {src_file} to {dest_file}")
        else:
            print(f"{src_file} does not exist, skipping {directory}")
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
        os.chdir(repo_dir)
        for file in files_to_commit:
            subprocess.run(["git", "add", file], check=True)
        # subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print("Changes committed to git.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


# Example usage
search_name = "Eplus-env-sub_run1"
required_dir = "final_runs_v2-res1"
file_name = "monitor.csv"
root_directory = "."
destination_directory = "/home/jjjl4/rds/hpc-work/CUBES/exp/jack/paper/thermostat_experiment/Eplus_files/final_runs_v2/monitor"
commit_message = "Copied monitor.csv files to final_runs_v2"

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
