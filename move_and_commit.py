# pylint: disable=all
import os
import shutil
import subprocess


def find_dirs_with_name(root_dir, part1):
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if part1 in dirpath:  # Search directories containing part1 in their path
            matching_dirs.append(dirpath)
    return matching_dirs


def copy_and_move_progress_csv(directories, destination):
    final_results_dir = destination
    os.makedirs(
        final_results_dir, exist_ok=True
    )  # Create the final_results_v0 directory if it doesn't exist

    copied_files = []
    for directory in directories:
        source_file = os.path.join(directory, "progress.csv")
        if os.path.exists(source_file):
            final_dest_file = os.path.join(
                final_results_dir, os.path.basename(directory) + "_progress.csv"
            )
            shutil.copy(source_file, final_dest_file)
            copied_files.append(final_dest_file)
            print(f"Copied {source_file} to {final_dest_file}")
        else:
            print(f"{source_file} does not exist, skipping {directory}")
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
destination_directory = "/home/jjjl4/rds/hpc-work/CUBES/exp/jack/paper/thermostat_experiment/Eplus_files/final_runs_v0"
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
