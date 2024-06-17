# pylint: disable=all
import os
import shutil
import subprocess


def find_dirs_with_name(root_dir, part1):
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath == root_dir:  # Only search in the root directory, not subdirectories
            for dirname in dirnames:
                if part1 in dirname:
                    matching_dirs.append(os.path.join(dirpath, dirname))
    return matching_dirs


def move_progress_csv(directories, destination):
    moved_files = []
    for directory in directories:
        source_file = os.path.join(directory, "progress.csv")
        if os.path.exists(source_file):
            dest_file = os.path.join(destination, os.path.basename(directory) + ".csv")
            shutil.move(source_file, dest_file)
            moved_files.append(dest_file)
            print(f"Moved {source_file} to {dest_file}")
        else:
            print(f"{source_file} does not exist, skipping {directory}")
    return moved_files


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

        print("Changes added to git.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


# Start searching from the current directory
root_directory = "."
search_part1 = "single_zone_control_manual"
destination_directory = "/home/jjjl4/rds/hpc-work/CUBES/exp/jack/paper/thermostat_experiment/Eplus_files/single_zoning_experiment"
commit_message = "Moved progress.csv files to single_zoning_experiment"


# Find matching directories
matching_directories = find_dirs_with_name(root_directory, search_part1)

# Move progress.csv files to the destination
moved_files = move_progress_csv(matching_directories, destination_directory)

# Find the root of the Git repository
repo_directory = get_git_repo_root()

if repo_directory and moved_files:
    # Commit the move to git
    git_commit(commit_message, repo_directory, moved_files)
else:
    if not repo_directory:
        print("Could not find the Git repository root. Skipping Git commit.")
    if not moved_files:
        print("No files were moved. Skipping Git commit.")
