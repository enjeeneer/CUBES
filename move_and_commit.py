# pylint: disable=all
import os
import shutil
import subprocess


def find_dirs_with_name(root_dir, part1, part2):
    matching_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if part1 in dirname and part2 in dirname:
                matching_dirs.append(os.path.join(dirpath, dirname))
    return matching_dirs


def move_dirs(directories, destination):
    for directory in directories:
        dest_dir = os.path.join(destination, os.path.basename(directory))
        if not os.path.exists(dest_dir):
            shutil.move(directory, dest_dir)
            print(f"Moved {directory} to {dest_dir}")
        else:
            print(f"Destination {dest_dir} already exists, skipping {directory}")


def get_git_repo_root():
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
        ).strip()
        return repo_root
    except subprocess.CalledProcessError as e:
        print(f"Error finding Git repo root: {e}")
        return None


def git_commit(commit_message, repo_dir):
    try:
        # Navigate to the repository directory
        os.chdir(repo_dir)

        # Add changes to staging
        subprocess.run(["git", "add", "."], check=True)

        # Commit changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        print("Changes committed to git.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


# Start searching from the current directory
root_directory = "."
search_part1 = "zoning_influence_test"
search_part2 = "manual"
destination_directory = "/home/jjjl4/rds/hpc-work/CUBES/exp/jack/paper/thermostat_experiment/Eplus_files/single_zoning_experiment"
commit_message = "Moved zoning influence test directories to single_zoning_experiment"

# Find matching directories
matching_directories = find_dirs_with_name(root_directory, search_part1, search_part2)

# Move matching directories to the destination
move_dirs(matching_directories, destination_directory)

# Find the root of the Git repository
repo_directory = get_git_repo_root()

if repo_directory:
    # Commit the move to git
    git_commit(commit_message, repo_directory)
else:
    print("Could not find the Git repository root. Skipping Git commit.")
