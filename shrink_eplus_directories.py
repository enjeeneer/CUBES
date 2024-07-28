# pylint: disable=all
import os
import glob
import shutil


def delete_files_and_dirs(base_dir):
    # Use glob to find all directories matching 'Eplus-env-*'
    for dir_path in glob.iglob(os.path.join(base_dir, "Eplus-env-*")):
        if os.path.isdir(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    # Check if the file is progress.csv in the base directory or monitor.csv in the specific subdirectory
                    if (file_name == "progress.csv" and root == dir_path) or (
                        file_name == "monitor.csv"
                        and os.path.basename(root) == "Eplus-env-sub_run1"
                    ):
                        continue
                    # Delete the file if it is not progress.csv or monitor.csv
                    print(f"Deleting file: {file_path}")
                    os.remove(file_path)
                # Avoid deleting directories that are 'Eplus-env-sub_run1'
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if dir_name == "Eplus-env-sub_run1":
                        continue
                    # Delete the directory if it is not 'Eplus-env-sub_run1'
                    print(f"Deleting directory: {dir_path}")
                    shutil.rmtree(dir_path)


if __name__ == "__main__":
    # Replace 'your_base_directory_path' with the path to your base directory
    base_directory = "/home/jjjl4/rds/hpc-work/CUBES/Eplus_runs"
    delete_files_and_dirs(base_directory)
