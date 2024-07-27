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
                    if file_name not in ("progress.csv", "monitor.csv"):
                        file_path = os.path.join(root, file_name)
                        print(f"Deleting file: {file_path}")
                        os.remove(file_path)
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    print(f"Deleting directory: {dir_path}")
                    shutil.rmtree(dir_path)


if __name__ == "__main__":

    base_directory = "/home/jjjl4/rds/hpc-work/CUBES/Eplus_runs"
    delete_files_and_dirs(base_directory)
