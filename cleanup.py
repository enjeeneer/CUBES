# pylint: disable=all
import os
import glob
import shutil
import sys


def find_dir(search_string, base_dir):
    # Use glob to find directories that match the search string pattern
    matched_dirs = glob.glob(os.path.join(base_dir, f"*{search_string}*"))
    if matched_dirs:
        return matched_dirs[0]  # Return the first matched directory
    return None


def delete_files_and_dirs(base_dir, search_string):
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


if __name__ == "__main__":
    base_directory = "."
    if len(sys.argv) != 2:
        print("Usage: python cleanup.py <search_string>")
        sys.exit(1)

    search_string = sys.argv[1]
    delete_files_and_dirs(base_directory, search_string)
