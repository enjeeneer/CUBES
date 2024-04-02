# pylint: disable-all
import pandas as pd
import numpy as np
import datetime
import glob
import os

cwd = os.getcwd()
pir_data_path = os.path.dirname(cwd)

cwd = os.getcwd()
pir_data_path = os.path.dirname(cwd)

# I will need to get all of the data


def get_raw_pir_data():
    pir_files = []
    for file in glob.glob(os.path.join(pir_data_path, "*CLEAN.txt")):
        pir_files.append(file)
    pir_files = sorted(pir_files)

    head_files = []
    for file in glob.glob(os.path.join(pir_data_path, "*.head")):
        head_files.append(file)
    head_files = sorted(head_files)

    return pir_files, head_files


def check_zone_data_coverage(dataframe):
    positive_counts = (dataframe >= 0).sum()
    total_steps = len(dataframe)

    coverage = positive_counts / total_steps

    list_coverage = coverage.to_list()

    return list_coverage


# get raw pir into a usable format with correct columns names


def format_pir_data(pir_files, head_files):

    data_dict = {"dataframe": []}

    for i, file in enumerate(pir_files):

        house_id = file.split("-")[1]

        data = pd.read_csv(file, sep=" ", header=None, index_col=0)

        # data = data.iloc[::5]

        with open(head_files[i], "r") as f:
            lines = f.readlines()

        column_names = lines[-1].split("\t")
        column_names = column_names[0].split("   ")

        if column_names[-1] == " ":
            del column_names[-1]

        rename = column_names[2:]
        new_cols = column_names[:2]

        for col in rename:
            col = house_id + "_" + col.split(",")[-1]
            new_cols.append(col)

        data.columns = new_cols

        data["UTC_Time"] = pd.to_datetime(data["unixtime"], unit="s", origin="unix")
        data = data.drop(columns=["unixtime", "unixtimeDST"])
        data = data.set_index("UTC_Time")

        zone_data_coverage = check_zone_data_coverage(data)

        # Collect columns to drop
        columns_to_drop = []

        for i, coverage in enumerate(zone_data_coverage):
            if coverage < 0.8:
                # Append the column name to the list of columns to drop
                columns_to_drop.append(data.columns[i])

        # Drop the columns outside the loop
        data_dropped = data.drop(columns=columns_to_drop, axis=1)

        if len(data_dropped.columns) > 0:
            data_dict["dataframe"].append(data_dropped)

    return data_dict


# Function to generate priority datetime offsets
def generate_datetime_priorities(base_datetime):
    offsets = [
        pd.DateOffset(years=0),  # same year
        pd.DateOffset(years=1),  # next year
        pd.DateOffset(years=-1),  # previous year
        pd.DateOffset(months=-1),  # previous month, same year
        pd.DateOffset(months=1),  # next month, same year
        pd.DateOffset(years=-1, months=-1),  # previous year, previous month
        pd.DateOffset(years=-1, months=1),  # previous year, next month
        pd.DateOffset(years=1, months=-1),  # next year, previous month
        pd.DateOffset(years=1, months=1),  # next year, next month
    ]
    return [base_datetime + offset for offset in offsets]


def replace_missing_data(df):
    print("df: ", df)
    print("house id: ", df.columns[0].split("_"))
    print("df type: ", type(df))
    replacement_error = {"id": [], "index": []}

    # Replace -99 with NaN for easier handling
    df.replace(-99, np.nan, inplace=True)

    # Iterate over the DataFrame to find and replace NaN values
    for (row_index, row) in df.iterrows():
        for col in df.columns:
            if pd.isnull(row[col]):  # If the value is NaN (originally -99)
                datetime_priorities = generate_datetime_priorities(row_index)

                # Attempt to find a replacement value
                replacement_found = False
                for priority_datetime in datetime_priorities:
                    if priority_datetime in df.index:
                        replacement_value = df.at[priority_datetime, col]
                        if not pd.isnull(replacement_value):
                            df.at[row_index, col] = replacement_value
                            replacement_found = True
                            break

                if not replacement_found:
                    replacement_error["id"].append(df.columns[0].split("_"))
                    replacement_error["index"].append(row_index)
                    # Optional: Handle the case where no replacement is found, e.g., keep as NaN or set a default value
                    pass

    return df, replacement_error


def resample_pir(dataframe, limit):
    resampled_df = dataframe.resample("10T").apply(
        lambda x: 1 if (x != 0).sum() >= limit else 0
    )

    return resampled_df


def load_pir_csv_file():
    csv_files = []
    for file in glob.glob("*.csv"):
        csv_files.append(file)
    csv_files = sorted(csv_files)
    return csv_files


def main():
    csv_files = load_pir_csv_file()
    resampling_values = [1, 2, 5]

    p, h = get_raw_pir_data()
    df_dict = format_pir_data(p, h)

    print(p, h)
    for i, csv in enumerate(csv_files):

        df = pd.read_csv(csv)

        df.index = df_dict["dataframe"][i].index

        house_id = df.columns[0].split("_")[0]
        print(house_id)
        os.makedirs(house_id)

        for r_v in resampling_values:
            print(r_v)
            resampled_df = resample_pir(df, r_v)

            file_path = os.path.join(
                house_id, "{}_{}_{}.csv".format(house_id, r_v, "schedule")
            )

            resampled_df.to_csv(file_path)

    return


# Function to apply formatting rules
def format_column_names(indexes):
    formatted_indexes = []
    for index in indexes:
        # Replace spaces with "_" outside of parentheses
        outside, *inside = index.split("(")
        outside_formatted = outside.replace(" ", "_")
        if inside:  # Check if there is a part inside parentheses
            inside_formatted = "(".join(inside).replace(
                " ", ""
            )  # Remove space inside parentheses
            formatted_index = f"{outside_formatted}({inside_formatted}"
        else:
            formatted_index = outside_formatted
        # Remove brackets and replace "(" with "_"
        formatted_index = formatted_index.replace("(", "_").replace(")", "")
        formatted_indexes.append(formatted_index)
    return formatted_indexes


def clean_column_names():
    # Specify the directory path
    directory_path = (
        "/workspaces/CUBES/exp/jack/Data/LEEDR/LEEDR_data_minute_pir/csv_schedules"
    )

    # Get all directories within the specified directory
    directories = [
        d
        for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d))
    ]

    # Iterate through each directory
    for directory in directories:
        limits = [1, 2, 5]
        for lim in limits:
            # Construct the path to the CSV file within the current directory
            filename = directory + "_" + str(lim) + "_schedule.csv"

            csv_file_path = os.path.join(directory_path, directory, filename)

            # Check if the CSV file exists
            if os.path.exists(csv_file_path):
                # Open the CSV file using pandas
                df = pd.read_csv(csv_file_path, index_col=0)

                index_labels = df.columns

                index_labels = [label.strip('"') for label in index_labels]

                index_labels = format_column_names(index_labels)

                df.columns = index_labels

                df.to_csv(csv_file_path)

            else:
                print(f"No CSV file found in {directory}")

    return


# def main():
#    p, h = get_raw_pir_data()
#    df = format_pir_data(p, h)
#
#    for fd in df["dataframe"]:
#        replaced_fd, replacement_error = replace_missing_data(fd)
#        house_id = df.columns[0].split("_")
#        replaced_fd.to_csv(house_id+".csv", index=False, sep=',', encoding='utf-8', header=True)
#
#    return

# main()
