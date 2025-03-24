import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# Directories for house-level and room-level schedules
house_schedule_dir = "/workspaces/CUBES/exp/jack/paper/thermostat_experiment/SI/generate_occupancy/mapped_schedules"
room_schedule_dir = "/workspaces/CUBES/exp/jack/paper/thermostat_experiment/SI/generate_occupancy/room_schedules"

# Define the list of house files and room files
house_files = glob.glob(os.path.join(house_schedule_dir, "*.csv"))
room_files = glob.glob(os.path.join(room_schedule_dir, "*.csv"))

# Create a dictionary to load all house schedules
house_schedules = {
    os.path.basename(f)
    .replace(".csv", ""): pd.read_csv(f, parse_dates=["UTC_Time"])
    .set_index("UTC_Time")
    for f in house_files
}

# Create a dictionary to load all room schedules
room_schedules = {
    os.path.basename(f)
    .replace("_schedule.csv", ""): pd.read_csv(f, parse_dates=["UTC_Time"])
    .set_index("UTC_Time")
    for f in room_files
}

# Mapping of which rooms to use for each house
house_room_availability = {
    "mapped_H09_cleaned_2013": ["hall_upstairs"],
    "mapped_H11_cleaned_2013": ["hall_upstairs", "bedroom_1"],
    "mapped_H28_cleaned_2013": ["hall_upstairs"],
    "mapped_H30_cleaned_2013": ["bedroom_3"],
    "mapped_H33_cleaned_2013": ["hall_downstairs", "hall_upstairs"],
    "mapped_H43_cleaned_2013": "all",  # Use full H43 schedule as it has complete coverage
}

# List of all rooms to ensure complete coverage
all_rooms = [
    "front_room",
    "kitchen",
    "backroom",
    "hall_downstairs",
    "bedroom_1",
    "bedroom_2",
    "bedroom_3",
    "bathroom",
    "hall_upstairs",
]

# Generate a complete minute-by-minute time index for an entire year (non-leap year)
time_index = pd.date_range(start="2013-01-01 00:00", end="2013-12-31 23:59", freq="min")

# Number of days in a non-leap year
num_days = len(time_index) // (24 * 60)
days_per_week = 7
minutes_per_week = days_per_week * 1440

# Number of synthetic schedules to generate
num_schedules = 10

# Create output directory for synthetic schedules
output_directory = ""

# Precompute week indices for slicing
week_slices = [
    (week_idx * minutes_per_week, (week_idx + 1) * minutes_per_week)
    for week_idx in range(num_days // days_per_week)
]

# Generate synthetic schedules
for replicate in tqdm(range(num_schedules), desc="Generating synthetic schedules"):
    synthetic_schedule = pd.DataFrame(
        index=time_index, columns=all_rooms
    )  # Initialise with full time index and rooms

    # Process each week, selecting a random house for each week
    for week_idx, (week_start_idx, week_end_idx) in tqdm(
        enumerate(week_slices),
        total=len(week_slices),
        desc=f"Processing weeks for replicate {replicate + 1}",
        leave=False,
    ):
        selected_house = np.random.choice(list(house_room_availability.keys()))
        available_rooms = house_room_availability[selected_house]

        # If the house has complete coverage, copy the full week's schedule directly
        if available_rooms == "all":
            week_data = (
                house_schedules[selected_house]
                .iloc[week_start_idx:week_end_idx]
                .reindex(columns=all_rooms)
            )
            synthetic_schedule.iloc[week_start_idx:week_end_idx] = week_data.values
        else:
            # Construct the weekly schedule for all rooms using a combination of house and room-level schedules
            week_data = {}
            for room in all_rooms:
                if (
                    room in available_rooms
                    and room in house_schedules[selected_house].columns
                ):
                    # Use the room's schedule from the selected house for this week
                    week_data[room] = (
                        house_schedules[selected_house][room]
                        .iloc[week_start_idx:week_end_idx]
                        .values
                    )
                else:
                    # Fallback to the room-level schedule using a random column for this week
                    random_column = np.random.choice(room_schedules[room].columns)
                    week_data[room] = (
                        room_schedules[room][random_column]
                        .iloc[week_start_idx:week_end_idx]
                        .values
                    )

            # Vectorised assignment of weekly data
            synthetic_schedule.iloc[week_start_idx:week_end_idx] = pd.DataFrame(
                week_data,
                index=time_index[week_start_idx:week_end_idx],
                columns=all_rooms,
            )

    synthetic_schedule = synthetic_schedule.clip(0, 1)  # Clip values between 0 and 1

    # Calculate row sums
    row_sums = synthetic_schedule.sum(axis=1)

    # Avoid division by zero by replacing zeros in row_sums with 1
    row_sums = row_sums.replace(0, 1)

    # Fractionalise each row using broadcasting
    synthetic_schedule = synthetic_schedule.div(row_sums, axis=0)

    # Save the complete synthetic schedule for the current replicate
    output_filename = os.path.join(output_directory, f"rep_{replicate + 1}.sch")
    synthetic_schedule.to_csv(output_filename, index=True, index_label="UTC_Time")

    print(
        f"Synthetic house schedule saved to '{output_filename}' with optimised sampling for replicate {replicate + 1}."
    )
