import pandas as pd
import numpy as np
from tqdm import tqdm  # Use standard tqdm for terminal scripts
import sys

# Function to create a fast occupancy dictionary using zip
def create_occupancy_dict(prob_distribution):
    time_keys = zip(
        prob_distribution['hour'],
        prob_distribution['minute'],
        prob_distribution['day_of_week'],
        prob_distribution['month']
    )
    return dict(zip(time_keys, prob_distribution['average_occupancy']))

# Function to generate synthetic schedules using the occupancy dictionary
def generate_synthetic_schedule(occupancy_dict, num_minutes_in_year):
    synthetic_schedule = []
    for minute in tqdm(range(num_minutes_in_year), desc="Generating Schedule", unit="min", leave=True):
        current_hour = (minute // 60) % 24
        current_minute = minute % 60
        current_day_of_week = (minute // 1440) % 7  # 0=Monday, 6=Sunday
        current_month = ((minute // (1440 * 30)) % 12) + 1
        time_key = (current_hour, current_minute, current_day_of_week, current_month)

        # Lookup occupancy probability
        occupancy_probability = occupancy_dict.get(time_key, 0)  # Default to 0 if key not found
        is_occupied = np.random.binomial(1, occupancy_probability)
        synthetic_schedule.append(is_occupied)
    return synthetic_schedule

# Main script to generate schedules for all rooms
room_names = [
    'bedroom_1_schedule', 'backroom_schedule', 'front_room_schedule',
    'bathroom_schedule', 'kitchen_schedule', 'bedroom_3_schedule',
    'hall_upstairs_schedule', 'bedroom_2_schedule', 'hall_downstairs_schedule'
]

# Load probability distribution data from CSV file
print("Loading probabilities from CSV file...", flush=True)
final_probability_distribution = pd.read_csv("/workspaces/CUBES/exp/jack/paper/thermostat_experiment/SI/generate_occupancy/final_probability_distributions_numeric.csv")

# Create a precomputed dictionary for each room using the optimised zip approach
print("Creating occupancy dictionaries for each room...", flush=True)
room_occupancy_dicts = {
    room_name: create_occupancy_dict(final_probability_distribution[final_probability_distribution['room'] == room_name])
    for room_name in room_names
}

# Define the number of synthetic schedules to generate
num_schedules = 10
num_minutes_in_year = 365 * 1440  # Non-leap year

# Generate and save multiple schedules
for schedule_index in range(num_schedules):
    print(f"Starting schedule generation for repetition {schedule_index}...", flush=True)
    room_schedules = {}

    # Generate synthetic schedules for each room
    for room_name in room_names:
        print(f"Generating schedule for: {room_name}", flush=True)
        sys.stdout.flush()  # Ensure print statements show up immediately
        room_schedule = generate_synthetic_schedule(room_occupancy_dicts[room_name], num_minutes_in_year)
        room_schedules[room_name] = room_schedule

    # Convert the dictionary to a DataFrame for easy analysis
    schedules_df = pd.DataFrame(room_schedules)

    # Remove the '_schedule' suffix from the column names
    schedules_df.columns = [name.replace('_schedule', '') for name in schedules_df.columns]

   # Step to fractionalise rows (Vectorised)
    row_sums = schedules_df.sum(axis=1)  # Compute the sum for each row
    schedules_df = schedules_df.div(row_sums, axis=0).fillna(0)  # Divide by row sum and handle division by zero

    # Save to a CSV file with the appropriate filename
    filename = f"weekly_median_rep_{schedule_index}.sch"
    schedules_df.to_csv(filename, index=False)

    print(f"Saved schedule to {filename}", flush=True)

print("All schedules have been generated and saved successfully.", flush=True)
