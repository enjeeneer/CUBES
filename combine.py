# pylint: disable=all
import os
import pandas as pd
import json

# Define the directory containing the JSON files
directory = "/home/jjjl4/rds/hpc-work/CUBES/results/"

# Initialise an empty list to hold DataFrames
aggregated_dfs = []

# Iterate through all JSON files in the directory
for filename in os.listdir(directory):
    if filename.endswith("final-paper2.json"):
        file_path = os.path.join(directory, filename)

        # Load the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

        # Convert JSON data to a DataFrame
        df = pd.DataFrame(data)

        # Calculate the mean across all zones for each metric
        aggregated_data = df.mean(axis=0)

        # Convert the aggregated series back to a DataFrame
        aggregated_df = aggregated_data.to_frame().T

        # Add a label for this aggregated entry, using the filename as identifier
        aggregated_df.insert(0, "filename", filename)

        # Append the aggregated DataFrame to the list
        aggregated_dfs.append(aggregated_df)

# Combine all aggregated DataFrames into a single DataFrame
final_df = pd.concat(aggregated_dfs, ignore_index=True)

# Optionally, save the final aggregated DataFrame to a CSV file
final_df.to_csv(
    "/home/jjjl4/rds/hpc-work/CUBES/exp/jack/paper/thermostat_experiment/Eplus_files/combined_results.csv",
    index=False,
)
