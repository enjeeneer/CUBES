#!/bin/bash

# Arrays of different components of your directory structure
locations=("southampton" "london" "cambridge" "manchester" "newcastle")
controls=("manual" "eco")
cases=(0 1 2 3 4)
reps=(0 1 2 3 4)

# Loop through each combination of location, control, case, and rep
for location in "${locations[@]}"; do
    for control in "${controls[@]}"; do
        for case in "${cases[@]}"; do
            for rep in "${reps[@]}"; do
                # Construct the directory path
                dir_path="Eplus-env-2024-04-30_${location}_${control}_2022_case_${case}_rep_${rep}-res1"
                file_path="${dir_path}/progress.csv"

                # Check if the file exists
                if [ -f "$file_path" ]; then
                    # Add the file to the staging area
                    git add "$file_path"
                    echo "Added $file_path"
                else
                    echo "File not found: $file_path"
                fi
            done
        done
    done
done

# Commit the changes
git commit -m "Updated progress.csv files across multiple configurations"
