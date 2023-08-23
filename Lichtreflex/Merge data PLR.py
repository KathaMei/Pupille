#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 19:48:38 2023

@author: Katharina
"""
###############################unused


import os
import pandas as pd

# Define the path to the directory containing the CSV files
path = "/Users/Katharina/Desktop/Beispieldaten/"

# Initialize an empty list to store the individual dataframes
dfs = []

# loop through all subdirectories
for subdir, dirs, files in os.walk(path):
    for file in files:
        # check if the file is a csv file and has the desired name
        if file in ("PLR1-results.csv", "PLR2-results.csv", "PLR3-results.csv"):
            # read the file into a dataframe
            df = pd.read_csv(os.path.join(subdir, file))
            dfs.append(df)

# Concatenate all the dataframes into one
result = pd.concat(dfs)





# Define the mapping of Proband IDs to conditions
#let all conditions here and add the new ones
condition_mapping = {
    "PJ04_1": 1,
    "PJ04_2": 4,
    "PJ04_3": 3,
    "PJ04_4": 2
    # Add the rest of the mappings here...
}

# Define a function to extract the condition from the Proband ID
def get_condition(proband_id):
    # Extract the relevant part of the ID
    prefix = proband_id.split("_")[0] + "_" + proband_id.split("_")[1]
    # Look up the condition in the mapping
    condition = condition_mapping.get(prefix)
    return str(condition)

# Apply the function to the Proband ID column to create a new column called "condition"
df["condition"] = df["ID"].apply(get_condition)

# Save the updated dataframe to a new csv file
output_path = "/Users/Katharina/Desktop/Beispieldaten/merged_PLR_results_with_condition.csv"
df.to_csv(output_path, index=False)


