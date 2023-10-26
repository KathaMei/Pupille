
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 08:58:14 2023

@author: Katharina
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessfunction2 import PLR2d
from preprocessfunction3 import PLR3d
import tkinter as tk
from tkinter import simpledialog
import csv
import os.path
import shutil
from datetime import datetime

# create a Tkinter window
root = tk.Tk()
root.withdraw()

# ask for subject id
subject_id = simpledialog.askstring(title="Subject ID",
                                    prompt="Enter subject ID:")

# ask for stimulation condition
stimulation_condition = simpledialog.askstring(title="Stimulation Condition",
                                               prompt="Enter stimulation condition:")
# Load the CSV file
df = pd.read_csv(f"/Users/Katharina/Desktop/Beispieldaten/{subject_id[:4]}/{subject_id}/exports/000/pupil_positions.csv", index_col=False)

# Load the annotation timestamps
annotation_timestamps = np.load(f"/Users/Katharina/Desktop/Beispieldaten/{subject_id[:4]}/{subject_id}/annotation_timestamps.npy")

# Define the duration of the window
window_duration = 60  # in seconds

# Define the columns to include in the CSV file
csv_cols = ['pupil_timestamp', 'diameter_3d', 'diameter','eye_id','confidence']

# Create an empty DataFrame to hold the sliced data
sliced_data = pd.DataFrame(columns=csv_cols)


# Loop through each annotation timestamp and slice the data
for annotation_timestamp in annotation_timestamps:
    # Calculate the start and end timestamps for the window after the annotation
    window_start = annotation_timestamp - 5.0
    window_end = window_start + window_duration
    
    # Select the rows that fall within the window
    df_sliced = df[(df['pupil_timestamp'] >= window_start) & (df['pupil_timestamp'] <= window_end)]
    
    # Add the sliced data to the DataFrame
    sliced_data = pd.concat([sliced_data, df_sliced[csv_cols]])
   
    
# Save the sliced data to a CSV file separate for each eye
sliced_data_0 = sliced_data[sliced_data['eye_id'] == 0]
sliced_data_1 = sliced_data[sliced_data['eye_id'] == 1]


# Interpolate missing values for diameter and diameter_3d
sliced_data_0['diameter'] = sliced_data_0['diameter'].interpolate(limit_direction='both')
sliced_data_0['diameter_3d'] = sliced_data_0['diameter_3d'].interpolate(limit_direction='both')


# Normalize the 'diameter' column using min-max normalization
sliced_data_0['diameter'] = (sliced_data_0['diameter'] - sliced_data_0['diameter'].min()) / (sliced_data_0['diameter'].max() - sliced_data_0['diameter'].min())

# Normalize the 'diameter_3d' column using min-max normalization
sliced_data_0['diameter_3d'] = (sliced_data_0['diameter_3d'] - sliced_data_0['diameter_3d'].min()) / (sliced_data_0['diameter_3d'].max() - sliced_data_0['diameter_3d'].min())

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the 'diameter' column on the first subplot
ax1.plot(sliced_data_0['pupil_timestamp'], sliced_data_0['diameter'], color='blue')
ax1.set_xlabel('Pupil Timestamp')
ax1.set_ylabel('Diameter')

# Plot the 'diameter_3d' column on the second subplot
ax2.plot(sliced_data_0['pupil_timestamp'], sliced_data_0['diameter_3d'], color='red')
ax2.set_xlabel('Pupil Timestamp')
ax2.set_ylabel('Diameter 3D')

# Show the plot
plt.show()


# Interpolate missing values for diameter and diameter_3d
sliced_data_1['diameter'] = sliced_data_1['diameter'].interpolate(limit_direction='both')
sliced_data_1['diameter_3d'] = sliced_data_1['diameter_3d'].interpolate(limit_direction='both')


# Normalize the 'diameter' column using min-max normalization
sliced_data_1['diameter'] = (sliced_data_1['diameter'] - sliced_data_1['diameter'].min()) / (sliced_data_1['diameter'].max() - sliced_data_1['diameter'].min())

# Normalize the 'diameter_3d' column using min-max normalization
sliced_data_1['diameter_3d'] = (sliced_data_1['diameter_3d'] - sliced_data_1['diameter_3d'].min()) / (sliced_data_1['diameter_3d'].max() - sliced_data_1['diameter_3d'].min())

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot the 'diameter' column on the first subplot
ax1.plot(sliced_data_1['pupil_timestamp'], sliced_data_1['diameter'], color='blue')
ax1.set_xlabel('Pupil Timestamp')
ax1.set_ylabel('Diameter')

# Plot the 'diameter_3d' column on the second subplot
ax2.plot(sliced_data_1['pupil_timestamp'], sliced_data_1['diameter_3d'], color='red')
ax2.set_xlabel('Pupil Timestamp')
ax2.set_ylabel('Diameter 3D')

# Show the plot
plt.show()

# Create an empty list to store the sliced dataframes and baselines
df_list_eye_id0 = []
df_list_eye_id1 = []
 
 # Loop through each annotation timestamp and create a list with the dataframes and a variable for each dataframe, eye 0
for i, annotation_timestamp in enumerate(annotation_timestamps):
    # Calculate the start and end timestamps for the window after the annotation
    window_start = annotation_timestamp - 5
    window_end = window_start + window_duration
    
    # Select the rows that fall within the window
    df_sliced_0 = sliced_data_0[(sliced_data_0['pupil_timestamp'] >= window_start) & (sliced_data_0['pupil_timestamp'] <= window_end)]
    df_sliced_1 = sliced_data_1[(sliced_data_1['pupil_timestamp'] >= window_start) & (sliced_data_1['pupil_timestamp'] <= window_end)]
    
   
    # Append the sliced dataframe to the list
    df_list_eye_id0.append(df_sliced_0)
    df_list_eye_id1.append(df_sliced_1)
    


# Iterate over each dataframe in the list
for i, df in enumerate(df_list_eye_id0):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot diameter vs pupil_timestamp on the left subplot
    ax1.plot(df["pupil_timestamp"], df["diameter"])
    ax1.set_xlabel("Pupil Timestamp")
    ax1.set_ylabel("Diameter")

    # Plot diameter_3d vs pupil_timestamp on the right subplot
    ax2.plot(df["pupil_timestamp"], df["diameter_3d"])
    ax2.set_xlabel("Pupil Timestamp")
    ax2.set_ylabel("Diameter 3D")

    # Show the plot
    plt.show()



#df_list_eye_id0[17].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe.csv', index=False)


#
#
#
#
#look at the normalisation

# Filter the data to include only rows with confidence >= 0.6
sliced_data_0_confidence = sliced_data_0[sliced_data_0['confidence'] >= 0.6]

# Calculate the 0.02 quantile on the filtered data
confidence_threshold_0 = sliced_data_0_confidence['confidence'].quantile(0.02)

# Print the result
print(f"The 0.02 quantile of pupil size for confidence >= 0.6 is {confidence_threshold_0:.2f}.")


# Filter the data to include only rows with confidence >= 0.6
sliced_data_1_confidence = sliced_data_1[sliced_data_1['confidence'] >= 0.6]

# Calculate the 0.02 quantile on the filtered data
confidence_threshold_1 = sliced_data_1_confidence['confidence'].quantile(0.02)

# Print the result
print(f"The 0.02 quantile of pupil size for confidence >= 0.6 is {confidence_threshold_1:.2f}.")


# Create an empty list to store the sliced dataframes and baselines
df_list_eye_id0 = []
df_list_eye_id1 = []
 
 # Loop through each annotation timestamp and create a list with the dataframes and a variable for each dataframe, eye 0
for i, annotation_timestamp in enumerate(annotation_timestamps):
    # Calculate the start and end timestamps for the window after the annotation
    window_start = annotation_timestamp - 5
    window_end = window_start + window_duration
    
    # Select the rows that fall within the window
    df_sliced_0 = sliced_data_0_confidence[(sliced_data_0_confidence['pupil_timestamp'] >= window_start) & (sliced_data_0_confidence['pupil_timestamp'] <= window_end)]
    df_sliced_1 = sliced_data_1_confidence[(sliced_data_1_confidence['pupil_timestamp'] >= window_start) & (sliced_data_1_confidence['pupil_timestamp'] <= window_end)]
    
    # Append the sliced dataframe to the list
    df_list_eye_id0.append(df_sliced_0)
    df_list_eye_id1.append(df_sliced_1)


# Loop through each dataframe in df_list_eye_id0
for i, df_sliced_0 in enumerate(df_list_eye_id0):
    # Create a histogram of the confidence values
    plt.hist(df_sliced_0['confidence'], bins=100, density=True)
    # Add a title and axis labels
    plt.title(f'Dataframe {i} - Eye 0')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    # Show the plot
    plt.show()

# Loop through each dataframe in df_list_eye_id0
for i, df_sliced_0 in enumerate(df_list_eye_id0):
    # Create a histogram of the confidence values
    plt.hist(df_sliced_0['diameter'], bins=100, density=True)
    # Add a title and axis labels
    plt.title(f'Dataframe {i} - Eye 0')
    plt.xlabel('Diameter')
    plt.ylabel('Density')
    # Show the plot
    plt.show()

# Loop through each dataframe in df_list_eye_id0
for i, df_sliced_0 in enumerate(df_list_eye_id0):
    # Create a histogram of the confidence values
    plt.hist(df_sliced_0['diameter_3d'], bins=100, density=True)
    # Add a title and axis labels
    plt.title(f'Dataframe {i} - Eye 0')
    plt.xlabel('Diameter_3D')
    plt.ylabel('Density')
    # Show the plot
    plt.show()
    



# Load the annotation timestamps
annotation_timestamps = np.load(f"/Users/Katharina/Desktop/Beispieldaten/{subject_id[:4]}/{subject_id}/annotation_timestamps.npy")

df_list_eye_id0_preprocessed = []

for i, df in enumerate(df_list_eye_id0):
    
    # Store the original unprocessed dataframe in a new variable
    df_preprocessed_eye_id0_i = df.copy()

    # Calculate masked first derivative of the dataframe
    df_preprocessed_eye_id0_i = PLR2d.mask_pupil_confidence([df_preprocessed_eye_id0_i], threshold=confidence_threshold_0)[0]
    df_preprocessed_eye_id0_i = PLR2d.iqr_threshold([df_preprocessed_eye_id0_i], iqr_factor=4, mask_cols=['diameter'])[0]
    df_preprocessed_eye_id0_i = PLR2d.mask_pupil_first_derivative([df_preprocessed_eye_id0_i])[0]
    df_preprocessed_eye_id0_i = PLR2d.mask_pupil_zscore([df_preprocessed_eye_id0_i], threshold=3.0, mask_cols=["diameter"])[0]
  
    # Find the index of the closest annotation to the current dataframe
    annotation_index = np.abs(annotation_timestamps - df_preprocessed_eye_id0_i.iloc[0]['pupil_timestamp']).argmin()
    
    # Define the baseline, stimulation, and after_var time ranges based on the annotation
    baseline_start = annotation_timestamps[annotation_index] - 5
    baseline_end = annotation_timestamps[annotation_index]
    stim_start = annotation_timestamps[annotation_index]
    stim_end = stim_start + 30
    after_var_start = stim_end
    after_var_end = after_var_start + 25
    
    # Add a new column to the dataframe with the appropriate label based on the time ranges
    df_preprocessed_eye_id0_i.loc[(df_preprocessed_eye_id0_i['pupil_timestamp'] >= baseline_start) & (df_preprocessed_eye_id0_i['pupil_timestamp'] <= baseline_end), 'label'] = 1
    df_preprocessed_eye_id0_i.loc[(df_preprocessed_eye_id0_i['pupil_timestamp'] >= stim_start) & (df_preprocessed_eye_id0_i['pupil_timestamp'] <= stim_end), 'label'] = 2
    df_preprocessed_eye_id0_i.loc[(df_preprocessed_eye_id0_i['pupil_timestamp'] >= after_var_start) & (df_preprocessed_eye_id0_i['pupil_timestamp'] <= after_var_end), 'label'] = 3
    
    # Add a new column to the dataframe containing its own index number
    df_preprocessed_eye_id0_i['index'] = i
    
    # Append the preprocessed dataframe to the list
    df_list_eye_id0_preprocessed.append(df_preprocessed_eye_id0_i)


# Initialize counter variable
counter = 0

# Loop through each dataframe in the list and plot the data for eye 0
for i, df in enumerate(df_list_eye_id0):
    # Extract the timestamp and diameter data from the dataframe
    pupil_timestamp = df['pupil_timestamp']
    diameter = df['diameter']
    
    # Create a new figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the data before preprocessing in the left subplot
    ax1.plot(pupil_timestamp, diameter, marker=".", linestyle="")
    ax1.set_title(f"Eye0 Dataframe {i}")
    ax1.set_xlabel("Pupil Timestamp")
    ax1.set_ylabel("Diameter in mm/pixels")
    
    
    # Extract the preprocessed dataframe from the preprocessed list
    df_preprocessed_eye_id0_i = df_list_eye_id0_preprocessed[i]
    
    # Extract the timestamp and diameter data from the preprocessed dataframe
    pupil_timestamp_pre = df_preprocessed_eye_id0_i['pupil_timestamp']
    diameter_pre = df_preprocessed_eye_id0_i['diameter']
    
    # Plot the data after preprocessing in the right subplot
    label1 = df_preprocessed_eye_id0_i[df_preprocessed_eye_id0_i['label']==1]
    label2 = df_preprocessed_eye_id0_i[df_preprocessed_eye_id0_i['label']==2]
    label3 = df_preprocessed_eye_id0_i[df_preprocessed_eye_id0_i['label']==3]
    
    ax2.plot(label1['pupil_timestamp'], label1['diameter'], color='red', label='label 1')
    ax2.plot(label2['pupil_timestamp'], label2['diameter'], color='blue', label='label 2')
    ax2.plot(label3['pupil_timestamp'], label3['diameter'], color='green', label='label 3')
   
   
    # Plot the data after preprocessing in the right subplot
    ax2.set_title(f"Preprocessed Eye0 Dataframe {i}")
    ax2.set_xlabel("Pupil Timestamp")
    ax2.set_ylabel("Diameter in mm/pixels")
    
    # Show the plot
    counter += 1
    plt.show()


removed_diameter_cols = []
df_list_eye_id0_preprocessed_1 = []

for i, df in enumerate(df_list_eye_id0_preprocessed):
    count_preprocessed_diameter = df['diameter'].count()
    count_diameter = df_list_eye_id0[i]['diameter'].count()

    diameter_diff = count_diameter - count_preprocessed_diameter
    diameter_pct_diff = count_preprocessed_diameter / count_diameter * 100

    if diameter_pct_diff < 50:
        df_copy = df.copy() # make a copy of the dataframe
        df_copy['diameter'] = np.nan # replace diameter column with NaNs
        df_list_eye_id0_preprocessed_1.append(df_copy) # append the preprocessed dataframe to the new list
        removed_diameter_cols.append(i)
    else:
        df_list_eye_id0_preprocessed_1.append(df) # append the original dataframe to the new list

    print(f"Removed 'diameter' column from DataFrame {i}")
    print(f"\nDataFrame {i}:")
    print(f"Difference in number of non-NaN values for diameter: {diameter_diff}")
    print(f"Percentage difference in number of non-NaN values for diameter: {diameter_pct_diff:.2f}%")

    
# Print summary message for removed 'diameter' columns
if removed_diameter_cols:
    print(f"\nThe 'diameter' column has been removed from the following DataFrames: {removed_diameter_cols}")
else:
    print("\nNo 'diameter' columns were removed.")



#df_list_eye_id0_preprocessed_1[17].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe.csv', index=False)



PLR3d.wait_for_input()


#diameter_3d

# Load the annotation timestamps
annotation_timestamps = np.load(f"/Users/Katharina/Desktop/Beispieldaten/{subject_id[:4]}/{subject_id}/annotation_timestamps.npy")

df_list_eye_id0_preprocessed_2 = []

for i, df in enumerate(df_list_eye_id0_preprocessed_1):

    # Store the original unprocessed dataframe in a new variable
    df_preprocessed_eye_id0_1_i = df.copy()
    
    # Calculate masked first derivative of the dataframe
    df_preprocessed_eye_id0_1_i = PLR3d.mask_pupil_confidence_3d([df_preprocessed_eye_id0_1_i], threshold=confidence_threshold_1)[0]
    df_preprocessed_eye_id0_1_i = PLR3d.iqr_threshold_3d([df_preprocessed_eye_id0_1_i], iqr_factor=4, mask_cols=['diameter_3d'])[0]
    df_preprocessed_eye_id0_1_i = PLR3d.mask_pupil_first_derivative_3d([df_preprocessed_eye_id0_1_i])[0]
    df_preprocessed_eye_id0_1_i = PLR3d.mask_pupil_zscore_3d([df_preprocessed_eye_id0_1_i], threshold=3.0, mask_cols=["diameter_3d"])[0]
  
        
    # Find the index of the closest annotation to the current dataframe
    annotation_index = np.abs(annotation_timestamps - df_preprocessed_eye_id0_1_i.iloc[0]['pupil_timestamp']).argmin()
    
    # Define the baseline, stimulation, and after_var time ranges based on the annotation
    baseline_start = annotation_timestamps[annotation_index] - 5
    baseline_end = annotation_timestamps[annotation_index]
    stim_start = annotation_timestamps[annotation_index]
    stim_end = stim_start + 30
    after_var_start = stim_end
    after_var_end = after_var_start + 25
    
    # Add a new column to the dataframe with the appropriate label based on the time ranges
    df_preprocessed_eye_id0_1_i.loc[(df_preprocessed_eye_id0_1_i['pupil_timestamp'] >= baseline_start) & (df_preprocessed_eye_id0_1_i['pupil_timestamp'] <= baseline_end), 'label'] = 1
    df_preprocessed_eye_id0_1_i.loc[(df_preprocessed_eye_id0_1_i['pupil_timestamp'] >= stim_start) & (df_preprocessed_eye_id0_1_i['pupil_timestamp'] <= stim_end), 'label'] = 2
    df_preprocessed_eye_id0_1_i.loc[(df_preprocessed_eye_id0_1_i['pupil_timestamp'] >= after_var_start) & (df_preprocessed_eye_id0_1_i['pupil_timestamp'] <= after_var_end), 'label'] = 3
    
    # Add a new column to the dataframe containing its own index number
    df_preprocessed_eye_id0_1_i['index'] = i
    
    # Append the preprocessed dataframe to the list
    df_list_eye_id0_preprocessed_2.append(df_preprocessed_eye_id0_1_i)


# Initialize counter variable
counter = 0

# Loop through each dataframe in the list and plot the data for eye 0
for i, df in enumerate(df_list_eye_id0_preprocessed_1):
    # Extract the timestamp and diameter data from the dataframe
    pupil_timestamp = df['pupil_timestamp']
    diameter_3d = df['diameter_3d']
    
    # Create a new figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the data before preprocessing in the left subplot
    ax1.plot(pupil_timestamp, diameter_3d, marker=".", linestyle="")
    ax1.set_title(f"Eye0 Dataframe 3D {i}")
    ax1.set_xlabel("Pupil Timestamp")
    ax1.set_ylabel("Diameter in mm/pixels")
    
    # Extract the preprocessed dataframe from the preprocessed list
    df_preprocessed_eye_id0_2_i = df_list_eye_id0_preprocessed_2[i]
    
    # Extract the timestamp and diameter data from the preprocessed dataframe
    pupil_timestamp_pre = df_preprocessed_eye_id0_2_i['pupil_timestamp']
    diameter_3d_pre = df_preprocessed_eye_id0_2_i['diameter_3d']
    
    # Plot the data after preprocessing in the right subplot
    label1 = df_preprocessed_eye_id0_2_i[df_preprocessed_eye_id0_2_i['label']==1]
    label2 = df_preprocessed_eye_id0_2_i[df_preprocessed_eye_id0_2_i['label']==2]
    label3 = df_preprocessed_eye_id0_2_i[df_preprocessed_eye_id0_2_i['label']==3]
    
    ax2.plot(label1['pupil_timestamp'], label1['diameter_3d'], color='red', label='label 1')
    ax2.plot(label2['pupil_timestamp'], label2['diameter_3d'], color='blue', label='label 2')
    ax2.plot(label3['pupil_timestamp'], label3['diameter_3d'], color='green', label='label 3')
   
   
    # Plot the data after preprocessing in the right subplot
    ax2.set_title(f"Preprocessed Eye0 Dataframe 3D {i}")
    ax2.set_xlabel("Pupil Timestamp")
    ax2.set_ylabel("Diameter in mm/pixels")
    
    # Show the plot
    counter += 1
    plt.show()


#df_list_eye_id0_preprocessed_2[17].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe.csv', index=False)



removed_diameter_cols_3d = []
df_list_eye_id0_preprocessed_3 = []

for i, df in enumerate(df_list_eye_id0_preprocessed_2):
    count_preprocessed_diameter = df['diameter_3d'].count()
    count_diameter = df_list_eye_id0_preprocessed_1[i]['diameter_3d'].count()

    diameter_diff = count_diameter - count_preprocessed_diameter
    diameter_pct_diff = count_preprocessed_diameter / count_diameter * 100

    if diameter_pct_diff < 50:
        df_copy = df.copy() # make a copy of the dataframe
        df_copy['diameter_3d'] = np.nan # replace diameter column with NaNs
        df_list_eye_id0_preprocessed_3.append(df_copy) # append the preprocessed dataframe to the new list
        removed_diameter_cols_3d.append(i)
    else:
        df_list_eye_id0_preprocessed_3.append(df) # append the original dataframe to the new list

    print(f"Removed 'diameter_3d' column from DataFrame {i}")
    print(f"\nDataFrame {i}:")
    print(f"Difference in number of non-NaN values for diameter_3d: {diameter_diff}")
    print(f"Percentage difference in number of non-NaN values for diameter_3d: {diameter_pct_diff:.2f}%")

    
# Print summary message for removed 'diameter' columns
if removed_diameter_cols_3d:
    print(f"\nThe 'diameter_3d' column has been removed from the following DataFrames: {removed_diameter_cols_3d}")
else:
    print("\nNo 'diameter_3d' columns were removed.")


#df_list_eye_id0_preprocessed_1[19].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe.csv', index=False)

#df_list_eye_id0_preprocessed_2[19].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe2.csv', index=False)





#df_list_eye_id0_preprocessed_filtered[4].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe.csv', index=False)

   
removed_diameter_cols_indices = []

removed_diameter_3d_cols_indices= []
# Create a new list to store the filtered data frames
df_list_eye_id0_preprocessed_filtered = []

# Ask the user whether to remove the 'diameter' column from any data frames
choice = simpledialog.askstring("Input", "Do you want to remove the 'diameter' column from any data frames? (y/n)", parent=root)
if choice.lower() == "y":
    # Ask the user which data frames to remove the 'diameter' column from
    index_str = simpledialog.askstring("Input", "Enter a list of data frame indices to remove the 'diameter' column from (separated by commas):", parent=root)
    try:
        index_list = [int(x.strip()) for x in index_str.split(",")]
    except:
        tk.messagebox.showerror("Error", "Invalid input. Please enter a comma-separated list of integers.")
    else:
        # Loop through all the data frames
        for i, df in enumerate(df_list_eye_id0_preprocessed_3):
            # If the current data frame is in the list of indices to remove the 'diameter' column from, set the values to NaN
            if i in index_list and "diameter" in df.columns:
                df["diameter"] = np.nan
                removed_diameter_cols_indices.append(i)
            # Append the filtered data frame to the new list
            df_list_eye_id0_preprocessed_filtered.append(df)
else:
    df_list_eye_id0_preprocessed_filtered = df_list_eye_id0_preprocessed_3.copy()

# Ask the user whether to remove the 'diameter_3d' column from any data frames
choice = simpledialog.askstring("Input", "Do you want to remove the 'diameter_3d' column from any data frames? (y/n)", parent=root)
if choice.lower() == "y":
    # Ask the user which data frames to remove the 'diameter_3d' column from
    index_str = simpledialog.askstring("Input", "Enter a list of data frame indices to remove the 'diameter_3d' column from (separated by commas):", parent=root)
    try:
        index_list = [int(x.strip()) for x in index_str.split(",")]
    except:
        tk.messagebox.showerror("Error", "Invalid input. Please enter a comma-separated list of integers.")
    else:
        # Loop through all the data frames
        for i, df in enumerate(df_list_eye_id0_preprocessed_filtered):
            # If the current data frame is in the list of indices to remove the 'diameter_3d' column from, set the values to NaN
            if i in index_list and "diameter_3d" in df.columns:
                df["diameter_3d"] = np.nan
                removed_diameter_3d_cols_indices.append(i)
else:
    pass # Do nothing if the user doesn't want to remove the 'diameter_3d' column



#df_list_eye_id1_preprocessed_filtered[3].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe.csv', index=False)




for i in range(len(df_list_eye_id0_preprocessed_filtered)):
    df = df_list_eye_id0_preprocessed_filtered[i]
    
    # Assign time_slot 0 to the baseline data (label=1)
    df.loc[df['label']==1, 'time_slot'] = 0
    
    # Divide remaining data into 1000 time slots (label=2 or 3)
    df.loc[df['label'].isin([2,3]), 'time_slot'] = pd.cut(df.loc[df['label'].isin([2,3]), 'pupil_timestamp'], bins=1000, labels=False) + 1
    
    df_list_eye_id0_preprocessed_filtered[i] = df


# Concatenate all the data frames into a single data frame
df_combined = pd.concat(df_list_eye_id0_preprocessed_filtered)
    
#
# use the subject_id and stimulation_condition to create the file name
output_path = f"/Users/Katharina/Desktop/Beispieldaten/{stimulation_condition}/"
eye_id = "eye_id0"
file_contents = "list"
file_name = f"{output_path}{subject_id}_{stimulation_condition}_{eye_id}_{file_contents}.csv"

df_combined.to_csv(f'{output_path}{subject_id}_{stimulation_condition}_{eye_id}_list.csv', index=False)



# Calculate the mean of the diameter and diameter_3d columns separately for each time slot
mean_diameter = df_combined.groupby('time_slot')['diameter'].mean()
mean_diameter_3d = df_combined.groupby('time_slot')['diameter_3d'].mean()


# Create a new data frame with the means and other columns
df_means = pd.DataFrame({
    'time_slot': mean_diameter.index,
    'eye_id': ['eye_id0'] * len(mean_diameter),
    'diameter': mean_diameter.values,
    'diameter_3d': mean_diameter_3d.values
})


output_path = f"/Users/Katharina/Desktop/Beispieldaten/{stimulation_condition}/"
eye_id = "eye_id0"
file_contents = "mean"
file_name = f"{output_path}{subject_id}_{stimulation_condition}_{eye_id}_{file_contents}.csv"

df_means.to_csv(f'{output_path}{subject_id}_{stimulation_condition}_{eye_id}_means.csv', index=False)



# Print the data frame to the console
print(df_means)




PLR3d.wait_for_input()




# Loop through each dataframe in df_list_eye_id0
for i, df_sliced_1 in enumerate(df_list_eye_id1):
    # Create a histogram of the confidence values
    plt.hist(df_sliced_1['confidence'], bins=100, density=True)
    # Add a title and axis labels
    plt.title(f'Dataframe {i} - Eye 1')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    # Show the plot
    plt.show()

# Loop through each dataframe in df_list_eye_id0
for i, df_sliced_1 in enumerate(df_list_eye_id1):
    # Create a histogram of the confidence values
    plt.hist(df_sliced_1['diameter'], bins=100, density=True)
    # Add a title and axis labels
    plt.title(f'Dataframe {i} - Eye 1')
    plt.xlabel('Diameter')
    plt.ylabel('Density')
    # Show the plot
    plt.show()

# Loop through each dataframe in df_list_eye_id0
for i, df_sliced_1 in enumerate(df_list_eye_id1):
    # Create a histogram of the confidence values
    plt.hist(df_sliced_1['diameter_3d'], bins=100, density=True)
    # Add a title and axis labels
    plt.title(f'Dataframe {i} - Eye 1')
    plt.xlabel('Diameter_3D')
    plt.ylabel('Density')
    # Show the plot
    plt.show()
    




# Load the annotation timestamps
annotation_timestamps = np.load(f"/Users/Katharina/Desktop/Beispieldaten/{subject_id[:4]}/{subject_id}/annotation_timestamps.npy")

df_list_eye_id1_preprocessed = []

for i, df in enumerate(df_list_eye_id1):
    
    # Store the original unprocessed dataframe in a new variable
    df_preprocessed_eye_id1_i = df.copy()

    # Calculate masked first derivative of the dataframe
    df_preprocessed_eye_id1_i = PLR2d.mask_pupil_confidence([df_preprocessed_eye_id1_i], threshold=confidence_threshold_1)[0]
    df_preprocessed_eye_id1_i = PLR2d.iqr_threshold([df_preprocessed_eye_id1_i], iqr_factor=4, mask_cols=['diameter'])[0]
    df_preprocessed_eye_id1_i = PLR2d.mask_pupil_first_derivative([df_preprocessed_eye_id1_i])[0]
    df_preprocessed_eye_id0_1_i = PLR3d.mask_pupil_zscore([df_preprocessed_eye_id0_1_i], threshold=3.0, mask_cols=["diameter_3d"])[0]
  

    # Find the index of the closest annotation to the current dataframe
    annotation_index = np.abs(annotation_timestamps - df_preprocessed_eye_id1_i.iloc[0]['pupil_timestamp']).argmin()
    
    # Define the baseline, stimulation, and after_var time ranges based on the annotation
    baseline_start = annotation_timestamps[annotation_index] - 5
    baseline_end = annotation_timestamps[annotation_index]
    stim_start = annotation_timestamps[annotation_index]
    stim_end = stim_start + 30
    after_var_start = stim_end
    after_var_end = after_var_start + 25
    
    # Add a new column to the dataframe with the appropriate label based on the time ranges
    df_preprocessed_eye_id1_i.loc[(df_preprocessed_eye_id1_i['pupil_timestamp'] >= baseline_start) & (df_preprocessed_eye_id1_i['pupil_timestamp'] <= baseline_end), 'label'] = 1
    df_preprocessed_eye_id1_i.loc[(df_preprocessed_eye_id1_i['pupil_timestamp'] >= stim_start) & (df_preprocessed_eye_id1_i['pupil_timestamp'] <= stim_end), 'label'] = 2
    df_preprocessed_eye_id1_i.loc[(df_preprocessed_eye_id1_i['pupil_timestamp'] >= after_var_start) & (df_preprocessed_eye_id1_i['pupil_timestamp'] <= after_var_end), 'label'] = 3
    
    # Add a new column to the dataframe containing its own index number
    df_preprocessed_eye_id1_i['index'] = i
    
    # Append the preprocessed dataframe to the list
    df_list_eye_id1_preprocessed.append(df_preprocessed_eye_id1_i)


# Initialize counter variable
counter = 0

# Loop through each dataframe in the list and plot the data for eye 0
for i, df in enumerate(df_list_eye_id1):
    # Extract the timestamp and diameter data from the dataframe
    pupil_timestamp = df['pupil_timestamp']
    diameter = df['diameter']
    
    # Create a new figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the data before preprocessing in the left subplot
    ax1.plot(pupil_timestamp, diameter, marker=".", linestyle="")
    ax1.set_title(f"Eye1 Dataframe {i}")
    ax1.set_xlabel("Pupil Timestamp")
    ax1.set_ylabel("Diameter in mm/pixels")
    
    
    # Extract the preprocessed dataframe from the preprocessed list
    df_preprocessed_eye_id1_i = df_list_eye_id1_preprocessed[i]
    
    # Extract the timestamp and diameter data from the preprocessed dataframe
    pupil_timestamp_pre = df_preprocessed_eye_id1_i['pupil_timestamp']
    diameter_pre = df_preprocessed_eye_id1_i['diameter']
    
    # Plot the data after preprocessing in the right subplot
    label1 = df_preprocessed_eye_id1_i[df_preprocessed_eye_id1_i['label']==1]
    label2 = df_preprocessed_eye_id1_i[df_preprocessed_eye_id1_i['label']==2]
    label3 = df_preprocessed_eye_id1_i[df_preprocessed_eye_id1_i['label']==3]
    
    ax2.plot(label1['pupil_timestamp'], label1['diameter'], color='red', label='label 1')
    ax2.plot(label2['pupil_timestamp'], label2['diameter'], color='blue', label='label 2')
    ax2.plot(label3['pupil_timestamp'], label3['diameter'], color='green', label='label 3')
   
   
    # Plot the data after preprocessing in the right subplot
    ax2.set_title(f"Preprocessed Eye1 Dataframe {i}")
    ax2.set_xlabel("Pupil Timestamp")
    ax2.set_ylabel("Diameter in mm/pixels")
    
    # Show the plot
    counter += 1
    plt.show()


removed_diameter_cols_eye1 = []
df_list_eye_id1_preprocessed_1 = []

for i, df in enumerate(df_list_eye_id1_preprocessed):
    count_preprocessed_diameter = df['diameter'].count()
    count_diameter = df_list_eye_id1[i]['diameter'].count()

    diameter_diff = count_diameter - count_preprocessed_diameter
    diameter_pct_diff = count_preprocessed_diameter / count_diameter * 100

    if diameter_pct_diff < 50:
        df_copy = df.copy() # make a copy of the dataframe
        df_copy['diameter'] = np.nan # replace diameter column with NaNs
        df_list_eye_id1_preprocessed_1.append(df_copy) # append the preprocessed dataframe to the new list
        removed_diameter_cols_eye1.append(i)
    else:
        df_list_eye_id1_preprocessed_1.append(df) # append the original dataframe to the new list

    print(f"Removed 'diameter' column from DataFrame {i}")
    print(f"\nDataFrame {i}:")
    print(f"Difference in number of non-NaN values for diameter: {diameter_diff}")
    print(f"Percentage difference in number of non-NaN values for diameter: {diameter_pct_diff:.2f}%")

    
# Print summary message for removed 'diameter' columns
if removed_diameter_cols_eye1:
    print(f"\nThe 'diameter' column has been removed from the following DataFrames: {removed_diameter_cols_eye1}")
else:
    print("\nNo 'diameter' columns were removed.")



#df_list_eye_id0_preprocessed_1[17].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe.csv', index=False)


PLR3d.wait_for_input()


#diameter_3d

# Load the annotation timestamps
annotation_timestamps = np.load(f"/Users/Katharina/Desktop/Beispieldaten/{subject_id[:4]}/{subject_id}/annotation_timestamps.npy")

df_list_eye_id1_preprocessed_2 = []

for i, df in enumerate(df_list_eye_id1_preprocessed_1):

    # Store the original unprocessed dataframe in a new variable
    df_preprocessed_eye_id1_1_i = df.copy()
    
    # Calculate masked first derivative of the dataframe
    df_preprocessed_eye_id1_1_i = PLR3d.mask_pupil_confidence_3d([df_preprocessed_eye_id1_1_i], threshold=confidence_threshold_1)[0]
    df_preprocessed_eye_id1_1_i = PLR3d.iqr_threshold_3d([df_preprocessed_eye_id1_1_i], iqr_factor=4, mask_cols=['diameter_3d'])[0]
    df_preprocessed_eye_id1_1_i = PLR3d.mask_pupil_first_derivative_3d([df_preprocessed_eye_id1_1_i])[0]
    df_preprocessed_eye_id1_1_i = PLR3d.mask_pupil_zscore_3d([df_preprocessed_eye_id1_1_i], threshold=3.0, mask_cols=["diameter_3d"])[0]

        
    # Find the index of the closest annotation to the current dataframe
    annotation_index = np.abs(annotation_timestamps - df_preprocessed_eye_id1_1_i.iloc[0]['pupil_timestamp']).argmin()
    
    # Define the baseline, stimulation, and after_var time ranges based on the annotation
    baseline_start = annotation_timestamps[annotation_index] - 5
    baseline_end = annotation_timestamps[annotation_index]
    stim_start = annotation_timestamps[annotation_index]
    stim_end = stim_start + 30
    after_var_start = stim_end
    after_var_end = after_var_start + 25
    
    # Add a new column to the dataframe with the appropriate label based on the time ranges
    df_preprocessed_eye_id1_1_i.loc[(df_preprocessed_eye_id1_1_i['pupil_timestamp'] >= baseline_start) & (df_preprocessed_eye_id1_1_i['pupil_timestamp'] <= baseline_end), 'label'] = 1
    df_preprocessed_eye_id1_1_i.loc[(df_preprocessed_eye_id1_1_i['pupil_timestamp'] >= stim_start) & (df_preprocessed_eye_id1_1_i['pupil_timestamp'] <= stim_end), 'label'] = 2
    df_preprocessed_eye_id1_1_i.loc[(df_preprocessed_eye_id1_1_i['pupil_timestamp'] >= after_var_start) & (df_preprocessed_eye_id1_1_i['pupil_timestamp'] <= after_var_end), 'label'] = 3
    
    # Add a new column to the dataframe containing its own index number
    df_preprocessed_eye_id1_1_i['index'] = i
    
    # Append the preprocessed dataframe to the list
    df_list_eye_id1_preprocessed_2.append(df_preprocessed_eye_id1_1_i)


# Initialize counter variable
counter = 0

# Loop through each dataframe in the list and plot the data for eye 0
for i, df in enumerate(df_list_eye_id1_preprocessed_1):
    # Extract the timestamp and diameter data from the dataframe
    pupil_timestamp = df['pupil_timestamp']
    diameter_3d = df['diameter_3d']
    
    # Create a new figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the data before preprocessing in the left subplot
    ax1.plot(pupil_timestamp, diameter_3d, marker=".", linestyle="")
    ax1.set_title(f"Eye1 Dataframe 3D {i}")
    ax1.set_xlabel("Pupil Timestamp")
    ax1.set_ylabel("Diameter in mm/pixels")
    
    # Extract the preprocessed dataframe from the preprocessed list
    df_preprocessed_eye_id1_2_i = df_list_eye_id1_preprocessed_2[i]
    
    # Extract the timestamp and diameter data from the preprocessed dataframe
    pupil_timestamp_pre = df_preprocessed_eye_id1_2_i['pupil_timestamp']
    diameter_3d_pre = df_preprocessed_eye_id1_2_i['diameter_3d']
    
    # Plot the data after preprocessing in the right subplot
    label1 = df_preprocessed_eye_id1_2_i[df_preprocessed_eye_id1_2_i['label']==1]
    label2 = df_preprocessed_eye_id1_2_i[df_preprocessed_eye_id1_2_i['label']==2]
    label3 = df_preprocessed_eye_id1_2_i[df_preprocessed_eye_id1_2_i['label']==3]
    
    ax2.plot(label1['pupil_timestamp'], label1['diameter_3d'], color='red', label='label 1')
    ax2.plot(label2['pupil_timestamp'], label2['diameter_3d'], color='blue', label='label 2')
    ax2.plot(label3['pupil_timestamp'], label3['diameter_3d'], color='green', label='label 3')
   
   
    # Plot the data after preprocessing in the right subplot
    ax2.set_title(f"Preprocessed Eye1 Dataframe 3D {i}")
    ax2.set_xlabel("Pupil Timestamp")
    ax2.set_ylabel("Diameter in mm/pixels")
    
    # Show the plot
    counter += 1
    plt.show()


#df_list_eye_id0_preprocessed_2[17].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe.csv', index=False)


removed_diameter_cols_3d_eye1 = []
df_list_eye_id1_preprocessed_3 = []

for i, df in enumerate(df_list_eye_id1_preprocessed_2):
    count_preprocessed_diameter = df['diameter_3d'].count()
    count_diameter = df_list_eye_id1_preprocessed_1[i]['diameter_3d'].count()

    diameter_diff = count_diameter - count_preprocessed_diameter
    diameter_pct_diff = count_preprocessed_diameter / count_diameter * 100

    if diameter_pct_diff < 50:
        df_copy = df.copy() # make a copy of the dataframe
        df_copy['diameter_3d'] = np.nan # replace diameter column with NaNs
        df_list_eye_id1_preprocessed_3.append(df_copy) # append the preprocessed dataframe to the new list
        removed_diameter_cols_3d_eye1.append(i)
    else:
        df_list_eye_id1_preprocessed_3.append(df) # append the original dataframe to the new list

    print(f"Removed 'diameter_3d' column from DataFrame {i}")
    print(f"\nDataFrame {i}:")
    print(f"Difference in number of non-NaN values for diameter_3d: {diameter_diff}")
    print(f"Percentage difference in number of non-NaN values for diameter_3d: {diameter_pct_diff:.2f}%")

    
# Print summary message for removed 'diameter' columns
if removed_diameter_cols_3d_eye1:
    print(f"\nThe 'diameter_3d' column has been removed from the following DataFrames: {removed_diameter_cols_3d_eye1}")
else:
    print("\nNo 'diameter_3d' columns were removed.")


#df_list_eye_id0_preprocessed_1[19].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe.csv', index=False)

removed_diameter_cols_indices_eye1 = []

removed_diameter_3d_cols_indices_eye1 = []

# Create a new list to store the filtered data frames
df_list_eye_id1_preprocessed_filtered = []

# Ask the user whether to remove the 'diameter' column from any data frames
choice = simpledialog.askstring("Input", "Do you want to remove the 'diameter' column from any data frames? (y/n)", parent=root)
if choice.lower() == "y":
    # Ask the user which data frames to remove the 'diameter' column from
    index_str = simpledialog.askstring("Input", "Enter a list of data frame indices to remove the 'diameter' column from (separated by commas):", parent=root)
    try:
        index_list = [int(x.strip()) for x in index_str.split(",")]
    except:
        tk.messagebox.showerror("Error", "Invalid input. Please enter a comma-separated list of integers.")
    else:
        # Loop through all the data frames
        for i, df in enumerate(df_list_eye_id1_preprocessed_3):
            # If the current data frame is in the list of indices to remove the 'diameter' column from, set the values to NaN
            if i in index_list and "diameter" in df.columns:
                df["diameter"] = np.nan
                removed_diameter_cols_indices_eye1.append(i)
                
            # Append the filtered data frame to the new list
            df_list_eye_id1_preprocessed_filtered.append(df)
else:
    df_list_eye_id1_preprocessed_filtered = df_list_eye_id1_preprocessed_3.copy()

# Ask the user whether to remove the 'diameter_3d' column from any data frames
choice = simpledialog.askstring("Input", "Do you want to remove the 'diameter_3d' column from any data frames? (y/n)", parent=root)
if choice.lower() == "y":
    # Ask the user which data frames to remove the 'diameter_3d' column from
    index_str = simpledialog.askstring("Input", "Enter a list of data frame indices to remove the 'diameter_3d' column from (separated by commas):", parent=root)
    try:
        index_list = [int(x.strip()) for x in index_str.split(",")]
    except:
        tk.messagebox.showerror("Error", "Invalid input. Please enter a comma-separated list of integers.")
    else:
        # Loop through all the data frames
        for i, df in enumerate(df_list_eye_id1_preprocessed_filtered):
            # If the current data frame is in the list of indices to remove the 'diameter_3d' column from, set the values to NaN
            if i in index_list and "diameter_3d" in df.columns:
                df["diameter_3d"] = np.nan
                removed_diameter_3d_cols_indices_eye1.append(i)
else:
    pass # Do nothing if the user doesn't want to remove the 'diameter_3d' column



df_list_eye_id0_preprocessed_filtered[7].to_csv('/Users/Katharina/Desktop/Beispieldaten/example_dataframe.csv', index=False)




for i in range(len(df_list_eye_id1_preprocessed_filtered)):
    df = df_list_eye_id1_preprocessed_filtered[i]
    
    # Assign time_slot 0 to the baseline data (label=1)
    df.loc[df['label']==1, 'time_slot'] = 0
    
    # Divide remaining data into 1000 time slots (label=2 or 3)
    df.loc[df['label'].isin([2,3]), 'time_slot'] = pd.cut(df.loc[df['label'].isin([2,3]), 'pupil_timestamp'], bins=1000, labels=False) + 1
    
    df_list_eye_id1_preprocessed_filtered[i] = df


# Concatenate all the data frames into a single data frame
df_combined = pd.concat(df_list_eye_id1_preprocessed_filtered)
    
#
#
#
# use the subject_id and stimulation_condition to create the file name
output_path = f"/Users/Katharina/Desktop/Beispieldaten/{stimulation_condition}/"
eye_id = "eye_id1"
file_contents = "list"
file_name = f"{output_path}{subject_id}_{stimulation_condition}_{eye_id}_{file_contents}.csv"

df_combined.to_csv(f'{output_path}{subject_id}_{stimulation_condition}_{eye_id}_list.csv', index=False)


# Calculate the mean of the diameter and diameter_3d columns separately for each time slot
mean_diameter_eye1 = df_combined.groupby('time_slot')['diameter'].mean()
mean_diameter_3d_eye1 = df_combined.groupby('time_slot')['diameter_3d'].mean()


# Create a new data frame with the means and other columns
df_means = pd.DataFrame({
    'time_slot': mean_diameter_eye1.index,
    'eye_id': ['eye_id0'] * len(mean_diameter_eye1),
    'diameter': mean_diameter_eye1.values,
    'diameter_3d': mean_diameter_3d_eye1.values
})

output_path = f"/Users/Katharina/Desktop/Beispieldaten/{stimulation_condition}/"
eye_id = "eye_id1"
file_contents = "mean"
file_name = f"{output_path}{subject_id}_{stimulation_condition}_{eye_id}_{file_contents}.csv"

df_means.to_csv(f'{output_path}{subject_id}_{stimulation_condition}_{eye_id}_means.csv', index=False)

# Print the data frame to the console
print(df_means)




# Define the file name and path for the CSV file
file_name = "my_data.csv"
file_path = f"/Users/Katharina/Desktop/Beispieldaten/{file_name}"

# Backup the existing CSV file
if os.path.isfile(file_path):
    backup_dir = 'csv_backups'
    if not os.path.isdir(backup_dir):
        os.makedirs(backup_dir)
    backup_path = os.path.join(backup_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{os.path.basename(file_path)}")
    shutil.copy2(file_path, backup_path)

# Write the data to the CSV file
if not os.path.isfile(file_path):
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["subject", "confidence_threshold_0", "diameter_threshold", "diameter_3d_threshold",
                             "removed_diameter_cols", "removed_diameter_cols_3d", "manually_removed_diameter_cols_eye0",
                             "manually_removed_diameter_3d_cols_eye0", "confidence_threshold_1",
                             "diameter_threshold_eye1", "diameter_3d_threshold_eye1", "removed_diameter_cols_eye1",
                             "removed_diameter_cols_3d_eye1", "manually_removed_diameter_cols_eye1", "manually_removed_diameter_3d_cols_eye1"])

with open(file_path, mode="a", newline="") as file:
    writer = csv.writer(file)

    # Write the data for the current subject to a new row in the CSV file
    row_data = [
        f"{subject_id}_{stimulation_condition}",
        confidence_threshold_0,
        removed_diameter_cols, 
        removed_diameter_cols_3d, 
        str(removed_diameter_cols_indices), 
        str(removed_diameter_3d_cols_indices), 
        confidence_threshold_1, 
        removed_diameter_cols_eye1, 
        removed_diameter_cols_3d_eye1, 
        str(removed_diameter_cols_indices_eye1), 
        str(removed_diameter_3d_cols_indices_eye1), 
        ",".join(removed_diameter_cols)
    ]
    writer.writerow(row_data)
    
