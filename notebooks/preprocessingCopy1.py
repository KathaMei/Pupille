import sys
sys.path.append("../Pupillengröße/Skripte/")
# preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os.path
import shutil
from datetime import datetime
from preprocessfunction2 import PLR2d
from preprocessfunction3 import PLR3d


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 08:58:14 2023

@author: Katharina
"""
from dataclasses import dataclass
@dataclass 
class ProcessConfig: 
    eyenum:int=0 #eye0 right, eye1 left
    column:str="unknown" #diameter or diameter_3d
    sfactor:float=1 #the factor the velocity (vt) should be divided with to detect blinks in the blink_reconstruct
    data_path:str="" #path the data is taken
    subject_id:str="" #subject_id
    condition:str="" #1,2,3 or 4: 30Stim, 30Placebo, 3.4Sham, 3.4Placebo
    timebase:str="" #30s stimulation/placebo or 3.4s stimulation/placebo
    start_time_offset:float=0 #stimulation duration: 3.4s or 30s
    end_time_offset:float=0 #time after stimulation: 26.6s or 30s
    window_duration:float=0  #the time of the whole dataset: start_time_offset + after_var_start_offset
    upper_threshold:float=0 #upper diameter threshold, different for diameter (pixel) and diameter_3d (mm)
    diameter_threshold:float=0 #lower diameter threshold, different for diameter (pixel) and diameter_3d (mm)

# blinkreconstruct for a pandas series. Returns a numpy array.
# see https://pydatamatrix.eu/0.15/series/#function-blinkreconstructseries-vt5-vt_start10-vt_end5-maxdur500-margin10-smooth_winlen21-std_thr3-gap_margin20-gap_vt10-modeuoriginal
def blinkreconstruct(df, vt=5, vt_start=10, vt_end=5, maxdur=500, margin=10, smooth_winlen=21, std_thr=3, gap_margin=20, gap_vt=10, mode=u'advanced'):
    display(type(df))
    import datamatrix
    import datamatrix.series
    import datamatrix.operations
    dm=datamatrix.convert.from_pandas(df).series
    return datamatrix.series.blinkreconstruct(dm, vt,vt_start,vt_end,maxdur,margin,smooth_winlen,std_thr,gap_margin,gap_vt,mode)

def reconstruct(config: ProcessConfig, eye, window_size=100):
    # Remove blinks.
    plot_all = {
    99: True  # Specify the index of the data frame you want to plot
    }
    col = config.column
    interp_col = f'{col}_interp'
    rec_col = f'{col}_rec'

    eye[interp_col] = eye[col].interpolate(method='linear')
    eye[rec_col] = blinkreconstruct(eye[interp_col],
                                    vt_start=10 / config.sfactor, vt_end=5 / config.sfactor, maxdur=800,
                                    mode='advanced')

    
    # Apply the mask_pupil_first_derivative function to eye_rec column
    eye[rec_col] = PLR2d.mask_pupil_first_derivative(eye, mask_cols=[rec_col])[0]

    # blinkreconstruct replaces the blinks with NaN with mode='advanced',
    # so we interpolate the gaps and low pass the result to obtain something.
    # Detect and exclude datasets with artifacts
    #interp_100(config, eye)


#def reconstruct(config:ProcessConfig, eye, window_size=100):
 #   # Remove blinks.
   # plot_all = {
  #  99: True  # Specify the index of the data frame you want to plot
#}
    #col=config.column
    #eye[f'{col}_interp']=eye[col].interpolate(method='linear')
    #eye[f'{col}_rec']=blinkreconstruct(eye[f'{col}_interp'], 
                                                    #  vt_start=10/config.sfactor,vt_end=5/config.sfactor, maxdur=800, 
                                                     # mode='advanced')
    #eye[f'{col}_rec']= PLR2d.mask_pupil_first_derivative([eye], mask_cols=eye[f'{col}_rec'])[0]

    # blinkreconstruct replaces the blinks with NaN with mode='advanced',
    # so we interpolate the gaps and low pass the result to obtain something. 
    # Detect and exclude datasets with artifacts

def interp_100(config:ProcessConfig, eye, window_size=100):
    plot_all = {
    99: True  # Specify the index of the data frame you want to plot
}
    col=config.column
    eye[f'{col}_rec_interp']=eye[f'{col}_rec'].interpolate(method='linear')
    # Use moving average + recenter as low pass.
    eye[f'{col}_rec_interp_100']=eye[f'{col}_rec_interp'].rolling(window=window_size).mean().shift(-window_size//2)
    

def process(eyenum,column,subject_id,condition,timebase,data_path,progress):
    import preprocessingCopy1
    config=ProcessConfig()
    config.eyenum=eyenum
    config.column=column
    config.subject_id=subject_id
    config.condition=condition
    config.timebase=timebase
    config.data_path=data_path
    
    if config.column=="diameter_3d": 
        config.sfactor=50
        config.upper_threshold=12
    elif column=="diameter": 
        sfactor=1
        config.upper_threshold=200
    else: 
        raise ValueError("column")
    if config.timebase=="30":
         config.stime_start_offset=30
         config.after_var_start_offset=29
         config.window_duration=59
         config.diameter_threshold=30
    elif timebase=="3.4":
         config.stime_start_offset=3.4
         config.after_var_start_offset=25
         config.window_duration=29
         config.diameter_threshold=1
    else: 
        raise ValueError("timebase")
    return preprocessingCopy1.process2(config,progress)

    
def process2(config:ProcessConfig,progress):

    progress("Starting process2")
    subject_id = config.subject_id    
    stimulation_condition = config.condition
    
    measurement_path=f"{config.data_path}/{config.subject_id[:4]}/{config.subject_id}"
    
    # Load the CSV file
    df = pd.read_csv(f"{measurement_path}/exports/000/pupil_positions.csv", index_col=False)

    # Load the annotation timestamps
    annotation_timestamps = np.load(f"{measurement_path}/annotation_timestamps.npy")

    # Define the duration of the window after the annotations
    window_duration = config.window_duration

    # Define the columns to include in the CSV file
    csv_cols = ['pupil_timestamp', 'diameter_3d', 'diameter','eye_id','confidence']

    # Create an empty DataFrame to hold the sliced data
    sliced_data = pd.DataFrame(columns=csv_cols)

    progress('Loop through each annotation timestamp and slice the data')
    # Loop through each annotation timestamp and slice the data
    for annotation_timestamp in annotation_timestamps:
        # Calculate the start and end timestamps for the window after the annotation
        window_start = annotation_timestamp - 1.0
        window_end = window_start + window_duration

        # Select the rows that fall within the window
        df_sliced = df[(df['pupil_timestamp'] >= window_start) & (df['pupil_timestamp'] <= window_end)]

        # Add the sliced data to the DataFrame
        sliced_data = pd.concat([sliced_data, df_sliced[csv_cols]])

    # Save the sliced data to a CSV file separate for each eye
    sliced_data = sliced_data[sliced_data['eye_id'] == config.eyenum]

    # Filter the data to include only rows with confidence >= 0.6
    sliced_data_confidence = sliced_data[sliced_data['confidence'] >= 0.6]

    # Calculate the 0.02 quantile on the filtered data
    confidence_threshold = sliced_data_confidence['confidence'].quantile(0.02)


    # Print the result
    progress(f"The 0.02 quantile of pupil size for confidence >= 0.6 is {confidence_threshold:.2f}.")


    # Create an empty list to store the sliced dataframes and baselines
    df_list_eye_id = []

    progress('Loop through each annotation timestamp and create a list with the dataframes and a variable for each dataframe')
    for i, annotation_timestamp in enumerate(annotation_timestamps):
        # Calculate the start and end timestamps for the window after the annotation
        window_start = annotation_timestamp - 1
        window_end = window_start + window_duration

        # Select the rows that fall within the window
        df_sliced = sliced_data_confidence[(sliced_data_confidence['pupil_timestamp'] >= window_start) & (sliced_data_confidence['pupil_timestamp'] <= window_end)]
        # Append the sliced dataframe to the list
        df_list_eye_id.append(df_sliced)

    #this does not work for all subjects as the data quality is very different:
    # Calculate the lower threshold value for 0.1% exclusion
    #diameter_threshold = sliced_data_0['diameter'].quantile(0.02)
    #print(f"The lower diameter threshold value to exclude 1% of the data is {diameter_threshold:.2f}.")

    #preprocess the data and label
    # Load the annotation timestamps

    progress('preprocess and slice data')
    annotation_timestamps = np.load(f"{measurement_path}/annotation_timestamps.npy")
    df_list_eye_id_preprocessed = []
    for i, df in enumerate(df_list_eye_id):

        # Store the original unprocessed dataframe in a new variable
        df_preprocessed_eye_id_i = df.copy()
        # Call the reconstruct function to remove blinks, interpolate, and smooth the data
        reconstruct(config,df_preprocessed_eye_id_i) 
        #df_preprocessed_eye_id_i = PLR2d.mask_pupil_confidence([df_preprocessed_eye_id_i], threshold=confidence_threshold)[0]
        #df_preprocessed_eye_id_i = PLR2d.mask_pupil_first_derivative([df_preprocessed_eye_id_i])[0]
        #df_preprocessed_eye_id_i = PLR2d.mask_pupil_zscore([df_preprocessed_eye_id_i], threshold=2.0, mask_cols=[config.column])[0]
        
        interp_100(config,df_preprocessed_eye_id_i)
        
         # remove blinks, interpolate, smooth 
        df_preprocessed_eye_id_i[f"{config.column}_original"]=df_preprocessed_eye_id_i[f"{config.column}"]
        df_preprocessed_eye_id_i[f"{config.column}"]=df_preprocessed_eye_id_i[f"{config.column}_rec_interp_100"]
                
        # Calculate masked first derivative of the dataframe
        
      #  df_preprocessed_eye_id_i = PLR2d.remove_threshold([df_preprocessed_eye_id_i], lower_threshold=config.diameter_threshold, upper_threshold=config.upper_threshold, mask_cols=[config.column])[0]
      #  df_preprocessed_eye_id_i = PLR2d.iqr_threshold([df_preprocessed_eye_id_i], iqr_factor=4, mask_cols=[config.column])[0]
     #   df_preprocessed_eye_id_i = PLR2d.mask_pupil_zscore([df_preprocessed_eye_id_i], threshold=3.0, mask_cols=[config.column])[0]
      #  df_preprocessed_eye_id_i = PLR2d.mask_pupil_first_derivative([df_preprocessed_eye_id_i])[0]

        # Find the index of the closest annotation to the current dataframe
        annotation_index = np.abs(annotation_timestamps - df_preprocessed_eye_id_i.iloc[0]['pupil_timestamp']).argmin()

        # Define the baseline, stimulation, and after_var time ranges based on the annotation
        baseline_start = annotation_timestamps[annotation_index] - 1
        baseline_end = annotation_timestamps[annotation_index]
        stim_start = annotation_timestamps[annotation_index]
        stim_end = stim_start + config.stime_start_offset
        after_var_start = stim_end
        after_var_end = after_var_start + config.after_var_start_offset

        # Add a new column to the dataframe with the appropriate label based on the time ranges
        df_preprocessed_eye_id_i.loc[(df_preprocessed_eye_id_i['pupil_timestamp'] >= baseline_start) & (df_preprocessed_eye_id_i['pupil_timestamp'] <= baseline_end), 'label'] = 1
        df_preprocessed_eye_id_i.loc[(df_preprocessed_eye_id_i['pupil_timestamp'] >= stim_start) & (df_preprocessed_eye_id_i['pupil_timestamp'] <= stim_end), 'label'] = 2
        df_preprocessed_eye_id_i.loc[(df_preprocessed_eye_id_i['pupil_timestamp'] >= after_var_start) & (df_preprocessed_eye_id_i['pupil_timestamp'] <= after_var_end), 'label'] = 3

        # Add a new column to the dataframe containing its own index number
        df_preprocessed_eye_id_i['index'] = i

        # Append the preprocessed dataframe to the list
        df_list_eye_id_preprocessed.append(df_preprocessed_eye_id_i)

    counter = 0

    # Loop through each dataframe in the list and plot the data for eye 0
    for i, df in enumerate(df_list_eye_id):
        # Extract the timestamp and diameter data from the dataframe
        pupil_timestamp = df['pupil_timestamp']
        diameter = df['diameter']

        # Create a new figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the data before preprocessing in the left subplot
        ax1.plot(pupil_timestamp, diameter, marker=".", linestyle="")
        ax1.set_title(f"Eye Dataframe {i}")
        ax1.set_xlabel("Pupil Timestamp")
        ax1.set_ylabel("Diameter in mm/pixels")

        # Extract the preprocessed dataframe from the preprocessed list
        df_preprocessed_eye_id_i = df_list_eye_id_preprocessed[i]

        # Extract the timestamp and diameter data from the preprocessed dataframe
        pupil_timestamp_pre = df_preprocessed_eye_id_i['pupil_timestamp']
        diameter_pre = df_preprocessed_eye_id_i['diameter']

        # Plot the data after preprocessing in the right subplot
        label1 = df_preprocessed_eye_id_i[df_preprocessed_eye_id_i['label']==1]
        label2 = df_preprocessed_eye_id_i[df_preprocessed_eye_id_i['label']==2]
        label3 = df_preprocessed_eye_id_i[df_preprocessed_eye_id_i['label']==3]

        ax2.plot(label1['pupil_timestamp'], label1['diameter'], color='red', label='label 1')
        ax2.plot(label2['pupil_timestamp'], label2['diameter'], color='blue', label='label 2')
        ax2.plot(label3['pupil_timestamp'], label3['diameter'], color='green', label='label 3')


        # Plot the data after preprocessing in the right subplot
        ax2.set_title(f"Preprocessed Eye Dataframe {i}")
        ax2.set_xlabel("Pupil Timestamp")
        ax2.set_ylabel("Diameter in mm/pixels")

        # Show the plot
        counter += 1
        plt.show()


    #remove dataframes wheen a lot of data is deleted
   # if False: 
    #    progress('remove dataframes wheen a lot of data is deleted')
     #   removed_diameter_cols = []
      #  df_list_eye_id_preprocessed_1 = []
#
 #       for i, df in enumerate(df_list_eye_id_preprocessed):
  #          count_preprocessed_diameter = df[config.column].count()
   #         count_diameter = df_list_eye_id[i][config.column].count()
#
 #           diameter_diff = count_diameter - count_preprocessed_diameter
  #          diameter_pct_diff = count_preprocessed_diameter / count_diameter * 100
#
 #           if diameter_pct_diff < 50:
  #              df_copy = df.copy() # make a copy of the dataframe
   #             df_copy[config.column] = np.nan # replace diameter column with NaNs
    #            df_list_eye_id_preprocessed_1.append(df_copy) # append the preprocessed dataframe to the new list
     #           removed_diameter_cols.append(i)
      #      else:
       #         df_list_eye_id_preprocessed_1.append(df) # append the original dataframe to the new list
#
 #           progress(f"Removed 'diameter' column from DataFrame {i}")
  #          progress(f"\nDataFrame {i}:")
   #         progress(f"Difference in number of non-NaN values for diameter: {diameter_diff}")
    #        progress(f"Percentage difference in number of non-NaN values for diameter: {diameter_pct_diff:.2f}%")

        # Print summary message for removed 'diameter' columns
     #   if removed_diameter_cols:
      #      progress(f"\nThe 'diameter' column has been removed from the following DataFrames: {removed_diameter_cols}")
       # else:
        #    progress("\nNo 'diameter' columns were removed.")



    #When I look at the plots, I maybe want to exclude more dataframes, here I am asked.
    # Create a new list to store the filtered data frames

    removed_diameter_cols_indices = []

    removed_diameter_3d_cols_indices= []

    df_list_eye_id_preprocessed_filtered = df_list_eye_id_preprocessed.copy()

    #apply time_slots according to the pupil_timestamps. Each time_slot should have the same range of pupil_timestamps
    #important: label 1 marks the baseline, which should be the 5 seconds before a stimulation start
    #here it gets the time_slot 0
    


   # progress('apply timeslots')
    #for i in range(len(df_list_eye_id_preprocessed_filtered)):
     #   df = df_list_eye_id_preprocessed_filtered[i]

        # Assign time_slot 0 to the baseline data (label=1)
      #  df.loc[df['label']==1, 'time_slot'] = 0

        # Divide remaining data into 1000 time slots (label=2 or 3)
       # df.loc[df['label'].isin([2,3]), 'time_slot'] = pd.cut(df.loc[df['label'].isin([2,3]), 'pupil_timestamp'], bins=1000, labels=False) + 1

        #df_list_eye_id_preprocessed_filtered[i] = df
        
        
        
    for i in range(len(df_list_eye_id_preprocessed_filtered)):
        df = df_list_eye_id_preprocessed_filtered[i]

        # Assign time_slot 0 to the baseline data (label=1)
        df.loc[df['label'] == 1, 'time_slot'] = 0

        # Subset the DataFrame for label 2 or 3 data
        subset = df[df['label'].isin([2, 3])].copy()

        # Reset the index of the subset DataFrame
        subset.reset_index(drop=True, inplace=True)

        # Calculate the time range for label 2 and 3 data
        min_time = subset['pupil_timestamp'].min()
        max_time = subset['pupil_timestamp'].max()
        time_range = max_time - min_time

        # Divide the subset data into 1000 time slots
        subset['time_slot'], bins = pd.cut(subset['pupil_timestamp'], bins=1000, labels=False, retbins=True)

        # Adjust the time_slot values to start from 1
        subset['time_slot'] = subset['time_slot'] + 1

        # Assign the time_slot values back to the original DataFrame
        df.loc[df['label'].isin([2, 3]), 'time_slot'] = subset['time_slot'].values

        df_list_eye_id_preprocessed_filtered[i] = df




    progress('merge dataframes')
    # Concatenate all the data frames into a single data frame
    df_combined = pd.concat(df_list_eye_id_preprocessed_filtered)

    #
    # use the subject_id and stimulation_condition to create the file name
    output_path = config.data_path
    eye_id = f"eye_id{config.eyenum}"    
    list_file_name = f"{output_path}/{subject_id}_{stimulation_condition}_{eye_id}_{config.column}_list.csv"
    means_file_name = f"{output_path}/{subject_id}_{stimulation_condition}_{eye_id}_{config.column}_mean.csv"
    progress(f"save processed data to {list_file_name} and {means_file_name}")
    
    df_combined.to_csv(list_file_name, index=False)

    # Calculate the mean of the diameter and diameter_3d columns separately for each time slot
    mean_diameter = df_combined.groupby('time_slot')[config.column].mean()

    # Create a new data frame with the means and other columns
    df_means = pd.DataFrame({
        'time_slot': mean_diameter.index,
        'eye_id': [f"eye_id{config.eyenum}"] * len(mean_diameter),
        config.column: mean_diameter.values,
    })

    df_means.to_csv(means_file_name, index=False)

    # Print the data frame to the console
    progress(df_means)

    
