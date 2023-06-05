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
    noise_threshold_factor:float=12 # Threshold factor for MAD noise rejection
    noise_rejection_percent:float=5 # A measurent is rejected if it contains more than this percent of NaN after noise detection.
    validate_only:bool=False # just validate the data

# Return condition for randomized condition code of subject
def get_condition(subject_id):
    f=pd.read_csv('zuordnungen.csv',index_col='proband')
    prob=subject_id[:4]
    msr=subject_id[5:6]    
    q=f.loc[prob][int(msr)-1]
    names=[("3.4","3.4Stim"),("3.4","3.4Placebo"),("30","30Stim"),("30","30Placebo")]
    return names[int(q)-1]
    
# Median Absolute Deviation of series data.
def mad(col):
    med=col.median()
    return (col-med).abs().median()

def nan_pct(series):
    total=len(series.index) # total number of items
    count=series.isna().sum() # number of NaN items
    r=100*count/total
    return r
    
# Compute a new column with data exceeding the threshold replaced by NaN, return the percentage of new NaN values in the new column.
def compute_and_reject_noise(df,thresf,col,col_out):
    import math
    col_diff=df[col].diff()
    mad_do=mad(col_diff)
    threshold=col_diff.median()+thresf*mad_do
    df[col_out]=df[col]
    df.loc[col_diff.abs()>threshold,col_out]=math.nan
    print(f"nan% of {col}={nan_pct(df[col])}")
    print(f"nan% of {col_out}={nan_pct(df[col_out])}")
    nan_pct_grow=nan_pct(df[col_out])-nan_pct(df[col])
    return nan_pct_grow

# blinkreconstruct for a pandas series. Returns a numpy array.
# see https://pydatamatrix.eu/0.15/series/#function-blinkreconstructseries-vt5-vt_start10-vt_end5-maxdur500-margin10-smooth_winlen21-std_thr3-gap_margin20-gap_vt10-modeuoriginal
def blinkreconstruct(df, vt=5, vt_start=10, vt_end=5, maxdur=500, margin=10, smooth_winlen=21, std_thr=3, gap_margin=20, gap_vt=10, mode=u'advanced'):
    import datamatrix
    import datamatrix.series
    import datamatrix.operations
    dm=datamatrix.convert.from_pandas(df).series
    return datamatrix.series.blinkreconstruct(dm, vt,vt_start,vt_end,maxdur,margin,smooth_winlen,std_thr,gap_margin,gap_vt,mode)

def reconstruct(config: ProcessConfig, eye, col_in, col_out, window_size=100):
    interpolated=eye[col_in].interpolate(method='linear')
    eye[col_out] = blinkreconstruct(interpolated,
                                    vt_start=10 / config.sfactor, vt_end=5 / config.sfactor, maxdur=800,
                                    mode='advanced')

    
def interp_100(config:ProcessConfig, eye, col_in, col_interp, col_out, window_size=100):
    plot_all = {
    99: True  # Specify the index of the data frame you want to plot
}
    col=config.column
    eye[f'{col_interp}']=eye[f'{col_in}'].interpolate(method='linear')
    # Use moving average + recenter as low pass.
    eye[f'{col_out}']=eye[f'{col_interp}'].rolling(window=window_size).mean().shift(-window_size//2)
    

def create_baseline_column(df, col, newcol):
    m=df.loc[df['label'] == 1, col].mean()
    df[newcol]=df[col]-m


def create_process_config(eyenum,column,subject_id,data_path):
    import preprocessing
    config=ProcessConfig()
    config.eyenum=eyenum
    config.column=column
    config.subject_id=subject_id
    (timebase,cond)=get_condition(subject_id)
    config.condition=cond
    config.timebase=timebase
    config.data_path=data_path
    
    if config.column=="diameter_3d": 
        config.sfactor=50
        config.upper_threshold=12
    elif column=="diameter": 
        config.sfactor=1
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
    return config

def process(config:ProcessConfig,progress):
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

    # ------------------------------------------------------------------------------------------------
    # Create an empty DataFrame to hold the sliced data
    df_list_eye_id = []
    good_bad=[]
    progress('Loop through each annotation timestamp and slice the data')
    # Loop through each annotation timestamp and slice the data
    for annotation_timestamp in annotation_timestamps:
        # Calculate the start and end timestamps for the window after the annotation
        window_start = annotation_timestamp - 1.0
        window_end = window_start + window_duration

        # Select the rows that fall within the window
        df_sliced = df[
            (df['pupil_timestamp'] >= window_start) 
            & (df['pupil_timestamp'] <= window_end) 
            & (df['eye_id'] == config.eyenum) 
            & (df['confidence']>=0.6)]            
        df_sliced=df_sliced.copy()
        based_timestamps=df_sliced['pupil_timestamp']-(window_start+1)
        # pupil_timestamp_based -1..0 = baseline, 0=stimulation
        df_sliced['pupil_timestamp_based']=based_timestamps
        
        nan_percent=compute_and_reject_noise(df_sliced,config.noise_threshold_factor,f"{config.column}",f"{config.column}_gated")
        if (nan_percent > config.noise_rejection_percent): 
            progress(f"measurement @{annotation_timestamp} has {nan_percent}% noise data. Rejecting")
            good_bad.append((subject_id,annotation_timestamp,nan_percent,False))
        else:
            df_list_eye_id.append(df_sliced)
            good_bad.append((subject_id,annotation_timestamp,nan_percent,True))
    
    if config.validate_only:
        return good_bad
    
    if not df_list_eye_id:
        return []
        
    # ------------------------------------------------------------------------------------------------
    progress("Label the data")
    for df in df_list_eye_id:
        # df[0..1] = label 1
        # df[1..config.stime_start_offset] = Label 2
        # df[rest]=Label 3 
#         config.stime_start_offset=3.4
#         config.after_var_start_offset=25

        df['label']=0
        df.loc[(df['pupil_timestamp_based'] < 0), 'label'] = 1
        df.loc[
            (df['pupil_timestamp_based'] >= 0) 
            & (df['pupil_timestamp_based']<config.stime_start_offset), 
            'label'] = 2
        df.loc[
            (df['pupil_timestamp_based'] >= config.stime_start_offset) 
            & (df['pupil_timestamp_based']<config.stime_start_offset+config.after_var_start_offset), 
            'label'] = 3        
        
    # ------------------------------------------------------------------------------------------------
    progress('preprocess and slice data')
    df_list_eye_id_preprocessed = []
    for i, df in enumerate(df_list_eye_id):

        # Store the original unprocessed dataframe in a new variable
        df_preprocessed_eye_id_i = df.copy()
        # Call the reconstruct function to remove blinks, interpolate, and smooth the data
        reconstruct(config,df_preprocessed_eye_id_i,f"{config.column}",f"{config.column}_rec") 

        nanp_before=nan_pct(df_preprocessed_eye_id_i[f"{config.column}_gated"])
        nanp_after=nan_pct(df_preprocessed_eye_id_i[f"{config.column}_rec"])
        progress(f"nanp before={nanp_before}, nanp after={nanp_after}")
        
        if (nanp_after-nanp_before)>10:
            progress(f"measurement @{annotation_timestamp} has {(nanp_after-nanp_before)}% more noise data after blinkreconstruct. Rejecting")            
        else:
            #df_preprocessed_eye_id_i = PLR2d.mask_pupil_confidence([df_preprocessed_eye_id_i], threshold=confidence_threshold)[0]
            #df_preprocessed_eye_id_i = PLR2d.mask_pupil_first_derivative([df_preprocessed_eye_id_i])[0]
            #df_preprocessed_eye_id_i = PLR2d.mask_pupil_zscore([df_preprocessed_eye_id_i], threshold=2.0, mask_cols=[config.column])[0]

             # remove blinks, interpolate, smooth 
            interp_100(config,df_preprocessed_eye_id_i, f'{config.column}_rec',f'{config.column}_rec_interp',f'{config.column}_rec_interp_100')  

            df_preprocessed_eye_id_i[f"{config.column}_original"]=df_preprocessed_eye_id_i[f"{config.column}"]
            df_preprocessed_eye_id_i[f"{config.column}"]=df_preprocessed_eye_id_i[f"{config.column}_rec_interp_100"]

            # Calculate masked first derivative of the dataframe

          #  df_preprocessed_eye_id_i = PLR2d.remove_threshold([df_preprocessed_eye_id_i], lower_threshold=config.diameter_threshold, upper_threshold=config.upper_threshold, mask_cols=[config.column])[0]
          #  df_preprocessed_eye_id_i = PLR2d.iqr_threshold([df_preprocessed_eye_id_i], iqr_factor=4, mask_cols=[config.column])[0]
         #   df_preprocessed_eye_id_i = PLR2d.mask_pupil_zscore([df_preprocessed_eye_id_i], threshold=3.0, mask_cols=[config.column])[0]
          #  df_preprocessed_eye_id_i = PLR2d.mask_pupil_first_derivative([df_preprocessed_eye_id_i])[0]

            # Add a new column to the dataframe containing its own index number
            df_preprocessed_eye_id_i['index'] = i
            # Create a baseline column for config.column.
            create_baseline_column(df_preprocessed_eye_id_i, config.column, f'{config.column}_baseline')
            # Append the preprocessed dataframe to the list                    
            df_list_eye_id_preprocessed.append(df_preprocessed_eye_id_i)

        
    # ------------------------------------------------------------------------------------------------
    df_list_eye_id_preprocessed_filtered = df_list_eye_id_preprocessed.copy()        
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
    list_file_name = f"{output_path}/{subject_id}-{stimulation_condition}-{eye_id}-{config.column}_list.csv"
    means_file_name = f"{output_path}/{subject_id}-{stimulation_condition}-{eye_id}-{config.column}_mean.csv"
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

    return df_list_eye_id_preprocessed_filtered
