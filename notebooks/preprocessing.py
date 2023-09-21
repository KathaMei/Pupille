# preprocessing.py
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os.path
import shutil
from datetime import datetime
import math
sys.path.append("../Pupillengröße/Skripte/")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 08:58:14 2023

@author: Katharina
"""
from dataclasses import dataclass
from typing import List

@dataclass 
class ProcessConfig: 
    '''
    Data class to present attributes used in application.

    attributes
    ---------
        eyenum:int:                      Eye side: left (eye1) or right (eye0).
        column:str:                      Column of pupil size data: diameter or diameter_3d.
        sfactor:float:                   The factor the velocity (vt) should be divided with to detect blinks in the blinkreconstruct function.
        data_path:str:                   File path from which the data is taken.
        subject_id:str:                  The subject_id contains the subject number and the number of study cycles which is used to select the corresponding condition. 
        condition:str:                   Stimulation conditionen: 3.4Stim, 30Stim, 3.4Placebo, 30Placebo
        timebase:str:                    Time of stimulation: 30s or 3.4s.
        stime_time_offset:float:         Stimulation offset after: 30s or 3.4s.
        after_var_start_offset:float:    Time after stimulation: 26.6s (3.4Stim/Placebo) or 30s (30Stim/Placebo).
        window_duration:float:           The time of the whole dataset: stime_time_offset + after_var_start_offset.
        nan_reconstruct_threshold:float: When percentage of NaNs is above the threshold percentage after blinkreconstruct function, data is removed from further steps. Used for column 'diameter'.
        nan_before_threshold:float:      Threshold percentage of NaNs before using blinkreconstruct function. When percentage of NaNs is above threhold percentage, data is removed from further steps. High percentage hints a lot of missing data. Used for column 'diameter_3d'.
        nan_after_threshold:float:       When percentage of NaNs is above the threshold percentage after blinkreconstruct function, data is removed from further steps. Used for column 'diameter_3d'.
        noise_threshold_factor:float:    Threshold factor for MAD noise rejection.
        noise_rejection_percent:float:   A measurent is rejected if it contains more than this percent of NaNs after noise detection.
        validate_only:bool:              For data validation. Used to check which parameter values fit best to the data to remove artefacts.
        survive_threshold:int:           Minimum number of surviving frames required for a result set. If number is below threshold, all results for this study cycle are removed.
        baseline_length:float:           The length of the baseline periode the mean is calculated for.
    '''
    eyenum:int=0 #eye0 right, eye1 left
    column:str="unknown" #diameter or diameter_3d
    sfactor:float="" #the factor the velocity (vt) should be divided with to detect blinks in the blink_reconstruct
    data_path:str="" #path the data is taken
    subject_id:str="" #subject_id
    condition:str="" #1,2,3 or 4: 30Stim, 30Placebo, 3.4Sham, 3.4Placebo
    timebase:str="" #30s stimulation/placebo or 3.4s stimulation/placebo
    stime_time_offset:float=0 #stimulation duration: 3.4s or 30s
    after_var_start_offset:float=0 #time after stimulation: 26.6s or 30s
    window_duration:float=0  #the time of the whole dataset: stime_time_offset + after_var_start_offset
    nan_reconstruct_threshold:float=0 #threshold percentage of nans after reconstruct function for diameter, above threshold removed
    nan_before_threshold:float=0 #threshold percentage of nans before reconstruct function for diameter_3d, high percentage hint a lot of missing data, above threshold removed
    nan_after_threshold:float=0 #threshold percentage of nans after reconstruct function for dia,eter_3d,above threshold removed
    noise_threshold_factor:float="" # Threshold factor for MAD noise rejection
    noise_rejection_percent:float="" # A measurent is rejected if it contains more than this percent of NaN after noise detection.
    validate_only:bool=False # just validate the data
    survive_threshold:int=0 # minimum number of surviving frames required for a result set, in this case when less than 10% of dataframes left - removal
    baseline_length:float=2.0 
    
@dataclass
class ProcessFrame:
    '''
    Data class to present attributes used for preprocessed dataframes?.

    attributes
    ---------
        index:int:
        baseline_mean:float: The mean of pupil size in the baseline period before annotation_timestamp.
        baseline_std:float:  The standard deviation of pupil size in the baseline period before annotation_timestamp.
        annotation_ts:float: The timestamp when the stimulation starts.
        zscore:float:        Statistical measure that indicates how many standard deviations data is away from the mean. In this code it is used to calculate how many standard deviations the baseline means of all frames are away from the overall baseline mean of this study cycle.
        valid:bool:          Attribute that shows if the frame is still valid after running the code. 
        stage:str:           Processing steps of the code.
        remark:str:          Comments and information about the data.
        data:pd.DataFrame:   Dataframe used.
    '''
    index:int=None
    baseline_mean:float=None
    baseline_std:float=None
    annotation_ts:float=None
    zscore:float=None
    valid:bool=True
    stage:str=""
    remark:str=""
    data:pd.DataFrame=None
    
@dataclass
class ProcessResult:
    '''
    Data class to present attributes used for preprocessed dataframes?.

    attributes
    ---------
        config: ProcessConfig:     Dataclass ProcessConfig used.
        num_valid:int:             Number of valid dataframes.
        frames:list[ProcessFrame]: List of objects from type "ProcessFrame".
        percentage_valid:int:      Percentage of valid dataframes.
    '''
    config: ProcessConfig=None
    num_valid:int=0
    frames:list[ProcessFrame]=None
    percentage_valid:int=0



def save_pickle(filename,obj):
    '''
    Save the results as pickle file.

    parameter
    ---------
        filename: The name of the file.
        obj:      Object which is saved as pickle file.
    '''
    import pup_util
    return pup_util.save_pickle(filename,obj)
        
def load_pickle(filename):
    '''
    Load the pickle file.

    parameter
    ---------
        filename: The name of the file.
    '''
    import pup_util
    return pup_util.load_pickle(filename)
   
#Return condition for randomized condition code of subject
def get_condition(subject_id):
    '''
    Get the condition for the subject_id.

    parameter
    ---------
        subject_id: The subject_id contains the subject number and the number of study cycles which is used to select the corresponding condition. 
    '''
    import pup_util
    return pup_util.get_condition(subject_id)
    
# Median Absolute Deviation of series data.
def mad(col):
    '''
    The Median absolute deviaton is a statistical measure to quantify the variability of dispersion of data points in relation to the median. First the median of the frame for the data of the selected column is calculated. Then the absolute value of every data point is subtracted from the median. After that, the median of the results of substraction is calculated which is the median absolute deviation of the selected column. 

    parameter
    ---------
        col: Select the column of data points to run the mad.
    '''
    med=col.median()
    return (col-med).abs().median()

def nan_pct(series):
    '''
    The percentage of NaNs is calculated by counting the total number of NaN values and dividing it by the total number of indices.
    
    parameter
    ---------
        series: Data structure: list of data points with indices.
    '''
    total=len(series.index) # total number of items
    count=series.isna().sum() # number of NaN items
    r=100*count/total
    return r
    
# Compute a new column with data exceeding the threshold replaced by NaN, return the percentage of new NaN values in the new column.
def compute_and_reject_noise(df,thresf,col,col_out):
    '''
    The MAD for the difference between consecutive values in the column col is calculated. Setting a threshold by multiplying the median of the col_diff with the noise_threshold_factor from dataclass ProcessConfig. When the absolute difference of col_diff is above the threshold, the values in col_out turn to NaNs. The percentage change of NaN values in the new column (col_out) in comparison to the original column (col) is returned. A new column with data points exceeding the threshold and being replaced by NaNs is computed. 
    
    parameter
    ---------
        df:      Dataframe to which the function is applied.
        thresf:  Threshold factor for MAD noide rejection, attribute defined as noise_threshold_factor in dataclass ProcessConfig.
        col:     Selected column to which the function is applied.
        col_out: The computed column after applying the function.
    '''
    import math
    col_diff=df[col].diff()
    mad_do=mad(col_diff)
    threshold=col_diff.median()+thresf*mad_do
    df[col_out]=df[col]
    df.loc[col_diff.abs()>threshold,col_out]=math.nan
    #print(f"nan% of {col}={nan_pct(df[col])}")
    #print(f"nan% of {col_out}={nan_pct(df[col_out])}")
    nan_pct_grow=nan_pct(df[col_out])-nan_pct(df[col])
    return nan_pct_grow

    
# blinkreconstruct for a pandas series. Returns a numpy array.
# see https://pydatamatrix.eu/0.15/series/#function-blinkreconstructseries-vt5-vt_start10-vt_end5-maxdur500-margin10-smooth_winlen21-std_thr3-gap_margin20-gap_vt10-modeuoriginal
def blinkreconstruct(df, vt=5, vt_start=10, vt_end=5, maxdur=800, margin=20, smooth_winlen=21, std_thr=3, gap_margin=20, gap_vt=10, mode=u'advanced'):
    '''
    Blinkreconstruct for a pandas series. Converting the dataframe to a datamatrix series object. The function is invoked with several parameters. Returns suiteable data structure for functions to reconstruct blinks. The blinkreconstruct function is then applied to the prepared data object dm.
    see https://pydatamatrix.eu/0.15/series/#function-blinkreconstructseries-vt5-vt_start10-vt_end5-maxdur500-margin10-smooth_winlen21-std_thr3-gap_margin20-gap_vt10-modeuoriginal
    parameter
    ---------
        df:            Dataframe to which the function is applied.
        vt_start:      Pupil velocity threshold to detect the onset of a blink. 
        vt_end:        Pupil velocity threshold to detect the offset of a blink.
        maxdur:        Maximum duration for a blink.
        margin:        Margin around missing data that is reconstructed.
        smooth_winlen: The window length that is used to smooth the velocity profile.
        std_thr:       Threshold for standard deviation when the data is considered invalid.
        gap_margin:    Margin around missing data that is not reconstructed.
        gap_vt:        Pupil velocity threshold to detect invalid data.
        mode:          The algorithm used for blink reconstruction: original or advanced. Advanced is new and recommended.        
    '''
    import datamatrix
    import datamatrix.series
    import datamatrix.operations
    dm=datamatrix.convert.from_pandas(df).series
    return datamatrix.series.blinkreconstruct(dm,vt,vt_start,vt_end,maxdur,margin,smooth_winlen,std_thr,gap_margin,gap_vt,mode) #apply blinkreconstruct function from Mathot
        

def reconstruct(config: ProcessConfig, eye, col_in, col_out, window_size=100):
    '''
    Interpolates the data of the selected column of data object eye with method linear and after that applies the blinkreconstruct function to the prepared data object dm. The results are stored in col_out. Only the values of column diameter_3d are interpolated, as they contain less values than column diameter_2d.
    
    parameter
    ---------
        config: ProcessConfig: Use of the attributes defined in dataclass ProcessConfig.
        eye:                   Selected object to which the function is applied.
        col_in:                Selected column to which the function is applied.
        col_out:               Computed column after applying the function.
        window_size:           Number of data points used for interpolation.????
    '''
    try:
        interpolated=eye[col_in].interpolate(method='linear')
        eye[col_out] = blinkreconstruct(interpolated,
                                        vt_start=10 / config.sfactor, vt_end=5 / config.sfactor, maxdur=800,
                                        mode='advanced')
        return True
    except RecursionError as re: 
        return re

    
def interp_100(config:ProcessConfig, eye, col_in, col_interp, col_out, window_size=100):
    '''
    Creates a dictionary to specify the index of the dataframes which should be plotted. Interpolates the data for a selected column. Then, it calculates the rolling average over the window_size. Additionally, the computed rolling average is shifted by half of the window size forward in order to move the central point of the rolling average closer to the original data points. The smoothed data is returned in col_out.
    
    parameter
    ---------
        config: ProcessConfig: Use of the attributes defined in dataclass ProcessConfig.
        eye:                   Selected object to which the function is applied.
        col_in:                Selected column to which the function is applied.
        col_interp:            Interpolated data stored in this column.
        col_out:               Computed column after applying the function.
        window_size:           Number of data points used for rolling average.
    '''
    plot_all = {
    99: True  # Specify the index of the data frame you want to plot
}
    col=config.column
    eye[f'{col_interp}']=eye[f'{col_in}'].interpolate(method='linear')
    # Use moving average + recenter as low pass.
    eye[f'{col_out}']=eye[f'{col_interp}'].rolling(window=window_size).mean().shift(-window_size//2)
    

def create_baseline_column(df, col, newcol):
    '''
    For the data in the column col with label=1, the mean and standard deviation is calculated. Every data point in columm col is substracted from the calculated mean. The mean and standard deviation are returned.

    parameter
    ---------
        df:     Dataframe to which the function is applied.
        col:    Selected column to which the function is applied.
        newcol: Computed column after subtracting every data point from column col with the mean of data points with label 1.
    '''
    m=df.loc[df['label'] == 1, col].mean()
    s=df.loc[df['label'] == 1, col].std()
    df[newcol]=df[col]-m
    return (m,s)

        
def create_process_config(eyenum,column,subject_id,data_path):
    '''
    Creates a function with values for the attributes of the dataclass ProcessConfing. The values are selected based on column and timebase of the data.
    
    parameter
    ---------
        eyenum:        Eye side: left (eye1) or right (eye0).
        column:str:    Selected column to which the function is applied.
        subject_id:    The subject_id contains the subject number and the number of study cycles which is used to select the corresponding condition.
        data_path:str: File path from which the data is taken.
    '''
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
        config.sfactor=1000
        config.noise_threshold_factor=6
        config.noise_rejection_percent=2
        config.nan_before_threshold=60
        config.nan_after_threshold=5
    elif column=="diameter": 
        config.sfactor=1
        config.noise_threshold_factor=16
        config.noise_rejection_percent=20
        config.nan_reconstruct_threshold=30
    else: 
        raise ValueError("column")
    if config.timebase=="30":
         config.stime_start_offset=30
         config.after_var_start_offset=29
         config.window_duration=59
         config.survive_threshold=3
    elif timebase=="3.4":
         config.stime_start_offset=3.4
         config.after_var_start_offset=25.5
         config.window_duration=29
         config.survive_threshold=5
    else: 
        raise ValueError("timebase")
    return config

def process(config:ProcessConfig,progress):
    '''
    The dataframes are loaded. Various functions are applied in different stages. The following stages are defined:
    -stage="slice"
    -stage="label"
    -stage="preprocess"
    -stage="time_slot"
    -stage="zscore"
    -stage="Number of frames under threshold"
    -stage="finished"

    Variable description:
    -diameter_original = raw data
    -diameter_interp = interpolated data to remove NaNs in order to use the blink reconstruction
    -diameter_rec = data with removed blinks
    -diameter_rec_interp = interpolation after blink removal
    -diameter_rec_interp_100 = low pass filter, moving average over a window duration of 100, smoothing the data
    
    
    parameter
    ---------
        config:ProcessConfig: Use of the attributes defined in dataclass ProcessConfig.
        progress: Written feedback on the progress of the script.
    '''
    progress("Starting process2")

    result=ProcessResult()
    result.config=config
    result.frames=[]
    
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
    good_bad=[]

    progress('Loop through each annotation timestamp and slice the data')
    index=0
    # Loop through each annotation timestamp and slice the data
    for annotation_timestamp in annotation_timestamps:
        # Calculate the start and end timestamps for the window after the annotation
        window_start = annotation_timestamp - config.baseline_length
        window_end = window_start + window_duration

#class ProcessFrame:
#    annotation_pos:float:""
#    data:pd.DataFrame=None
#    valid:bool=False
        pf=ProcessFrame()
        pf.annotation_ts=annotation_timestamp
        pf.stage="slice"
        '''
        stage="slice"
        The data of the ProcessFrame was sliced into frames according to the annotation_timestamps and window_duration.
        '''
        pf.index=index
        index=index+1
        # Select the rows that fall within the window
        df_sliced = df[
            (df['pupil_timestamp'] >= window_start) 
            & (df['pupil_timestamp'] <= window_end) 
            & (df['eye_id'] == config.eyenum)] 
            #& (df['confidence']>=0.6)]
        df_sliced=df_sliced.copy()
        based_timestamps=df_sliced['pupil_timestamp']-(window_start+config.baseline_length)
        # pupil_timestamp_based -1..0 = baseline, 0=stimulation
        df_sliced['pupil_timestamp_based']=based_timestamps
        
        nan_percent=compute_and_reject_noise(df_sliced,config.noise_threshold_factor,f"{config.column}",f"{config.column}_gated")
        
        if (nan_percent > config.noise_rejection_percent): 
            pf.remark=f"measurement @{annotation_timestamp} has {nan_percent}% noise data. Rejecting"
            good_bad.append((subject_id,annotation_timestamp,nan_percent,False))
            pf.valid=False
            pf.data=None
        else:
            # df_sliced[f"{config.column}"]=df_sliced[f"{config.column}_gated"]
            good_bad.append((subject_id,annotation_timestamp,nan_percent,True))
            pf.valid=True
            pf.data=df_sliced
            
            
        result.frames.append(pf)
        
    if config.validate_only:
        return good_bad
        
    # ------------------------------------------------------------------------------------------------
    progress("Label the data")
    for pf in result.frames:
        if not(pf.valid): 
            continue
        df=pf.data
        pf.stage="label"
        '''
        stage="label"
        The baseline period for the data of the ProcessFrame was defined and the pupil_timestamps were calculated in relation to the baseline period, creating the pupil_timestamp_based column.
        The compute_and_reject_noise function was applied and the results stored in a new column in the frames called f"{config.column}_gated". If the nan_percent of the compute_and_reject_noise factor was above the config.noise_rejection_percent threshold, the dataframe was removed.
        '''
        
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
    for pf in result.frames:
        # Store the original unprocessed dataframe in a new variable
        if not(pf.valid): 
            continue
            
        pf.stage="preprocess"
        '''
        stage="preprocess"
        The data of the ProcessFrame has been labeled according to the pupil_timestamp_based and the config.stime_start_offset + config.after_var_start_offset parameters. Pupil_timestamp_based below 0 received label = 1 (before stimulation onset). Pupil_timestamp_based equal or above 0 and below config.stime_start_offset received label = 2 (during stimulation). Pupil_timestamp_based equal or above config.stime_start_offset and below config.stime_start_offset+config.after_var_start_offset) received label = 3 (after stimulation).
        '''
        
        df=pf.data.copy()
        
        # Call the reconstruct function to remove blinks, interpolate, and smooth the data
        status=reconstruct(config,df,f"{config.column}",f"{config.column}_rec")
        if isinstance(status,RecursionError): 
            pf.remark="Recursion error in blinkreconstruct."
            pf.stage="Recursion error in blinkreconstruct."
            pf.valid=False
            continue
        
         # Define the range for the column
        if config.column=="diameter":
            column_range = (40, 150)#min_value and max_value
        elif  config.column=="diameter_3d":
            column_range = (1.5, 9)
        else:
        # Replace values outside the range with NaN
            df.loc[~df[f"{config.column}_rec"].between(*column_range), f"{config.column}_rec"] = np.nan

        nanp_before=nan_pct(df[f"{config.column}"])
        nanp_after=nan_pct(df[f"{config.column}_rec"])
        progress(f"nanp before={nanp_before}, nanp after={nanp_after}")
        if config.column=="diameter" and (nanp_after-nanp_before)>config.nan_reconstruct_threshold:
            pf.remark=f"measurement @{pf.annotation_ts} has {(nanp_after-nanp_before)}% more noise data after blinkreconstruct. Rejecting" 
            pf.valid=False            
        elif config.column=="diameter_3d" and (nanp_before>config.nan_before_threshold or nanp_after>config.nan_after_threshold):
            pf.remark=f"measurement @{pf.annotation_ts} has {nanp_before} nan_before and {nanp_after} nan_after after blinkreconstruct. Rejecting"
            pf.valid=False         
        else:
            # remove blinks, interpolate, smooth 
            interp_100(config,df, f'{config.column}_rec',f'{config.column}_rec_interp',f'{config.column}_rec_interp_100')  

            df[f"{config.column}_original"]=df[f"{config.column}"]
            df[f"{config.column}"]=df[f"{config.column}_rec_interp_100"]

            # Create a baseline column for config.column.
            (pf.baseline_mean,pf.baseline_std)=create_baseline_column(df, f'{config.column}', f'{config.column}_baseline')
            if math.isnan(pf.baseline_mean):
                pf.valid=False
                pf.remark="baseline is nan. Check length of df.loc[df['label'] == 1]"
        pf.data=df
        
    # ------------------------------------------------------------------------------------------------

# Store the original unprocessed dataframe in a new variable
    for pf in result.frames:
        if not(pf.valid): 
            continue            
        df=pf.data.copy()
        pf.stage="time_slot"
        '''
        stage="time_slot"
        The reconstruct function was applied to the original data of the ProcessFrame. Ranges for the 
different columns were defined. If data points were above or below the ranges, they turned to NaNs. The results of both steps were stored in a new column in the frames called f"{config.column}_rec". With different values for the columns, the NaN percentage before and after the reconstruct function and data range definition was calculated. If the difference of percentage after the calculations was above the config.nan_reconstruct_threshold, the frame was removed.
The interp_100 function was applied to the data of the remaining frames. The results of the first step of the function, to interpolate the data, was stored in a new column in the frames called f'{config.column}_rec_interp. The results of the further code elements of the function, to calculate the rollowing average and shifting it by half of the window size forward were stored in a new column in the frames called f'{config.column}_rec_interp_100. Furthermore, the create_baseline_column function was applied to calculate the mean and standard deviation of the baseline periode. If the baseline_mean was only contained NaN values, the frame was removed.
        '''
        
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

        pf.data=df

    # compute and assign zscore for baseline_mean. mark frame as invalid if is falls outside the accepted range.
    bv=[f.baseline_mean for f in result.frames if f.valid]
    import scipy.stats as stats
    scores=stats.zscore(bv)
    for (frame,z) in zip([f for f in result.frames if f.valid],scores):
        frame.stage="zscore"
        '''
        stage="zscore"
        The column time_slot was assigned to the baseline data, the indeces were adapted and the time range for the data of label 2 and 3 were calculated. The data with label 2 or 3 was divided into 1000 time_slots. Zscore as statistical measure was used to calculate the number of standard deviations the baseline_means of every frame deviate from the average baseline mean. If the zscore of the baseline_mean from one repliacte deviated more than -2.5 or above 2.5 from the average baseline mean, it was removed. 
        '''    
        frame.zscore=z
        if (z<-2.5 or z>2.5):
            frame.valid=False
            frame.remark="zscore not in range -2.5 to 2.5"
    result.num_valid=sum([1 for f in result.frames if f.valid])           

            
    # ------------------------------------------------------------------------------------------------
  
    # Check if the number of valid frames is below the threshold
    result.num_valid = sum([1 for f in result.frames if f.valid])
    if result.num_valid < config.survive_threshold:
        for frame in result.frames:
            if frame.valid:
                frame.valid = False
                frame.stage = "Number of frames under threshold"
                '''
                stage="Number of frames under threshold"
                The sum of valid frames was calculated. If the sum was below the config.survive_threshold, all other frames of the subject_id were removed. 
                '''
                frame.remark = "num_valid < threshold"

    # ------------------------------------------------------------------------------------------------


    for pf in result.frames:
        if not(pf.valid): 
            continue            
        pf.stage="finished"
        '''
        stage="finished"
        The valid frames reached the final stage and was used for further statistical analyses.
        '''

    return result



    for pf in result.frames:

        '''
        Loop to combine, save the files, calculate the mean for the diameter and diameter_3d columns for each time slot and save files with the mean values.
        '''
        
        progress('merge dataframes')

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
# ------------------------------------------------------------------------------------------------

def average_frames_by_binning(pr:ProcessResult, field:str, interval_ms=10)->pd.DataFrame:
    '''
    Function to average the values of the remaining valid frames. First the valid frames are concatenated into one dataframe and the baseline values are removed. Then a new timestamp column is created. It contains quantized timestamps based on the provided 'interval_ms'. The quantizied timestamps are calculated by rounding the values in the 'pupil_timestamp_based' column to the nearest multiple of the 'interval_ms' and converting it back to seconds. The data is grouped by the new timestamp column and the mean of values in the field column is calculated. After the grouping, the index of the resulting df is resetted. It returns the dataframe av_df with averaged data based on the quantized timestamps or `None` if pr.frame does not contain any valid dataframes.
    
    parameter
    ---------
        pr:ProcessResult: Data class to present attributes used for preprocessed dataframes.
        field:str:        Diameter or diameter_3d.
        interval_ms:      The time interval the data is grouped by.
    '''
    # all the valid frames
    valids=[f.data for f in pr.frames if f.valid]
    if len(valids)==0:
        return None
    df=pd.concat(valids)
    # remove baseline values
    df=df.loc[df.label!=1,['pupil_timestamp_based',field,'label']]
    df=df.copy()
    # a timestamp column which is quantized by interval_ms
    df['ts']=(interval_ms/1000.0)*np.round(df.pupil_timestamp_based*1000/interval_ms)
    # df['ts']=np.round(df.pupil_timestamp_based,decimals=1)
    # average over all data with the same timestamp
    av_df = df.groupby('ts')[field].mean().reset_index()
    return av_df



'''
Unused functions with different method to average data points.
'''

def average_frames_by_resample(pr:ProcessResult, field:str, interval="10ms")->pd.DataFrame:
    '''
    Function to average the values of the remaining valid frames. With a loop, the valid frames are stored in a dataframe and the baseline values are removed. A new timestamp column is created which converts values from the pupil_timestamp_based column to datetime values. The new timestamp column is set as new index for the dataframe. After that, the dataframe is resampeled based on the interval. For each interval, the mean is calculated. To fikk in any missing values between the reseampled intervales, interpolation is used. All frames are concatenated into one dataframe.
The data is grouped by the new timestamp column and the means of values in the field column are calculated. After the grouping, the index of the resulting df is resetted. It returns the dataframe av_df with averaged data based on the new timestamp column.

    parameter
    ---------
        pr:ProcessResult: Data class to present attributes used for preprocessed dataframes.
        field:str:        Diameter or diameter_3d.
        interval_ms:      The time interval the data is grouped by.
    '''
    # collect resampled frames here
    ret=[]
    for f in pr.frames:
        if f.valid:
            df=f.data
            # remove baseline values pre copy
            df=df.loc[df.label!=1,['pupil_timestamp_based',field,'label']].copy()
            # create a timestamp column with the right data time
            df['ts']=pd.to_datetime(df['pupil_timestamp_based'], unit='s')
            # resample data according to interval
            df.set_index('ts', inplace=True) 
            df=df.resample(interval).mean().interpolate()
            # print(df)
            # df=df.resample(interval).count() #.interpolate(method='polynomial', order=3)
            ret.append(df)        
    # now average all rows with the same timestamp
    ret=pd.concat(ret)
    # av_df = ret.groupby('ts')[field].sum() # mean().reset_index()
    av_df = ret.groupby('ts')[field].mean().reset_index()
    return av_df



