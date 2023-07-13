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
    nan_reconstruct_threshold: float=0 #threshold percentage of nans after reconstruct function for diameter, above threshold removed
    nan_before_threshold:float=0 #threshold percentage of nans before reconstruct function for diameter_3d, high percentage hint a lot of missing data, above threshold removed
    nan_after_threshold:float=0 #threshold percentage of nans after reconstruct function for dia,eter_3d,above threshold removed
    noise_threshold_factor:float="" # Threshold factor for MAD noise rejection
    noise_rejection_percent:float="" # A measurent is rejected if it contains more than this percent of NaN after noise detection.
    validate_only:bool=False # just validate the data
    survive_threshold:int=0 # minimum number of surviving frames required for a result set, in this case when less than 10% of dataframes left - removal
    baseline_length:float=2.0 
    
@dataclass
class ProcessFrame:
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
    config: ProcessConfig=None
    num_valid:int=0
    frames:list[ProcessFrame]=None
    percentage_valid:int=0

def save_pickle(filename,obj):
    import pickle
    with open(filename,"wb") as h: 
        pickle.dump(obj,h,protocol=5)
        
def load_pickle(filename):
    import pickle
    with open(filename,"rb") as h:
        return pickle.load(h)
   

#Return condition for randomized condition code of subject
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
    #print(f"nan% of {col}={nan_pct(df[col])}")
    #print(f"nan% of {col_out}={nan_pct(df[col_out])}")
    nan_pct_grow=nan_pct(df[col_out])-nan_pct(df[col])
    return nan_pct_grow


# blinkreconstruct for a pandas series. Returns a numpy array.
# see https://pydatamatrix.eu/0.15/series/#function-blinkreconstructseries-vt5-vt_start10-vt_end5-maxdur500-margin10-smooth_winlen21-std_thr3-gap_margin20-gap_vt10-modeuoriginal
def blinkreconstruct(df, vt=5, vt_start=10, vt_end=5, maxdur=800, margin=20, smooth_winlen=21, std_thr=3, gap_margin=20, gap_vt=10, mode=u'advanced'):
    import datamatrix
    import datamatrix.series
    import datamatrix.operations
    dm=datamatrix.convert.from_pandas(df).series
    return datamatrix.series.blinkreconstruct(dm,vt,vt_start,vt_end,maxdur,margin,smooth_winlen,std_thr,gap_margin,gap_vt,mode)

def reconstruct(config: ProcessConfig, eye, col_in, col_out, window_size=100):
    interpolated=eye[col_in].interpolate(method='cubic')
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
    s=df.loc[df['label'] == 1, col].std()
    df[newcol]=df[col]-m
    return (m,s)

        
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
        
        df=pf.data.copy()
        
        # Call the reconstruct function to remove blinks, interpolate, and smooth the data
        reconstruct(config,df,f"{config.column}",f"{config.column}_rec")
        
         # Define the range for the column
        if config.column=="diameter":
            column_range = (40, 150)# Replace min_value and max_value with your desired range
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
        
        
        #diameter_original = Rohdaten
        #diameter_interp = Interpolarisation, um fehlende Werte zu verhindern, da sonst Funktion nicht ausführbar ist
        #diameter_rec = entfernte Blinzler
        #diameter_rec_interp = Interpolarisation nach Entfernung der Blinzler
        #diameter_rec_interp_100 = low pass filter, moving average über 100 Werte zum Smoothem
        
    # ------------------------------------------------------------------------------------------------

# Store the original unprocessed dataframe in a new variable
    for pf in result.frames:
        if not(pf.valid): 
            continue            
        df=pf.data.copy()
        pf.stage="time_slot"
        
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
                frame.remark = "num_valid < threshold"

    # ------------------------------------------------------------------------------------------------


    for pf in result.frames:
        if not(pf.valid): 
            continue            
        pf.stage="finished"

    return result



    for pf in result.frames:

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

def average_frames_by_resample(pr:ProcessResult, field:str, interval="10ms")->pd.DataFrame:
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

def average_frames_by_binning(pr:ProcessResult, field:str, interval_ms=10)->pd.DataFrame:
    # all the valid frames
    valids=[f.data for f in pr.frames if f.valid]
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

