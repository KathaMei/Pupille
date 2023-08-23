#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 14:43:40 2023

@author: Katharina
"""
##############unused

import numpy as np
from pyplr import utils
import pandas as pd
from classPLRfromGitHub import PLR
from pyplr import graphing
from pyplr import preproc
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class ProcessConfigPLR: 
    eyenum:int=0 #eye0 right, eye1 left
    column:str="unknown" #diameter or diameter_3d
    sfactor:float=1 #the factor the velocity (vt) should be divided with to detect blinks in the blink_reconstruct
    rec_dir:str="" #path the data is taken
    subject_id:str="" #subject_id
    condition:str="" #1,2,3 or 4: 30Stim, 30Placebo, 3.4Sham, 3.4Placebo
    timebase:str="" #30s stimulation/placebo or 3.4s stimulation/placebo
    start_time_offset:float=0 #stimulation duration: 3.4s or 30s
    after_var_start_offset:float=0 #time after stimulation: 26.6s or 30s
    window_duration:float=0  #the time of the whole dataset: start_time_offset + after_var_start_offset
    upper_threshold:float=0 #upper diameter threshold, different for diameter (pixel) and diameter_3d (mm)
    diameter_threshold:float=0 #lower diameter threshold, different for diameter (pixel) and diameter_3d (mm)

@dataclass
class ProcessFrame:
    annotation_ts:float=None
    valid:bool=True
    stage:str=""
    remark:str=""
    data:pd.DataFrame=None
    
@dataclass
class ProcessResult:
    config: ProcessConfigPLR=None
    frames:list[ProcessFrame]=None
    
    
# blinkreconstruct for a pandas series. Returns a numpy array.
# see https://pydatamatrix.eu/0.15/series/#function-blinkreconstructseries-vt5-vt_start10-vt_end5-maxdur500-margin10-smooth_winlen21-std_thr3-gap_margin20-gap_vt10-modeuoriginal
def blinkreconstruct(df, vt=5, vt_start=10, vt_end=5, maxdur=500, margin=10, smooth_winlen=21, std_thr=3, gap_margin=20, gap_vt=10, mode=u'advanced'):
    display(type(df))
    import datamatrix
    import datamatrix.series
    import datamatrix.operations
    dm=datamatrix.convert.from_pandas(df).series
    return datamatrix.series.blinkreconstruct(dm, vt,vt_start,vt_end,maxdur,margin,smooth_winlen,std_thr,gap_margin,gap_vt,mode)

def reconstruct(config:ProcessConfigPLR, eye, window_size=100):
    # Remove blinks.
    col=config.column
    eye[f'{col}_interp']=eye[col].interpolate(method='linear')
    eye[f'{col}_rec']=blinkreconstruct(eye[f'{col}_interp'], 
                                                      vt_start=10/config.sfactor,vt_end=5/config.sfactor,
                                                      mode='advanced')
    # blinkreconstruct replaces the bliks with NaN with mode='advanced',
    # so we interpolate the gaps and low pass the result to obtain something. 
    eye[f'{col}_rec_interp']=eye[f'{col}_rec'].interpolate(method='linear')
    # Use moving average + recenter as low pass.
    eye[f'{col}_rec_interp_100']=eye[f'{col}_rec_interp'].rolling(window=window_size).mean().shift(-window_size//2)

    
def progress(eyenum,column,subject_id,condition,timebase,rec_dir,progress):
    import preprocessing
    config=ProcessConfigPLR()
    config.eyenum=eyenum
    config.column=column
    config.subject_id=subject_id
    config.condition=condition
    config.timebase=timebase
    config.rec_dir=rec_dir
    
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
         config.after_var_start_offset=25
         config.window_duration=55
         config.diameter_threshold=40
    elif timebase=="3.4":
         config.stime_start_offset=25
         config.after_var_start_offset=21.6
         config.window_duration=30
         config.diameter_threshold=1
    else: 
        raise ValueError("timebase")
    return preprocessing.process2(config,progress)


def process(config:ProcessConfigPLR,progress):
    progress("Starting process2")

    result=ProcessResult()
    result.config=config
    result.frames=[]
    
    subject_id = config.subject_id    
    stimulation_condition = config.condition
    
    measurement_path=f"{config.data_path}/{config.subject_id[:4]}/{config.subject_id}"
   

# Load the data from the pupil player
# Pupil Labs recording directory

# define the patient ID for the dataframe and assign the 4 light strenghts

Light_strenght_1 = 1
Light_strenght_2 = 2
Light_strenght_3 = 3
Light_strenght_4 = 4


# Columns to load
use_cols = ['confidence',
            'method',
            'pupil_timestamp',
            'eye_id',
            'diameter_3d',
            'diameter']

rec_dir = '/Users/Katharina/Desktop/Beispieldaten/PJ04/PJ04_A_PLR2/'
# Get a handle on a subject
s = utils.new_subject(
    rec_dir, export='000')

# Load pupil data, method has to be changed to '3d' otherwise multiple repeat error
#eye_id=best takes the eye with the best confidence, eye_id=0 takes right, 1 takes left
samples = utils.load_pupil(
    rec_dir, eye_id='best', method='3d', cols=use_cols)
samples

#get rid of the blink artefacts and plot the pupil diameter over time 


# Sampling frequency
SAMPLE_RATE = 120

# Pupil columns to analyse - 3d is in mm and diameter is in pixel 
pupil_cols = ['diameter_3d', 'diameter']


# Make figure for processing, append figure to pupil_preprocessing
f, axs = graphing.pupil_preprocessing_figure(nrows=5, subject='Example')

samples[pupil_cols].plot(title='Raw', ax=axs[0], legend=True)


# Plot the raw data
#samples[pupil_cols].plot(title='Raw', ax=axs[0], legend=True)
#axs[0].legend(loc='center right', labels=['mm', 'pixels'])

# Mask first derivative
#Default is a threshold of 3 SD from the mean first derivate
# If there are a lot of blinks the mean first derivate is higher
# Therefore set the threshold lower 
#samples = preproc.mask_pupil_first_derivative(
#    samples, threshold=3.0, mask_cols=pupil_cols)
#samples[pupil_cols].plot(
#    title='Masked 1st deriv (3*SD)', ax=axs[1], legend=False)

# Mask confidence
samples = preproc.mask_pupil_confidence(
    samples, threshold=0.8, mask_cols=pupil_cols)
samples[pupil_cols].plot(
    title='Masked confidence (<0.8)', ax=axs[2], legend=False)
 

# Apply z-score filter to diameter_3d column
#diameter_zscore = (samples['diameter'] - samples['diameter'].mean()) / samples['diameter'].std()
#zscore_threshold = 3
#samples = samples[abs(diameter_zscore) <= zscore_threshold]


# Interpolate
#samples = preproc.interpolate_pupil(
#    samples, interp_cols=pupil_cols)
#samples[pupil_cols].plot(
#    title='Linear interpolation', ax=axs[3], legend=False)

# Smooth
#samples = preproc.butterworth_series(
#    samples, fields=pupil_cols, filt_order=3,
#    cutoff_freq=4/(SAMPLE_RATE/2))
#samples[pupil_cols].plot(
#    title='3rd order Butterworth filter with 4 Hz cut-off',
#    ax=axs[4], legend=False);

events = utils.load_annotations(s['data_dir'])
events

# Number of samples to extract and which sample
# should mark the onset of the event
# 120 data points equal one second with a sample rate of 120 
# Real duration is Duration-ONSET_IDX, example 240 equals 1 second
#ONSET_IDX is the time before light stimulus that gets sampled 
DURATION = 1600
ONSET_IDX = 120



# Extract the event ranges, gets range of total of Duration (720)
#and shifts it Onset_idx to the left in time, therefore 120 is the 0 point
ranges = utils.extract(
    samples,
    events,
    offset=-ONSET_IDX,
    duration=DURATION,
    borrow_attributes=['color'])
ranges

# Calculate baselines, the time before the onset is 0 to 120  
baselines = ranges.loc[:, range(0, ONSET_IDX), :].mean(level=0)

# New columns for percent signal change, simply adds 2 new columns
# with percent change 
ranges = preproc.percent_signal_change(
    ranges, baselines, pupil_cols)
ranges

ranges.index
print(ranges.loc[0:3])
 
ranges1 = ranges.loc[0]
ranges2 = ranges.loc[1]  
ranges3 = ranges.loc[2]
ranges4 = ranges.loc[3]   

print(ranges4)



# function to get the average plr from the input_range 


def get_average_plr(input_range):
    average_plr = input_range.mean(level=0)['diameter'].to_numpy()
    return PLR(average_plr,
              sample_rate=SAMPLE_RATE,
              onset_idx=ONSET_IDX,
              stim_duration=1)

plr1 = get_average_plr(ranges1)
plr2 = get_average_plr(ranges2)
plr3 = get_average_plr(ranges3)
plr4 = get_average_plr(ranges4)

plr_all = [plr1, plr2, plr3, plr4]

#plot graphs for vel = velocity in green, acc = acceleration in red and parameters
fig1 = plr1.plot(vel=True, acc=True, print_params=True)
fig2 = plr2.plot(vel=True, acc=True, print_params=True)
fig3 = plr3.plot(vel=True, acc=True, print_params=True)
fig4 = plr4.plot(vel=True, acc=True, print_params=True)

params1 = plr1.parameters()
params1

params2 = plr2.parameters()
params2

params3 = plr3.parameters()
params3

params4 = plr4.parameters()
params4

 


def get_pyplr_results(plr):
    #D1 = baseline pupilsize, in mm
    D1 = plr.baseline()
    print("D1: ",D1)
    
    #D2 = minimum pupilsize, in mm 
    D2 = plr.peak_constriction()
    print("D2: ",D2)
    
    #AMP = constriction amplitude, in mm
    AMP = D1-D2
    print("AMP: ",AMP)
    
    #VCmax = maximum velocity of constriction, in mm/s
    VCmax = plr.max_constriction_velocity()
    print("VCmax: ",VCmax)
    
    #ACmax = maximum acceleration, in mm/s ?
    ACmax = plr.max_constriction_acceleration()
    print("ACmax: ",ACmax)
    
    #T1 = latency from the onset of the light stimulus to the maximum acceleration
    # in milliseconds - looks like in seconds on the graph  
    T1 = plr.latency_to_constriction_b()
    print("T1: ",T1)
    
    #T2 = time to maximum velocity, looks like in seconds on the graph 
    T2 = plr.time_to_max_velocity()
    print("T2: ",T2)
    
    #T3 = time to maximum constriction, in milliseconds - looks like in seconds on graph 
    T3 = plr.time_to_max_constriction()
    print("T3: ",T3)
    
    #relative constriction amplitude: AMP/D1
    rel_AMP = AMP/D1
    print("rel_AMP: ", rel_AMP)
    
    #time to 75% redilation 
    redil_75 = plr.time_to_75pc_recovery()
    print("redil_75", redil_75)
    
    #time to 50% redilation
    redil_50 = plr.time_to_50pc_recovery()
    print('redil_50', redil_50)
    
    redil_25 = plr.time_to_25pc_recovery()
    print('redil_25', redil_25)
    
    # create a CSV file
    return {
            'D1':D1,
            'D2': D2,
            'AMP': AMP,
            'VCmax':VCmax,
            'ACmax': ACmax,
            'T1': T1,
            'T2':T2,
            'T3':T3,
            "rel_AMP": rel_AMP,
            "redil_75": redil_75,
            'redil_50': redil_50,
            'redil_25': redil_25
            }

pyplr_results = {'ID': [ID, ID, ID, ID],
                 'D1':[],
                 'D2':[],
                 'AMP': [],
                 'VCmax':[],
                 'ACmax': [],
                 'T1': [],
                 'T2':[],
                 'T3':[],
                 "rel_AMP": [],
                 "redil_75": [],
                 'redil_50': [],
                 'redil_25': [],
                 'Light_strenght': [Light_strenght_1, Light_strenght_2, Light_strenght_3, Light_strenght_4]}
#pyplr_results = pd.DataFrame([ID, D1, D2, AMP. VCmax, ACmax, T1, T2, T3, Light_strenght], columns=["ID", "D1", "D2", "AMP", "VCmax", "ACmax", "T1", "T2", "T3", "Light_strenght"])

for plr in plr_all:
    pyplr_result = get_pyplr_results(plr)
    pyplr_results["D1"].append(pyplr_result["D1"])
    pyplr_results["D2"].append(pyplr_result["D2"])
    pyplr_results["AMP"].append(pyplr_result["AMP"])
    pyplr_results["VCmax"].append(pyplr_result["VCmax"])
    pyplr_results["ACmax"].append(pyplr_result["ACmax"])
    pyplr_results["T1"].append(pyplr_result["T1"])
    pyplr_results["T2"].append(pyplr_result["T2"])
    pyplr_results["T3"].append(pyplr_result["T3"])
    pyplr_results["rel_AMP"].append(pyplr_result["rel_AMP"])
    pyplr_results["redil_75"].append(pyplr_result["redil_75"])
    pyplr_results["redil_50"].append(pyplr_result["redil_50"])
    pyplr_results["redil_25"].append(pyplr_result["redil_25"])


df = pd.DataFrame(pyplr_results)

#always change the directory to what the sample gets saved 
df.to_csv('/Users/Katharina/Desktop/Beispieldaten/PJ04/PJ04_A_PLR2/PLR2-results.csv', index=False)

print(df)
