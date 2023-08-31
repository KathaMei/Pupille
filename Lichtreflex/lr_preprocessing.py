# lr_preprocessing.py
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import os.path
import shutil
from datetime import datetime
import math
sys.path.append("../Pupillengröße/Skripte/")
from dataclasses import dataclass
from typing import List
import pup_util

@dataclass 
class ProcessConfig:
    '''
    Data class to present attributes used in application.

    attributes
    ---------
        eyenum:int:                      Eye side: left (eye1) or right (eye0).
        column:str:                      Column of pupil size data: diameter or diameter_3d.
        sfactor:float:                   The factor the velocity (vt) should be divided with to detect blinks in the blinkreconstruct function.
        data_dir:str:                    File path from which the data is taken.
        subject_id:str:                  The subject_id contains the subject number and the number of study cycles which is used to select the corresponding condition. 
        condition:str:                   Stimulation conditionen: 3.4Stim, 30Stim, 3.4Placebo, 30Placebo
        out_dir:str:                     The output directory.
    '''
    eyenum:int=0 #eye0 right, eye1 left
    column:str="unknown" #diameter or diameter_3d
    sfactor:float="" #the factor the velocity (vt) should be divided with to detect blinks in the blink_reconstruct
    data_dir:str="" #path the data is taken
    subject_id:str="" #subject_id
    condition:str="" #1,2,3 or 4: 30Stim, 30Placebo, 3.4Sham, 3.4Placebo
    out_dir:str=None
    
def create_process_config(eyenum,column,subject_id,data_dir):
    '''
    Creates a function with values for the attributes of the dataclass ProcessConfing. The values are selected based on column and timebase of the data.
    
    parameter
    ---------
        eyenum:        Eye side: left (eye1) or right (eye0).
        column:str:    Selected column to which the function is applied.
        subject_id:    The subject_id contains the subject number and the number of study cycles which is used to select the corresponding condition.
        data_path:str: File path from which the data is taken.
    '''
    config=ProcessConfig()
    config.eyenum=eyenum
    config.column=column
    config.subject_id=subject_id
    (_,cond)=pup_util.get_condition(subject_id)
    config.condition=cond
    config.data_dir=data_dir    
    if config.column=="diameter_3d": 
        config.sfactor=1000
    elif column=="diameter": 
        config.sfactor=1
    else: 
        raise ValueError("column")
    return config

def blinkreconstruct(df, vt=5, vt_start=10, vt_end=5, maxdur=5000, margin=10, smooth_winlen=21, std_thr=3, gap_margin=20, gap_vt=10, mode=u'advanced'):
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
    return datamatrix.series.blinkreconstruct(dm, vt,vt_start,vt_end,maxdur,margin,smooth_winlen,std_thr,gap_margin,gap_vt,mode)

def reconstruct(eye, col_in, col_out, window_size=100):
        '''
        Applies the blinkreconstruct function to the prepared data object dm. Interpolates the data of the selected column of data object eye with method linear. The results are stored in col_out. 
        
        parameter
        ---------
            config: ProcessConfig: Use of the attributes defined in dataclass ProcessConfig.
            eye:                   Selected object to which the function is applied.
            col_in:                Selected column to which the function is applied.
            col_out:               Computed column after applying the function.
            window_size:           Number of data points used for interpolation.????
        '''
        if col_in==col_out:
            eye[f'{col_in}_original']=eye[col_in]
        # Remove blinks.
        eye[f'{col_in}_rec']=blinkreconstruct(eye[col_in], 
                                                       vt_start=50,vt_end=5, mode='advanced')
        # blinkreconstruct replaces the bliks with NaN with mode='advanced',
        # so we interpolate the gaps and low pass the result to obtain something. 
        eye[f'{col_in}_rec_interp']=eye[f'{col_in}_rec'].interpolate(method='linear').copy()
        # Use moving average + recenter as low pass.
        # eye[col_out]=eye[f'{col_in}_rec_interp'].rolling(window=window_size).mean().shift(-window_size//2)
        eye[col_out]=eye[f'{col_in}_rec_interp']

def process(config:ProcessConfig,progress=print):
    '''
    The dataframes are loaded. Various functions are applied to preprocess the data and calculate PLR values.
    First various variables are assigned and the data loaded.
    
    -Light_strength: the intensity of the light the probands were exposed to.
    -SAMPLE_RATE: The frequency of measurements of the eye tracker.

    Then the reconstruct function is applied and a low pass filter is used to smooth the data. The light reflexes of one dataset are sliced according to the annotation timestamps. The baseline mean is calculated. The class PLR from Github is used to plot the data, calculate parameter values and create csv files with results. 

    parameter
    ---------
        config:ProcessConfig: Use of the attributes defined in dataclass ProcessConfig.
        progress: Written feedback on the progress of the script.
    '''
    from pyplr import utils
    #from pyplr import graphing
    from pyplr import preproc
    
    # define the patient ID for the dataframe and assign the 4 light strenghts
    Light_strenght_1 = 1
    Light_strenght_2 = 2
    Light_strenght_3 = 3
    Light_strenght_4 = 4
    
    # Sampling frequency
    SAMPLE_RATE = 120
    
    # Columns to load
    use_cols = ['confidence',
               'method',
                'pupil_timestamp',
                'eye_id',
                config.column]
    
    pupil_cols = [config.column]

    rec_dir=f"{config.data_dir}/{config.subject_id[:4]}/{config.subject_id}"

    ####wird gar nicht mehr gemacht???
    ###
    ##
    # Check if the pyplr_analysis directory exists
    if not os.path.exists(os.path.join(rec_dir, 'pyplr_analysis')):
        # Create the directory if it doesn't exist
        os.makedirs(os.path.join(rec_dir, 'pyplr_analysis'))

    
    # Get a handle on a subject
    subject = utils.new_subject(
        rec_dir, export='000', out_dir_nm='pyplr_analysis')
    if config.eyenum==0: 
        eye_id_key='right'
    else:
        eye_id_key='left'
        
    if config.column=='diameter':
        method_key='2d' 
    else:
        method_key='3d' 
    s=subject
    progress("SUBJECT:",subject)
    samples = utils.load_pupil(s['data_dir'], eye_id=eye_id_key, method=method_key, cols=use_cols)
    reconstruct(samples,config.column,config.column)

    subject_id=config.subject_id
    
    # Make figure for processing, append figure to pupil_preprocessing
    #f, axs = graphing.pupil_preprocessing_figure(nrows=5, subject=f"{subject_id}" )
    # Plot the raw data
    # Plot the raw data
    #samples[pupil_cols].plot(title=f"{subject_id}_left_2d", ax=axs[0], legend=True)
    #axs[0].legend(loc='center right', labels=['pixels'])
    # Mask first derivative
    #Default is a threshold of 3 SD from the mean first derivate
    # If there are a lot of blinks the mean first derivate is higher
    # Therefore set the threshold lower 
    #samples = preproc.mask_pupil_first_derivative(
        #samples, threshold=3.0, mask_cols=pupil_cols)
    #samples[pupil_cols].plot(
       # title='Masked 1st deriv (3*SD)', ax=axs[1], legend=False)
    
    # Mask confidence
    #samples = preproc.mask_pupil_confidence(
    #    samples, threshold=0.8, mask_cols=pupil_cols)
    #samples[pupil_cols].plot(
    #   title='Masked confidence (<0.8)', ax=axs[2], legend=False)

    # Interpolate
    #samples = preproc.interpolate_pupil(
    #    samples, interp_cols=pupil_cols)
    #samples[pupil_cols].plot(
    #    title='Linear interpolation', ax=axs[3], legend=False)
    
    # Smooth
    samples = preproc.butterworth_series(
        samples, fields=pupil_cols, filt_order=3,
        cutoff_freq=4/(SAMPLE_RATE/2))
    #samples[pupil_cols].plot(
    #    title='3rd order Butterworth filter with 4 Hz cut-off',
     #   ax=axs[4], legend=False);
    '''
    Annotation timestamps are loaded and the data is sliced into four separate light reflexes. The baseline mean is calculated and the percentage change frame baseline.
    '''
    events = utils.load_annotations(s['data_dir'])
    progress(events)

    DURATION = 1200
    ONSET_IDX = 120
    
    # Extract the event ranges, gets range of total duration (720) and shifts the onset_idx to the left in time, therefore 120 is the 0 point
    ranges = utils.extract(
        samples,
        events,
        offset=-ONSET_IDX,
        duration=DURATION,
        borrow_attributes=['color'])
    
    # Convert data to numeric format
    ranges = ranges.apply(pd.to_numeric, errors='coerce')
    
    # Calculate baselines, the time before the onset is 0 to 120
    baselines = ranges.loc[:, range(0, ONSET_IDX), :].groupby(level=0, axis=1).mean()
    
    # New columns for percent signal change, simply adds 2 new columns with percent change
    ranges = preproc.percent_signal_change(
        ranges, baselines, pupil_cols)
    ranges
    
    progress(ranges.loc[0:3])
    
    ranges1 = ranges.loc[0]
    ranges2 = ranges.loc[1]
    ranges3 = ranges.loc[2]
    ranges4 = ranges.loc[3]
    
    progress(ranges4)

    def get_average_plr(input_range):
        '''
        The data class PLR from classPLRfromGithub module is imported. The PLR data is taken to calculate the average PLR for each group in the input data. The data is grouped and the mean is calculated. It is converted to a NumPy array. A PLR object is created containing the average_plr data and relevant characeristics.
        
        parameter
        ---------
            input_range: The PLR data.
      
        '''
        from classPLRfromGitHub import PLR
        average_plr = input_range.groupby(level=0)[config.column].mean().to_numpy()
        return PLR(average_plr,
                  sample_rate=SAMPLE_RATE,
                  onset_idx=ONSET_IDX,
                  stim_duration=1)
    
    plr1 = get_average_plr(ranges1)
    plr2 = get_average_plr(ranges2)
    plr3 = get_average_plr(ranges3)
    plr4 = get_average_plr(ranges4)
    
    plr_all = [plr1, plr2, plr3, plr4]

    '''
    Plot the light reflexes in separate plots, showing the data, the velocitiy and acceleration.
    '''
    
    #plot graphs for vel = velocity in green, acc = acceleration in red and parameters
    fig1 = plr1.plot(vel=True, acc=True, print_params=False)
    fig2 = plr2.plot(vel=True, acc=True, print_params=False)
    fig3 = plr3.plot(vel=True, acc=True, print_params=False)
    fig4 = plr4.plot(vel=True, acc=True, print_params=False)
    
    #params1 = plr1.parameters()    
    #params2 = plr2.parameters()    
    #params3 = plr3.parameters()    
    #params4 = plr4.parameters()

    def get_pyplr_results(plr):
        '''
        Appling the functions to calculate PLR parameters. Create a csv file to store all the values of the parameters for all four light reflexes.
        '''
    #D1 = baseline pupilsize, in mm
        D1 = plr.baseline()
        progress("D1: ",D1)
        
        #D2 = minimum pupilsize, in mm 
        D2 = plr.peak_constriction()
        progress("D2: ",D2)
        
        #AMP = constriction amplitude, in mm
        AMP = D1-D2
        progress("AMP: ",AMP)
        
        #VCmax = maximum velocity of constriction, in mm/s
        VCmax = plr.max_constriction_velocity()
        progress("VCmax: ",VCmax)
        
        #ACmax = maximum acceleration, in mm/s ?
        ACmax = plr.max_constriction_acceleration()
        progress("ACmax: ",ACmax)
        
        #T1 = latency from the onset of the light stimulus to the maximum acceleration
        # in milliseconds - looks like in seconds on the graph  
        T1 = plr.latency_to_constriction_b()
        progress("T1: ",T1)
        
        #T2 = time to maximum velocity, looks like in seconds on the graph 
        T2 = plr.time_to_max_velocity()
        progress("T2: ",T2)
        
        #T3 = time to maximum constriction, in milliseconds - looks like in seconds on graph 
        T3 = plr.time_to_max_constriction()
        progress("T3: ",T3)
        
        #relative constriction amplitude: AMP/D1
        rel_AMP = AMP/D1
        progress("rel_AMP: ", rel_AMP)
        
        #time to 75% redilation 
        redil_75 = plr.time_to_75pc_recovery()
        progress("redil_75", redil_75)
        
        #time to 50% redilation
        redil_50 = plr.time_to_50pc_recovery()
        progress('redil_50', redil_50)
        
        redil_25 = plr.time_to_25pc_recovery()
        progress('redil_25', redil_25)
        
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
    '''
    Creates a dictionary to store the calculated vales of the PLR in separate lists.
    '''
    pyplr_results = {'Subject ID': [subject_id, subject_id, subject_id, subject_id],
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
    progress(pyplr_results)

    plr_all = [plr1, plr2, plr3, plr4]
    
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
    
    '''
    Save the results as csv files.
    '''
    df = pd.DataFrame(pyplr_results)
    df["eye_id"] = config.eyenum
    df["method"] = method_key
    df["condition"]=config.condition
    #always change the directory to what the sample gets saved 
    if config.out_dir!=None: 
        out_dir=config.out_dir
    else:
        out_dir=subject['out_dir']
    outfile=f"{out_dir}/PLR_{eye_id_key}_{method_key}_results.csv"
    
    os.makedirs(out_dir,exist_ok=True)
    df.to_csv(outfile, index=False)
    
    progress(f'output written to {outfile}')
    # print(df)
