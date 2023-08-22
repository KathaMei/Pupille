import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging as log
from dataclasses import dataclass
import sys

@dataclass 
class DataConfig: 
    window_duration:float=30

def plot(df, title):
        '''
        Plotting the dataframe with pupil_timestamps on x-axis and diameter values on y-axis in a line chart fro checkdata notebook. Creating histograms showing the distribution of the variables confidence, diameter and diameter_3d.

        parameter
        ---------
          df:    Dataframe which is plotted. 
          title: Title for the plot.
        '''
        fig, ax = plt.subplots(2,2)
        ax[0,0].set_title(title)
        sub=df.plot(ax=ax[0,0],x='pupil_timestamp', y='diameter', kind='line')
        df.plot(ax=sub,x='pupil_timestamp', y='diameter_100', kind='line')
        ax[0,1].set_title("confidence")
        ax[0,1].hist(df['confidence'], bins=100, density=True)
        ax[1,0].set_title("diameter")
        ax[1,0].hist(df['diameter'], bins=100, density=True)
        ax[1,1].set_title("diameter_3d")
        ax[1,1].hist(df['diameter_3d'], bins=100, density=True)
        plt.show()

def load_df(path, usecols):
    '''
    Controlling csv dataframe if pickle file already exists. If already created, pickle file is loaded. If no pickle file is found, it is created and the csv dataframe included.

    parameter
    ---------
        path:    Path to files which are loaded. 
        usecols: Columns which are selected.
    '''
    import os
    import pandas as pd
    if os.path.exists(f"{path}.pickle"):
        log.info("read pickle")
        return pd.read_pickle(f"{path}.pickle")        
    df=pd.read_csv(path,index_col=False, usecols=usecols)
    df.to_pickle(f"{path}.pickle")
    return df

def prepare(data_dir,subject_id,eye_id, config:DataConfig):
    '''
    Load dataframes used for the code. Select the important columns. Load the annotation_timestamps and slice the dataframes into replicates. Add a timeslot column.

    ###Add Moving Average???
    ###only used in ckeckdata notebook
    
    parameter
    ---------
        data_dir:          Path to files which are loaded. 
        subject_id:        Subject_id, important for selecting the files.
        eye_id:            Eye side.
        config:DataConfig: Selected dataclass.
    '''
    csv_cols = ['pupil_timestamp', 'diameter_3d', 'diameter','eye_id','confidence']
    df=load_df(f"{data_dir}/{subject_id[:4]}/{subject_id}/exports/000/pupil_positions.csv", csv_cols)
    # df = pd.read_csv(f"{data_dir}/{subject_id[:4]}/{subject_id}/exports/000/pupil_positions.csv", index_col=False, usecols=csv_cols)
    # add moving average for the whole dataset
    df['diameter_100']=df['diameter'].rolling(window=100).mean()
    annotation_timestamps = np.load(f"{data_dir}/{subject_id[:4]}/{subject_id}/annotation_timestamps.npy")
    res=[]
    for annotation_timestamp in annotation_timestamps:
        # Calculate the start and end timestamps for the window after the annotation
        window_start = annotation_timestamp - 5.0
        window_end = window_start + config.window_duration
        
        # Select the rows that fall within the window
        df_sliced = df[(df['pupil_timestamp'].between(window_start,window_end)) & (df['eye_id'] == eye_id)]
        df_sliced=df_sliced.copy()
        df_sliced['pupil_timestamp_based'] = df_sliced['pupil_timestamp'] - window_start
        # Do more cleanup

        # Add a timeslot column
        df_sliced['rowid'] = range(len(df_sliced)) 
        df_sliced['timeslot'] = df_sliced['rowid'] // 1000
        res.append(df_sliced)        
    return res

# blinkreconstruct for a pandas series. Returns a numpy array.
# see https://pydatamatrix.eu/0.15/series/#function-blinkreconstructseries-vt5-vt_start10-vt_end5-maxdur500-margin10-smooth_winlen21-std_thr3-gap_margin20-gap_vt10-modeuoriginal
def blinkreconstruct(df, vt=5, vt_start=10, vt_end=5, maxdur=500, margin=10, smooth_winlen=21, std_thr=3, gap_margin=20, gap_vt=10, mode=u'advanced'):
    '''
    Blinkreconstruct for a pandas series. Returns a numpy array.
    see https://pydatamatrix.eu/0.15/series/#function-blinkreconstructseries-vt5-vt_start10-vt_end5-maxdur500-margin10-smooth_winlen21-std_thr3-gap_margin20-gap_vt10-modeuoriginal
    '''
    
    display(type(df))
    import datamatrix
    import datamatrix.series
    import datamatrix.operations
    dm=datamatrix.convert.from_pandas(df).series
    return datamatrix.series.blinkreconstruct(dm, vt,vt_start,vt_end,maxdur,margin,smooth_winlen,std_thr,gap_margin,gap_vt,mode)





        

        
     
    

if __name__=="__main__": 
    import pandas as pd
    subject_id="PJ02_1_Ruhe"
    window_duration=30
    data_dir="./Users/Katharina/Desktop/Beispieldaten/"
    config=DataConfig(window_duration=30)
    prepare(data_dir,subject_id,0,config)
