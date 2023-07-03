import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging as log
from dataclasses import dataclass

@dataclass 
class DataConfig: 
    window_duration:float=14
    eyenum:int=0 #eye0 right, eye1 left
    column:str="unknown" #diameter or diameter_3d
    subject_id:str="" #subject_id
    condition:str="" #1,2,3 or 4: 30Stim, 30Placebo, 3.4Sham, 3.4Placebo

def create_process_data_config(eyenum,column,subject_id,data_path):
    #import preprocessing
    config=DataConfig()
    config.eyenum=eyenum
    config.column=column
    config.subject_id=subject_id
    (timebase,cond)=get_condition(subject_id)
    config.condition=cond
    
    if config.column=="diameter_3d":
        pass
    elif column=="diameter": 
        pass
    else: 
        raise ValueError("column")
    return config

    
def plot(df, title):
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
    import os
    import pandas as pd
    if os.path.exists(f"{path}.pickle"):
        log.info("read pickle")
        return pd.read_pickle(f"{path}.pickle")        
    df=pd.read_csv(path,index_col=False, usecols=usecols)
    df.to_pickle(f"{path}.pickle")
    return df

def prepare(data_dir,subject_id,eye_id, config:DataConfig):
    csv_cols = ['pupil_timestamp', 'diameter_3d', 'diameter','eye_id','confidence']
    df=load_df(f"{data_dir}/{subject_id[:4]}/{subject_id}/exports/000/pupil_positions.csv", csv_cols)
    # df = pd.read_csv(f"{data_dir}/{subject_id[:4]}/{subject_id}/exports/000/pupil_positions.csv", index_col=False, usecols=csv_cols)
    # add moving average for the whole dataset


# blinkreconstruct for a pandas series. Returns a numpy array.
# see https://pydatamatrix.eu/0.15/series/#function-blinkreconstructseries-vt5-vt_start10-vt_end5-maxdur500-margin10-smooth_winlen21-std_thr3-gap_margin20-gap_vt10-modeuoriginal
def blinkreconstruct(df, vt=1, vt_start=10, vt_end=5, maxdur=800, margin=10, smooth_winlen=21, std_thr=3, gap_margin=20, gap_vt=10, mode=u'advanced'):
    display(type(df))
    import datamatrix
    import datamatrix.series
    import datamatrix.operations
    dm=datamatrix.convert.from_pandas(df).series
    return datamatrix.series.blinkreconstruct(dm, vt,vt_start,vt_end,maxdur,margin,smooth_winlen,std_thr,gap_margin,gap_vt,mode)





        

        
     
    

#if __name__=="__main__": 
#    import pandas as pd
#    subject_id="PJ02_1_Ruhe"
#    window_duration=30
#    data_dir="./Users/Katharina/Desktop/Beispieldaten/"
#    config=DataConfig(window_duration=30)
#    prepare(data_dir,subject_id,0,config)
