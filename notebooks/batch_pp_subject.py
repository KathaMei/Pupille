#!/usr/bin/env python3
import preprocessing
import sys

display=print
def noprint(x):
    pass

def pps(fn, field,eye):
    import os
    fn=fn.rstrip("/")
    d,subject_id=os.path.split(fn)
    data_dir,_=os.path.split(d)
    out_dir="/Users/Katharina/Desktop/ErgebnissePreprocessing"
    #out_dir="./"
    config=preprocessing.create_process_config(eye,field,subject_id,data_dir)
    res=preprocessing.process(config,noprint)    
    outfn=f"{out_dir}/{subject_id}_{config.condition}_{field}_{eye}.pickle"
    print(f"saving results to {outfn}")
    preprocessing.save_pickle(outfn,res)
    av_df=preprocessing.average_frames_by_binning(res,f'{field}_baseline',interval_ms=100)
    csvfn=f"{out_dir}/averaged_{subject_id}_{config.condition}_{field}_{eye}.csv"
    av_df.to_csv(csvfn,index_label="recno")

def pp(fn):
    pps(fn,"diameter",0)
    pps(fn,"diameter_3d",0)
    pps(fn,"diameter",1)
    pps(fn,"diameter_3d",1)
    
for d in sys.argv[1:]: 
    pp(d)
    
    
