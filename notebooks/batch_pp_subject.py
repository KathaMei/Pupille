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
    config=preprocessing.create_process_config(eye,field,subject_id,data_dir)
    res=preprocessing.process(config,noprint)
    outfn=f"{data_dir}/{subject_id}_{field}_{eye}.pickle"
    print(f"saving results to {outfn}")
    preprocessing.save_pickle(outfn,res)
    
def pp(fn):
    pps(fn,"diameter",0)
    pps(fn,"diameter_3d",0)
    pps(fn,"diameter",1)
    pps(fn,"diameter_3d",1)
    
for d in sys.argv[1:]: 
    pp(d)
    
    
