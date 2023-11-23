#!/usr/bin/env python3
import lr_preprocessing
import sys
import pup_config

display=print
def noprint(*x):
    pass

def pps(fn, field,eye):
    '''
    Repeatedly executing functions. Save the results as csv files.

    parameter
    ---------
        fn:    Functions used. Removing /. 
        field: Diameter or diameter_3d.
        eye:   Variable name.
    '''
    try:
        import os
        fn=fn.rstrip("/")
        d,subject_id=os.path.split(fn)
        data_dir,_=os.path.split(d)
        config=lr_preprocessing.create_process_config(eye,field,subject_id,data_dir)
        config.out_dir=f'{pup_config.obj_dir}/PLRPrüfung/{subject_id}/'
        print(f"processing source={fn}, field={field}, eye={eye}")
        res=lr_preprocessing.process(config,noprint)
    except Exception as ex: 
        print(f"ERROR: processing of file {fn} failed with exception {type(ex)}{ex}",file=sys.stderr)

def pp(fn):
    pps(fn,"diameter",0)
    pps(fn,"diameter_3d",0)
    pps(fn,"diameter",1)
    pps(fn,"diameter_3d",1)
    
for d in sys.argv[1:]: 
    pp(d)
    
    
