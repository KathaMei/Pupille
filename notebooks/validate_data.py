#!/usr/bin/env python3
import preprocessing
import sys
import pandas

display=print
def noprint(x):
    pass

def validate_field(fn, field):
    '''
    Repeatedly executing functions. Save specific information to select the best fitting thresholds values for the data to remove artefacts.
    
    parameter
        ---------
          fn:    Functions used. 
        field:   Diameter or diameter_3d.
    '''
    import os
    d,subject_id=os.path.split(fn)
    data_dir,_=os.path.split(d)
    config=preprocessing.create_process_config(0,field,subject_id,data_dir)
    config.validate_only=True
    eye0=preprocessing.process(config,noprint)
    for k in eye0:
        subject,annotation,nan_percent,good=k
        if good: v=1
        else: v=0
        print(f'{subject},{field},{annotation},{nan_percent},{v}')
    
def validate(fn):
    '''
    Run the functions for different fields.

    parameter
    ---------
        fn:    Functions used. 
    '''
    validate_field(fn,"diameter")
    #validate_field(fn,"diameter_3d")
    
for d in sys.argv[1:]: 
    validate(d)
    
    
