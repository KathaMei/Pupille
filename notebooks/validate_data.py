#!/usr/bin/env python3
import preprocessing
import sys

display=print
def noprint(x):
    pass

def validate(fn):
    import os
    d,subject_id=os.path.split(fn)
    data_dir,_=os.path.split(d)
    config=preprocessing.create_process_config(0,"diameter",subject_id,"Ruhe","30",data_dir)
    config.validate_only=True
    eye0=preprocessing.process(config,noprint)
    for k in eye0:
        subject,annotation,nan_percent,good=k
        if good: v=1
        else: v=0
        print(f'{subject},{annotation},{nan_percent},{v}')
    
for d in sys.argv[1:]: 
    validate(d)
    
    
