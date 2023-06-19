#!/usr/bin/env python3
import preprocessing as pp
import sys

def show(d):
    result=pp.load_pickle(d)
    for f in result.frames:
        print(f.valid,f.stage,f.remark)
    
for d in sys.argv[1:]: 
    show(d)
