import preprocessing
import sys
import pandas
# python3 count_fails.py ~/Nextcloud/KatharinaBeispieldaten/*.pickle
for d in sys.argv[1:]: 
    eye=preprocessing.load_pickle(d)
    print(eye.config.subject_id)
    r=[(x.stage,x.remark) for x in eye.frames]
    pd=pandas.DataFrame(r,columns=["stage","remark"])
    print(pd.groupby("stage")["stage"].count())
    
