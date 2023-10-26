def save_pickle(filename,obj):
    import pickle
    with open(filename,"wb") as h: 
        pickle.dump(obj,h,protocol=5)
        
def load_pickle(filename):
    import pickle
    with open(filename,"rb") as h:
        return pickle.load(h)
   

#Return condition for randomized condition code of subject
def get_condition(subject_id):
    import pandas as pd
    import os
    zuordnungen=f'{os.path.dirname(__file__)}/zuordnungen.csv'
    f=pd.read_csv(zuordnungen,index_col='proband')
    prob=subject_id[:4]
    msr=subject_id[5:6]
    if msr=='A': 
        return (0,"BaselineA")
    if msr=='B': 
        return (0,"BaselineB")
    q=f.loc[prob][int(msr)-1]
    names=[("3.4","3.4Stim"),("3.4","3.4Placebo"),("30","30Stim"),("30","30Placebo")]
    return names[int(q)-1]
