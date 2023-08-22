#import preprocessing
#import sys
#import pandas
# python3 count_fails.py ~/Nextcloud/KatharinaBeispieldaten/*.pickle
#for d in sys.argv[1:]: 
    #eye=preprocessing.load_pickle(d)
    #print(eye.config.subject_id, eye.config.eyenum, eye.config.column)
    #print(eye.config.condition)
    #r=[(x.stage,x.remark) for x in eye.frames]
    #pd=pandas.DataFrame(r,columns=["stage","remark"])
    #print(pd.groupby("stage")["stage"].count())
    
#python3 /Users/Katharina/Desktop/Pupille/notebooks/count_fails.py /Users/Katharina/Desktop/Beispieldaten/Auswertungsergebnisse/*.pickle


'''
Counting the replicates in every stage for different combinations for condition, eyenum, column.
'''

import sys
import preprocessing
import pandas as pd

total_pickle_files = 0
total_condition_3_4_stim = 0
total_condition_30_stim = 0
total_condition_3_4_placebo = 0
total_condition_30_placebo = 0
total_eyenum_0 = 0
total_eyenum_1 = 0
total_column_diameter = 0
total_column_diameter_3d = 0

data_counts_by_eyenum_column = {
    ("eyenum_0", "diameter"): {"count": 0, "stage_relics": {}},
    ("eyenum_0", "diameter_3d"): {"count": 0, "stage_relics": {}},
    ("eyenum_1", "diameter"): {"count": 0, "stage_relics": {}},
    ("eyenum_1", "diameter_3d"): {"count": 0, "stage_relics": {}}
}

for d in sys.argv[1:]:
    eye = preprocessing.load_pickle(d)
    
    total_pickle_files += 1
    
    subject_id = eye.config.subject_id
    eyenum = eye.config.eyenum
    column = eye.config.column
    condition = eye.config.condition
    
    if condition == '3.4Stim':
        total_condition_3_4_stim += 1
    elif condition == '30Stim':
        total_condition_30_stim += 1
    elif condition == '3.4Placebo':
        total_condition_3_4_placebo += 1
    elif condition == '30Placebo':
        total_condition_30_placebo += 1
    
    if eyenum == 0:
        total_eyenum_0 += 1
    elif eyenum == 1:
        total_eyenum_1 += 1
    
    if column == 'diameter':
        total_column_diameter += 1
        data_counts_by_eyenum_column[("eyenum_0", "diameter")]["count"] += len(eye.frames)
        for frame in eye.frames:
            stage = frame.stage
            remark = frame.remark
            if stage not in data_counts_by_eyenum_column[("eyenum_0", "diameter")]["stage_relics"]:
                data_counts_by_eyenum_column[("eyenum_0", "diameter")]["stage_relics"][stage] = []
            data_counts_by_eyenum_column[("eyenum_0", "diameter")]["stage_relics"][stage].append(remark)
    elif column == 'diameter_3d':
        total_column_diameter_3d += 1
        data_counts_by_eyenum_column[("eyenum_0", "diameter_3d")]["count"] += len(eye.frames)
        for frame in eye.frames:
            stage = frame.stage
            remark = frame.remark
            if stage not in data_counts_by_eyenum_column[("eyenum_0", "diameter_3d")]["stage_relics"]:
                data_counts_by_eyenum_column[("eyenum_0", "diameter_3d")]["stage_relics"][stage] = []
            data_counts_by_eyenum_column[("eyenum_0", "diameter_3d")]["stage_relics"][stage].append(remark)
    
    if eyenum == 1 and column == 'diameter':
        data_counts_by_eyenum_column[("eyenum_1", "diameter")]["count"] += len(eye.frames)
        for frame in eye.frames:
            stage = frame.stage
            remark = frame.remark
            if stage not in data_counts_by_eyenum_column[("eyenum_1", "diameter")]["stage_relics"]:
                data_counts_by_eyenum_column[("eyenum_1", "diameter")]["stage_relics"][stage] = []
            data_counts_by_eyenum_column[("eyenum_1", "diameter")]["stage_relics"][stage].append(remark)
    elif eyenum == 1 and column == 'diameter_3d':
        data_counts_by_eyenum_column[("eyenum_1", "diameter_3d")]["count"] += len(eye.frames)
        for frame in eye.frames:
            stage = frame.stage
            remark = frame.remark
            if stage not in data_counts_by_eyenum_column[("eyenum_1", "diameter_3d")]["stage_relics"]:
                data_counts_by_eyenum_column[("eyenum_1", "diameter_3d")]["stage_relics"][stage] = []
            data_counts_by_eyenum_column[("eyenum_1", "diameter_3d")]["stage_relics"][stage].append(remark)
    
    r = [(x.stage, x.remark) for x in eye.frames]
    pd_data = pd.DataFrame(r, columns=["stage", "remark"])
    stage_counts = pd_data.groupby("stage")["stage"].count()
    
    print(f"Subject ID: {subject_id}")
    print(f"Condition: {condition}")
    print(f"Eyenum: {eyenum}")
    print(f"Column: {column}")
    print("Stage Counts:")
    print(stage_counts)
    print("\n")

print(f"Total Pickle Files: {total_pickle_files}")
print(f"Total 3.4Stim Conditions: {total_condition_3_4_stim}")
print(f"Total 30Stim Conditions: {total_condition_30_stim}")
print(f"Total 3.4Placebo Conditions: {total_condition_3_4_placebo}")
print(f"Total 30Placebo Conditions: {total_condition_30_placebo}")
print(f"Total Eyenum = 0: {total_eyenum_0}")
print(f"Total Eyenum = 1: {total_eyenum_1}")
print(f"Total Column = diameter: {total_column_diameter}")
print(f"Total Column = diameter_3d: {total_column_diameter_3d}")

#print("Data Counts by Eyenum and Column:")
#for key, value in data_counts_by_eyenum_column.items():
    #eyenum, column = key
    #count = value["count"]
    #print(f"Eyenum: {eyenum}, Column: {column}, Data Count: {count}")
    #print("Stage Relics:")
    #for stage, relics in value["stage_relics"].items():
        #print(f"Stage: {stage}, Relics: {relics}")

import sys
import preprocessing
import pandas as pd
from collections import defaultdict
import os

output_path = '/Users/Katharina/Desktop/Beispieldaten'

data_counts_by_combination_and_stage = defaultdict(dict)

for d in sys.argv[1:]:
    eye = preprocessing.load_pickle(d)
    
    subject_id = eye.config.subject_id
    eyenum = eye.config.eyenum
    column = eye.config.column
    condition = eye.config.condition
    
    for frame in eye.frames:
        stage = frame.stage
        if stage not in data_counts_by_combination_and_stage[(eyenum, column)]:
            data_counts_by_combination_and_stage[(eyenum, column)][stage] = 0
        data_counts_by_combination_and_stage[(eyenum, column)][stage] += 1

print("Data Counts by Eyenum and Column in Each Stage:")

for (eyenum, column), stage_counts in data_counts_by_combination_and_stage.items():
    print(f"Eyenum: {eyenum}, Column: {column}")
    for stage, count in stage_counts.items():
        print(f"Stage: {stage}, Data Count: {count}")
    print("\n")


# Erstelle DataFrames und speichere CSV-Dateien
rows_eyenum_column = []

for (eyenum, column), stage_counts in data_counts_by_combination_and_stage.items():
    for stage, count in stage_counts.items():
        rows_eyenum_column.append({'Eyenum': eyenum, 'Column': column, 'Stage': stage, 'Data Count': count})

df_eyenum_column = pd.DataFrame(rows_eyenum_column)
csv_filename_1 = 'data_counts_eyenum_column.csv'
output_filepath_1 = os.path.join(output_path, csv_filename_1)
df_eyenum_column.to_csv(output_filepath_1, index=False)
print(f"DataFrame wurde als '{output_filepath_1}' gespeichert.")


# Dein zweiter Code
data_counts_by_combination_stage_condition = defaultdict(dict)

for d in sys.argv[1:]:
    eye = preprocessing.load_pickle(d)
    
    subject_id = eye.config.subject_id
    eyenum = eye.config.eyenum
    column = eye.config.column
    condition = eye.config.condition
    
    for frame in eye.frames:
        stage = frame.stage
        if (eyenum, column, condition) not in data_counts_by_combination_stage_condition:
            data_counts_by_combination_stage_condition[(eyenum, column, condition)] = defaultdict(int)
        data_counts_by_combination_stage_condition[(eyenum, column, condition)][stage] += 1

print("Data Counts by Eyenum, Column, and Condition in Each Stage:")

for (eyenum, column, condition), stage_counts in data_counts_by_combination_stage_condition.items():
    print(f"Eyenum: {eyenum}, Column: {column}, Condition: {condition}")
    for stage, count in stage_counts.items():
        print(f"Stage: {stage}, Data Count: {count}")
    print("\n")

# Erstelle DataFrames und speichere CSV-Dateien
rows_eyenum_column_condition = []

for (eyenum, column, condition), stage_counts in data_counts_by_combination_stage_condition.items():
    for stage, count in stage_counts.items():
        rows_eyenum_column_condition.append({'Eyenum': eyenum, 'Column': column, 'Condition': condition, 'Stage': stage, 'Data Count': count})

df_eyenum_column_condition = pd.DataFrame(rows_eyenum_column_condition)
csv_filename_2 = 'data_counts_eyenum_column_condition.csv'
output_filepath_2 = os.path.join(output_path, csv_filename_2)
df_eyenum_column_condition.to_csv(output_filepath_2, index=False)
print(f"DataFrame wurde als '{output_filepath_2}' gespeichert.")

