# plotting.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_preprocessed(df,title,ts_name,col_name):
    # Extract the timestamp and diameter data from the dataframe
    pupil_timestamp = df[ts_name]
    diameter = df[col_name]

    # Create a new figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the data before preprocessing in the left subplot
    ax1.plot(pupil_timestamp, diameter, marker=".", linestyle="")
    ax1.set_title(title)
    ax1.set_xlabel("Pupil Timestamp")
    ax1.set_ylabel("Diameter in mm/pixels")


    # Extract the timestamp and diameter data from the preprocessed dataframe
    pupil_timestamp_pre = df[ts_name]
    diameter_pre = df[col_name]

    # Plot the data after preprocessing in the right subplot
    label1 = df[df['label']==1]
    label2 = df[df['label']==2]
    label3 = df[df['label']==3]

    ax2.plot(label1[ts_name], label1[col_name], color='red', label='label 1')
    ax2.plot(label2[ts_name], label2[col_name], color='blue', label='label 2')
    ax2.plot(label3[ts_name], label3[col_name], color='green', label='label 3')
                         
    # Plot the data after preprocessing in the right subplot
    ax2.set_title(title)
    ax2.set_xlabel("Pupil Timestamp")
    ax2.set_ylabel("Diameter in mm/pixels")
