# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:43:04 2020

@author: dongq
"""

#%% Import Packages
import pandas as pd 
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#from peakutils.plot import plot as pplot
#import peakutils
from scipy import signal
from scipy.interpolate import interp1d
#from moviepy.editor import *
from scipy import stats
import copy
#%% Read in dataset for segmentation
df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\reconstructed-3d-data\output_3d_data.csv')

#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')


#%% Read in the file for trial segmentation, set parameters
f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\Ground_truth_segments_20200804_RT.txt", "r") 

frames_per_second = 25
seconds_per_minute = 60

#%% Use the file to segment the dataset
ground_truth_experiment_segments = f.read()
f_temp = ground_truth_experiment_segments.split(" ")
f_seg = np.zeros((int(len(f_temp)/4),4))

for i in range(len(f_seg)):
    f_seg[i,0] = int(f_temp[i*4+0])
    f_seg[i,1] = int(f_temp[i*4+1])
    f_seg[i,2] = int(f_temp[i*4+2])
    f_seg[i,3] = int(f_temp[i*4+3])

f_second = np.zeros((len(f_seg),2))

for i in range(len(f_second)):
    f_second[i,0] = f_seg[i,0]*seconds_per_minute + f_seg[i,1]
    f_second[i,1] = f_seg[i,2]*seconds_per_minute + f_seg[i,3]
    
f_frame = f_second*frames_per_second

#ground_truth_segment = np.zeros((len(df_speed)))
ground_truth_segment = np.zeros((df.shape[0]))

f_frame_list = list()

for i in range(len(f_frame)):
    #f_frame_list.append(list(range(int(f_frame[i,0]),int(f_frame[i,1]+1))))
    #print(list(range(int(f_frame[i,0]),int(f_frame[i,1]+1))))
    f_frame_list = f_frame_list + list(range(int(f_frame[i,0]),int(f_frame[i,1]+1)))
    
for i in range(len(f_frame_list)):
    #print(i)
    ground_truth_segment[f_frame_list[i]] = 1
    
#%% Append a row to the new DataFrame if ground_truth_segment[i] == 1
    
#%% Save the DataFrame as a .csv file in the same folder as the original dataset

#%%
"""
def count_non_confident_points_3D(df):
    #df_head = df.head()
    #df_col_name = df.columns
    likelihood_limit = 0.1
    #df_likelihood = df.xs('likelihood',axis=1)
    
    df_likelihood = [col for col in df.columns if 'score' in col]
    
    df_likelihood_copy = df_likelihood
    df_dropouts = []
    for i in df_likelihood_copy:
        df_likelihood_i = df_likelihood_copy[i]
        dropout_nums = int(df_likelihood_i[df_likelihood_i<likelihood_limit].count().values)
        df_dropouts.append(dropout_nums)
    return df_dropouts
"""
#%%
"""
non_confident_3D_points = count_non_confident_points_3D(df)
"""

#%%