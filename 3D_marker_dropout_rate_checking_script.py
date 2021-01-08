# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 00:01:01 2020

@author: dongq
"""

"""
Converts DLC (actually Min's 3D reconstruction Output File) (in .csv format)
to OpenSim (which is in .trc format)
"""
import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt

#%%Import 3D reconstructed marker file from Min's 3D Reconstruction code
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\3D-data\output_3d_data_rotate4.csv')
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\reconsturcted-3d-data\output_3d_data.csv')

df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\reconstructed-3d-data-RT3D\output_3d_data.csv')
#%% Set parameters for this dataset
"""
IMPORTANT: SPECIFY IF THIS DATASET HAS STATIC "WORLD GROUND TRUTH" POINTS OR NOT
"""
has_static = True

frameRate = 24
numFrames = df.shape[0]-1
numMarkers = 8

mm_to_m_conversion = 1000

#%% Delete the static points if this dataset has them
if has_static == True:
    list_to_delete = ['pointX_x','pointX_y','pointX_z','pointX_error','pointX_ncams','pointX_score','pointY_x','pointY_y','pointY_z','pointY_error','pointY_ncams','pointY_score','pointZ_x','pointZ_y','pointZ_z','pointZ_error','pointZ_ncams','pointZ_score',]
    df = df.drop(columns = list_to_delete)
    
#%% TEMP, check dropout rate of the 3D dataset
df_array = df.to_numpy()

num_nans = 0
for i in range(df_array.shape[0]):
    for j in range(df_array.shape[1]):
        if np.isnan(df_array[i,j]):
            num_nans += 1
actual_num_nans = num_nans/2 #x,y,z are needed, but the following 3 are not needed
total_points = len(df_array)*3*numMarkers #3 in terms of X,Y,Z
dropout_percentage = actual_num_nans/total_points
print("Number of markers in all frames with NaNs", num_nans)
print("Total points in all frames", total_points)
print("Marker dropout percentage", dropout_percentage)


#%% Collect the dropout result

#%% Preset values

frames_per_second = 25
seconds_per_minute = 60


sample_session_start = 300 #in seconds
sample_session_end = 320 #in seconds
sample_start_frame = sample_session_start * frames_per_second
sample_end_frame = sample_session_end * frames_per_second

sample_session_start_2D = 400 #in seconds
sample_session_end_2D = 420 #in seconds
sample_start_frame_2D = sample_session_start_2D * frames_per_second
sample_end_frame_2D = sample_session_end_2D * frames_per_second


font = {'family' : 'normal',
#        'weight' : 'bold',
        'size'   : 22}

font_medium = {'family' : 'normal',
 #       'weight' : 'bold',
        'size'   : 16}

X = np.linspace(0,df_array.shape[0]-1,df_array.shape[0])/frames_per_second
small_X = list(np.linspace(sample_session_start,sample_session_end,(sample_session_end-sample_session_start)*frames_per_second))

#X_2D = np.linspace(0,df_array_2D.shape[0]-1,df_array_2D.shape[0])/frames_per_second
#ssmall_X_2D = list(np.linspace(sample_session_start_2D,sample_session_end_2D,(sample_session_end_2D-sample_session_start_2D)*frames_per_second))



#%% Collect the likelihood result and plot as histogram

likelihood_array = df_array[:,[5,11,17,23,29,35,41,47,53,59]]

plt.figure()
plt.hist(likelihood_array)
plt.title("Likelihood Score of 3D data")
plt.show()



