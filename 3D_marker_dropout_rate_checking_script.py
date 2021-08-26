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

df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\Iteration_3_results\reconstructed-3d-data-RT3D\output_3d_data.csv')
df_array = df.to_numpy()
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
    

#%% Get the array for trial segmentation
#if experiment_phase_only == 1:
    #df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')

# =============================================================================
# seconds_per_minute = 60
# f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\Ground_truth_segments_2020-12-03-RT3D-2.txt", "r") 
# ground_truth_experiment_segments = f.read()
# f_temp = ground_truth_experiment_segments.split(" ")
# f_seg = np.zeros((int(len(f_temp)/4),4))
# 
# for i in range(len(f_seg)):
#     f_seg[i,0] = int(f_temp[i*4+0])
#     f_seg[i,1] = int(f_temp[i*4+1])
#     f_seg[i,2] = int(f_temp[i*4+2])
#     f_seg[i,3] = int(f_temp[i*4+3])
# 
# f_second = np.zeros((len(f_seg),2))
# 
# for i in range(len(f_second)):
#     f_second[i,0] = f_seg[i,0]*seconds_per_minute + f_seg[i,1]
#     f_second[i,1] = f_seg[i,2]*seconds_per_minute + f_seg[i,3]
#     
# f_frame = f_second*frameRate
# 
# #ground_truth_segment = np.zeros((len(df_speed)))
# 
# f_frame_list = list()
# 
# for i in range(len(f_frame)):
#     f_frame_list = f_frame_list + list(range(int(f_frame[i,0]),int(f_frame[i,1]+1)))
#         
# 
#
#         
# ground_truth_segment = np.zeros((df_array.shape[0]))        
# for i in range(len(f_frame_list)):
#     #print(i)
#     ground_truth_segment[f_frame_list[i]-1] = 1
# =============================================================================
#%%
    
    
ground_truth_file_dir = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\Ground_truth_segments_2020-12-03-RT3D-2.txt'

# Wrap up the ground truth part of the code
def get_ground_truth_data(ground_truth_file_dir):
    seconds_per_minute = 60
    f = open(ground_truth_file_dir, "r") 
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
        
    f_frame = f_second*frameRate
    
    #ground_truth_segment = np.zeros((len(df_speed)))
    
    f_frame_list = list()
    
    for i in range(len(f_frame)):
        f_frame_list = f_frame_list + list(range(int(f_frame[i,0]),int(f_frame[i,1]+1)))
        
    df_array = df.to_numpy()
        
    ground_truth_segment = np.zeros((df_array.shape[0]))        
    for i in range(len(f_frame_list)):
        #print(i)
        ground_truth_segment[f_frame_list[i]-1] = 1
        
    return f_frame_list, ground_truth_segment

f_frame_list, ground_truth_segment = get_ground_truth_data(ground_truth_file_dir)

    
    
#%% TEMP, check dropout rate of the 3D dataset
df_array = df_array[f_frame_list]

#%%
num_nans = 0
for i in range(df_array.shape[0]):
    for j in range(df_array.shape[1]):
        if np.isnan(df_array[i,j]):
            num_nans += 1
actual_num_nans = num_nans/(2*3) #x,y,z are needed, but the following 3 are not needed
total_points = len(df_array)*numMarkers #3 in terms of X,Y,Z
dropout_percentage = actual_num_nans/total_points
#print("Number of markers in all frames with NaNs (", num_nans)
print("Total points in all frames: ", len(df_array), "frames *", numMarkers, "markers =", total_points, "markers")
print("Total dropout markers (not total frames) ", actual_num_nans)
print("Marker dropout percentage", round(dropout_percentage*100,2), "%")


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

likelihood_array = df_array[:,[5,11,17,23,29,35,41,47]]

plt.figure()
plt.hist(likelihood_array)
plt.title("Likelihood Score of 3D data")
plt.show()

#%% Calculate number of dropout markers available to be interpolated

interpolatable_markers = []
interpolatable_markers_numbers = []
total_number_of_dropouts = []
percentage_of_interpolatable_dropouts_amongst_all_dropouts = []
percentage_of_interpolatable_dropouts_amongst_all_markers = []
interpolate_limit = 2 #consecutive frames

for i in range(likelihood_array.shape[1]): #8 markers
    interpolatable_frames = []
    total_number_of_dropouts_per_marker = 0
    j = 1
    while j < (likelihood_array.shape[0] - interpolate_limit - 1): #32302 frames in experiment phase
        #print(j)
        if np.isnan(likelihood_array[j,i]):
            total_number_of_dropouts_per_marker += 1
        if np.isnan(likelihood_array[j,i]) and ~np.isnan(likelihood_array[j-1,i]): #the first nan
            if ~np.isnan(likelihood_array[j+1,i]): #only one dropout frame
                interpolatable_frames.append(j)
            elif np.isnan(likelihood_array[j+1,i]) and ~np.isnan(likelihood_array[j+2,i]): #only two consecutive dropout frames
                interpolatable_frames.extend((j,j+1))
                j+=1
            #else: #both second and third consecutive frames are nans
            #    print("too many consecutive nans, not interpolatable")       
        j += 1
    print(interpolatable_frames)        
    interpolatable_markers.append(interpolatable_frames)
    interpolatable_markers_numbers.append(len(interpolatable_frames))
    total_number_of_dropouts.append(total_number_of_dropouts_per_marker)
    percentage_of_interpolatable_dropouts_amongst_all_dropouts.append((len(interpolatable_frames)/total_number_of_dropouts_per_marker)*100) #in percentage
    percentage_of_interpolatable_dropouts_amongst_all_markers.append((len(interpolatable_frames)/likelihood_array.shape[0])*100) #in percentage
    
#%% Calcuate percentage of interpolatable dropout markers amongst all dropout
    #markers, and all markers in total
print("Total number of interpolatable dropouts per marker:")
print("shoulder, elbow1, elbow2, wrist1, wrist2, hand1, hand2, hand3")
print(interpolatable_markers_numbers)
print('\n')

print("Total number of dropouts per marker:")
print("shoulder, elbow1, elbow2, wrist1, wrist2, hand1, hand2, hand3")
print(total_number_of_dropouts)
print('\n')

print("Percentage of interpolatable dropouts amongst all dropouts:")
print("shoulder, elbow1, elbow2, wrist1, wrist2, hand1, hand2, hand3")
print([round(num,4) for num in percentage_of_interpolatable_dropouts_amongst_all_dropouts])
print('\n')

print("Percentage of interpolatable dropouts amongst all markers:")
print("shoulder, elbow1, elbow2, wrist1, wrist2, hand1, hand2, hand3")
print([round(num,4) for num in percentage_of_interpolatable_dropouts_amongst_all_markers])
print('\n')
