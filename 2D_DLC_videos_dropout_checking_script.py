# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:14:54 2020

@author: dongq
"""
import pandas
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.image as mpimg

import cv2
import os
import math
import pandas as pd 
import scipy
from scipy.io import savemat

#%% Define if we want to look at experiment phase only or not

experiment_phase_only = 1

#%% Define function count_non_confident_points
"""
input:
    df: dataframe for 1 camera
    
output:
    df_dropouts (list): number of frames that have likelihood less than 0.9, 
    for each marker.
    
The function takes in the dataframe for 1 camera, iterates through the dataframe
to find the number of frames that have a likelihood value less than 0.9 for
each marker in the dataframe as a list, and returns this list.
"""

def count_non_confident_points(df):
    #df_head = df.head()
    #df_col_name = df.columns
    likelihood_limit = 0.9
    df_likelihood = df.xs('likelihood',level='coords',axis=1)
    df_likelihood_copy = df_likelihood
    df_dropouts = []
    #print(df_likelihood_copy)
    for i in df_likelihood_copy:
        df_likelihood_i = df_likelihood_copy.xs(i[1],level='bodyparts',axis=1)
        #print(df_likelihood_i[df_likelihood_i<likelihood_limit].count().values)
        dropout_nums = int(df_likelihood_i[df_likelihood_i<likelihood_limit].count().values)
        df_dropouts.append(dropout_nums)
    #Returns the number of dropouts for each marker of this camera 
    return df_dropouts


#%% Read in datasets and the file that contains the ground truth segmentation for this dataset (written by hand)

#cam1 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\exp00001DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
#cam2 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\exp00002DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
#cam3 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\exp00003DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
#cam4 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\exp00004DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')

#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\Ground_truth_segments_20200804_FR.txt", "r") 


#cam1 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\exp00001DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
#cam2 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\exp00002DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
#cam3 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\exp00003DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
#cam4 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\exp00004DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')

#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\Ground_truth_segments_20200804_RT.txt", "r") 

#cam1 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT3D\videos\exp_han_00009DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
#cam2 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT3D\videos\exp_han_00010DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
#cam3 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT3D\videos\exp_han_00011DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
#cam4 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT3D\videos\exp_han_00012DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')

#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT3D\videos\Ground_truth_segments_2020-09-22-RT3D.txt", "r") 

#cam1 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D-IncreasedLighting\videos\exp_han_00017_section_higher_lightingDLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
#cam2 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D-IncreasedLighting\videos\exp_han_00018_section_higher_lightingDLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
#cam3 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D-IncreasedLighting\videos\exp_han_00019_section_higher_lightingDLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
#cam4 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D-IncreasedLighting\videos\exp_han_00020_section_higher_lightingDLC_resnet50_HanSep22shuffle1_1030000filtered.h5')

#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D-IncreasedLighting\videos\Ground_truth_segments_2020-09-22-RT2D-IncreasedLighting.txt", "r") 

#cam1 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\videos\exp_han_00017DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
#cam2 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\videos\exp_han_00018DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
#cam3 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\videos\exp_han_00019DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
#cam4 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\videos\exp_han_00020DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')

#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\videos\Ground_truth_segments_2020-09-22-RT2D.txt", "r") 

#folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\videos'

#cam1 = pandas.read_hdf(folder + r'\Crackle_20201203_00001DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
#cam2 = pandas.read_hdf(folder + r'\Crackle_20201203_00002DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
#cam3 = pandas.read_hdf(folder + r'\Crackle_20201203_00003DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
#cam4 = pandas.read_hdf(folder + r'\Crackle_20201203_00004DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')

#cam1 = pandas.read_hdf(folder + r'\Crackle_20201203_00007DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
#cam2 = pandas.read_hdf(folder + r'\Crackle_20201203_00008DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
#cam3 = pandas.read_hdf(folder + r'\Crackle_20201203_00009DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
#cam4 = pandas.read_hdf(folder + r'\Crackle_20201203_00010DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')


folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\videos'

# =============================================================================
# cam1 = pandas.read_hdf(folder + r'\Crackle_20201203_00007DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Crackle_20201203_00008DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Crackle_20201203_00009DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Crackle_20201203_00010DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# 
# f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\Ground_truth_segments_2020-12-03-RT3D-2.txt", "r") 
# =============================================================================

cam1 = pandas.read_hdf(folder + r'\Crackle_20201203_00001DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
cam2 = pandas.read_hdf(folder + r'\Crackle_20201203_00002DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
cam3 = pandas.read_hdf(folder + r'\Crackle_20201203_00003DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
cam4 = pandas.read_hdf(folder + r'\Crackle_20201203_00004DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')

f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\Ground_truth_segments_2020-12-03-RT2D.txt", "r") 

# =============================================================================
# folder = r'D:\DLC_Folders_Currently_In_Use\Han-Qiwei-2020-08-04-FreeReaching\videos'
# 
# cam1 = pandas.read_hdf(folder + r'\exp00001DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\exp00002DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\exp00003DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\exp00004DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\videos'
# 
# cam1 = pandas.read_hdf(folder + r'\Han_20201204_00005DLC_resnet50_Han_202012Dec14shuffle1_1030000_filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Han_20201204_00006DLC_resnet50_Han_202012Dec14shuffle1_1030000_filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Han_20201204_00007DLC_resnet50_Han_202012Dec14shuffle1_1030000_filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Han_20201204_00008DLC_resnet50_Han_202012Dec14shuffle1_1030000_filtered.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\Han_20201204_RT3D_groundTruth.txt", "r")
# =============================================================================


# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\videos'
# 
# cam1 = pandas.read_hdf(folder + r'\Han_20201204_00005DLC_resnet50_Han_202012Dec14shuffle1_1030000_filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Han_20201204_00006DLC_resnet50_Han_202012Dec14shuffle1_1030000_filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Han_20201204_00007DLC_resnet50_Han_202012Dec14shuffle1_1030000_filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Han_20201204_00008DLC_resnet50_Han_202012Dec14shuffle1_1030000_filtered.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\Han_20201204_RT3D_groundTruth.txt", "r")
# =============================================================================

 




"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Remember to check this for EACH VIDEO cuz they'll vary
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
#frames_per_second = 25
frames_per_second = 24

store_Bento_data = False

#%% Define Trial segmenting function
"""
input:
    df: dataframe for one camera
    ground_truth_list: array, showing that for each frame in the camera, is 
    this frame in experiment phase (1) or not. In other words, is the monkey
    doing experiment in this frame or not.
    
output:
    df2: dataframe with experiment phase included only.

The function uses the ground_truth_list to segment the experiment phase in the
input dataframe, and then returns the result.
"""
def experiment_trial_segment(df,ground_truth_list):
    df2 = pd.DataFrame(np.zeros((0,df.shape[1])),columns = df.columns)
    for i in range(len(ground_truth_list)):
        df2.loc[i] = df.iloc[ground_truth_list[i]-1]
    #print(df2)
    return df2


#%% Get the array for trial segmentation
#if experiment_phase_only == 1:
    #df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')

seconds_per_minute = 60

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

f_frame_list = list()

for i in range(len(f_frame)):
    f_frame_list = f_frame_list + list(range(int(f_frame[i,0]),int(f_frame[i,1]+1)))
        
#%% get ground truth segment 
        
ground_truth_segment = np.zeros((cam1.shape[0]))        
for i in range(len(f_frame_list)):
    #print(i)
    ground_truth_segment[f_frame_list[i]-1] = 1
    
#%% Use the experiment_trial_segment() for each camera, to get experiment_only dataframe

#cam1_exp_only = experiment_trial_segment(cam1,f_frame_list)
#cam2_exp_only = experiment_trial_segment(cam2,f_frame_list)
#cam3_exp_only = experiment_trial_segment(cam3,f_frame_list)
#cam4_exp_only = experiment_trial_segment(cam4,f_frame_list)

cam1_exp_only = cam1.iloc[f_frame_list]
cam2_exp_only = cam2.iloc[f_frame_list]
cam3_exp_only = cam3.iloc[f_frame_list]
cam4_exp_only = cam4.iloc[f_frame_list]

#%%
cam1_exp_only = cam1
cam2_exp_only = cam2
cam3_exp_only = cam3
cam4_exp_only = cam4

#%% Use the count_non_confident_points() for each camera, both the whole dataset version and the experiment-only version.

cam1_dropouts_whole_dataset = count_non_confident_points(cam1)
cam2_dropouts_whole_dataset = count_non_confident_points(cam2)
cam3_dropouts_whole_dataset = count_non_confident_points(cam3)
cam4_dropouts_whole_dataset = count_non_confident_points(cam4)

cam1_dropouts = count_non_confident_points(cam1_exp_only)
cam2_dropouts = count_non_confident_points(cam2_exp_only)
cam3_dropouts = count_non_confident_points(cam3_exp_only)
cam4_dropouts = count_non_confident_points(cam4_exp_only)

print("\n Dropout frames of the whole dataset (row=each cam, col = each marker)")
print(cam1_dropouts_whole_dataset,'\n',cam2_dropouts_whole_dataset,'\n',cam3_dropouts_whole_dataset,'\n',cam4_dropouts_whole_dataset)

print("\n Dropout frames of experiment phae in the dataset (row=each cam, col = each marker)")
print(cam1_dropouts,'\n',cam2_dropouts,'\n',cam3_dropouts,'\n',cam4_dropouts)


#%% Define count_hard_to_see_points function
#https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

"""
input:
    cam1, cam2, cam3, cam4: marker positions for one marker throughout the 
    whole video, each for one camera

output:
    hard_to_see_points: an array representing the whole recorded video per
    frame, each frame with 0 meaning that this point is seen by at least 2
    cams with high likelihood, and 1 meaning this point is not seen by at
    least 2 cameras with high likelihood.
    
The function takes in one marker position throughout the whole dataset for
all 4 cameras, and decide that throughout the whole dataset, for each frame,
are there at least 2 camera views that have high likelihood values. If yes,
put 0 on this frame for this marker. If not, put 1.

***: And yes, this is a hard-coded function, in which I assume there are 4
cams and 4 cams only. I'm not a big fan of dataframe, but I'll try to see how
I can fix this.
"""

def count_hard_to_see_points(cam1, cam2, cam3, cam4):
    #for index,row in cam1.iterrows():
    #    print(row)
    cam1_array = cam1.values
    cam2_array = cam2.values
    cam3_array = cam3.values
    cam4_array = cam4.values
    cam_array_min_size = min(cam1_array.size, cam2_array.size, cam3_array.size,
                             cam4_array.size)
    print(cam_array_min_size)
    hard_to_see_points = np.zeros(cam1_array.size)
    print(cam_array_min_size)
    for i in range(cam_array_min_size):
        result1 = bool(float(cam1_array[i]) > 0.9)
        result2 = bool(cam2_array[i] > 0.9)
        result3 = bool(cam3_array[i] > 0.9)
        result4 = bool(cam4_array[i] > 0.9)
        if result1:
            cam1_seen = 1
        else:
            cam1_seen = 0
        if result2:
            cam2_seen = 1
        else:
            cam2_seen = 0
        if result3:
            cam3_seen = 1
        else:
            cam3_seen = 0
        if result4:
            cam4_seen = 1
        else:
            cam4_seen = 0
            
        if cam1_seen + cam2_seen + cam3_seen + cam4_seen < 3:
            hard_to_see_points[i] = 1
    return hard_to_see_points
            
            
    
#%% How often do I not have two cameras seeing one marker
    
cam1_likelihood = cam1_exp_only.xs('likelihood',level='coords',axis=1)
cam2_likelihood = cam2_exp_only.xs('likelihood',level='coords',axis=1)
cam3_likelihood = cam3_exp_only.xs('likelihood',level='coords',axis=1)
cam4_likelihood = cam4_exp_only.xs('likelihood',level='coords',axis=1)

shoulder1_cam1 = cam1_likelihood.xs('shoulder1',level='bodyparts',axis=1)
shoulder1_cam2 = cam2_likelihood.xs('shoulder1',level='bodyparts',axis=1)
shoulder1_cam3 = cam3_likelihood.xs('shoulder1',level='bodyparts',axis=1)
shoulder1_cam4 = cam4_likelihood.xs('shoulder1',level='bodyparts',axis=1)
shoulder1_unseen_points = count_hard_to_see_points(shoulder1_cam1,shoulder1_cam2,shoulder1_cam3,shoulder1_cam4)
# =============================================================================
# 
# shoulder1_cam1 = cam1_likelihood.xs('shoulder',level='bodyparts',axis=1)
# shoulder1_cam2 = cam2_likelihood.xs('shoulder',level='bodyparts',axis=1)
# shoulder1_cam3 = cam3_likelihood.xs('shoulder',level='bodyparts',axis=1)
# shoulder1_cam4 = cam4_likelihood.xs('shoulder',level='bodyparts',axis=1)
# shoulder1_unseen_points = count_hard_to_see_points(shoulder1_cam1,shoulder1_cam2,shoulder1_cam3,shoulder1_cam4)
# =============================================================================



# =============================================================================
# arm1_cam1 =  cam1_likelihood.xs('arm1',level='bodyparts',axis=1)
# arm1_cam2 =  cam2_likelihood.xs('arm1',level='bodyparts',axis=1)
# arm1_cam3 =  cam3_likelihood.xs('arm1',level='bodyparts',axis=1)
# arm1_cam4 =  cam4_likelihood.xs('arm1',level='bodyparts',axis=1)
# arm1_unseen_points = count_hard_to_see_points(arm1_cam1,arm1_cam2,arm1_cam3,arm1_cam4)
# 
# arm2_cam1 =  cam1_likelihood.xs('arm2',level='bodyparts',axis=1)
# arm2_cam2 =  cam2_likelihood.xs('arm2',level='bodyparts',axis=1)
# arm2_cam3 =  cam3_likelihood.xs('arm2',level='bodyparts',axis=1)
# arm2_cam4 =  cam4_likelihood.xs('arm2',level='bodyparts',axis=1)
# arm2_unseen_points = count_hard_to_see_points(arm2_cam1,arm2_cam2,arm2_cam3,arm2_cam4)
# =============================================================================

elbow1_cam1 = cam1_likelihood.xs('elbow1',level='bodyparts',axis=1)
elbow1_cam2 = cam2_likelihood.xs('elbow1',level='bodyparts',axis=1)
elbow1_cam3 = cam3_likelihood.xs('elbow1',level='bodyparts',axis=1)
elbow1_cam4 = cam4_likelihood.xs('elbow1',level='bodyparts',axis=1)
elbow1_unseen_points = count_hard_to_see_points(elbow1_cam1,elbow1_cam2,elbow1_cam3,elbow1_cam4)

elbow2_cam1 = cam1_likelihood.xs('elbow2',level='bodyparts',axis=1)
elbow2_cam2 = cam2_likelihood.xs('elbow2',level='bodyparts',axis=1)
elbow2_cam3 = cam3_likelihood.xs('elbow2',level='bodyparts',axis=1)
elbow2_cam4 = cam4_likelihood.xs('elbow2',level='bodyparts',axis=1)
elbow2_unseen_points = count_hard_to_see_points(elbow2_cam1,elbow2_cam2,elbow2_cam3,elbow2_cam4)

wrist1_cam1 = cam1_likelihood.xs('wrist1',level='bodyparts',axis=1)
wrist1_cam2 = cam2_likelihood.xs('wrist1',level='bodyparts',axis=1)
wrist1_cam3 = cam3_likelihood.xs('wrist1',level='bodyparts',axis=1)
wrist1_cam4 = cam4_likelihood.xs('wrist1',level='bodyparts',axis=1)
wrist1_unseen_points = count_hard_to_see_points(wrist1_cam1,wrist1_cam2,wrist1_cam3,wrist1_cam4)

wrist2_cam1 = cam1_likelihood.xs('wrist2',level='bodyparts',axis=1)
wrist2_cam2 = cam2_likelihood.xs('wrist2',level='bodyparts',axis=1)
wrist2_cam3 = cam3_likelihood.xs('wrist2',level='bodyparts',axis=1)
wrist2_cam4 = cam4_likelihood.xs('wrist2',level='bodyparts',axis=1)
wrist2_unseen_points = count_hard_to_see_points(wrist2_cam1,wrist2_cam2,wrist2_cam3,wrist2_cam4)

hand1_cam1 = cam1_likelihood.xs('hand1',level='bodyparts',axis=1)
hand1_cam2 = cam2_likelihood.xs('hand1',level='bodyparts',axis=1)
hand1_cam3 = cam3_likelihood.xs('hand1',level='bodyparts',axis=1)
hand1_cam4 = cam4_likelihood.xs('hand1',level='bodyparts',axis=1)
hand1_unseen_points = count_hard_to_see_points(hand1_cam1,hand1_cam2,hand1_cam3,hand1_cam4)

hand2_cam1 = cam1_likelihood.xs('hand2',level='bodyparts',axis=1)
hand2_cam2 = cam2_likelihood.xs('hand2',level='bodyparts',axis=1)
hand2_cam3 = cam3_likelihood.xs('hand2',level='bodyparts',axis=1)
hand2_cam4 = cam4_likelihood.xs('hand2',level='bodyparts',axis=1)
hand2_unseen_points = count_hard_to_see_points(hand2_cam1,hand2_cam2,hand2_cam3,hand2_cam4)

hand3_cam1 = cam1_likelihood.xs('hand3',level='bodyparts',axis=1)
hand3_cam2 = cam2_likelihood.xs('hand3',level='bodyparts',axis=1)
hand3_cam3 = cam3_likelihood.xs('hand3',level='bodyparts',axis=1)
hand3_cam4 = cam4_likelihood.xs('hand3',level='bodyparts',axis=1)
hand3_unseen_points = count_hard_to_see_points(hand3_cam1,hand3_cam2,hand3_cam3,hand3_cam4)

#%% Set the plotting parameters

font = {'family' : 'normal',
#        'weight' : 'bold',
        'size'   : 8}

font_medium = {'family' : 'normal',
 #       'weight' : 'bold',
        'size'   : 8}

#%% Plot line plots for the likelihood distribution of all 4 cameras

# =============================================================================
# marker_name_list = ['shoulder1','arm1','arm2','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
# marker_pos_list_cam1 = [shoulder1_cam1,arm1_cam1,arm2_cam1,elbow1_cam1,elbow2_cam1,wrist1_cam1,wrist2_cam1,hand1_cam1,hand2_cam1,hand3_cam1]
# marker_pos_list_cam2 = [shoulder1_cam2,arm1_cam2,arm2_cam2,elbow1_cam2,elbow2_cam2,wrist1_cam2,wrist2_cam2,hand1_cam2,hand2_cam2,hand3_cam2]
# marker_pos_list_cam3 = [shoulder1_cam3,arm1_cam3,arm2_cam3,elbow1_cam3,elbow2_cam3,wrist1_cam3,wrist2_cam3,hand1_cam3,hand2_cam3,hand3_cam3]
# marker_pos_list_cam4 = [shoulder1_cam4,arm1_cam4,arm2_cam4,elbow1_cam4,elbow2_cam4,wrist1_cam4,wrist2_cam4,hand1_cam4,hand2_cam4,hand3_cam4]
# =============================================================================

marker_name_list = ['shoulder1','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
marker_pos_list_cam1 = [shoulder1_cam1,elbow1_cam1,elbow2_cam1,wrist1_cam1,wrist2_cam1,hand1_cam1,hand2_cam1,hand3_cam1]
marker_pos_list_cam2 = [shoulder1_cam2,elbow1_cam2,elbow2_cam2,wrist1_cam2,wrist2_cam2,hand1_cam2,hand2_cam2,hand3_cam2]
marker_pos_list_cam3 = [shoulder1_cam3,elbow1_cam3,elbow2_cam3,wrist1_cam3,wrist2_cam3,hand1_cam3,hand2_cam3,hand3_cam3]
marker_pos_list_cam4 = [shoulder1_cam4,elbow1_cam4,elbow2_cam4,wrist1_cam4,wrist2_cam4,hand1_cam4,hand2_cam4,hand3_cam4]


#marker_pos_list = [marker_pos_list_cam1,marker_pos_list_cam2,marker_pos_list_cam3,marker_pos_list_cam4]
#X_axis_ref_list = [arm1_cam1,arm1_cam2,arm1_cam3,arm1_cam4]


marker_pos_list = [marker_pos_list_cam1,marker_pos_list_cam2,marker_pos_list_cam3,marker_pos_list_cam4]
X_axis_ref_list = [elbow1_cam1,elbow1_cam2,elbow1_cam3,elbow1_cam4]

for i in range(len(marker_pos_list)):
    for j in range(len(marker_pos_list[i])):
        marker_pos_list[i][j] = marker_pos_list[i][j].values.tolist()


"""
!!!: If you want to plot a section of the data, set the start and end frame number here
"""
start_frame = 0
end_frame = len(marker_pos_list_cam4[0])

#Experiment phase
#start_frame = 370*frames_per_second
#end_frame = 400*frames_per_second

#Non-exp phase
#start_frame = 510*frames_per_second
#end_frame = 540*frames_per_second

#Just randomly first 20 seconds
#start_frame = 0
#end_frame = 20 *frames_per_second

num_of_markers = len(marker_name_list)

overall_likelihood = []

#fig, ax = plt.subplots(nrows=11, ncols=1)


for j in range (len(marker_pos_list)):
    
    X = np.linspace(0,X_axis_ref_list[j].shape[0]-1,X_axis_ref_list[j].shape[0])/25
    fig=plt.figure(figsize=(8,20))
    for i in range(num_of_markers):
        if i == 1:
            plt.title("20201203_Crackle_RT3D_Cam" + str(j+1) + "_DLC_Tracing_Probability",fontsize=20)
            #plt.title("20200922_Han_RT2D_Cam" + str(j+1) + "_DLC_Tracing_Probability",fontsize=20)
            #plt.title("DLC Tracking Probability " + "cam " + str() ,fontsize=20)
        
        ax = plt.subplot(num_of_markers+1,1,i+1)
        #ax = fig.add_subplot(num_of_markers+1,1,i+1)
        
        ax.plot(X[start_frame:end_frame],marker_pos_list[j][i][start_frame:end_frame])
        #overall_likelihood.append(marker_pos_list[j][i][start_frame:end_frame])
        if np.mean(marker_pos_list[j][i][start_frame:end_frame]) > 0.5:
            overall_likelihood += marker_pos_list[j][i][start_frame:end_frame]
        
        plt.ylim(-0.05,1.05)
        plt.ylabel(marker_name_list[i],fontsize=20)
        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        labelsize=16) 
    #ax = plt.subplot(num_of_markers+1,1,num_of_markers+1)
    
    #ax.plot(X[start_frame:end_frame],ground_truth_segment[start_frame:end_frame],c='r')
    
    #plt.ylim(-0.05,1.05)
    plt.tick_params(axis='x',which='both',labelbottom=True,labelsize=14)
    #plt.ylabel("truth",fontsize=20)
    plt.xlabel("seconds",fontsize=20)


#%% Plot overall likelihood histogram (for all 4 cameras and all the markers)
import random
import matplotlib.ticker as ticker
#choose 1/20 of the overall_likelihood list
overall_likelihood_shrinked_size = int(len(overall_likelihood)/10)
overall_likelihood_shrinked_choice = random.sample(range(len(overall_likelihood)),overall_likelihood_shrinked_size)
overall_likelihood = np.array(overall_likelihood)
overall_likelihood_shrinked = overall_likelihood[overall_likelihood_shrinked_choice]


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(12,7))
plt.hist(overall_likelihood_shrinked, bins=20)
ax.yaxis.set_major_locator(ticker.MultipleLocator(int(overall_likelihood_shrinked_size/5)))
#ax.yaxis.set_minor_locator(ticker.MultipleLocator())
ax.yaxis.set_major_formatter(ticker.PercentFormatter(overall_likelihood_shrinked_size))
ax.set_xlabel("likelihood")
ax.set_ylabel("percentage of landmarks")
ax.set_ylabel("number of landmarks")
vals = ax.get_yticks()
#ax.set_yticklabels(['{:.0%}'.format(x) for x in vals])
#ax.set_yscale('log')
#ax.set_title("Percentage of landmarks with certain likelihood")

    



#%% For each marker, find best two likelihood among 4 cams


#parameter using: 
#marker_pos_list_cam1: a list of lists 10(markers)*length_of_recording
#marker_pos_list_cam2    
#marker_pos_list_cam3    
#marker_pos_list_cam4
    
len_X = len(marker_pos_list_cam1[1]) #which markers
len_Y = len(marker_pos_list_cam1) #which frame

likelihood_best = np.zeros((len_X, len_Y))
likelihood_best_cam = np.zeros((len_X, len_Y))
likelihood_second_best = np.zeros((len_X, len_Y))
likelihood_second_best_cam = np.zeros((len_X, len_Y))
i=0

#Find the array with the least number of markers (usually it's going to be a difference of 1 to 2 frames)
min_len_marker_pos_list = min(len(marker_pos_list_cam1[0]),
                              len(marker_pos_list_cam2[0]),
                              len(marker_pos_list_cam3[0]),
                              len(marker_pos_list_cam4[0]))

for i in range(len(marker_pos_list_cam1)): #iter of 10
    for j in range(min_len_marker_pos_list): #iter of frames (length_of_recording)
        #hard_coding, assuming only 4 cameras
        temp_list = np.array([[marker_pos_list_cam1[i][j], 
                               marker_pos_list_cam2[i][j], 
                               marker_pos_list_cam3[i][j], 
                               marker_pos_list_cam4[i][j]],
                              [1,2,3,4]])
        
        sorted_temp_list = temp_list[:,np.argsort(temp_list[0,:])]
        for k in range(len(sorted_temp_list[0,:])):
            sorted_temp_list[0,k] = sorted_temp_list[0,k][0]
        #sorted_temp_list[0,:] = [float(i) for i in sorted_temp_list[0,:]]
        
        print(str(sorted_temp_list))
        
        likelihood_best[j,i] = sorted_temp_list[0,-1]
        likelihood_second_best[j,i] = sorted_temp_list[0,-2]
        likelihood_best_cam[j,i] = sorted_temp_list[1,-1]
        likelihood_second_best_cam[j,i] = sorted_temp_list[1,-2]
        
        print(str(i) + " " + str(j))
        #print(str(sorted_temp_list[-1]) + " " + str( sorted_temp_list_cam_nums[-1]))
        #print(str( sorted_temp_list[-2]) + " " + str(sorted_temp_list_cam_nums[-2]) + "\n")
        
        #print(str(sorted_temp_list_cam_nums) + '\n')
        
        
#%% Plot these data
        

#fig, ax = plt.subplots(nrows=11, ncols=1)

#plot_dataset = likelihood_best

plot_dataset = likelihood_second_best

start_frame = 0
end_frame = len(plot_dataset)

if likelihood_best.shape[1] == 10:
    print("yes")
    likelihood_best = likelihood_best[:,(0,3,4,5,6,7,8,9)]
    likelihood_best_cam = likelihood_best_cam[:,(0,3,4,5,6,7,8,9)]
    likelihood_second_best = likelihood_second_best[:,(0,3,4,5,6,7,8,9)]
    likelihood_second_best_cam = likelihood_second_best_cam[:,(0,3,4,5,6,7,8,9)]

marker_name_list = ['shoulder1','elbow1','elbow2','wrist1','wrist2',
                    'hand1','hand2','hand3']
marker_pos_list_cam1 = [shoulder1_cam1,elbow1_cam1,elbow2_cam1,wrist1_cam1,
                        wrist2_cam1,hand1_cam1,hand2_cam1,hand3_cam1]
marker_pos_list_cam2 = [shoulder1_cam2,elbow1_cam2,elbow2_cam2,wrist1_cam2,
                        wrist2_cam2,hand1_cam2,hand2_cam2,hand3_cam2]
marker_pos_list_cam3 = [shoulder1_cam3,elbow1_cam3,elbow2_cam3,wrist1_cam3,
                        wrist2_cam3,hand1_cam3,hand2_cam3,hand3_cam3]
marker_pos_list_cam4 = [shoulder1_cam4,elbow1_cam4,elbow2_cam4,wrist1_cam4,
                        wrist2_cam4,hand1_cam4,hand2_cam4,hand3_cam4]


num_of_markers = len(marker_name_list)

X = np.linspace(0,plot_dataset.shape[0]-1,plot_dataset.shape[0])/frames_per_second
#fig=plt.figure(figsize=(8,20))
fig=plt.figure(figsize=(16,6))
for i in range(num_of_markers):
    if i == 1:
        #plt.title("Top_1_DLC_Tracing_Likelihood",fontsize=20)
        plt.title("Top_2_DLC_Tracing_Likelihood",fontsize=20)
        #plt.title("20200922_Han_RT2D_Cam" + str(j+1) + "_DLC_Tracing_Probability",fontsize=20)
        #plt.title("DLC Tracking Probability " + "cam " + str() ,fontsize=20)
    #if i == 1 or i == 2:
    #    continue
    
    ax = plt.subplot(int((num_of_markers+1)/2),2,i+1)
    #ax = fig.add_subplot(num_of_markers+1,1,i+1)
    
    #ax.plot(X[start_frame:end_frame],plot_dataset[start_frame:end_frame,i])
    #ax.scatter(X[start_frame:end_frame],likelihood_second_best_cam[start_frame:end_frame,i]/4,alpha=0.2)
    ax.hist(likelihood_best[start_frame:end_frame,i],bins=10,alpha=.8)
    ax.hist(likelihood_second_best[start_frame:end_frame,i],bins=10,alpha=.5)
    #ax.set_yscale('log')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(likelihood_best.shape[0]))
    
    #plt.ylim(-0.05,1.05)
    plt.ylabel(marker_name_list[i],fontsize=20)
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False, # labels along the bottom edge are off
    labelsize=16) 
#ax = plt.subplot(num_of_markers+1,1,num_of_markers+1)

#ax.plot(X[start_frame:end_frame],ground_truth_segment[start_frame:end_frame],c='r')

#plt.ylim(-0.05,1.05)
plt.tick_params(axis='x',which='both',labelbottom=True,labelsize=14)
#plt.ylabel("truth",fontsize=20)
plt.xlabel("likelihood",fontsize=20)
        
#%%
#fig=plt.figure()

#plt.boxplot(likelihood_best)

#%% Percentage of top-two dropouts.

#best_shoulder = likelihood_best[:,0]
#best_shoulder_bad = best_shoulder[best_shoulder <=10]

likelihood_best_percentage = 100 - ((sum(likelihood_best < 0.9)/
                                     sum(likelihood_best <= 2))*100)
likelihood_second_best_percentage  = 100 - ((sum(likelihood_second_best < 0.9)/
                                             sum(likelihood_second_best <= 2))*100)
#The highest likelihood is 1, so <=2 contains all
print(likelihood_best_percentage)
print(likelihood_second_best_percentage)


#%% Get the camera numbers with low likelihood values for the best and second best likelihood

likelihood_best_low_likelihood_cams = likelihood_best_cam*(likelihood_best < 0.1)
likelihood_second_best_low_likelihood_cams = likelihood_second_best_cam*(likelihood_second_best < 0.1)

#%% Store these data using the standard from Bento
#Reference: https://github.com/annkennedy/bento/wiki/Support-for-Neural-Recording-Data
if store_Bento_data == True:
    """
    From nd array to dataframe, and then from dataframe to dictionary
    https://blog.csdn.net/rosefun96/article/details/78970239
    """
    rows_int = np.array(range(likelihood_best.shape[0]))
    rows_str = rows_int.astype(str)
    col = ['shoulder','arm1','arm2','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
    
    likelihood_best_df = pd.DataFrame(likelihood_best, index = rows_str, columns = col)
    likelihood_best_cam_df = pd.DataFrame(likelihood_best_cam, index = rows_str, columns = col)
    likelihood_second_best_df = pd.DataFrame(likelihood_second_best, index = rows_str, columns = col)
    likelihood_second_best_cam_df = pd.DataFrame(likelihood_second_best_cam, index = rows_str, columns = col)
    
    likelihood_best_dict = likelihood_best_df.to_dict(orient="list")
    likelihood_best_cam_dict = likelihood_best_cam_df.to_dict(orient="list")
    likelihood_second_best_dict = likelihood_second_best_df.to_dict(orient="list")
    likelihood_second_best_cam_dict = likelihood_second_best_cam_df.to_dict(orient="list")
    
    file_dir = folder + r'\best_2_likelihood'
    
    likelihood_best_name = r'\likelihood_best.mat'
    likelihood_best_cam_name = r'\likelihood_best_cam.mat'
    likelihood_second_best_name = r'\likelihood_second_best.mat'
    likelihood_second_best_cam_name = r'\likelihood_second_best_cam.mat'
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        
    savemat(file_dir + likelihood_best_name, likelihood_best_dict)
    savemat(file_dir + likelihood_best_cam_name, likelihood_best_cam_dict)
    savemat(file_dir + likelihood_second_best_name, likelihood_second_best_dict)
    savemat(file_dir + likelihood_second_best_cam_name, likelihood_second_best_cam_dict)

#%% 












#%% plot these points above
# =============================================================================
# fig, ax = plt.subplots()
# ax.plot(shoulder1_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for shoulder1')
# fig.savefig('unseen_points_shoulder1.png')
# plt.show()
# 
# fig, ax = plt.subplots()
# ax.plot(arm1_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for arm1')
# fig.savefig('unseen_points_arm1.png')
# plt.show()
# 
# fig, ax = plt.subplots()
# ax.plot(arm2_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for arm2')
# fig.savefig('unseen_points_arm2.png')
# plt.show()
# 
# fig, ax = plt.subplots()
# ax.plot(elbow1_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for elbow1')
# fig.savefig('unseen_points_elbow1.png')
# plt.show()
# 
# fig, ax = plt.subplots()
# ax.plot(elbow2_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for elbow2')
# fig.savefig('unseen_points_elbow2.png')
# plt.show()
# 
# fig, ax = plt.subplots()
# ax.plot(wrist1_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for wrist1')
# fig.savefig('unseen_points_wrist1.png')
# plt.show()
# 
# fig, ax = plt.subplots()
# ax.plot(wrist2_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for wrist2')
# fig.savefig('unseen_points_wrist2.png')
# plt.show()
# 
# fig, ax = plt.subplots()
# ax.plot(hand1_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for hand1')
# fig.savefig('unseen_points_hand1.png')
# plt.show()
# 
# fig, ax = plt.subplots()
# ax.plot(hand2_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for hand2')
# fig.savefig('unseen_points_hand2.png')
# plt.show()
# 
# fig, ax = plt.subplots()
# ax.plot(hand3_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for hand3')
# fig.savefig('unseen_points_hand3.png')
# plt.show()
# =============================================================================

#%% XXX Try histogram plot
# =============================================================================
# fig, ax = plt.subplots()
# ax.hist(hand3_unseen_points)
# ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
#        title='Points not seen by 2 cams at once for hand3')
# fig.savefig('unseen_points_hand3.png')
# plt.show()
# =============================================================================

#%% XXX Try pie chart
# =============================================================================
# labels = 'Seen Points','Unseen Points'
# sizes = [np.count_nonzero(hand3_unseen_points==0),np.count_nonzero(hand3_unseen_points==1)]
# explode = (0,0.1)
# 
# fig1,ax1 = plt.subplots()
# ax1.pie(sizes,explode=explode,labels=labels,autopct='%1.1f%%',
#         shadow=True, startangle=90)
# ax1.axis('equal')
# plt.title('Points not seen by 2 cams at once for hand3')
# plt.show()
# =============================================================================
# =============================================================================
# fig, ax = plt.subplots()
# ax.plot(t, s)
# 
# ax.set(xlabel='time (s)', ylabel='voltage (mV)',
#        title='About as simple as it gets, folks')
# ax.grid()
# 
# fig.savefig("test.png")
# plt.show()
# =============================================================================

#%% XXX
# =============================================================================
# print(sum(shoulder1_unseen_points),
# sum(arm1_unseen_points),
# sum(arm2_unseen_points),
# sum(elbow1_unseen_points),
# sum(elbow2_unseen_points),
# sum(wrist1_unseen_points),
# sum(wrist2_unseen_points),
# sum(hand1_unseen_points),
# sum(hand2_unseen_points),
# sum(hand3_unseen_points))
# 
# print(sum(shoulder1_unseen_points)/cam1.shape[0],
# sum(arm1_unseen_points)/cam1.shape[0],
# sum(arm2_unseen_points)/cam1.shape[0],
# sum(elbow1_unseen_points)/cam1.shape[0],
# sum(elbow2_unseen_points)/cam1.shape[0],
# sum(wrist1_unseen_points)/cam1.shape[0],
# sum(wrist2_unseen_points)/cam1.shape[0],
# sum(hand1_unseen_points)/cam1.shape[0],
# sum(hand2_unseen_points)/cam1.shape[0],
# sum(hand3_unseen_points)/cam1.shape[0])
# =============================================================================

#%% XXX
"""
0:shoulder1
1:arm1
2:arm2
3:elbow1
4:elbow2
5:wrist1
6:wrist2
7:hand1
8:hand2
9:hand3
"""

#%% Functions for determing which frames are seen by only one single camera for each Marker
# =============================================================================
# 
# def label_hard_to_see_points(cam1, cam2, cam3):
#     cam1_array = cam1.values
#     cam2_array = cam2.values
#     cam3_array = cam3.values
#     hard_to_see_points_cam1 = np.zeros(cam1_array.size)
#     hard_to_see_points_cam2 = np.zeros(cam2_array.size)
#     hard_to_see_points_cam3 = np.zeros(cam3_array.size)
#     
#     for i in range(cam1_array.size):
#         result1 = bool(cam1_array[i] > 0.9)
#         result2 = bool(cam2_array[i] > 0.9)
#         result3 = bool(cam3_array[i] > 0.9)
#         if result1 == True and result2 == False and result3 == False:
#             hard_to_see_points_cam1[i] = 1
#         if result2 == True and result1 == False and result3 == False:
#             hard_to_see_points_cam2[i] = 1
#         if result3 == True and result1 == False and result2 == False:
#             hard_to_see_points_cam3[i] = 1
#     return hard_to_see_points_cam1, hard_to_see_points_cam2, hard_to_see_points_cam3
# =============================================================================

#%%Plot unseen frames for each Marker in each camera

# =============================================================================
# shoulder1_unseen_points_cam1,shoulder1_unseen_points_cam2,shoulder1_unseen_points_cam3  = label_hard_to_see_points(shoulder1_cam1,shoulder1_cam2,shoulder1_cam3)
# shoulder1_unseen_points_cam1_loc = [i for i,x in enumerate(shoulder1_unseen_points_cam1) if x == 1]
# shoulder1_unseen_points_cam2_loc = [i for i,x in enumerate(shoulder1_unseen_points_cam2) if x == 1]
# shoulder1_unseen_points_cam3_loc = [i for i,x in enumerate(shoulder1_unseen_points_cam3) if x == 1]
# 
# arm1_unseen_points_cam1, arm1_unseen_points_cam2, arm1_unseen_points_cam3 = label_hard_to_see_points(arm1_cam1,arm1_cam2,arm1_cam3)
# arm1_unseen_points_cam1_loc = [i for i,x in enumerate(arm1_unseen_points_cam1) if x == 1]
# arm1_unseen_points_cam2_loc = [i for i,x in enumerate(arm1_unseen_points_cam2) if x == 1]
# arm1_unseen_points_cam3_loc = [i for i,x in enumerate(arm1_unseen_points_cam3) if x == 1]
# 
# arm2_unseen_points_cam1, arm2_unseen_points_cam2, arm2_unseen_points_cam3 = label_hard_to_see_points(arm2_cam1,arm2_cam2,arm2_cam3)
# arm2_unseen_points_cam1_loc = [i for i,x in enumerate(arm2_unseen_points_cam1) if x == 1]
# arm2_unseen_points_cam2_loc = [i for i,x in enumerate(arm2_unseen_points_cam2) if x == 1]
# arm2_unseen_points_cam3_loc = [i for i,x in enumerate(arm2_unseen_points_cam3) if x == 1]
# 
# elbow1_unseen_points_cam1, elbow1_unseen_points_cam2, elbow1_unseen_points_cam3 = label_hard_to_see_points(elbow1_cam1,elbow1_cam2,elbow1_cam3)
# elbow1_unseen_points_cam1_loc = [i for i,x in enumerate(elbow1_unseen_points_cam1) if x == 1]
# elbow1_unseen_points_cam2_loc = [i for i,x in enumerate(elbow1_unseen_points_cam2) if x == 1]
# elbow1_unseen_points_cam3_loc = [i for i,x in enumerate(elbow1_unseen_points_cam3) if x == 1]
# 
# elbow2_unseen_points_cam1, elbow2_unseen_points_cam2, elbow2_unseen_points_cam3 = label_hard_to_see_points(elbow2_cam1,elbow2_cam2,elbow2_cam3)
# elbow2_unseen_points_cam1_loc = [i for i,x in enumerate(elbow2_unseen_points_cam1) if x == 1]
# elbow2_unseen_points_cam2_loc = [i for i,x in enumerate(elbow2_unseen_points_cam2) if x == 1]
# elbow2_unseen_points_cam3_loc = [i for i,x in enumerate(elbow2_unseen_points_cam3) if x == 1]
# 
# wrist1_unseen_points_cam1, wrist1_unseen_points_cam2, wrist1_unseen_points_cam3 = label_hard_to_see_points(wrist1_cam1,wrist1_cam2,wrist1_cam3)
# wrist1_unseen_points_cam1_loc = [i for i,x in enumerate(wrist1_unseen_points_cam1) if x == 1]
# wrist1_unseen_points_cam2_loc = [i for i,x in enumerate(wrist1_unseen_points_cam2) if x == 1]
# wrist1_unseen_points_cam3_loc = [i for i,x in enumerate(wrist1_unseen_points_cam3) if x == 1]
# 
# wrist2_unseen_points_cam1, wrist2_unseen_points_cam2, wrist2_unseen_points_cam3 = label_hard_to_see_points(wrist2_cam1,wrist2_cam2,wrist2_cam3)
# wrist2_unseen_points_cam1_loc = [i for i,x in enumerate(wrist2_unseen_points_cam1) if x == 1]
# wrist2_unseen_points_cam2_loc = [i for i,x in enumerate(wrist2_unseen_points_cam2) if x == 1]
# wrist2_unseen_points_cam3_loc = [i for i,x in enumerate(wrist2_unseen_points_cam3) if x == 1]
# 
# hand1_unseen_points_cam1, hand1_unseen_points_cam2, hand1_unseen_points_cam3 = label_hard_to_see_points(hand1_cam1,hand1_cam2,hand1_cam3)
# hand1_unseen_points_cam1_loc = [i for i,x in enumerate(hand1_unseen_points_cam1) if x == 1]
# hand1_unseen_points_cam2_loc = [i for i,x in enumerate(hand1_unseen_points_cam2) if x == 1]
# hand1_unseen_points_cam3_loc = [i for i,x in enumerate(hand1_unseen_points_cam3) if x == 1]
# 
# hand2_unseen_points_cam1, hand2_unseen_points_cam2, hand2_unseen_points_cam3 = label_hard_to_see_points(hand2_cam1,hand2_cam2,hand2_cam3)
# hand2_unseen_points_cam1_loc = [i for i,x in enumerate(hand2_unseen_points_cam1) if x == 1]
# hand2_unseen_points_cam2_loc = [i for i,x in enumerate(hand2_unseen_points_cam2) if x == 1]
# hand2_unseen_points_cam3_loc = [i for i,x in enumerate(hand2_unseen_points_cam3) if x == 1]
# 
# hand3_unseen_points_cam1, hand3_unseen_points_cam2, hand3_unseen_points_cam3 = label_hard_to_see_points(hand3_cam1,hand3_cam2,hand3_cam3)
# hand3_unseen_points_cam1_loc = [i for i,x in enumerate(hand3_unseen_points_cam1) if x == 1]
# hand3_unseen_points_cam2_loc = [i for i,x in enumerate(hand3_unseen_points_cam2) if x == 1]
# hand3_unseen_points_cam3_loc = [i for i,x in enumerate(hand3_unseen_points_cam3) if x == 1]
# =============================================================================

#%%Print the numbers and check if they're right or not
# =============================================================================
# print(shoulder1_unseen_points_cam1)
# print(shoulder1_unseen_points_cam2)
# print(shoulder1_unseen_points_cam3)
# print(shoulder1_unseen_points_cam1_loc)
# print(shoulder1_unseen_points_cam2_loc)
# print(shoulder1_unseen_points_cam3_loc)
# 
# print(arm1_unseen_points_cam1)
# print(arm1_unseen_points_cam2)
# print(arm1_unseen_points_cam3)
# print(arm1_unseen_points_cam1_loc)
# print(arm1_unseen_points_cam2_loc)
# print(arm1_unseen_points_cam3_loc)
# 
# print(arm2_unseen_points_cam1)
# print(arm2_unseen_points_cam2)
# print(arm2_unseen_points_cam3)
# print(arm2_unseen_points_cam1_loc)
# print(arm2_unseen_points_cam2_loc)
# print(arm2_unseen_points_cam3_loc)
# 
# print(elbow1_unseen_points_cam1)
# print(elbow1_unseen_points_cam2)
# print(elbow1_unseen_points_cam3)
# print(elbow1_unseen_points_cam1_loc)
# print(elbow1_unseen_points_cam2_loc)
# print(elbow1_unseen_points_cam3_loc)
# 
# print(elbow2_unseen_points_cam1)
# print(elbow2_unseen_points_cam2)
# print(elbow2_unseen_points_cam3)
# print(elbow2_unseen_points_cam1_loc)
# print(elbow2_unseen_points_cam2_loc)
# print(elbow2_unseen_points_cam3_loc)
# 
# print(wrist1_unseen_points_cam1)
# print(wrist1_unseen_points_cam2)
# print(wrist1_unseen_points_cam3)
# print(wrist1_unseen_points_cam1_loc)
# print(wrist1_unseen_points_cam2_loc)
# print(wrist1_unseen_points_cam3_loc)
# 
# print(wrist2_unseen_points_cam1)
# print(wrist2_unseen_points_cam2)
# print(wrist2_unseen_points_cam3)
# print(wrist2_unseen_points_cam1_loc)
# print(wrist2_unseen_points_cam2_loc)
# print(wrist2_unseen_points_cam3_loc)
# 
# print(hand1_unseen_points_cam1)
# print(hand1_unseen_points_cam2)
# print(hand1_unseen_points_cam3)
# print(hand1_unseen_points_cam1_loc)
# print(hand1_unseen_points_cam2_loc)
# print(hand1_unseen_points_cam3_loc)
# 
# print(hand2_unseen_points_cam1)
# print(hand2_unseen_points_cam2)
# print(hand2_unseen_points_cam3)
# print(hand2_unseen_points_cam1_loc)
# print(hand2_unseen_points_cam2_loc)
# print(hand2_unseen_points_cam3_loc)
# 
# print(hand3_unseen_points_cam1)
# print(hand3_unseen_points_cam2)
# print(hand3_unseen_points_cam3)
# print(hand3_unseen_points_cam1_loc)
# print(hand3_unseen_points_cam2_loc)
# print(hand3_unseen_points_cam3_loc)
# =============================================================================

#Yeah this is shit code, I'm thinking about combining them into dataframe or something
#to make more sense out of it

#%%Get the coordinates for each unseen points
# =============================================================================
# def marker_coords(unseen_points_array ,cam, cam_name, marker_name, dataframe_level):
#     marker_coord = cam.xs(marker_name,level=dataframe_level,axis=1)
#     
#     marker_x_coords = unseen_points_array * np.transpose(marker_coord.xs('x',level='coords',axis=1).values.tolist())[0]
#     marker_y_coords = unseen_points_array * np.transpose(marker_coord.xs('y',level='coords',axis=1).values.tolist())[0]
#     #marker_z_coords = np.transpose(marker_coord.xs('z',level='coords',axis=1).values.tolist())[0]
#     print(marker_x_coords)
#     print(marker_y_coords)
#     print(marker_x_coords.shape)
#     print(marker_y_coords.shape)
#     plt.plot(marker_x_coords,marker_y_coords,'o')
#     plt.title(marker_name + ' unseen points by ' + cam_name)
#     plt.show()
#     return marker_x_coords, marker_y_coords
# =============================================================================
    
#%%
# =============================================================================
# shoulder1_cam1_unseen_points_x,shoulder1_cam1_unseen_points_y = marker_coords(shoulder1_unseen_points_cam1,cam1,'cam1','shoulder1','bodyparts')
# shoulder1_cam2_unseen_points_x,shoulder1_cam2_unseen_points_y = marker_coords(shoulder1_unseen_points_cam2,cam2,'cam2','shoulder1','bodyparts')
# shoulder1_cam3_unseen_points_x,shoulder1_cam3_unseen_points_y = marker_coords(shoulder1_unseen_points_cam3,cam3,'cam3','shoulder1','bodyparts')
# 
# arm1_cam1_unseen_points_x,arm1_cam1_unseen_points_y = marker_coords(arm1_unseen_points_cam1,cam1,'cam1','arm1','bodyparts')
# arm1_cam2_unseen_points_x,arm1_cam2_unseen_points_y = marker_coords(arm1_unseen_points_cam2,cam2,'cam2','arm1','bodyparts')
# arm1_cam3_unseen_points_x,arm1_cam3_unseen_points_y = marker_coords(arm1_unseen_points_cam3,cam3,'cam3','arm1','bodyparts')
# 
# arm2_cam1_unseen_points_x,arm2_cam1_unseen_points_y = marker_coords(arm2_unseen_points_cam1,cam1,'cam1','arm2','bodyparts')
# arm2_cam2_unseen_points_x,arm2_cam2_unseen_points_y = marker_coords(arm2_unseen_points_cam2,cam2,'cam2','arm2','bodyparts')
# arm2_cam3_unseen_points_x,arm2_cam3_unseen_points_y = marker_coords(arm2_unseen_points_cam3,cam3,'cam3','arm2','bodyparts')
# 
# elbow1_cam1_unseen_points_x,elbow1_cam1_unseen_points_y = marker_coords(elbow1_unseen_points_cam1,cam1,'cam1','elbow1','bodyparts')
# elbow1_cam2_unseen_points_x,elbow1_cam2_unseen_points_y = marker_coords(elbow1_unseen_points_cam2,cam2,'cam2','elbow1','bodyparts')
# elbow1_cam3_unseen_points_x,elbow1_cam3_unseen_points_y = marker_coords(elbow1_unseen_points_cam3,cam3,'cam3','elbow1','bodyparts')
# 
# elbow2_cam1_unseen_points_x,elbow2_cam1_unseen_points_y = marker_coords(elbow2_unseen_points_cam1,cam1,'cam1','elbow2','bodyparts')
# elbow2_cam2_unseen_points_x,elbow2_cam2_unseen_points_y = marker_coords(elbow2_unseen_points_cam2,cam2,'cam2','elbow2','bodyparts')
# elbow2_cam3_unseen_points_x,elbow2_cam3_unseen_points_y = marker_coords(elbow2_unseen_points_cam3,cam3,'cam3','elbow2','bodyparts')
# 
# wrist1_cam1_unseen_points_x,wrist1_cam1_unseen_points_y = marker_coords(wrist1_unseen_points_cam1,cam1,'cam1','wrist1','bodyparts')
# wrist1_cam2_unseen_points_x,wrist1_cam2_unseen_points_y = marker_coords(wrist1_unseen_points_cam2,cam2,'cam2','wrist1','bodyparts')
# wrist1_cam3_unseen_points_x,wrist1_cam3_unseen_points_y = marker_coords(wrist1_unseen_points_cam3,cam3,'cam3','wrist1','bodyparts')
# 
# wrist2_cam1_unseen_points_x,wrist2_cam1_unseen_points_y = marker_coords(wrist2_unseen_points_cam1,cam1,'cam1','wrist2','bodyparts')
# wrist2_cam2_unseen_points_x,wrist2_cam2_unseen_points_y = marker_coords(wrist2_unseen_points_cam2,cam2,'cam2','wrist2','bodyparts')
# wrist2_cam3_unseen_points_x,wrist2_cam3_unseen_points_y = marker_coords(wrist2_unseen_points_cam3,cam3,'cam3','wrist2','bodyparts')
# 
# hand1_cam1_unseen_points_x,hand1_cam1_unseen_points_y = marker_coords(hand1_unseen_points_cam1,cam1,'cam1','hand1','bodyparts')
# hand1_cam2_unseen_points_x,hand1_cam2_unseen_points_y = marker_coords(hand1_unseen_points_cam2,cam2,'cam2','hand1','bodyparts')
# hand1_cam3_unseen_points_x,hand1_cam3_unseen_points_y = marker_coords(hand1_unseen_points_cam3,cam3,'cam3','hand1','bodyparts')
# 
# hand2_cam1_unseen_points_x,hand2_cam1_unseen_points_y = marker_coords(hand2_unseen_points_cam1,cam1,'cam1','hand2','bodyparts')
# hand2_cam2_unseen_points_x,hand2_cam2_unseen_points_y = marker_coords(hand2_unseen_points_cam2,cam2,'cam2','hand2','bodyparts')
# hand2_cam3_unseen_points_x,hand2_cam3_unseen_points_y = marker_coords(hand2_unseen_points_cam3,cam3,'cam3','hand2','bodyparts')
# 
# hand3_cam1_unseen_points_x,hand3_cam1_unseen_points_y = marker_coords(hand3_unseen_points_cam1,cam1,'cam1','hand3','bodyparts')
# hand3_cam2_unseen_points_x,hand3_cam2_unseen_points_y = marker_coords(hand3_unseen_points_cam2,cam2,'cam2','hand3','bodyparts')
# hand3_cam3_unseen_points_x,hand3_cam3_unseen_points_y = marker_coords(hand3_unseen_points_cam3,cam3,'cam3','hand3','bodyparts')
# =============================================================================

#%% Try to extract frames from videos to plot the "unseen" points directly onto
#   the picture
#   https://www.geeksforgeeks.org/extract-images-from-video-in-python/
# =============================================================================
# """
# video_loc: video location, Windows users plz use "\\" instead of "\"
# unseen_frames: a list of the frame numbers in which only this camera can see this specific marker
# num_frames: how many frames totally there are in this video
# folder_name: what to name the new folder which you want to store the frames
# """
# def save_unseen_frames_per_cam(video_loc,unseen_frames,num_frames,folder_name):
#     camera = cv2.VideoCapture(video_loc)
#     print(os.getcwd())
#     try: 
#         # creating a folder named data 
#         if not os.path.exists(folder_name): 
#             print('here')
#             os.makedirs(folder_name)
#     # if not created then raise error 
#     except OSError: 
#         print ('Error: Creating directory of data') 
# 
#     # frame 
#     #currentframe = 0
#     unseen_frame_nums = 0
#     
#     for i in range(num_frames):
#         
#         # reading from frame 
#         ret,frame = camera.read() 
#         #print(unseen_frame_nums)
#         if unseen_frame_nums < len(unseen_frames):
#             if ret and i == unseen_frames[unseen_frame_nums]:
#                 # if video is still left continue creating images 
#                 name = './' +folder_name+ '/frame' + str(i) + '.jpg'
#                 print ('Creating...' + name) 
#           
#                 # writing the extracted images 
#                 cv2.imwrite(name, frame) 
#           
#                 # increasing counter so that it will 
#                 # show how many frames are created 
#                 #currentframe += 1
#                 unseen_frame_nums += 1
#             #else: 
#                 #currentframe += 1
#             
#     # Release all space and windows once done 
#     camera.release() 
#     cv2.destroyAllWindows()
#     return 1
# =============================================================================

#%%
"""
I've done running all these code (not only shoulder but also other markers) but
totally messed it up by covering it with an older version. This one left behind
as an example of how to use this function,but well, anyways.
"""
"""
test_shoulder1_cam1 = save_unseen_frames_per_cam("C:\\Users\\dongq\\DeepLabCut\\Han-Qiwei-2020-02-21\\videos\\exp00001DeepCut_resnet50_HanFeb21shuffle1_1030000_labeled.mp4",shoulder1_unseen_points_cam1_loc,46610,'shoulder1_cam1_unseen_points_unlabeled') 
"""

#%% function to plot all these "unseen" points on pictures to see where
#   their positions roughly are
# =============================================================================
# def plot_unseen_points_on_pics(img_dir,x,y,marker_name,cam_name):
#     img = mpimg.imread(img_dir)
#     fig,ax = plt.subplots()
#     ax.imshow(img)
#     ax.scatter(x,y)
#     ax.set(xlabel='x axis',ylabel='y axis',
#        title = marker_name + ' points only seen by ' + cam_name)
#     plt.show()
#     fig.savefig(marker_name+'_'+cam_name+'.png')
# =============================================================================

#%%And plot them out to see if there's anything interesting to learn about
# =============================================================================
# img_base_dir = 'C:\\Users\\dongq\\DeepLabCut\\Han-Qiwei-2020-02-21\\videos'
# 
# img = mpimg.imread(image_base_dir + '\\arm1_cam1_unseen_points_unlabeled\\frame29581.jpg')
# fig,ax = plt.subplots()
# ax.imshow(img)
# ax.scatter(arm1_cam1_unseen_points_x,arm1_cam1_unseen_points_y)
# ax.set(xlabel='x axis',ylabel='y axis',
#        title='Arm1 points only seen by Cam1')
# plt.show()
# 
# 
# #plt.colorbar()?
# 
# 
# plot_unseen_points_on_pics(img_base_dir+'\\shoulder1_cam1_unseen_points_unlabeled\\frame8950.jpg', shoulder1_cam1_unseen_points_x, shoulder1_cam1_unseen_points_y, 'shoulder1', 'cam1')
# plot_unseen_points_on_pics(img_base_dir+'\\shoulder1_cam2_unseen_points_unlabeled\\frame7357.jpg', shoulder1_cam2_unseen_points_x, shoulder1_cam2_unseen_points_y, 'shoulder1', 'cam2')
# plot_unseen_points_on_pics(img_base_dir+'\\shoulder1_cam3_unseen_points_unlabeled\\frame20497.jpg', shoulder1_cam3_unseen_points_x, shoulder1_cam3_unseen_points_y, 'shoulder1', 'cam3')
# 
# plot_unseen_points_on_pics(img_base_dir+'\\arm1_cam1_unseen_points_unlabeled\\frame29581.jpg', arm1_cam1_unseen_points_x, arm1_cam1_unseen_points_y, 'arm1', 'cam1')
# plot_unseen_points_on_pics(img_base_dir+'\\arm1_cam2_unseen_points_unlabeled\\frame19040.jpg', arm1_cam2_unseen_points_x, arm1_cam2_unseen_points_y, 'arm1', 'cam2')
# plot_unseen_points_on_pics(img_base_dir+'\\arm1_cam3_unseen_points_unlabeled\\frame34094.jpg', arm1_cam3_unseen_points_x, arm1_cam3_unseen_points_y, 'arm1', 'cam3')
# 
# plot_unseen_points_on_pics(img_base_dir+'\\arm2_cam1_unseen_points_unlabeled\\frame30068.jpg', arm2_cam1_unseen_points_x, arm2_cam1_unseen_points_y, 'arm2', 'cam1')
# plot_unseen_points_on_pics(img_base_dir+'\\arm2_cam2_unseen_points_unlabeled\\frame19120.jpg', arm2_cam2_unseen_points_x, arm2_cam2_unseen_points_y, 'arm2', 'cam2')
# plot_unseen_points_on_pics(img_base_dir+'\\arm2_cam3_unseen_points_unlabeled\\frame14161.jpg', arm2_cam3_unseen_points_x, arm2_cam3_unseen_points_y, 'arm2', 'cam3')
# 
# plot_unseen_points_on_pics(img_base_dir+'\\elbow1_cam1_unseen_points_unlabeled\\frame40382.jpg', elbow1_cam1_unseen_points_x, elbow1_cam1_unseen_points_y, 'elbow1', 'cam1')
# plot_unseen_points_on_pics(img_base_dir+'\\elbow1_cam2_unseen_points_unlabeled\\frame4846.jpg', elbow1_cam2_unseen_points_x, elbow1_cam2_unseen_points_y, 'elbow1', 'cam2')
# plot_unseen_points_on_pics(img_base_dir+'\\elbow1_cam3_unseen_points_unlabeled\\frame45315.jpg', elbow1_cam3_unseen_points_x, elbow1_cam3_unseen_points_y, 'elbow1', 'cam3')
# 
# plot_unseen_points_on_pics(img_base_dir+'\\elbow2_cam1_unseen_points_unlabeled\\frame40969.jpg', elbow2_cam1_unseen_points_x, elbow2_cam1_unseen_points_y, 'elbow2', 'cam1')
# plot_unseen_points_on_pics(img_base_dir+'\\elbow2_cam2_unseen_points_unlabeled\\frame2038.jpg', elbow2_cam2_unseen_points_x, elbow2_cam2_unseen_points_y, 'elbow2', 'cam2')
# plot_unseen_points_on_pics(img_base_dir+'\\elbow2_cam3_unseen_points_unlabeled\\frame3241.jpg', elbow2_cam3_unseen_points_x, elbow2_cam3_unseen_points_y, 'elbow2', 'cam3')
# 
# plot_unseen_points_on_pics(img_base_dir+'\\wrist1_cam1_unseen_points_unlabeled\\frame299.jpg', wrist1_cam1_unseen_points_x, wrist1_cam1_unseen_points_y, 'wrist1', 'cam1')
# plot_unseen_points_on_pics(img_base_dir+'\\wrist1_cam2_unseen_points_unlabeled\\frame40090.jpg', wrist1_cam2_unseen_points_x, wrist1_cam2_unseen_points_y, 'wrist1', 'cam2')
# plot_unseen_points_on_pics(img_base_dir+'\\wrist1_cam3_unseen_points_unlabeled\\frame23669.jpg', wrist1_cam3_unseen_points_x, wrist1_cam3_unseen_points_y, 'wrist1', 'cam3')
# 
# plot_unseen_points_on_pics(img_base_dir+'\\wrist2_cam1_unseen_points_unlabeled\\frame2909.jpg', wrist2_cam1_unseen_points_x, wrist2_cam1_unseen_points_y, 'wrist2', 'cam1')
# plot_unseen_points_on_pics(img_base_dir+'\\wrist2_cam2_unseen_points_unlabeled\\frame45507.jpg', wrist2_cam2_unseen_points_x, wrist2_cam2_unseen_points_y, 'wrist2', 'cam2')
# plot_unseen_points_on_pics(img_base_dir+'\\wrist2_cam3_unseen_points_unlabeled\\frame3489.jpg', wrist2_cam3_unseen_points_x, wrist2_cam3_unseen_points_y, 'wrist2', 'cam3')
# 
# plot_unseen_points_on_pics(img_base_dir+'\\hand1_cam1_unseen_points_unlabeled\\frame828.jpg', hand1_cam1_unseen_points_x, hand1_cam1_unseen_points_y, 'hand1', 'cam1')
# plot_unseen_points_on_pics(img_base_dir+'\\hand1_cam2_unseen_points_unlabeled\\frame19228.jpg', hand1_cam2_unseen_points_x, hand1_cam2_unseen_points_y, 'hand1', 'cam2')
# plot_unseen_points_on_pics(img_base_dir+'\\hand1_cam3_unseen_points_unlabeled\\frame9668.jpg', hand1_cam3_unseen_points_x, hand1_cam3_unseen_points_y, 'hand1', 'cam3')
# 
# plot_unseen_points_on_pics(img_base_dir+'\\hand2_cam1_unseen_points_unlabeled\\frame9550.jpg', hand2_cam1_unseen_points_x, hand2_cam1_unseen_points_y, 'hand2', 'cam1')
# plot_unseen_points_on_pics(img_base_dir+'\\hand2_cam2_unseen_points_unlabeled\\frame40384.jpg', hand2_cam2_unseen_points_x, hand2_cam2_unseen_points_y, 'hand2', 'cam2')
# plot_unseen_points_on_pics(img_base_dir+'\\hand2_cam3_unseen_points_unlabeled\\frame3891.jpg', hand2_cam3_unseen_points_x, hand2_cam3_unseen_points_y, 'hand2', 'cam3')
# 
# plot_unseen_points_on_pics(img_base_dir+'\\hand3_cam1_unseen_points_unlabeled\\frame40400.jpg', hand3_cam1_unseen_points_x, hand3_cam1_unseen_points_y, 'hand3', 'cam1')
# plot_unseen_points_on_pics(img_base_dir+'\\hand3_cam2_unseen_points_unlabeled\\frame40335.jpg', hand3_cam2_unseen_points_x, hand3_cam2_unseen_points_y, 'hand3', 'cam2')
# plot_unseen_points_on_pics(img_base_dir+'\\hand3_cam3_unseen_points_unlabeled\\frame5529.jpg', hand3_cam3_unseen_points_x, hand3_cam3_unseen_points_y, 'hand3', 'cam3')
# =============================================================================

#%% Next step: get the 2D coordinates of the points above, plot
#   them and see if there are any significant results to learn about

#%% Define a function that bins the "likelihood" numbers for each points in each cams
def likelihood_binning(array):
    likelihood_array = np.zeros((10,array.shape[1]))
    binned_likelihood = np.zeros((array.shape[0],array.shape[1]))
    for i in range(array.shape[1]): #each marker (10)
        #for i in range(10): #each bin (0~10,10~20......90~100)
        bin_loc = 0
        for j in range(array.shape[0]):
            if array[j,i] < 0.1:
                bin_loc = 0
            elif array[j,i] > 0.9:
                bin_loc = 9
            else:
                bin_loc = int(math.floor(array[j,i]*10))
            binned_likelihood[j,i] = bin_loc/10
            likelihood_array[bin_loc,i] += 1
            #print(bin_loc)
    return likelihood_array,binned_likelihood

#%% Plot 

#overall_dropout_positions_cam1 = shoulder1_unseen_points_cam1 + arm1_unseen_points_cam1 + arm2_unseen_points_cam1 + elbow1_unseen_points_cam1 + elbow2_unseen_points_cam1 + wrist1_unseen_points_cam1 + wrist2_unseen_points_cam1 + hand1_unseen_points_cam1 + hand2_unseen_points_cam1 + hand3_unseen_points_cam1
#plt.plot(overall_dropout_positions_cam1)

cam1_likelihood = cam1_exp_only.xs('likelihood',level='coords',axis=1)
cam2_likelihood = cam2_exp_only.xs('likelihood',level='coords',axis=1)
cam3_likelihood = cam3_exp_only.xs('likelihood',level='coords',axis=1)
cam4_likelihood = cam4_exp_only.xs('likelihood',level='coords',axis=1)

cam1_likelihood_values = cam1_likelihood.values
cam2_likelihood_values = cam2_likelihood.values
cam3_likelihood_values = cam3_likelihood.values
cam4_likelihood_values = cam4_likelihood.values

cam1_likelihood_values_distribution, cam1_likelihood_values_binned = likelihood_binning(cam1_likelihood_values)
cam2_likelihood_values_distribution, cam2_likelihood_values_binned = likelihood_binning(cam2_likelihood_values)
cam3_likelihood_values_distribution, cam3_likelihood_values_binned = likelihood_binning(cam3_likelihood_values)
cam4_likelihood_values_distribution, cam4_likelihood_values_binned = likelihood_binning(cam4_likelihood_values)


cam1_all_likelihood_values_distribution, cam1_all_likelihood_values_binned = likelihood_binning(cam1.xs('likelihood',level='coords',axis=1).values)

#%% Hisogram plot likelihood distribution of all markers in all 4 cameras


# =============================================================================
# #fig,ax = plt.subplots()
# #plt.style.use('ggplot')
# plt.figure()
# plt.hist(cam1_likelihood_values_binned)
# plt.title('CAM1 likelihood distribution')
# plt.show()
# 
# plt.figure()
# plt.hist(cam2_likelihood_values_binned)
# plt.title('CAM2 likelihood distribution')
# plt.show()
# 
# plt.figure()
# plt.hist(cam3_likelihood_values_binned)
# plt.title('CAM3 likelihood distribution')
# plt.show()
# 
# plt.figure()
# plt.hist(cam4_likelihood_values_binned)
# plt.title('CAM4 likelihood distribution')
# plt.show()
# =============================================================================

#%% Bar plots

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

def calc_likelihood(cam_dataset, threshold):
    seen = []
    seen_percentage = []
    unseen = []
    unseen_percentage = []
    for i in range(cam_dataset.shape[1]):
        marker = cam_dataset[:,i]
        seen.append(len(marker[marker > threshold]))
        seen_percentage.append(len(marker[marker > threshold]) / len(marker))
        #unseen.append(len(marker[marker < threshold]))
        unseen.append(len(marker) - seen[i])
        #unseen_percentage.append(len(marker[marker < threshold]) / len(marker))
        unseen_percentage.append((len(marker) - len(marker[marker > threshold])) / len(marker))
    return seen, seen_percentage, unseen, unseen_percentage


cam_marker_results = calc_likelihood(cam1_likelihood_values_binned, 0.8)

cam_seen = tuple(cam_marker_results[0])
cam_seen_percentage = tuple(cam_marker_results[1])
cam_unseen = tuple(cam_marker_results[2])
cam_unseen_percentage = tuple(cam_marker_results[3])


cam_all_marker_results = calc_likelihood(cam1_all_likelihood_values_binned, 0.8)

cam_all_seen = tuple(cam_all_marker_results[0])
cam_all_seen_percentage = tuple(cam_all_marker_results[1])
cam_all_unseen = tuple(cam_all_marker_results[2])
cam_all_unseen_percentage = tuple(cam_all_marker_results[3])



cam1_seen_percentage = tuple(calc_likelihood(cam1_likelihood_values_binned[:,0:8], 0.8)[1])
cam2_seen_percentage = tuple(calc_likelihood(cam2_likelihood_values_binned[:,0:8], 0.8)[1])
cam3_seen_percentage = tuple(calc_likelihood(cam3_likelihood_values_binned[:,0:8], 0.8)[1])
cam4_seen_percentage = tuple(calc_likelihood(cam4_likelihood_values_binned[:,0:8], 0.8)[1])


N_markers = 8
ind = np.arange(N_markers)
width = 0.2

plt.figure(figsize=(14,7))

#Plot comparisons between EXP-ONLY and WHOLE-RECORDING
#p1 = plt.bar(ind, cam_all_seen_percentage, width)
#p2 = plt.bar(ind, cam_unseen_percentage, width, bottom = cam_seen_percentage)
#p2 = plt.bar(ind+0.35, cam_seen_percentage, width)

p1 = plt.bar(ind-0.3,cam1_seen_percentage,width)
p2 = plt.bar(ind-0.1,cam2_seen_percentage,width)
p3 = plt.bar(ind+0.1,cam3_seen_percentage,width)
p4 = plt.bar(ind+0.3,cam4_seen_percentage,width)

plt.ylabel('Percentage of Frames')
#plt.title('Percentage of high likelihood frames for each camera for RT3D task')
plt.title('Percentage of high likelihood frames for each camera for RT2D task')
plt.xlabel('Markers')
plt.xticks(ind,('shoulder1','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3'))
#plt.legend((p1[0],p2[0]),('Whole Video','Experiment Only'),loc='lower right')
plt.legend((p1[0],p2[0],p3[0],p4[0]),('Cam1','Cam2','Cam3','Cam4'),loc='lower right')
#plt.legend(p1[0],'High Likelihood',loc='lower right')
plt.ylim(0,1.01)
plt.show()

#%%
# =============================================================================
# img = mpimg.imread(img_dir)
# fig,ax = plt.subplots()
# ax.imshow(img)
# ax.scatter(x,y)
# ax.set(xlabel='x axis',ylabel='y axis',
#    title = marker_name + ' points only seen by ' + cam_name)
# plt.show()
# fig.savefig(marker_name+'_'+cam_name+'.png')
# 
# =============================================================================