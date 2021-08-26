# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:52:00 2020

@author: dongq
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.image as mpimg

import cv2
import os
import math
import pandas as pd 

import copy

import seaborn as sns

#%% Read in 4 DLC tracking files (2D) for the same project

#folder = 
# =============================================================================
# 
# cam1 = pandas.read_hdf(r'D:\DLC_Folders\Han-Qiwei-2020-09-22-RT2D\videos\exp_han_00017DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
# cam2 = pandas.read_hdf(r'D:\DLC_Folders\Han-Qiwei-2020-09-22-RT2D\videos\exp_han_00018DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
# cam3 = pandas.read_hdf(r'D:\DLC_Folders\Han-Qiwei-2020-09-22-RT2D\videos\exp_han_00019DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
# cam4 = pandas.read_hdf(r'D:\DLC_Folders\Han-Qiwei-2020-09-22-RT2D\videos\exp_han_00020DLC_resnet50_HanSep22shuffle1_1030000filtered.h5')
# 
# f = open(r"D:\DLC_Folders\Han-Qiwei-2020-09-22-RT2D\videos\Ground_truth_segments_2020-09-22-RT2D.txt", "r") 
# =============================================================================

# =============================================================================
# 
# cam1 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\videos\Han_20201204_00005DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam2 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\videos\Han_20201204_00006DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam3 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\videos\Han_20201204_00007DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam4 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\videos\Han_20201204_00008DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# 
# f = open(r"C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\Han_20201204_RT3D_groundTruth.txt", "r") 
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\videos'
# cam1 = pandas.read_hdf(folder + r'\Crackle_20201203_00001DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Crackle_20201203_00002DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Crackle_20201203_00003DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Crackle_20201203_00004DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\Ground_truth_segments_2020-12-03-RT2D.txt", "r") 
# frames_per_second = 24
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\videos'
# cam1 = pandas.read_hdf(folder + r'\Crackle_20201203_00007DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Crackle_20201203_00008DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Crackle_20201203_00009DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Crackle_20201203_00010DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\Ground_truth_segments_2020-12-03-RT3D-2.txt", "r") 
# frames_per_second = 25
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-15\videos'
# cam1 = pandas.read_hdf(folder + r'\Crackle_20201215_00001DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Crackle_20201215_00002DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Crackle_20201215_00003DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Crackle_20201215_00004DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
# #f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-15\Ground_truth_segments_2020-12-03-RT3D-2.txt", "r") 
# frames_per_second = 25
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Han_20201203_rwFreeReach\videos'
# cam1 = pandas.read_hdf(folder + r'\Han_20201203_00006DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam2 = pandas.read_hdf(folder + r'\Han_20201203_00007DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam3 = pandas.read_hdf(folder + r'\Han_20201203_00008DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam4 = pandas.read_hdf(folder + r'\Han_20201203_00009DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Han_20201203_rwFreeReach\Han_20201203_RT2D_groundTruth.txt", "r") 
# frames_per_second = 25
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Han_20201203_rwFreeReach\videos'
# cam1 = pandas.read_hdf(folder + r'\Han_20201203_00010DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam2 = pandas.read_hdf(folder + r'\Han_20201203_00011DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam3 = pandas.read_hdf(folder + r'\Han_20201203_00012DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam4 = pandas.read_hdf(folder + r'\Han_20201203_00013DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Han_20201203_rwFreeReach\Han_20201203_RT3D_groundTruth.txt", "r") 
# frames_per_second = 25
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\videos'
# cam1 = pandas.read_hdf(folder + r'\Han_20201204_00001DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam2 = pandas.read_hdf(folder + r'\Han_20201204_00002DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam3 = pandas.read_hdf(folder + r'\Han_20201204_00003DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam4 = pandas.read_hdf(folder + r'\Han_20201204_00004DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\Han_20201204_RT2D_groundTruth.txt", "r") 
# frames_per_second = 25
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\videos'
# cam1 = pandas.read_hdf(folder + r'\Han_20201204_00005DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam2 = pandas.read_hdf(folder + r'\Han_20201204_00006DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam3 = pandas.read_hdf(folder + r'\Han_20201204_00007DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# cam4 = pandas.read_hdf(folder + r'\Han_20201204_00008DLC_resnet50_Han_202012Dec14shuffle1_1030000.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\Han_20201204_RT3D_groundTruth.txt", "r") 
# frames_per_second = 25
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Han_20201217_rwFreeReach\videos'
# cam1 = pandas.read_hdf(folder + r'\Han_2020121700001DLC_resnet50_Han_202012Dec14shuffle1_1030000filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Han_2020121700002DLC_resnet50_Han_202012Dec14shuffle1_1030000filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Han_2020121700003DLC_resnet50_Han_202012Dec14shuffle1_1030000filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Han_2020121700004DLC_resnet50_Han_202012Dec14shuffle1_1030000filtered.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Han_20201217_rwFreeReach\Han_20201217_RT3D_groundTruth.txt", "r") 
# frames_per_second = 25
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Han_20201217_rwFreeReach\videos'
# cam1 = pandas.read_hdf(folder + r'\Han_2020121700005DLC_resnet50_Han_202012Dec14shuffle1_1030000filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Han_2020121700006DLC_resnet50_Han_202012Dec14shuffle1_1030000filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Han_2020121700007DLC_resnet50_Han_202012Dec14shuffle1_1030000filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Han_2020121700008DLC_resnet50_Han_202012Dec14shuffle1_1030000filtered.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Han_20201217_rwFreeReach\Han_20201217_RT2D_task1_groundTruth.txt", "r") 
# frames_per_second = 25
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-15\videos'
# cam1 = pandas.read_hdf(folder + r'\Crackle_20201215_00001DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Crackle_20201215_00002DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Crackle_20201215_00003DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Crackle_20201215_00004DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-15\Ground_truth_segments_2020-12-15-RT3D.txt", "r") 
# frames_per_second = 24
# =============================================================================

folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-15\videos'
cam1 = pandas.read_hdf(folder + r'\Crackle_20201215_00005DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
cam2 = pandas.read_hdf(folder + r'\Crackle_20201215_00006DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
cam3 = pandas.read_hdf(folder + r'\Crackle_20201215_00007DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
cam4 = pandas.read_hdf(folder + r'\Crackle_20201215_00008DLC_resnet50_TestDec14shuffle1_1030000filtered.h5')
f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-15\Ground_truth_segments_2020-12-15-RT2D-task1.txt", "r") 
frames_per_second = 25

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-16\videos'
# cam1 = pandas.read_hdf(folder + r'\Crackle_2020121600001DLC_resnet50_TestDec14shuffle1_650000filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Crackle_2020121600002DLC_resnet50_TestDec14shuffle1_650000filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Crackle_2020121600003DLC_resnet50_TestDec14shuffle1_650000filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Crackle_2020121600004DLC_resnet50_TestDec14shuffle1_650000filtered.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-16\Ground_truth_segments_2020-12-16-RT3D.txt", "r") 
# frames_per_second = 25
# =============================================================================

# =============================================================================
# folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-16\videos'
# cam1 = pandas.read_hdf(folder + r'\Crackle_2020121600005DLC_resnet50_TestDec14shuffle1_650000filtered.h5')
# cam2 = pandas.read_hdf(folder + r'\Crackle_2020121600006DLC_resnet50_TestDec14shuffle1_650000filtered.h5')
# cam3 = pandas.read_hdf(folder + r'\Crackle_2020121600007DLC_resnet50_TestDec14shuffle1_650000filtered.h5')
# cam4 = pandas.read_hdf(folder + r'\Crackle_2020121600008DLC_resnet50_TestDec14shuffle1_650000filtered.h5')
# f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-16\Ground_truth_segments_2020-12-16-RT2D-task1.txt", "r") 
# frames_per_second = 25
# =============================================================================

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

ground_truth_segment = np.zeros((cam1.shape[0]))        
for i in range(len(f_frame_list)):
    #print(i)
    ground_truth_segment[f_frame_list[i]] = 1

#%% Function counting non confident points

def count_low_likelihood_points(array, likelihood, threshold):
    likelihood_array = array[:,likelihood]
    likelihood_sum = []
    for i in range(likelihood_array.shape[1]):
        like_i = likelihood_array[:,i]
        like_thres = like_i[like_i < threshold]
        likelihood_sum.append(like_thres.shape[0])
    return likelihood_sum

#%% Function counting confident points
    
def count_high_likelihood_points(array, likelihood, threshold):
    likelihood_array = array[:,likelihood]
    likelihood_sum = []
    for i in range(likelihood_array.shape[1]):
        like_i = likelihood_array[:,i]
        like_thres = like_i[like_i > threshold]
        likelihood_sum.append(like_thres.shape[0])
    return likelihood_sum

#%% Trial segmenting function
    
def experiment_trial_segment(array,ground_truth_list):
    return array[ground_truth_list,:]

#%% Change dataframe to np array

cam1 = cam1.to_numpy()
cam2 = cam2.to_numpy()
cam3 = cam3.to_numpy()
cam4 = cam4.to_numpy()

# =============================================================================
# cam1_exp_only = cam1
# cam2_exp_only = cam2
# cam3_exp_only = cam3
# cam4_exp_only = cam4
# =============================================================================

cam1_exp_only = experiment_trial_segment(cam1,f_frame_list)
cam2_exp_only = experiment_trial_segment(cam2,f_frame_list)
cam3_exp_only = experiment_trial_segment(cam3,f_frame_list)
cam4_exp_only = experiment_trial_segment(cam4,f_frame_list)



#%% Call the count_low_likelihood_points() function

#speed_section = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29]

#likelihood_section = [2,5,8,11,14,17,20,23,26,29]
likelihood_section = [2,5,8,11,14,17,20,23]

threshold = 0.8

cam1_dropouts = count_low_likelihood_points(cam1,likelihood_section,threshold)
cam2_dropouts = count_low_likelihood_points(cam2,likelihood_section,threshold)
cam3_dropouts = count_low_likelihood_points(cam3,likelihood_section,threshold)
cam4_dropouts = count_low_likelihood_points(cam4,likelihood_section,threshold)

cam1_dropouts_exp_only = count_low_likelihood_points(cam1_exp_only,likelihood_section,threshold)
cam2_dropouts_exp_only = count_low_likelihood_points(cam2_exp_only,likelihood_section,threshold)
cam3_dropouts_exp_only = count_low_likelihood_points(cam3_exp_only,likelihood_section,threshold)
cam4_dropouts_exp_only = count_low_likelihood_points(cam4_exp_only,likelihood_section,threshold)


print("\n Dropout frames of the whole dataset (row=each cam, col = each marker)")
print(cam1_dropouts,'\n',cam2_dropouts,'\n',cam3_dropouts,'\n',cam4_dropouts)

print("\n Dropout frames of experiment phae in the dataset (row=each cam, col = each marker)")
print(cam1_dropouts_exp_only,'\n',cam2_dropouts_exp_only,'\n',cam3_dropouts_exp_only,'\n',cam4_dropouts_exp_only)


#%% function: count_hard_to_see_points
#https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

"""
If a marker is not tracked with high likelihood by at least two cameras, that
frame would be labeled 1 (hard to be 3D reconstructed)
"""
def count_hard_to_see_points(cam1, cam2, cam3, cam4):
    #for index,row in cam1.iterrows():
    #    print(row)
    cam1_array = cam1
    cam2_array = cam2
    cam3_array = cam3
    cam4_array = cam4
    hard_to_see_points = np.zeros(cam1_array.size)
    print(cam1_array.size)
    min_len = min(cam1_array.size,cam2_array.size,cam3_array.size,cam4_array.size)
    for i in range(min_len):
        result1 = bool(cam1_array[i] > 0.9)
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
            
        if cam1_seen + cam2_seen + cam3_seen + cam4_seen < 2:
            hard_to_see_points[i] = 1
    return hard_to_see_points

#%% Call the count_hard_to_see_points() function


allcam_unseen_points = []
allcam_unseen_points_exp_only = []

for i in range(int(cam1.shape[1]/3)):
    allcam_unseen_points.append(count_hard_to_see_points(
        cam1[:,likelihood_section[i]], 
        cam2[:,likelihood_section[i]], 
        cam3[:,likelihood_section[i]], 
        cam4[:,likelihood_section[i]]))
    allcam_unseen_points_exp_only.append(count_hard_to_see_points(
        cam1_exp_only[:,likelihood_section[i]], 
        cam2_exp_only[:,likelihood_section[i]], 
        cam3_exp_only[:,likelihood_section[i]], 
        cam4_exp_only[:,likelihood_section[i]]))

#%% Reverse the 1 and 0s in a new array, so that 1 means "arrays that are likely to be 3D reconstructed"
    
allcam_3D_reconstructable_points = copy.deepcopy(allcam_unseen_points)
allcam_3D_reconstructable_points_exp_only = copy.deepcopy(allcam_unseen_points_exp_only)

for i in range(len(allcam_3D_reconstructable_points)):
    for j in range(len(allcam_3D_reconstructable_points[i])):
        if allcam_3D_reconstructable_points[i][j] == 0.0:
            allcam_3D_reconstructable_points[i][j] = 1
        else:
            allcam_3D_reconstructable_points[i][j] = 0
    
for i in range(len(allcam_3D_reconstructable_points_exp_only)):
    for j in range(len(allcam_3D_reconstructable_points_exp_only[i])):
        if allcam_3D_reconstructable_points_exp_only[i][j] == 0.0:
            allcam_3D_reconstructable_points_exp_only[i][j] = 1
        else:
            allcam_3D_reconstructable_points_exp_only[i][j] = 0

#%% Sum up the arrays for plotting
            
sum_allcam_3D_recon = []
sum_allcam_3D_recon_exp = []

sum_allcam_3D_recon_percentage = []
sum_allcam_3D_recon_exp_percentage = []
std_allcam_3D_recon_exp_percentage = []

total_frames = len(allcam_3D_reconstructable_points[1])
total_frames_exp = len(allcam_3D_reconstructable_points_exp_only[1])

for i in range(len(allcam_3D_reconstructable_points)):
    sum_allcam_3D_recon.append(sum(allcam_3D_reconstructable_points[i]))
    sum_allcam_3D_recon_percentage.append(sum(allcam_3D_reconstructable_points[i])/total_frames)
    
    sum_allcam_3D_recon_exp.append(sum(allcam_3D_reconstructable_points_exp_only[i]))
    sum_allcam_3D_recon_exp_percentage.append(sum(allcam_3D_reconstructable_points_exp_only[i])/total_frames_exp)
    std_allcam_3D_recon_exp_percentage.append(np.std(allcam_3D_reconstructable_points_exp_only[i]))
    
    
#%% Set the plotting parameters

font = {'family' : 'normal',
#        'weight' : 'bold',
        'size'   : 16}

font_medium = {'family' : 'normal',
 #       'weight' : 'bold',
        'size'   : 16}

title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space

axis_font = {'fontname':'Arial', 'size':'16'}

#%% Plot the number/percentage of makers that are likely to be 3D reconed

#names = ['shoulder1','arm1','arm2','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
#names = ['shoulder1','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3',' ',' ']
names = ['shoulder1','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
N_markers = len(names)
ind = np.arange(N_markers)
width=0.4

plt.figure(figsize = (12,6))

#p1 = plt.bar(ind-0.2,sum_allcam_3D_recon_percentage,width)       
#p2 = plt.bar(ind+0.2,sum_allcam_3D_recon_exp_percentage,width)
p2 = plt.bar(ind,sum_allcam_3D_recon_exp_percentage[0:8],width)

plt.xlabel('Markers',**axis_font)
plt.ylabel('Percentage of Frames',**axis_font)
#plt.title('Percentage of high likelihood frames for RT3D task',**title_font)
plt.title('Percentage of high likelihood frames for RT2D task',**title_font)
plt.xticks(ind,names,**axis_font)
plt.yticks(np.arange(0,1.2,0.2),**axis_font)
plt.ylim((0,1))
#plt.legend((p1[0],p2[0]),('Whole Recording','Experiment Only'),loc='lower right')
#plt.set_fontsize(20)

plt.show()



#%% save in /video folder

save_file_name = folder + r'\2D_tracking_likelihood.csv'
df = pd.DataFrame([sum_allcam_3D_recon_exp_percentage[0:8],std_allcam_3D_recon_exp_percentage[0:8]])
df.index = ['mean','std']
df.to_csv(save_file_name)

#plt







