# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:42:32 2020

@author: dongq
"""


import scipy
from scipy.io import savemat

import pandas
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.image as mpimg

import cv2
import os
import math
import pandas as pd 
from matplotlib.ticker import PercentFormatter

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%%
folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03'
video_subfolder = r'\videos'

#cam1 = pandas.read_hdf(folder + r'\Crackle_20201203_00001DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
#cam2 = pandas.read_hdf(folder + r'\Crackle_20201203_00002DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
#cam3 = pandas.read_hdf(folder + r'\Crackle_20201203_00003DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')
#cam4 = pandas.read_hdf(folder + r'\Crackle_20201203_00004DLC_resnet50_TestDec3shuffle1_1030000filtered.h5')

cam1 = pandas.read_csv(folder + video_subfolder + r'\Crackle_20201203_00007DLC_resnet50_TestDec3shuffle1_1030000filtered.csv')
cam2 = pandas.read_csv(folder + video_subfolder + r'\Crackle_20201203_00008DLC_resnet50_TestDec3shuffle1_1030000filtered.csv')
cam3 = pandas.read_csv(folder + video_subfolder + r'\Crackle_20201203_00009DLC_resnet50_TestDec3shuffle1_1030000filtered.csv')
cam4 = pandas.read_csv(folder + video_subfolder + r'\Crackle_20201203_00010DLC_resnet50_TestDec3shuffle1_1030000filtered.csv')

f = open(folder + r"\Ground_truth_segments_2020-12-03-RT3D.txt", "r") 

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Remember to check this for EACH VIDEO cuz they'll vary
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
frames_per_second = 25

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


#def experiment_trial_segment(array, ground_truth_list):

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
    
cams = [cam1, cam2, cam3, cam4]
cams_arr = []
cams_arr_exp = []

for i in range(len(cams)):
    cams[i] = cams[i].drop([0,1],axis=0)
    cams_arr.append(cams[i].to_numpy().astype(np.float))
    cams_arr_exp.append(cams[i].to_numpy().astype(np.float)[f_frame_list])
         
#cam1_arr = cam1.to_numpy()
#cam2_arr = cam2.to_numpy()
#cam3_arr = cam3.to_numpy()
#cam4_arr = cam4.to_numpy()

#cam1_exp_only =         
         
#%% extract low likelihood frame numbers

#"ll" meaning "low likelihood"
#these datasets are list of lists
cam_arr_exp_likelihood = []
cam_arr_exp_ll = []

for i in range(len(cams_arr_exp)): #4
    markers = np.array([0,1,2,3,4,5,6,7]) #shoulder, elbow1,2, wrist1,2, hand 1,2,3
    if i == 0 or i == 2: 
        #SPECIFICALLY FOR CRACKLE20201203 DATASET, cam1 and 
        #cam3 can't see shoulders anyway, so we don't need to look at them anyways
        markers = np.array([1,2,3,4,5,6,7]) # elbow1,2, wrist1,2, hand 1,2,3
    likelihood_col = (markers + 1)*3
    
    likelihood = cams_arr_exp[i][:,likelihood_col]
    likelihood_framenum = np.hstack((likelihood,np.atleast_2d(cams_arr_exp[i][:,0]).T))
    cam_arr_exp_likelihood.append(likelihood_framenum) #store the results just in case
    
    low_likelihood = (likelihood <= 0.1)
    low_likelihood_framenum = np.hstack((low_likelihood,np.atleast_2d(cams_arr_exp[i][:,0]).T))
    cam_arr_exp_ll.append(low_likelihood_framenum)




#%% import the videos, construct the folders, and export the frames
    
#%% import the videos
    
#folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03'
#video_subfolder = r'\videos'

vid1 = r'Crackle_20201203_00007DLC_resnet50_TestDec3shuffle1_1030000_filtered_labeled'
vid2 = r'Crackle_20201203_00008DLC_resnet50_TestDec3shuffle1_1030000_filtered_labeled'
vid3 = r'Crackle_20201203_00009DLC_resnet50_TestDec3shuffle1_1030000_filtered_labeled'
vid4 = r'Crackle_20201203_00010DLC_resnet50_TestDec3shuffle1_1030000_filtered_labeled'
vid_type = r'.mp4'

vid1_dir = folder + video_subfolder + '\\' + vid1 + vid_type
vid2_dir = folder + video_subfolder + '\\' + vid2 + vid_type
vid3_dir = folder + video_subfolder + '\\' + vid3 + vid_type
vid4_dir = folder + video_subfolder + '\\' + vid4 + vid_type

vid_dirs = [vid1_dir, vid2_dir, vid3_dir, vid4_dir]


#vidcap1 = cv2.VideoCapture(vidDir1)
#%% Construct the folders

extract_frame_folder = 'low_likelihood_extracted_frames'
cam_names = ['cam1','cam2','cam3','cam4']
marker_names = ['shoulder','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
if not os.path.exists(folder + '\\' + extract_frame_folder):
    os.mkdir(folder + '\\' + extract_frame_folder)
for i in range(len(cam_names)):
    dir_layer1 = folder + '\\' +  extract_frame_folder + '\\' +  cam_names[i]
    if not os.path.exists(dir_layer1):
        os.mkdir(dir_layer1)
    
    for j in range(len(marker_names)):
        dir_layer2 = folder +'\\' + extract_frame_folder +'\\' + cam_names[i] + '\\' +  marker_names[j]
        if not os.path.exists(dir_layer2):
            os.mkdir(dir_layer2)

#%% read in each video, and export the frames
            #cam_arr_exp_ll
for i in range(len(vid_dirs)): #4 each camera
    
    #read in the video
    vid_dirs[i]    
    vidcap = cv2.VideoCapture(vid_dirs[i])
    
    for j in range(cam_arr_exp_ll[i].shape[1]): #8 or 9, for each marker
        
        for k in range(cam_arr_exp_ll[i].shape[0]): #35556, for each frame in the video
            #print("1")
            if cam_arr_exp_ll[i][k,j] == 1: #if it is a god damn low likelihood marker!!!
                
                arr_end = len(cam_arr_exp_ll[i][k,:])-1
                frame_num = cam_arr_exp_ll[i][k,arr_end]
                
                cam_name = cam_names[i]
                if cam_arr_exp_ll[i].shape[1] == 8:
                    marker_name = marker_names[j+1]
                else:
                    marker_name = marker_names[j]
                marker_folder = folder + '\\' + extract_frame_folder + '\\' + cam_name + '\\' + marker_name
                
                #read in the video
                vidcap.set(1, frame_num)
                ret, frame = vidcap.read()
                cv2.imwrite(marker_folder + '\\' + str(frame_num) + r'.png', frame)
                
                print(cam_name + ' ' + marker_name)



vidcap.release()
cv2.destroyAllWindows()



         
         
         
         
         
         
         
         
         
         
         
         
         
