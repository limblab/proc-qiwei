# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:14:54 2020

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

#%%Experiment phase only

experiment_phase_only = 1

#%%
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


#%%

#cam1 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\exp00001DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
#cam2 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\exp00002DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
#cam3 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\exp00003DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
#cam4 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\exp00004DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')

#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\Ground_truth_segments_20200804_FR.txt", "r") 


cam1 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\exp00001DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
cam2 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\exp00002DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
cam3 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\exp00003DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')
cam4 = pandas.read_hdf(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\exp00004DLC_resnet50_HanAug4shuffle1_1030000filtered.h5')

f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\Ground_truth_segments_20200804_RT.txt", "r") 


#%% Trial segmenting function
def experiment_trial_segment(df,ground_truth_list):
    df2 = pd.DataFrame(np.zeros((0,cam1.shape[1])),columns = df.columns)
    for i in range(len(ground_truth_list)):
        df2.loc[i] = df.iloc[ground_truth_list[i]]
    #print(df2)
    return df2


#%% Get the array for trial segmentation
if experiment_phase_only == 1:
    #df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')
    
    
    frames_per_second = 25
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
        
    #ground_truth_segment = np.zeros((cam1.shape[0]))        
    #for i in range(len(f_frame_list)):
    #    #print(i)
    #    ground_truth_segment[f_frame_list[i]] = 1
    
#%%
cam1_exp_only = experiment_trial_segment(cam1,f_frame_list)
cam2_exp_only = experiment_trial_segment(cam2,f_frame_list)
cam3_exp_only = experiment_trial_segment(cam3,f_frame_list)
cam4_exp_only = experiment_trial_segment(cam4,f_frame_list)
    
#%%
"""
cam1_dropouts = count_non_confident_points(cam1)
cam2_dropouts = count_non_confident_points(cam2)
cam3_dropouts = count_non_confident_points(cam3)
cam4_dropouts = count_non_confident_points(cam4)
"""
cam1_dropouts = count_non_confident_points(cam1_exp_only)
cam2_dropouts = count_non_confident_points(cam2_exp_only)
cam3_dropouts = count_non_confident_points(cam3_exp_only)
cam4_dropouts = count_non_confident_points(cam4_exp_only)

print(cam1_dropouts,'\n',cam2_dropouts,'\n',cam3_dropouts,'\n',cam4_dropouts)


#%% function: count_hard_to_see_points
#https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas

def count_hard_to_see_points(cam1, cam2, cam3, cam4):
    #for index,row in cam1.iterrows():
    #    print(row)
    cam1_array = cam1.values
    cam2_array = cam2.values
    cam3_array = cam3.values
    cam4_array = cam4.values
    hard_to_see_points = np.zeros(cam1_array.size)
    print(cam1_array.size)
    for i in range(cam1_array.size-1):
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

arm1_cam1 =  cam1_likelihood.xs('arm1',level='bodyparts',axis=1)
arm1_cam2 =  cam2_likelihood.xs('arm1',level='bodyparts',axis=1)
arm1_cam3 =  cam3_likelihood.xs('arm1',level='bodyparts',axis=1)
arm1_cam4 =  cam4_likelihood.xs('arm1',level='bodyparts',axis=1)
arm1_unseen_points = count_hard_to_see_points(arm1_cam1,arm1_cam2,arm1_cam3,arm1_cam4)


arm2_cam1 =  cam1_likelihood.xs('arm2',level='bodyparts',axis=1)
arm2_cam2 =  cam2_likelihood.xs('arm2',level='bodyparts',axis=1)
arm2_cam3 =  cam3_likelihood.xs('arm2',level='bodyparts',axis=1)
arm2_cam4 =  cam4_likelihood.xs('arm2',level='bodyparts',axis=1)
arm2_unseen_points = count_hard_to_see_points(arm2_cam1,arm2_cam2,arm2_cam3,arm2_cam4)

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


#%% Subplot the likelihood cam1 20200819

X = np.linspace(0,arm1_cam1.shape[0]-1,arm1_cam1.shape[0])/25
#fig, ax = plt.subplots(nrows=10, ncols=1)
fig=plt.figure()
plt.title("20200804_Han_FreeReaching_Cam1_DLC_Tracing_Probability_Exp_Only")
ax = fig.add_subplot(11,1,1)
ax.plot(X,shoulder1_cam1)
plt.ylabel("shoulder1")
ax.legend()
ax = fig.add_subplot(11,1,2)
ax.plot(X,arm1_cam1)
plt.ylabel("arm1")
ax.legend()
ax = fig.add_subplot(11,1,3)
ax.plot(X,arm2_cam1)
plt.ylabel("arm2")
ax.legend()
ax = fig.add_subplot(11,1,4)
ax.plot(X,elbow1_cam1)
plt.ylabel("elbow1")
ax.legend()
ax = fig.add_subplot(11,1,5)
ax.plot(X,elbow2_cam1)
plt.ylabel("elbow2")
ax.legend()
ax = fig.add_subplot(11,1,6)
ax.plot(X,wrist1_cam1)
plt.ylabel("wrist1")
ax.legend()
ax = fig.add_subplot(11,1,7)
ax.plot(X,wrist2_cam1)
plt.ylabel("wrist2")
ax.legend()
ax = fig.add_subplot(11,1,8)
ax.plot(X,hand1_cam1)
plt.ylabel("hand1")
ax.legend()
ax = fig.add_subplot(11,1,9)
ax.plot(X,hand2_cam1)
plt.ylabel("hand2")
ax.legend()
ax = fig.add_subplot(11,1,10)
ax.plot(X,hand3_cam1)
plt.ylabel("hand3")
ax.legend()
#ax = fig.add_subplot(11,1,11)
#ax.plot(X,ground_truth_segment)
#plt.ylabel("ground_truth")
#ax.legend()
plt.xlabel("seconds")

#%% Subplot the likelihood cam2 20200819

X = np.linspace(0,arm1_cam1.shape[0]-1,arm1_cam1.shape[0])/25
#fig, ax = plt.subplots(nrows=10, ncols=1)
fig=plt.figure()
plt.title("20200804_Han_FreeReaching_Cam2_DLC_Tracing_Probability_Exp_Only")
ax = fig.add_subplot(11,1,1)
ax.plot(X,shoulder1_cam2)
plt.ylabel("shoulder1")
ax.legend()
ax = fig.add_subplot(11,1,2)
ax.plot(X,arm1_cam2)
plt.ylabel("arm1")
ax.legend()
ax = fig.add_subplot(11,1,3)
ax.plot(X,arm2_cam2)
plt.ylabel("arm2")
ax.legend()
ax = fig.add_subplot(11,1,4)
ax.plot(X,elbow1_cam2)
plt.ylabel("elbow1")
ax.legend()
ax = fig.add_subplot(11,1,5)
ax.plot(X,elbow2_cam2)
plt.ylabel("elbow2")
ax.legend()
ax = fig.add_subplot(11,1,6)
ax.plot(X,wrist1_cam2)
plt.ylabel("wrist1")
ax.legend()
ax = fig.add_subplot(11,1,7)
ax.plot(X,wrist2_cam2)
plt.ylabel("wrist2")
ax.legend()
ax = fig.add_subplot(11,1,8)
ax.plot(X,hand1_cam2)
plt.ylabel("hand1")
ax.legend()
ax = fig.add_subplot(11,1,9)
ax.plot(X,hand2_cam2)
plt.ylabel("hand2")
ax.legend()
ax = fig.add_subplot(11,1,10)
ax.plot(X,hand3_cam2)
plt.ylabel("hand3")
ax.legend()
#ax = fig.add_subplot(11,1,11)
#ax.plot(X,ground_truth_segment)
#plt.ylabel("ground_truth")
#ax.legend()
plt.xlabel("seconds")

#%% Subplot the likelihood cam3 20200819

X = np.linspace(0,arm1_cam1.shape[0]-1,arm1_cam1.shape[0])/25
#fig, ax = plt.subplots(nrows=10, ncols=1)
fig=plt.figure()
plt.title("20200804_Han_FreeReaching_Cam3_DLC_Tracing_Probability_Exp_Only")
ax = fig.add_subplot(11,1,1)
ax.plot(X,shoulder1_cam3)
plt.ylabel("shoulder1")
ax.legend()
ax = fig.add_subplot(11,1,2)
ax.plot(X,arm1_cam3)
plt.ylabel("arm1")
ax.legend()
ax = fig.add_subplot(11,1,3)
ax.plot(X,arm2_cam3)
plt.ylabel("arm2")
ax.legend()
ax = fig.add_subplot(11,1,4)
ax.plot(X,elbow1_cam3)
plt.ylabel("elbow1")
ax.legend()
ax = fig.add_subplot(11,1,5)
ax.plot(X,elbow2_cam3)
plt.ylabel("elbow2")
ax.legend()
ax = fig.add_subplot(11,1,6)
ax.plot(X,wrist1_cam3)
plt.ylabel("wrist1")
ax.legend()
ax = fig.add_subplot(11,1,7)
ax.plot(X,wrist2_cam3)
plt.ylabel("wrist2")
ax.legend()
ax = fig.add_subplot(11,1,8)
ax.plot(X,hand1_cam3)
plt.ylabel("hand1")
ax.legend()
ax = fig.add_subplot(11,1,9)
ax.plot(X,hand2_cam3)
plt.ylabel("hand2")
ax.legend()
ax = fig.add_subplot(11,1,10)
ax.plot(X,hand3_cam3)
plt.ylabel("hand3")
ax.legend()
#ax = fig.add_subplot(11,1,11)
#ax.plot(X,ground_truth_segment)
#plt.ylabel("ground_truth")
#ax.legend()
plt.xlabel("seconds")

#%% Subplot the likelihood cam4 20200819

#fig, ax = plt.subplots(nrows=10, ncols=1)
fig=plt.figure()
plt.title("20200804_Han_FreeReaching_Cam4_DLC_Tracing_Probability_Exp_Only")
ax = fig.add_subplot(11,1,1)
ax.plot(X,shoulder1_cam4)
plt.ylabel("shoulder1")
ax.legend()
ax = fig.add_subplot(11,1,2)
ax.plot(X,arm1_cam4)
plt.ylabel("arm1")
ax.legend()
ax = fig.add_subplot(11,1,3)
ax.plot(X,arm2_cam4)
plt.ylabel("arm2")
ax.legend()
ax = fig.add_subplot(11,1,4)
ax.plot(X,elbow1_cam4)
plt.ylabel("elbow1")
ax.legend()
ax = fig.add_subplot(11,1,5)
ax.plot(X,elbow2_cam4)
plt.ylabel("elbow2")
ax.legend()
ax = fig.add_subplot(11,1,6)
ax.plot(X,wrist1_cam4)
plt.ylabel("wrist1")
ax.legend()
ax = fig.add_subplot(11,1,7)
ax.plot(X,wrist2_cam4)
plt.ylabel("wrist2")
ax.legend()
ax = fig.add_subplot(11,1,8)
ax.plot(X,hand1_cam4)
plt.ylabel("hand1")
ax.legend()
ax = fig.add_subplot(11,1,9)
ax.plot(X,hand2_cam4)
plt.ylabel("hand2")
ax.legend()
ax = fig.add_subplot(11,1,10)
ax.plot(X,hand3_cam4)
plt.ylabel("hand3")
ax.legend()
#ax = fig.add_subplot(11,1,11)
#ax.plot(X,ground_truth_segment)
#plt.ylabel("ground_truth")
#ax.legend()
plt.xlabel("seconds")


#%% plot these points above
fig, ax = plt.subplots()
ax.plot(shoulder1_unseen_points)
ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
       title='Points not seen by 2 cams at once for shoulder1')
fig.savefig('unseen_points_shoulder1.png')
plt.show()

fig, ax = plt.subplots()
ax.plot(arm1_unseen_points)
ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
       title='Points not seen by 2 cams at once for arm1')
fig.savefig('unseen_points_arm1.png')
plt.show()

fig, ax = plt.subplots()
ax.plot(arm2_unseen_points)
ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
       title='Points not seen by 2 cams at once for arm2')
fig.savefig('unseen_points_arm2.png')
plt.show()

fig, ax = plt.subplots()
ax.plot(elbow1_unseen_points)
ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
       title='Points not seen by 2 cams at once for elbow1')
fig.savefig('unseen_points_elbow1.png')
plt.show()

fig, ax = plt.subplots()
ax.plot(elbow2_unseen_points)
ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
       title='Points not seen by 2 cams at once for elbow2')
fig.savefig('unseen_points_elbow2.png')
plt.show()

fig, ax = plt.subplots()
ax.plot(wrist1_unseen_points)
ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
       title='Points not seen by 2 cams at once for wrist1')
fig.savefig('unseen_points_wrist1.png')
plt.show()

fig, ax = plt.subplots()
ax.plot(wrist2_unseen_points)
ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
       title='Points not seen by 2 cams at once for wrist2')
fig.savefig('unseen_points_wrist2.png')
plt.show()

fig, ax = plt.subplots()
ax.plot(hand1_unseen_points)
ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
       title='Points not seen by 2 cams at once for hand1')
fig.savefig('unseen_points_hand1.png')
plt.show()

fig, ax = plt.subplots()
ax.plot(hand2_unseen_points)
ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
       title='Points not seen by 2 cams at once for hand2')
fig.savefig('unseen_points_hand2.png')
plt.show()

fig, ax = plt.subplots()
ax.plot(hand3_unseen_points)
ax.set(xlabel='time(frame)',ylabel='seen by 2 cams or not',
       title='Points not seen by 2 cams at once for hand3')
fig.savefig('unseen_points_hand3.png')
plt.show()


"""

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test.png")
plt.show()
"""


#%%

print(sum(shoulder1_unseen_points),
sum(arm1_unseen_points),
sum(arm2_unseen_points),
sum(elbow1_unseen_points),
sum(elbow2_unseen_points),
sum(wrist1_unseen_points),
sum(wrist2_unseen_points),
sum(hand1_unseen_points),
sum(hand2_unseen_points),
sum(hand3_unseen_points))

print(sum(shoulder1_unseen_points)/cam1.shape[0],
sum(arm1_unseen_points)/cam1.shape[0],
sum(arm2_unseen_points)/cam1.shape[0],
sum(elbow1_unseen_points)/cam1.shape[0],
sum(elbow2_unseen_points)/cam1.shape[0],
sum(wrist1_unseen_points)/cam1.shape[0],
sum(wrist2_unseen_points)/cam1.shape[0],
sum(hand1_unseen_points)/cam1.shape[0],
sum(hand2_unseen_points)/cam1.shape[0],
sum(hand3_unseen_points)/cam1.shape[0])
#%%
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
def label_hard_to_see_points(cam1, cam2, cam3):
    cam1_array = cam1.values
    cam2_array = cam2.values
    cam3_array = cam3.values
    hard_to_see_points_cam1 = np.zeros(cam1_array.size)
    hard_to_see_points_cam2 = np.zeros(cam2_array.size)
    hard_to_see_points_cam3 = np.zeros(cam3_array.size)
    
    for i in range(cam1_array.size):
        result1 = bool(cam1_array[i] > 0.9)
        result2 = bool(cam2_array[i] > 0.9)
        result3 = bool(cam3_array[i] > 0.9)
        if result1 == True and result2 == False and result3 == False:
            hard_to_see_points_cam1[i] = 1
        if result2 == True and result1 == False and result3 == False:
            hard_to_see_points_cam2[i] = 1
        if result3 == True and result1 == False and result2 == False:
            hard_to_see_points_cam3[i] = 1
    return hard_to_see_points_cam1, hard_to_see_points_cam2, hard_to_see_points_cam3

#%%Plot unseen frames for each Marker in each camera

shoulder1_unseen_points_cam1,shoulder1_unseen_points_cam2,shoulder1_unseen_points_cam3  = label_hard_to_see_points(shoulder1_cam1,shoulder1_cam2,shoulder1_cam3)
shoulder1_unseen_points_cam1_loc = [i for i,x in enumerate(shoulder1_unseen_points_cam1) if x == 1]
shoulder1_unseen_points_cam2_loc = [i for i,x in enumerate(shoulder1_unseen_points_cam2) if x == 1]
shoulder1_unseen_points_cam3_loc = [i for i,x in enumerate(shoulder1_unseen_points_cam3) if x == 1]

arm1_unseen_points_cam1, arm1_unseen_points_cam2, arm1_unseen_points_cam3 = label_hard_to_see_points(arm1_cam1,arm1_cam2,arm1_cam3)
arm1_unseen_points_cam1_loc = [i for i,x in enumerate(arm1_unseen_points_cam1) if x == 1]
arm1_unseen_points_cam2_loc = [i for i,x in enumerate(arm1_unseen_points_cam2) if x == 1]
arm1_unseen_points_cam3_loc = [i for i,x in enumerate(arm1_unseen_points_cam3) if x == 1]

arm2_unseen_points_cam1, arm2_unseen_points_cam2, arm2_unseen_points_cam3 = label_hard_to_see_points(arm2_cam1,arm2_cam2,arm2_cam3)
arm2_unseen_points_cam1_loc = [i for i,x in enumerate(arm2_unseen_points_cam1) if x == 1]
arm2_unseen_points_cam2_loc = [i for i,x in enumerate(arm2_unseen_points_cam2) if x == 1]
arm2_unseen_points_cam3_loc = [i for i,x in enumerate(arm2_unseen_points_cam3) if x == 1]

elbow1_unseen_points_cam1, elbow1_unseen_points_cam2, elbow1_unseen_points_cam3 = label_hard_to_see_points(elbow1_cam1,elbow1_cam2,elbow1_cam3)
elbow1_unseen_points_cam1_loc = [i for i,x in enumerate(elbow1_unseen_points_cam1) if x == 1]
elbow1_unseen_points_cam2_loc = [i for i,x in enumerate(elbow1_unseen_points_cam2) if x == 1]
elbow1_unseen_points_cam3_loc = [i for i,x in enumerate(elbow1_unseen_points_cam3) if x == 1]

elbow2_unseen_points_cam1, elbow2_unseen_points_cam2, elbow2_unseen_points_cam3 = label_hard_to_see_points(elbow2_cam1,elbow2_cam2,elbow2_cam3)
elbow2_unseen_points_cam1_loc = [i for i,x in enumerate(elbow2_unseen_points_cam1) if x == 1]
elbow2_unseen_points_cam2_loc = [i for i,x in enumerate(elbow2_unseen_points_cam2) if x == 1]
elbow2_unseen_points_cam3_loc = [i for i,x in enumerate(elbow2_unseen_points_cam3) if x == 1]

wrist1_unseen_points_cam1, wrist1_unseen_points_cam2, wrist1_unseen_points_cam3 = label_hard_to_see_points(wrist1_cam1,wrist1_cam2,wrist1_cam3)
wrist1_unseen_points_cam1_loc = [i for i,x in enumerate(wrist1_unseen_points_cam1) if x == 1]
wrist1_unseen_points_cam2_loc = [i for i,x in enumerate(wrist1_unseen_points_cam2) if x == 1]
wrist1_unseen_points_cam3_loc = [i for i,x in enumerate(wrist1_unseen_points_cam3) if x == 1]

wrist2_unseen_points_cam1, wrist2_unseen_points_cam2, wrist2_unseen_points_cam3 = label_hard_to_see_points(wrist2_cam1,wrist2_cam2,wrist2_cam3)
wrist2_unseen_points_cam1_loc = [i for i,x in enumerate(wrist2_unseen_points_cam1) if x == 1]
wrist2_unseen_points_cam2_loc = [i for i,x in enumerate(wrist2_unseen_points_cam2) if x == 1]
wrist2_unseen_points_cam3_loc = [i for i,x in enumerate(wrist2_unseen_points_cam3) if x == 1]

hand1_unseen_points_cam1, hand1_unseen_points_cam2, hand1_unseen_points_cam3 = label_hard_to_see_points(hand1_cam1,hand1_cam2,hand1_cam3)
hand1_unseen_points_cam1_loc = [i for i,x in enumerate(hand1_unseen_points_cam1) if x == 1]
hand1_unseen_points_cam2_loc = [i for i,x in enumerate(hand1_unseen_points_cam2) if x == 1]
hand1_unseen_points_cam3_loc = [i for i,x in enumerate(hand1_unseen_points_cam3) if x == 1]

hand2_unseen_points_cam1, hand2_unseen_points_cam2, hand2_unseen_points_cam3 = label_hard_to_see_points(hand2_cam1,hand2_cam2,hand2_cam3)
hand2_unseen_points_cam1_loc = [i for i,x in enumerate(hand2_unseen_points_cam1) if x == 1]
hand2_unseen_points_cam2_loc = [i for i,x in enumerate(hand2_unseen_points_cam2) if x == 1]
hand2_unseen_points_cam3_loc = [i for i,x in enumerate(hand2_unseen_points_cam3) if x == 1]

hand3_unseen_points_cam1, hand3_unseen_points_cam2, hand3_unseen_points_cam3 = label_hard_to_see_points(hand3_cam1,hand3_cam2,hand3_cam3)
hand3_unseen_points_cam1_loc = [i for i,x in enumerate(hand3_unseen_points_cam1) if x == 1]
hand3_unseen_points_cam2_loc = [i for i,x in enumerate(hand3_unseen_points_cam2) if x == 1]
hand3_unseen_points_cam3_loc = [i for i,x in enumerate(hand3_unseen_points_cam3) if x == 1]

#%%Print the numbers and check if they're right or not
"""
print(shoulder1_unseen_points_cam1)
print(shoulder1_unseen_points_cam2)
print(shoulder1_unseen_points_cam3)
print(shoulder1_unseen_points_cam1_loc)
print(shoulder1_unseen_points_cam2_loc)
print(shoulder1_unseen_points_cam3_loc)

print(arm1_unseen_points_cam1)
print(arm1_unseen_points_cam2)
print(arm1_unseen_points_cam3)
print(arm1_unseen_points_cam1_loc)
print(arm1_unseen_points_cam2_loc)
print(arm1_unseen_points_cam3_loc)

print(arm2_unseen_points_cam1)
print(arm2_unseen_points_cam2)
print(arm2_unseen_points_cam3)
print(arm2_unseen_points_cam1_loc)
print(arm2_unseen_points_cam2_loc)
print(arm2_unseen_points_cam3_loc)

print(elbow1_unseen_points_cam1)
print(elbow1_unseen_points_cam2)
print(elbow1_unseen_points_cam3)
print(elbow1_unseen_points_cam1_loc)
print(elbow1_unseen_points_cam2_loc)
print(elbow1_unseen_points_cam3_loc)

print(elbow2_unseen_points_cam1)
print(elbow2_unseen_points_cam2)
print(elbow2_unseen_points_cam3)
print(elbow2_unseen_points_cam1_loc)
print(elbow2_unseen_points_cam2_loc)
print(elbow2_unseen_points_cam3_loc)

print(wrist1_unseen_points_cam1)
print(wrist1_unseen_points_cam2)
print(wrist1_unseen_points_cam3)
print(wrist1_unseen_points_cam1_loc)
print(wrist1_unseen_points_cam2_loc)
print(wrist1_unseen_points_cam3_loc)

print(wrist2_unseen_points_cam1)
print(wrist2_unseen_points_cam2)
print(wrist2_unseen_points_cam3)
print(wrist2_unseen_points_cam1_loc)
print(wrist2_unseen_points_cam2_loc)
print(wrist2_unseen_points_cam3_loc)

print(hand1_unseen_points_cam1)
print(hand1_unseen_points_cam2)
print(hand1_unseen_points_cam3)
print(hand1_unseen_points_cam1_loc)
print(hand1_unseen_points_cam2_loc)
print(hand1_unseen_points_cam3_loc)

print(hand2_unseen_points_cam1)
print(hand2_unseen_points_cam2)
print(hand2_unseen_points_cam3)
print(hand2_unseen_points_cam1_loc)
print(hand2_unseen_points_cam2_loc)
print(hand2_unseen_points_cam3_loc)

print(hand3_unseen_points_cam1)
print(hand3_unseen_points_cam2)
print(hand3_unseen_points_cam3)
print(hand3_unseen_points_cam1_loc)
print(hand3_unseen_points_cam2_loc)
print(hand3_unseen_points_cam3_loc)
"""
#Yeah this is shit code, I'm thinking about combining them into dataframe or something
#to make more sense out of it

#%%Get the coordinates for each unseen points

def marker_coords(unseen_points_array ,cam, cam_name, marker_name, dataframe_level):
    marker_coord = cam.xs(marker_name,level=dataframe_level,axis=1)
    
    marker_x_coords = unseen_points_array * np.transpose(marker_coord.xs('x',level='coords',axis=1).values.tolist())[0]
    marker_y_coords = unseen_points_array * np.transpose(marker_coord.xs('y',level='coords',axis=1).values.tolist())[0]
    #marker_z_coords = np.transpose(marker_coord.xs('z',level='coords',axis=1).values.tolist())[0]
    print(marker_x_coords)
    print(marker_y_coords)
    print(marker_x_coords.shape)
    print(marker_y_coords.shape)
    plt.plot(marker_x_coords,marker_y_coords,'o')
    plt.title(marker_name + ' unseen points by ' + cam_name)
    plt.show()
    return marker_x_coords, marker_y_coords
    
#%%
shoulder1_cam1_unseen_points_x,shoulder1_cam1_unseen_points_y = marker_coords(shoulder1_unseen_points_cam1,cam1,'cam1','shoulder1','bodyparts')
shoulder1_cam2_unseen_points_x,shoulder1_cam2_unseen_points_y = marker_coords(shoulder1_unseen_points_cam2,cam2,'cam2','shoulder1','bodyparts')
shoulder1_cam3_unseen_points_x,shoulder1_cam3_unseen_points_y = marker_coords(shoulder1_unseen_points_cam3,cam3,'cam3','shoulder1','bodyparts')

arm1_cam1_unseen_points_x,arm1_cam1_unseen_points_y = marker_coords(arm1_unseen_points_cam1,cam1,'cam1','arm1','bodyparts')
arm1_cam2_unseen_points_x,arm1_cam2_unseen_points_y = marker_coords(arm1_unseen_points_cam2,cam2,'cam2','arm1','bodyparts')
arm1_cam3_unseen_points_x,arm1_cam3_unseen_points_y = marker_coords(arm1_unseen_points_cam3,cam3,'cam3','arm1','bodyparts')

arm2_cam1_unseen_points_x,arm2_cam1_unseen_points_y = marker_coords(arm2_unseen_points_cam1,cam1,'cam1','arm2','bodyparts')
arm2_cam2_unseen_points_x,arm2_cam2_unseen_points_y = marker_coords(arm2_unseen_points_cam2,cam2,'cam2','arm2','bodyparts')
arm2_cam3_unseen_points_x,arm2_cam3_unseen_points_y = marker_coords(arm2_unseen_points_cam3,cam3,'cam3','arm2','bodyparts')

elbow1_cam1_unseen_points_x,elbow1_cam1_unseen_points_y = marker_coords(elbow1_unseen_points_cam1,cam1,'cam1','elbow1','bodyparts')
elbow1_cam2_unseen_points_x,elbow1_cam2_unseen_points_y = marker_coords(elbow1_unseen_points_cam2,cam2,'cam2','elbow1','bodyparts')
elbow1_cam3_unseen_points_x,elbow1_cam3_unseen_points_y = marker_coords(elbow1_unseen_points_cam3,cam3,'cam3','elbow1','bodyparts')

elbow2_cam1_unseen_points_x,elbow2_cam1_unseen_points_y = marker_coords(elbow2_unseen_points_cam1,cam1,'cam1','elbow2','bodyparts')
elbow2_cam2_unseen_points_x,elbow2_cam2_unseen_points_y = marker_coords(elbow2_unseen_points_cam2,cam2,'cam2','elbow2','bodyparts')
elbow2_cam3_unseen_points_x,elbow2_cam3_unseen_points_y = marker_coords(elbow2_unseen_points_cam3,cam3,'cam3','elbow2','bodyparts')

wrist1_cam1_unseen_points_x,wrist1_cam1_unseen_points_y = marker_coords(wrist1_unseen_points_cam1,cam1,'cam1','wrist1','bodyparts')
wrist1_cam2_unseen_points_x,wrist1_cam2_unseen_points_y = marker_coords(wrist1_unseen_points_cam2,cam2,'cam2','wrist1','bodyparts')
wrist1_cam3_unseen_points_x,wrist1_cam3_unseen_points_y = marker_coords(wrist1_unseen_points_cam3,cam3,'cam3','wrist1','bodyparts')

wrist2_cam1_unseen_points_x,wrist2_cam1_unseen_points_y = marker_coords(wrist2_unseen_points_cam1,cam1,'cam1','wrist2','bodyparts')
wrist2_cam2_unseen_points_x,wrist2_cam2_unseen_points_y = marker_coords(wrist2_unseen_points_cam2,cam2,'cam2','wrist2','bodyparts')
wrist2_cam3_unseen_points_x,wrist2_cam3_unseen_points_y = marker_coords(wrist2_unseen_points_cam3,cam3,'cam3','wrist2','bodyparts')

hand1_cam1_unseen_points_x,hand1_cam1_unseen_points_y = marker_coords(hand1_unseen_points_cam1,cam1,'cam1','hand1','bodyparts')
hand1_cam2_unseen_points_x,hand1_cam2_unseen_points_y = marker_coords(hand1_unseen_points_cam2,cam2,'cam2','hand1','bodyparts')
hand1_cam3_unseen_points_x,hand1_cam3_unseen_points_y = marker_coords(hand1_unseen_points_cam3,cam3,'cam3','hand1','bodyparts')

hand2_cam1_unseen_points_x,hand2_cam1_unseen_points_y = marker_coords(hand2_unseen_points_cam1,cam1,'cam1','hand2','bodyparts')
hand2_cam2_unseen_points_x,hand2_cam2_unseen_points_y = marker_coords(hand2_unseen_points_cam2,cam2,'cam2','hand2','bodyparts')
hand2_cam3_unseen_points_x,hand2_cam3_unseen_points_y = marker_coords(hand2_unseen_points_cam3,cam3,'cam3','hand2','bodyparts')

hand3_cam1_unseen_points_x,hand3_cam1_unseen_points_y = marker_coords(hand3_unseen_points_cam1,cam1,'cam1','hand3','bodyparts')
hand3_cam2_unseen_points_x,hand3_cam2_unseen_points_y = marker_coords(hand3_unseen_points_cam2,cam2,'cam2','hand3','bodyparts')
hand3_cam3_unseen_points_x,hand3_cam3_unseen_points_y = marker_coords(hand3_unseen_points_cam3,cam3,'cam3','hand3','bodyparts')

#%% Try to extract frames from videos to plot the "unseen" points directly onto
#   the picture
#   https://www.geeksforgeeks.org/extract-images-from-video-in-python/
"""
video_loc: video location, Windows users plz use "\\" instead of "\"
unseen_frames: a list of the frame numbers in which only this camera can see this specific marker
num_frames: how many frames totally there are in this video
folder_name: what to name the new folder which you want to store the frames
"""
def save_unseen_frames_per_cam(video_loc,unseen_frames,num_frames,folder_name):
    camera = cv2.VideoCapture(video_loc)
    print(os.getcwd())
    try: 
        # creating a folder named data 
        if not os.path.exists(folder_name): 
            print('here')
            os.makedirs(folder_name)
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 

    # frame 
    #currentframe = 0
    unseen_frame_nums = 0
    
    for i in range(num_frames):
        
        # reading from frame 
        ret,frame = camera.read() 
        #print(unseen_frame_nums)
        if unseen_frame_nums < len(unseen_frames):
            if ret and i == unseen_frames[unseen_frame_nums]:
                # if video is still left continue creating images 
                name = './' +folder_name+ '/frame' + str(i) + '.jpg'
                print ('Creating...' + name) 
          
                # writing the extracted images 
                cv2.imwrite(name, frame) 
          
                # increasing counter so that it will 
                # show how many frames are created 
                #currentframe += 1
                unseen_frame_nums += 1
            #else: 
                #currentframe += 1
            
    # Release all space and windows once done 
    camera.release() 
    cv2.destroyAllWindows()
    return 1

#%%
"""
I've done running all these code (not only shoulder but also other markers) but
totally messed it up by covering it with an older version. This one left behind
as an example of how to use this function,but well, anyways.
"""
test_shoulder1_cam1 = save_unseen_frames_per_cam("C:\\Users\\dongq\\DeepLabCut\\Han-Qiwei-2020-02-21\\videos\\exp00001DeepCut_resnet50_HanFeb21shuffle1_1030000_labeled.mp4",shoulder1_unseen_points_cam1_loc,46610,'shoulder1_cam1_unseen_points_unlabeled') 

#%% function to plot all these "unseen" points on pictures to see where
#   their positions roughly are
def plot_unseen_points_on_pics(img_dir,x,y,marker_name,cam_name):
    img = mpimg.imread(img_dir)
    fig,ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(x,y)
    ax.set(xlabel='x axis',ylabel='y axis',
       title = marker_name + ' points only seen by ' + cam_name)
    plt.show()
    fig.savefig(marker_name+'_'+cam_name+'.png')



#%%And plot them out to see if there's anything interesting to learn about
img_base_dir = 'C:\\Users\\dongq\\DeepLabCut\\Han-Qiwei-2020-02-21\\videos'
"""
img = mpimg.imread(image_base_dir + '\\arm1_cam1_unseen_points_unlabeled\\frame29581.jpg')
fig,ax = plt.subplots()
ax.imshow(img)
ax.scatter(arm1_cam1_unseen_points_x,arm1_cam1_unseen_points_y)
ax.set(xlabel='x axis',ylabel='y axis',
       title='Arm1 points only seen by Cam1')
plt.show()


#plt.colorbar()?
"""
plot_unseen_points_on_pics(img_base_dir+'\\shoulder1_cam1_unseen_points_unlabeled\\frame8950.jpg', shoulder1_cam1_unseen_points_x, shoulder1_cam1_unseen_points_y, 'shoulder1', 'cam1')
plot_unseen_points_on_pics(img_base_dir+'\\shoulder1_cam2_unseen_points_unlabeled\\frame7357.jpg', shoulder1_cam2_unseen_points_x, shoulder1_cam2_unseen_points_y, 'shoulder1', 'cam2')
plot_unseen_points_on_pics(img_base_dir+'\\shoulder1_cam3_unseen_points_unlabeled\\frame20497.jpg', shoulder1_cam3_unseen_points_x, shoulder1_cam3_unseen_points_y, 'shoulder1', 'cam3')

plot_unseen_points_on_pics(img_base_dir+'\\arm1_cam1_unseen_points_unlabeled\\frame29581.jpg', arm1_cam1_unseen_points_x, arm1_cam1_unseen_points_y, 'arm1', 'cam1')
plot_unseen_points_on_pics(img_base_dir+'\\arm1_cam2_unseen_points_unlabeled\\frame19040.jpg', arm1_cam2_unseen_points_x, arm1_cam2_unseen_points_y, 'arm1', 'cam2')
plot_unseen_points_on_pics(img_base_dir+'\\arm1_cam3_unseen_points_unlabeled\\frame34094.jpg', arm1_cam3_unseen_points_x, arm1_cam3_unseen_points_y, 'arm1', 'cam3')

plot_unseen_points_on_pics(img_base_dir+'\\arm2_cam1_unseen_points_unlabeled\\frame30068.jpg', arm2_cam1_unseen_points_x, arm2_cam1_unseen_points_y, 'arm2', 'cam1')
plot_unseen_points_on_pics(img_base_dir+'\\arm2_cam2_unseen_points_unlabeled\\frame19120.jpg', arm2_cam2_unseen_points_x, arm2_cam2_unseen_points_y, 'arm2', 'cam2')
plot_unseen_points_on_pics(img_base_dir+'\\arm2_cam3_unseen_points_unlabeled\\frame14161.jpg', arm2_cam3_unseen_points_x, arm2_cam3_unseen_points_y, 'arm2', 'cam3')

plot_unseen_points_on_pics(img_base_dir+'\\elbow1_cam1_unseen_points_unlabeled\\frame40382.jpg', elbow1_cam1_unseen_points_x, elbow1_cam1_unseen_points_y, 'elbow1', 'cam1')
plot_unseen_points_on_pics(img_base_dir+'\\elbow1_cam2_unseen_points_unlabeled\\frame4846.jpg', elbow1_cam2_unseen_points_x, elbow1_cam2_unseen_points_y, 'elbow1', 'cam2')
plot_unseen_points_on_pics(img_base_dir+'\\elbow1_cam3_unseen_points_unlabeled\\frame45315.jpg', elbow1_cam3_unseen_points_x, elbow1_cam3_unseen_points_y, 'elbow1', 'cam3')

plot_unseen_points_on_pics(img_base_dir+'\\elbow2_cam1_unseen_points_unlabeled\\frame40969.jpg', elbow2_cam1_unseen_points_x, elbow2_cam1_unseen_points_y, 'elbow2', 'cam1')
plot_unseen_points_on_pics(img_base_dir+'\\elbow2_cam2_unseen_points_unlabeled\\frame2038.jpg', elbow2_cam2_unseen_points_x, elbow2_cam2_unseen_points_y, 'elbow2', 'cam2')
plot_unseen_points_on_pics(img_base_dir+'\\elbow2_cam3_unseen_points_unlabeled\\frame3241.jpg', elbow2_cam3_unseen_points_x, elbow2_cam3_unseen_points_y, 'elbow2', 'cam3')

plot_unseen_points_on_pics(img_base_dir+'\\wrist1_cam1_unseen_points_unlabeled\\frame299.jpg', wrist1_cam1_unseen_points_x, wrist1_cam1_unseen_points_y, 'wrist1', 'cam1')
plot_unseen_points_on_pics(img_base_dir+'\\wrist1_cam2_unseen_points_unlabeled\\frame40090.jpg', wrist1_cam2_unseen_points_x, wrist1_cam2_unseen_points_y, 'wrist1', 'cam2')
plot_unseen_points_on_pics(img_base_dir+'\\wrist1_cam3_unseen_points_unlabeled\\frame23669.jpg', wrist1_cam3_unseen_points_x, wrist1_cam3_unseen_points_y, 'wrist1', 'cam3')

plot_unseen_points_on_pics(img_base_dir+'\\wrist2_cam1_unseen_points_unlabeled\\frame2909.jpg', wrist2_cam1_unseen_points_x, wrist2_cam1_unseen_points_y, 'wrist2', 'cam1')
plot_unseen_points_on_pics(img_base_dir+'\\wrist2_cam2_unseen_points_unlabeled\\frame45507.jpg', wrist2_cam2_unseen_points_x, wrist2_cam2_unseen_points_y, 'wrist2', 'cam2')
plot_unseen_points_on_pics(img_base_dir+'\\wrist2_cam3_unseen_points_unlabeled\\frame3489.jpg', wrist2_cam3_unseen_points_x, wrist2_cam3_unseen_points_y, 'wrist2', 'cam3')

plot_unseen_points_on_pics(img_base_dir+'\\hand1_cam1_unseen_points_unlabeled\\frame828.jpg', hand1_cam1_unseen_points_x, hand1_cam1_unseen_points_y, 'hand1', 'cam1')
plot_unseen_points_on_pics(img_base_dir+'\\hand1_cam2_unseen_points_unlabeled\\frame19228.jpg', hand1_cam2_unseen_points_x, hand1_cam2_unseen_points_y, 'hand1', 'cam2')
plot_unseen_points_on_pics(img_base_dir+'\\hand1_cam3_unseen_points_unlabeled\\frame9668.jpg', hand1_cam3_unseen_points_x, hand1_cam3_unseen_points_y, 'hand1', 'cam3')

plot_unseen_points_on_pics(img_base_dir+'\\hand2_cam1_unseen_points_unlabeled\\frame9550.jpg', hand2_cam1_unseen_points_x, hand2_cam1_unseen_points_y, 'hand2', 'cam1')
plot_unseen_points_on_pics(img_base_dir+'\\hand2_cam2_unseen_points_unlabeled\\frame40384.jpg', hand2_cam2_unseen_points_x, hand2_cam2_unseen_points_y, 'hand2', 'cam2')
plot_unseen_points_on_pics(img_base_dir+'\\hand2_cam3_unseen_points_unlabeled\\frame3891.jpg', hand2_cam3_unseen_points_x, hand2_cam3_unseen_points_y, 'hand2', 'cam3')

plot_unseen_points_on_pics(img_base_dir+'\\hand3_cam1_unseen_points_unlabeled\\frame40400.jpg', hand3_cam1_unseen_points_x, hand3_cam1_unseen_points_y, 'hand3', 'cam1')
plot_unseen_points_on_pics(img_base_dir+'\\hand3_cam2_unseen_points_unlabeled\\frame40335.jpg', hand3_cam2_unseen_points_x, hand3_cam2_unseen_points_y, 'hand3', 'cam2')
plot_unseen_points_on_pics(img_base_dir+'\\hand3_cam3_unseen_points_unlabeled\\frame5529.jpg', hand3_cam3_unseen_points_x, hand3_cam3_unseen_points_y, 'hand3', 'cam3')



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
#%%
#fig,ax = plt.subplots()
#plt.style.use('ggplot')
plt.figure()
plt.hist(cam1_likelihood_values_binned)
plt.title('CAM1 likelihood distribution')
plt.show()

plt.figure()
plt.hist(cam2_likelihood_values_binned)
plt.title('CAM2 likelihood distribution')
plt.show()

plt.figure()
plt.hist(cam3_likelihood_values_binned)
plt.title('CAM3 likelihood distribution')
plt.show()

plt.figure()
plt.hist(cam4_likelihood_values_binned)
plt.title('CAM4 likelihood distribution')
plt.show()

#%%
"""
    img = mpimg.imread(img_dir)
    fig,ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(x,y)
    ax.set(xlabel='x axis',ylabel='y axis',
       title = marker_name + ' points only seen by ' + cam_name)
    plt.show()
    fig.savefig(marker_name+'_'+cam_name+'.png')
"""

#%% Random Plots for PPT


