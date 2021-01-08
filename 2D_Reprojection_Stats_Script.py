# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:31:28 2020

@author: robin
"""
import os
import cv2
import numpy as np
import pandas as pd
from numpy import array as arr
from matplotlib import pyplot as plt
from utils.utils import load_config
from utils.calibration_utils import *
from triangulation.triangulate import *
from calibration.extrinsic import *
import math

config_path = 'C:/Users/dongq/DeepLabCut/Han-Qiwei-2020-02-21/config_20200221_static.toml'
config = load_config(config_path)

Recovery_3D_path = "C:/Users/dongq/DeepLabCut/Han-Qiwei-2020-02-21/Han-20200221.json"


file_dir = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos'

g_truth = r'\Ground_truth_segments_2.txt'

#%%
from triangulation.triangulate import reconstruct_3d
recovery = reconstruct_3d(config)

#%% Save 3d recovery json file
import numpy as np
from json import JSONEncoder
import json

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

with open(Recovery_3D_path, "w") as write_file:
    json.dump(recovery, write_file, cls=NumpyArrayEncoder)

#%% Load 3d recovery json file
import numpy as np
from json import JSONEncoder
import json

with open(Recovery_3D_path, "r") as read_file:
    print("Converting JSON encoded data into Numpy array")
    recovery = json.load(read_file)
    
recovery['registration_mat'] = np.array(recovery['registration_mat'])
recovery['center'] = np.array(recovery['center'])
#%% Do 2D reprojection
path, videos, vid_indices = get_video_path(config)
intrinsics = load_intrinsics(path, vid_indices)
extrinsics = load_extrinsics(path)

joints = config['labeling']['bodyparts_interested']

# Path to 3D data
#path_to_3d = 'C:/Users/dongq/DeepLabCut/Han-Qiwei-2020-02-21/videos/output_3d_data.csv'
path_to_3d = os.path.join(config['triangulation']['reconstruction_output_path'], 'output_3d_data.csv')

# Folder where you want to save 2D reprojected csv files
#path_to_save = 'F:/Han-Qiwei-2020-02-21-20201029T204821Z-001/Han-Qiwei-2020-02-21/static-data/reproject'
path_to_save = config['triangulation']['reconstruction_output_path']

# You should find a snapshot something like this:
# DeepCut_resnet50_Pop_freeReach_cam_1_0212Feb17shuffle1_620000  in 2.1 DLC
# If you use 2.2 DLC, the snapshot would look like the below:
snapshot = 'DLC_resnet50_HanFeb21shuffle1_1030000'

# Frame numbers you want to reproject.
# If you want to use the full data, put frame_counts = []
#frame_counts = np.array([1000,2000,3000,4000,5000,6000])
frame_counts = []

data_3d = pd.read_csv(path_to_3d)

if len(frame_counts) == 0:
    frame_counts = np.arange(len(data_3d))
else:
    data_3d = data_3d.iloc[frame_counts, :]
    
l = len(data_3d)

iterables = [[snapshot], joints, ['x', 'y']]
header = pd.MultiIndex.from_product(iterables, names=['scorer', 'bodyparts', 'coords'])

data_2d = {ind: [] for ind in vid_indices}

for video, vid_idx in zip(videos, vid_indices):
    df = pd.DataFrame(np.zeros((l, len(joints)*2)), index=frame_counts, columns=header)
    
    cameraMatrix = np.matrix(intrinsics[vid_idx]['camera_mat'])
    distCoeffs = np.array(intrinsics[vid_idx]['dist_coeff'])
    Rt = np.matrix(extrinsics[vid_idx])
    rvec, tvec = get_rtvec(Rt)
    
    for joint in joints:
        x = data_3d[joint+'_x']
        y = data_3d[joint+'_y']
        z = data_3d[joint+'_z']
        objectPoints = np.vstack([x,y,z]).T
        objectPoints = objectPoints.dot((np.linalg.inv(recovery['registration_mat'].T))) + recovery['center']
        coord_2d = np.squeeze(cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)[0], axis=1)
        
        df[snapshot][joint] = coord_2d
        
    data_2d[vid_idx].append(df)
    
    # df.to_csv(os.path.join(path_to_save, 
    #                        os.path.basename(video).split('.')[0] +'.csv'), mode='w')

#%% Calculate 2D reprojection error throughout the whole dataset

def l2_norm(x,y):
    #if 
    return math.sqrt((x2-x1)**2 + (y2-y1)**2) 

"""
err_list[0:12] cam1
err_list[13:25] cam2
err_list[26:38] cam3

    err_list[0]: cam1 shoulder1
    err_list1[1]: cam1 arm1
    err_list......
    err_list[10]: cam1 X
    err_list[11]:cam1 Y
    err_list[12]:cam1 Z
"""
err_list = []
err_mean_list = []
paths_to_2d_data = config['paths_to_2d_data']
for vid_idx, path in zip(vid_indices, paths_to_2d_data):
    df_dlc = pd.read_csv(path, header=[1,2], index_col=0)
    #frame_total = np.arange(df_dlc.shape[0])
    df_rep = data_2d[vid_idx][0]
    for joint in joints:
        x = np.array(df_dlc[joint]['x'][frame_counts] - 
                     df_rep[snapshot][joint]['x'][frame_counts])
        y = np.array(df_dlc[joint]['y'][frame_counts] - 
                     df_rep[snapshot][joint]['y'][frame_counts])
        
        coord = np.stack([x, y], axis=1)
        coord = np.ma.array(coord, mask=np.isnan(coord)) #https://stackoverflow.com/questions/37749900/how-to-disregard-the-nan-data-point-in-numpy-array-and-generate-the-normalized-d
        err = np.linalg.norm(coord, axis=1) #Calculate the error for each marker #Can't handle NaN
        err_list.append(err)
        err = np.ma.array(err,mask=np.isnan(err)) #https://stackoverflow.com/questions/37749900/how-to-disregard-the-nan-data-point-in-numpy-array-and-generate-the-normalized-d
        print(joint+': '+str(np.mean(err)))
        err_mean_list.append(np.mean(err))

#%% Isolate expeirment phase data out (but in 3D)
        

frames_per_second = 30
seconds_per_minute = 60

f = open(file_dir + g_truth, 'r')
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

f_frame_list = list()

for i in range(len(f_frame)):
    f_frame_list = f_frame_list + list(range(int(f_frame[i,0]),int(f_frame[i,1]+1)))
        
ground_truth_segment = np.zeros((data_3d.shape[0]))        
for i in range(len(f_frame_list)):
    #print(i)
    ground_truth_segment[f_frame_list[i]] = 1
    
"""
f_frame_list: list of frame numbers in which monkey is doing experiment
ground_truth_segment: list that shows whether in each frame monkey is doing experiment or not 
(1 means in experiment, 0 means not in experiment)
"""
#%% Define function calculating speed in 3D

def speed_calc_3D(X,Y,Z,fps):
    temp_df = np.empty((X.shape[0]))
    temp_df[:] = np.nan
    for i in range(X.shape[0]-1): #NOT SURE IF THIS IS GOING TO WORK
        if not math.isnan(X[i]) and not math.isnan(X[i+1]): #if one of the three coordinates are not NaN, the other two will not be NaN
            temp_speed = np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2) #cm per second, BUT the numbers aren't right.
            #temp_speed = np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)
            temp_df[i] = temp_speed
    #return temp_df/0.03333
    return temp_df


#%% Get the experiment only phase (and the corresponding speed data) using f_frame_list
try:
    list_to_delete = ['pointX_x','pointX_y','pointX_z','pointX_error','pointX_ncams','pointX_score','pointY_x','pointY_y','pointY_z','pointY_error','pointY_ncams','pointY_score','pointZ_x','pointZ_y','pointZ_z','pointZ_error','pointZ_ncams','pointZ_score','shoulder1_error','shoulder1_ncams','shoulder1_score','arm1_error','arm1_ncams','arm1_score','arm2_error','arm2_ncams','arm2_score','shoulder1_error','elbow1_ncams','elbow1_score','elbow1_error','elbow2_error','elbow2_ncams','elbow2_score','wrist1_error','wrist1_ncams','wrist1_score','wrist2_error','wrist2_ncams','wrist2_score','hand1_error','hand1_ncams','hand1_score','hand2_error','hand2_ncams','hand2_score','hand3_error','hand3_ncams','hand3_score']
    data_3d_cleaned = data_3d.drop(columns = list_to_delete)
except:
    print("error deleting unnecessary columns")
data_3d_np = data_3d_cleaned.to_numpy()
data_3d_np_exp = data_3d_np[f_frame_list] #data in 3d, numpy version, experiment phase only
#data_3d_np_exp_spd = np.diff(data_3d_np_exp,axis=0)

df_speed = np.zeros((data_3d_np.shape[0],math.floor(data_3d_np.shape[1]/3)))
for i in range(df_speed.shape[1]):
    X = i*3 + 0
    Y = i*3 + 1
    Z = i*3 + 2
    speed_3D = speed_calc_3D(data_3d_np[:,X],data_3d_np[:,Y],data_3d_np[:,Z],frames_per_second)
    print(speed_3D)
    df_speed[:,i] = speed_3D * 0.001 #TODO! Make sure this part is right

df_speed_exp = np.zeros((data_3d_np_exp.shape[0],math.floor(data_3d_np_exp.shape[1]/3)))
for i in range(df_speed_exp.shape[1]):
    X = i*3 + 0
    Y = i*3 + 1
    Z = i*3 + 2
    speed_3D = speed_calc_3D(data_3d_np_exp[:,X],data_3d_np_exp[:,Y],data_3d_np_exp[:,Z],frames_per_second)
    print(speed_3D)
    df_speed_exp[:,i] = speed_3D * 0.001 #TODO! Make sure this *0.001 is right from mm to m?
    

    
    
#%% add frame numbers to df_speed
    
df_speed_fnum = np.zeros((df_speed.shape[0],df_speed.shape[1]+1))
df_speed_fnum[:,0:df_speed_fnum.shape[1]-1] = df_speed
df_speed_fnum[:,-1] = data_3d_np[:,-1]
# = [df_speed, data_3d_np_exp[:,-1]]

df_speed_exp_fnum = np.zeros((df_speed_exp.shape[0],df_speed_exp.shape[1]+1))
df_speed_exp_fnum[:,0:df_speed_exp_fnum.shape[1]-1] = df_speed_exp
df_speed_exp_fnum[:,-1] = data_3d_np_exp[:,-1]

"""
df_speed: data speed in 3D, numpy, without frame numbers
df_speed_fnum:data speed in 3D, numpy, with frame numbers
data_3d_np: data in 3d, numpy version, (with frame numbers of course)
data_3d_np_exp: data in 3d, numpy version, experiment phase only
df_speed_exp: data speed in 3d, numpy, experiment phase only
df_speed_exp_fnum: data speed in 3d, numpy, experiment phase only, with frame number
"""
#%% Statistical plot of reprojection error
#plt.figure()
#plt.hist(err_list)
def average_markers(cam, num_avg_markers):
    average_cam = np.zeros((len(cam[0]),num_avg_markers))
    average_cam[:,0] = cam[0]
    average_cam[:,1] = (cam[1] + cam[2])/2
    #average_cam[:,1] = cam[:,2]
    average_cam[:,2] = (cam[3] + cam[4])/2
    average_cam[:,3] = (cam[5] + cam[6])/2
    average_cam[:,4] = (cam[7] + cam[8] + cam[9])/3
    return average_cam

avg_err_list_cam1 = average_markers(err_list[0:10], 5)

marker_names_averaged = ['shoulder','arm','elbow','wrist','hand']

bin_size=500
fig,ax = plt.subplots(figsize=(12,6))

for i in range(avg_err_list_cam1.shape[1]):
    plt.hist(avg_err_list_cam1[:,i],alpha=0.9,bins=bin_size, label=marker_names_averaged[i],linewidth = 3,histtype='step')

#cumulative='True' density=True stacked=True,,
plt.xlabel("reprojection error (in pixels)")
plt.ylabel('number of markers')
plt.xlim(0,25)
#ax.set_xscale('log')
plt.legend()
plt.show()
# =============================================================================
# def average_markers(cam, num_markers):
#     average_cam = np.zeros((cam.shape[0],num_markers))
#     average_cam[:,0] = cam[:,0]
#     average_cam[:,1] = (cam[:,1] + cam[:,2])/2
#     #average_cam[:,1] = cam[:,2]
#     average_cam[:,2] = (cam[:,3] + cam[:,4])/2
#     average_cam[:,3] = (cam[:,5] + cam[:,6])/2
#     average_cam[:,4] = (cam[:,7] + cam[:,8] + cam[:,9])/3
#     return average_cam
# 
# #reference: plt.hist(x1_limited,density=True,stacked=True,range=(0,1.5),bins=bin_size,alpha=0.5,label='2D ' + str(x1_limited.shape[0]) + ' frames',histtype='step',linewidth = 3)
# 
# marker_names_averaged = ['shoulder','arm','elbow','wrist','hand']
# 
# average_norm_c1 = average_markers(norm_c1,len(marker_names_averaged))
# average_norm_c2 = average_markers(norm_c2,len(marker_names_averaged))
# average_norm_c3 = average_markers(norm_c3,len(marker_names_averaged))
# average_norm_c4 = average_markers(norm_c4,len(marker_names_averaged))
# 
# bin_size = 100
# fig, ax = plt.subplots(figsize=(12,6))
# 
# for i in range(average_norm_c1.shape[1]):
#     #kde = stats.gaussian_kde(average_norm_c1[:,i])  
#     #xx = np.linspace(0,2,500)
#     #plt.hist(average_norm_c1[:,i],stacked=True,alpha=0.9,range=(0,1),bins=bin_size, label=marker_names_averaged[i],histtype='step',linewidth = 3)
#     plt.hist(average_norm_c1[:,i],stacked=True,alpha=0.9,range=(0,1),bins=bin_size, label=marker_names_averaged[i],histtype='step',linewidth = 3,density=True,cumulative='True')
#     #plt.hist(average_norm_c1[:,i][average_norm_c1[:,i]>0.001],stacked=True,alpha=0.9,range=(0,2),bins=bin_size, label=marker_names_averaged[i],histtype='step',linewidth = 3)
#     #plt.plot(xx,kde(xx),label=marker_names_averaged[i],linewidth=4)
# #plt.hist(norm_c1,alpha=0.5,range=(0,40),bins=bin_size, label='2D ',histtype='step',linewidth = 3)
# #plt.yscale('log')
# 
# #plt.hist(norm_c1,density=True,stacked=True,alpha=0.5,range=(0,50),bins=bin_size, label='2D ',histtype='step',linewidth = 3)
# 
# plt.xlabel("Speed (L2 norm of dx and dy)",**font_medium)
# plt.ylabel("Number of frames",**font_medium)
# plt.title("Marker dx dy",**font_medium)
# #plt.title("Wrist2 speed distribution",**font_medium)
# 
# y_ticks = np.arange(0,1.00000000001,0.05)
# 
# 
# plt.xticks(fontsize = 16)
# 
# ax.set_yticks(y_ticks)
# plt.yticks(fontsize = 16)
# #ax.get_xticklabels()[-2].set_color("red") (https://stackoverflow.com/questions/41924963/formatting-only-selected-tick-labels)
# ax.get_yticklabels()[-2].set_color('red')
# ax.get_yticklabels()[-2].set_weight('bold')
# #plt.ylim(0,400)
# #plt.yscale('log')
# ax.grid(True)
# #plt.gca().yaxis.set_major_formatter(PercentFormatter(bin_size))
# plt.legend(fontsize=12)
# plt.show()
# =============================================================================

#%% Plot a section of the data, with averaged marker reprojection errors, speed data, and position data

"""
take shoulder, arm's average, elbow's average, and wrist and hand's average.
"""
def average_markers_wristhand(cam, num_avg_markers):
    average_cam = np.zeros((len(cam[0]),num_avg_markers))
    average_cam[:,0] = cam[0]
    average_cam[:,1] = (cam[1] + cam[2])/2
    #average_cam[:,1] = cam[:,2]
    average_cam[:,2] = (cam[3] + cam[4])/2
    average_cam[:,3] = (cam[5] + cam[6] + cam[7] + cam[8] + cam[9])/5
    return average_cam

def average_speed_wristhand(data):
    average_spd = np.zeros((df_speed_fnum.shape[0],4)) #MAGIK NUMBER!!
    average_spd[:,0] = data[:,0]
    average_spd[:,1] = (data[:,1] + data[:,2]) / 2
    average_spd[:,2] = (data[:,3] + data[:,4]) / 2
    average_spd[:,3] = (data[:,5] + data[:,6] + data[:,7] + data[:,8] + data[:,9]) / 5
    return average_spd
"""
df_speed: data speed in 3D, numpy, without frame numbers
***df_speed_fnum:data speed in 3D, numpy, with frame numbers
data_3d_np: data in 3d, numpy version, (with frame numbers of course)
data_3d_np_exp: data in 3d, numpy version, experiment phase only
df_speed_exp: data speed in 3d, numpy, experiment phase only
df_speed_exp_fnum: data speed in 3d, numpy, experiment phase only, with frame number
"""

x_duration = 500
x_start = 7800
x_end  = x_start + x_duration
fps = frames_per_second

X = np.linspace(0,frame_counts.shape[0],frame_counts.shape[0])/fps

avg_err_list_cam1_wristhand = average_markers_wristhand(err_list[0:10], 4)

average_speed_wristhand = average_speed_wristhand(df_speed_fnum)

marker_names_averaged = ['shoulder','arm','elbow','wrist and hand']


#marker_names = ['shoulder1','arm1','arm2','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']

plt.figure(figsize=(6,9))
for i in range(len(marker_names_averaged)):    
    plt.subplot(len(marker_names_averaged),1,i+1)
    
    plt.plot(X[x_start:x_end],avg_err_list_cam1[x_start:x_end,i],label='error') #plot error
    plt.plot(X[x_start:x_end],average_speed_wristhand[x_start:x_end,i]*1000, alpha=0.35,label='speed')
    
    plt.legend(fontsize=10)
    plt.ylabel(marker_names_averaged[i])
    plt.ylim(0,25)
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.tick_params(axis='x',labelbottom=True)
plt.xlabel("time (in seconds)",fontsize=16)
#plt.yticks(fontsize=2)
plt.legend(fontsize=10)
plt.show()
    
#%% EXAMPLE 
# =============================================================================
# bin_size = 12
# #reference: plt.hist(x1_limited,density=True,stacked=True,range=(0,1.5),bins=bin_size,alpha=0.5,label='2D ' + str(x1_limited.shape[0]) + ' frames',histtype='step',linewidth = 3)
# plt.figure(figsize=(12,9))
# marker_names = ['shoulder1','arm1','arm2','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
# 
# for i in range(norm_c1.shape[1]):
#     kde = stats.gaussian_kde(norm_c1[:,i])  
#     xx = np.linspace(0,2,500)
#     plt.hist(norm_c1[:,i],stacked=True,alpha=0.9,range=(0,2),bins=bin_size, label=marker_names[i],histtype='step',linewidth = 3)
#     plt.plot(xx,kde(xx),label=marker_names[i])
# #plt.hist(norm_c1,alpha=0.5,range=(0,40),bins=bin_size, label='2D ',histtype='step',linewidth = 3)
# plt.yscale('log')
# 
# #plt.hist(norm_c1,density=True,stacked=True,alpha=0.5,range=(0,50),bins=bin_size, label='2D ',histtype='step',linewidth = 3)
# 
# plt.xlabel("",**font_medium)
# plt.ylabel("number of frames",**font_medium)
# plt.title("Marker dx dy",**font_medium)
# #plt.title("Wrist2 speed distribution",**font_medium)
# 
# 
# plt.xticks(fontsize = 16)
# plt.yticks(fontsize = 16)
# plt.ylim(0.001,20000)
# 
# #plt.gca().yaxis.set_major_formatter(PercentFormatter(bin_size))
# plt.legend(fontsize=12)
# plt.show()
# =============================================================================
