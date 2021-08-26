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

cam_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/videos/results_Qiwei_New_Iter3'
cam1_filename = '/Crackle_20201203_00001DLC_resnet50_TestDec3shuffle1_1030000filtered.csv'
cam2_filename = '/Crackle_20201203_00002DLC_resnet50_TestDec3shuffle1_1030000filtered.csv'
cam3_filename = '/Crackle_20201203_00003DLC_resnet50_TestDec3shuffle1_1030000filtered.csv'
cam4_filename = '/Crackle_20201203_00004DLC_resnet50_TestDec3shuffle1_1030000filtered.csv'


#config_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/config_Crackle_20201203_RT3D_static.toml'

#config_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/Iteration_2_results/config_Crackle_20201203_RT3D_static_Iter2.toml'
#config_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/Iteration_3_results/config_Crackle_20201203_RT3D_static_Iter3.toml'
config_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/Iteration_3_results/config_Crackle_20201203_RT2D_static_Iter3.toml'
config = load_config(config_path)

Recovery_3D_path = "C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/Iteration_3_results/Crackle_20201203_RT2D.json"

reprojected_file_prefix = 'Crackle_20201203_RT2D' #The file name of Recovery_3D_path

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
#%%
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
snapshot = 'DLC_resnet50_TestDec3shuffle1_1030000filtered'



#%% Get the array for trial segmentation
#if experiment_phase_only == 1:
    #df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')
f = open(r"C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\Ground_truth_segments_2020-12-03-RT2D-ForHandleData.txt", "r") 

frames_per_second = 24
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
##frame_counts = np.array([1,2,3,4,5,10,20,30,40,50,60,70,80,90,100,1000,2000,3000,4000,5000])
#%%
# Frame numbers you want to reproject.
# If you want to use the full data, put frame_counts = []
#frame_counts = np.array([1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000])

frame_counts = [] #All the data (including non-exp phase)
#frame_counts = np.array(f_frame_list) #Data in exp phase only

#%% Squeeze the likelihood data from the original dataset into the reinterpolated
#2D dataset, for the interpolation code. This is not the right solution, but 
#a fast one indeed.

data_3d = pd.read_csv(path_to_3d)

origin_cam1 = pd.read_csv(cam_path + cam1_filename)
origin_cam2 = pd.read_csv(cam_path + cam2_filename)
origin_cam3 = pd.read_csv(cam_path + cam3_filename)
origin_cam4 = pd.read_csv(cam_path + cam4_filename)

origin_cam = [origin_cam1,origin_cam2,origin_cam3,origin_cam4]
origin_cam_arr = []
likelihood_cols = [3,6,9,12,15,18,21,24,27,30,33]

for i in range(len(origin_cam)):
    origin_cam_np = origin_cam[i].to_numpy()
    origin_cam_np = np.delete(origin_cam_np,(0,1),axis=0)
    origin_cam_np_float = np.vstack(origin_cam_np[:,:]).astype(np.float)
    origin_cam_np_float_likelihood = origin_cam_np_float[:,likelihood_cols]
    
    origin_cam_arr.append(origin_cam_np_float_likelihood)
    



#%%

if len(frame_counts) == 0:
    frame_counts = np.arange(len(data_3d))
else:
    data_3d = data_3d.iloc[frame_counts, :]
    
l = len(data_3d)

iterables = [[snapshot], joints, ['x', 'y','likelihood']]
header = pd.MultiIndex.from_product(iterables, names=['scorer', 'bodyparts', 'coords'])

data_2d = {ind: [] for ind in vid_indices}

for video, vid_idx, cam_likelihood in zip(videos, vid_indices,origin_cam_arr):
    #df = pd.DataFrame(np.zeros((l, len(joints)*2)), index=frame_counts, columns=header)
    df = pd.DataFrame(np.zeros((l, len(joints)*3)), index=frame_counts, columns=header) #x,y, likelihood
    
    cameraMatrix = np.matrix(intrinsics[vid_idx]['camera_mat'])
    distCoeffs = np.array(intrinsics[vid_idx]['dist_coeff'])
    Rt = np.matrix(extrinsics[vid_idx])
    rvec, tvec = get_rtvec(Rt)
    
    #print(cam_likelihood)
    
    #for joint in joints:
    for i in range(len(joints)):  
        
        x = data_3d[joints[i]+'_x']
        y = data_3d[joints[i]+'_y']
        z = data_3d[joints[i]+'_z']
        objectPoints = np.vstack([x,y,z]).T
        objectPoints = objectPoints.dot((np.linalg.inv(recovery['registration_mat'].T))) + recovery['center']
        coord_2d = np.squeeze(cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)[0], axis=1)
        
        coord_2d_likelihood = np.zeros((coord_2d.shape[0],coord_2d.shape[1]+1))
        coord_2d_likelihood[:,0:2] = coord_2d
        coord_2d_likelihood[:,2] = cam_likelihood[:,i]
        
        df[snapshot][joints[i]] = coord_2d_likelihood
        
    data_2d[vid_idx].append(df)
    
    # df.to_csv(os.path.join(path_to_save, 
    #                        os.path.basename(video).split('.')[0] +'.csv'), mode='w')
#%% save data_2d

for i in range(len(data_2d)):
    data_2d[str(i+1)][0].to_csv(reprojected_file_prefix + "_reprojeted_2D_data_cam_" + str(i) + ".csv")


#%% 2D error

avg_err = []

paths_to_2d_data = config['paths_to_2d_data']
#paths_to_2d_data = config['paths_to_2d_data'][1]
for vid_idx, path in zip(vid_indices, paths_to_2d_data):
    df_dlc = pd.read_csv(path, header=[1,2], index_col=0)
    df_rep = data_2d[vid_idx][0]
    
    cam_err = []
    
    for joint in joints:
        x = np.array(df_dlc[joint]['x'][frame_counts] - 
                     df_rep[snapshot][joint]['x'][frame_counts])
        y = np.array(df_dlc[joint]['y'][frame_counts] - 
                     df_rep[snapshot][joint]['y'][frame_counts])
        x_nonan = np.unique(x[~np.isnan(x)])
        y_nonan = np.unique(y[~np.isnan(y)])
        #coord = np.stack([x, y], axis=1)
        coord = np.stack([x_nonan, x_nonan], axis=1)
        #print(str(x) + str(y))
        #coord_non_nan = np.unique(coord[~np.isnan(coord)])
        #print(coord_non_nan)
        #print(coord_non_nan.shape)
        #print(coord.shape)
        #err_non_nan = np.linalg.norm(coord_non_nan,axis=1)
        err = np.linalg.norm(coord, axis=1)
        print(joint+': '+str(np.mean(err)))
        cam_err.append(np.mean(err))
        #print(joint+': '+str(np.mean(err_non_nan)))
    print("\n")
    avg_err.append(cam_err)


#path, videos, vid_indices = get_video_path(config)
#%%
# =============================================================================
# #cams = ['Crackle_20201203_00007.avi','Crackle_20201203_00008.avi','Crackle_20201203_00009.avi','Crackle_20201203_00010.avi']
# cams = ['Crackle_20201203_00001.avi','Crackle_20201203_00002.avi','Crackle_20201203_00003.avi','Crackle_20201203_00004.avi']
# 
# #cams = ['Crackle_20201203_00008.avi']
# 
# paths_to_save = [os.path.join(path_to_save, cam.split('.')[0]) for cam in cams]
# #vidfolder = 'C:/Users/dongq/DeepLabCut/Han-Qiwei-2020-02-21/videos/'
# vidfolder = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/videos/'
# vidpaths = [os.path.join(vidfolder, cam) for cam in cams]
# 
# def extract_specific_frames(df_dlc, df_rep, vidpath, frame_counts, path_to_save): 
#     if not os.path.exists(vidpath):
#         print('Video does not exist.')
#         return
#             
#     cap = cv2.VideoCapture(vidpath) 
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration = frame_num/fps
#     count = len(frame_counts)
#     
#     colorclass = plt.cm.ScalarMappable(cmap='jet')
#     C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
#     colors = C[:, :3]
#     
#     with tqdm(total=count) as pbar:
#         for f, frame_count in enumerate(frame_counts):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
#             ret, frame = cap.read()
#             plt.figure()
#             plt.imshow(frame)
#             
#             for i, color in enumerate(colors):
#                 for joint, color in zip(joints, colors):
#                     plt.scatter(df_dlc[joint]['x'][frame_count],
#                                 df_dlc[joint]['y'][frame_count], 
#                                 alpha=0.1, s=20, color=color, marker="o")
#                     
#                     plt.scatter(df_rep[snapshot][joint]['x'][frame_count],
#                                 df_rep[snapshot][joint]['y'][frame_count], 
#                                 alpha=0.1,s=80, color=color, marker="+")
# 
#             plt.savefig(os.path.join(path_to_save, 'img' + str(frame_count).zfill(6) + '.png'),
#                         bbox_inches='tight', pad_inches=0)
#             plt.close()
#             pbar.update(1)
#             
#     print('\n{} frames were extracted.'.format(count))
# 
# #vid_indices = [vid_indices[1]]
# #paths_to_2d_data = [paths_to_2d_data[1]]
# #frame_counts = np.array([1000,2000,3000,4000,5000,10000,20000,30000])
# 
# for vid_idx, vidpath, dlc_path, img_path in zip(vid_indices, vidpaths, paths_to_2d_data, paths_to_save):
#     if not os.path.exists(img_path):
#         os.mkdir(img_path)
#         
#     df_dlc = pd.read_csv(dlc_path, header=[1,2], index_col=0)
#     df_rep = data_2d[vid_idx][0]
#     
#     extract_specific_frames(df_dlc, df_rep, vidpath, frame_counts, img_path)
#     
# =============================================================================
