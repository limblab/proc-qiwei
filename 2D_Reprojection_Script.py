# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 15:58:01 2020

@author: Min
"""

import os
import cv2
import numpy as np
import pandas as pd
from numpy import array as arr
from utils.calibration_utils import *
from triangulation.triangulate import *
from calibration.extrinsic import *
from utils.utils import load_config
import math
#%%
#config = load_config('config_20200804_FR_static.toml' )
#config = load_config('config_20200922_RT2D_static.toml' )
config = load_config('config_20200221_static.toml' )
#config = load_config('config_20200804_FR.toml' )
path, videos, vid_indices = get_video_path(config)
intrinsics = load_intrinsics(path, vid_indices)
extrinsics = load_extrinsics(path)
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
with open("Han-20200221.json", "w") as write_file:
    json.dump(recovery, write_file, cls=NumpyArrayEncoder)
    
#%% Load 3d recovery json file
import numpy as np
from json import JSONEncoder
import json
with open("Han-20200221.json", "r") as read_file:
    print("Converting JSON encoded data into Numpy array")
    recovery = json.load(read_file)
recovery['registration_mat'] = np.array(recovery['registration_mat'])
recovery['center'] = np.array(recovery['center'])

#%%
joints = config['labeling']['bodyparts_interested']
#joints = ['shoulder1','arm1','arm2']
#%%
# Path to 3D data
# path_to_3d = '/media/minyoungpark/Min/pop_0811_1/iteration-12/output_3d_data_trimmed.csv'
path_to_3d = os.path.join(config['triangulation']['reconstruction_output_path'], 'output_3d_data.csv')
# Folder where you want to save 2D reprojected csv files
path_to_save = config['triangulation']['reconstruction_output_path']
# You should find a snapshot something like this:
# DeepCut_resnet50_Pop_freeReach_cam_1_0212Feb17shuffle1_620000  in 2.1 DLC
# If you use 2.2 DLC, the snapshot would look like the below:
snapshot = 'DeepCut_resnet50_HanFeb21shuffle1_1030000_filtered'
"""
exp00001DLC_resnet50_RocketJul29shuffle1_1030000.csv
↓
DLC_resnet50_RocketJul29shuffle1_1030000
"""
# Frame numbers you want to reproject.
# If you want to use the full data, put frame_counts = []
#%%
#frame_counts = np.array([35, 36, 40])
frame_counts = [] #Reproject whole dataset

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
    df.to_csv(os.path.join(path_to_save,
                           os.path.basename(video).split('.')[0] +'.csv'), mode='w')
   
#%%
import os
import cv2
import numpy as np
import pandas as pd
from numpy import array as arr
from matplotlib import pyplot as plt
from utils.calibration_utils import *
from triangulation.triangulate import *
from calibration.extrinsic import *

#cams = ['exp00001.avi','exp00002.avi','exp00003.avi','exp00004.avi']
cams = ['exp00001.avi','exp00002.avi','exp00003.avi']
frame_counts = np.arange(1100,1110)
paths_to_save = [os.path.join(path_to_save, cam.split('.')[0]) for cam in cams]
vidfolder = 'C:/Users/dongq/DeepLabCut/Han-Qiwei-2020-02-21/videos/'
vidpaths = [os.path.join(vidfolder, cam) for cam in cams]
xyzs = []
for joint in joints:
    x = data_3d[joint+'_x']
    y = data_3d[joint+'_y']
    z = data_3d[joint+'_z']
    c = np.stack([x,y,z], axis=1)
    xyzs.append(c)
xyzs = np.array(xyzs)
xyzs = np.transpose(xyzs, [1,0,2])
xyzs = np.reshape(xyzs, (xyzs.shape[0], -1))
xyzs_3d = xyzs[frame_counts]

def extract_specific_frames(xyzs_3d, vidpath, frame_counts, path_to_save):
    if not os.path.exists(vidpath):
        print('Video does not exist.')
        return
    cap = cv2.VideoCapture(vidpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_num/fps
    count = len(frame_counts)
    colorclass = plt.cm.ScalarMappable(cmap='jet')
    C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
    colors = C[:, :3]
    with tqdm(total=count) as pbar:
        for f, frame_count in enumerate(frame_counts):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            plt.figure()
            plt.imshow(frame)
            for i, color in enumerate(colors):
                xyz_p = np.expand_dims(xyzs_3d[f, 3*i:3*i+3], axis=0)
                xyz_p = xyz_p.dot((np.linalg.inv(recovery['registration_mat'].T))) + recovery['center']
                coord_2d = np.squeeze(cv2.projectPoints(xyz_p, rvec, tvec, cameraMatrix, distCoeffs)[0], axis=1)
                x = coord_2d[0][0]
                y = coord_2d[0][1]
                plt.scatter(x, y, s=2, color=color, marker='o')
            plt.savefig(os.path.join(path_to_save, 'img' + str(frame_count).zfill(6) + '.png'),
                        bbox_inches='tight', pad_inches=0, dpi=600)
            plt.close()
            pbar.update(1)
    print('\n{} frames were extracted.'.format(count))
    
for vid_idx, vidpath, path_to_save in zip(vid_indices, vidpaths, paths_to_save):
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    extract_specific_frames(xyzs_3d, vidpath, frame_counts, path_to_save)
    
    
    
#%% Check euclidian distance of the points between 2d-reprojected videos and DLC inferred videos
#%% Read in files

# =============================================================================
# inferred_folder = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos'
# inferred_1 = 'exp00001DLC_resnet50_HanAug4shuffle1_1030000.csv'
# inferred_2 = 'exp00002DLC_resnet50_HanAug4shuffle1_1030000.csv'
# inferred_3 = 'exp00003DLC_resnet50_HanAug4shuffle1_1030000.csv'
# #inferred_4 = 'exp00004DLC_resnet50_HanAug4shuffle1_1030000.csv'
# 
# reprojected_folder = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data'
# reproj_1 = 'calib00001.csv'
# reproj_2 = 'calib00002.csv'
# reproj_3 = 'calib00003.csv'
# reproj_4 = 'calib00004.csv'
# =============================================================================

inferred_folder = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos'
inferred_1 = 'exp00001DLC_resnet50_HanFeb21shuffle1_1030000.csv'
inferred_2 = 'exp00002DLC_resnet50_HanFeb21shuffle1_1030000.csv'
inferred_3 = 'exp00003DLC_resnet50_HanFeb21shuffle1_1030000.csv'
#inferred_4 = 'exp00004DLC_resnet50_HanAug4shuffle1_1030000.csv'

reprojected_folder = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos'
reproj_1 = 'calib00001.csv'
reproj_2 = 'calib00002.csv'
reproj_3 = 'calib00003.csv'
#reproj_4 = 'calib00004.csv'

inferred_cam1 = pd.read_csv(inferred_folder + '\\' + inferred_1, header=[1,2], index_col=0)
inferred_cam2 = pd.read_csv(inferred_folder + '\\' + inferred_2, header=[1,2], index_col=0)
inferred_cam3 = pd.read_csv(inferred_folder + '\\' + inferred_3, header=[1,2], index_col=0)
#inferred_cam4 = pd.read_csv(inferred_folder + '\\' + inferred_4, header=[1,2], index_col=0)

reproj_cam1 = pd.read_csv(reprojected_folder + '\\' + reproj_1, header=[1,2], index_col=0)
reproj_cam2 = pd.read_csv(reprojected_folder + '\\' + reproj_2, header=[1,2], index_col=0)
reproj_cam3 = pd.read_csv(reprojected_folder + '\\' + reproj_3, header=[1,2], index_col=0)
#reproj_cam4 = pd.read_csv(reprojected_folder + '\\' + reproj_4, header=[1,2], index_col=0)

#%% Get the x and y for both inferred DLC points and reprojected points from the DataFrames

frame_num = 1100
#bp = 'shoulder1'
bp = ['shoulder1','arm1','arm2','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
inferred_cam1_x = []
inferred_cam2_x = []
inferred_cam3_x = []
#inferred_cam4_x = []

inferred_cam1_y = []
inferred_cam2_y = []
inferred_cam3_y = []
#inferred_cam4_y = []

reproj_cam1_x = []
reproj_cam2_x = []
reproj_cam3_x = []
#reproj_cam4_x = []

reproj_cam1_y = []
reproj_cam2_y = []
reproj_cam3_y = []
#reproj_cam4_y = []

#x = inferred_cam1[bp, 'x'][frame_num]
for i in range(len(bp)):
    inferred_cam1_x.append(inferred_cam1[bp[i],'x'][frame_num])
    inferred_cam2_x.append(inferred_cam2[bp[i],'x'][frame_num])
    inferred_cam3_x.append(inferred_cam3[bp[i],'x'][frame_num])
    #inferred_cam4_x.append(inferred_cam4[bp[i],'x'][frame_num])
    
    inferred_cam1_y.append(inferred_cam1[bp[i],'y'][frame_num])
    inferred_cam2_y.append(inferred_cam2[bp[i],'y'][frame_num])
    inferred_cam3_y.append(inferred_cam3[bp[i],'y'][frame_num])
    #inferred_cam4_y.append(inferred_cam4[bp[i],'y'][frame_num])
    
    reproj_cam1_x.append(reproj_cam1[bp[i],'x'][frame_num])
    reproj_cam2_x.append(reproj_cam2[bp[i],'x'][frame_num])
    reproj_cam3_x.append(reproj_cam3[bp[i],'x'][frame_num])
    #reproj_cam4_x.append(reproj_cam4[bp[i],'x'][frame_num])
    
    reproj_cam1_y.append(reproj_cam1[bp[i],'y'][frame_num])
    reproj_cam2_y.append(reproj_cam2[bp[i],'y'][frame_num])
    reproj_cam3_y.append(reproj_cam3[bp[i],'y'][frame_num])
    #reproj_cam4_y.append(reproj_cam4[bp[i],'y'][frame_num])

#%% Calculate the l2 norm between the pairs of points in all 4 cams

def l2_norm(x1,x2,y1,y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2) 


cam1_diff = []
cam2_diff = []
cam3_diff = []
#cam4_diff = []

for i in range(len(inferred_cam2_y)):
    cam1_diff.append(l2_norm(reproj_cam1_x[i],inferred_cam1_x[i],reproj_cam1_y[i],inferred_cam1_y[i]))
    cam2_diff.append(l2_norm(reproj_cam2_x[i],inferred_cam2_x[i],reproj_cam2_y[i],inferred_cam2_y[i]))
    cam3_diff.append(l2_norm(reproj_cam3_x[i],inferred_cam3_x[i],reproj_cam3_y[i],inferred_cam3_y[i]))
    #cam4_diff.append(l2_norm(reproj_cam4_x[i],inferred_cam4_x[i],reproj_cam4_y[i],inferred_cam4_y[i]))

#%%

frame_num = 1100
static_folder = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\static-data'
static_1 = 'cam_0.csv'
static_2 = 'cam_1.csv'
static_3 = 'cam_2.csv'
#static_4 = 'cam_3.csv'
#inferred_cam1 = pd.read_csv(inferred_folder + '\\' + inferred_1, header=[1,2], index_col=0)

static_cam1 = pd.read_csv(static_folder + '\\' + static_1, header=[1,2],index_col=0)
static_cam2 = pd.read_csv(static_folder + '\\' + static_2, header=[1,2],index_col=0)
static_cam3 = pd.read_csv(static_folder + '\\' + static_3, header=[1,2],index_col=0)
#static_cam4 = pd.read_csv(static_folder + '\\' + static_4, header=[1,2],index_col=0)

# =============================================================================
# inferred_cam1_x = []
# inferred_cam2_x = []
# inferred_cam3_x = []
# #inferred_cam4_x = []
# 
# inferred_cam1_y = []
# inferred_cam2_y = []
# inferred_cam3_y = []
# #inferred_cam4_y = []
# =============================================================================

bp_XYZ = ['pointX','pointY','pointZ']
for i in range(len(bp_XYZ)):
    inferred_cam1_x.append(static_cam1[bp_XYZ[i],'x'][frame_num])
    inferred_cam2_x.append(static_cam2[bp_XYZ[i],'x'][frame_num])
    inferred_cam3_x.append(static_cam3[bp_XYZ[i],'x'][frame_num])
    
    inferred_cam1_y.append(static_cam1[bp_XYZ[i],'y'][frame_num])
    inferred_cam2_y.append(static_cam2[bp_XYZ[i],'y'][frame_num])
    inferred_cam3_y.append(static_cam3[bp_XYZ[i],'y'][frame_num])



#%% Directly copy Min's plotting code and add results of 2D DLC Tracked Points on this 
# =============================================================================
# import os
# import cv2
# import numpy as np
# import pandas as pd
# from numpy import array as arr
# from matplotlib import pyplot as plt
# from utils.calibration_utils import *
# from triangulation.triangulate import *
# from calibration.extrinsic import *
# 
# path_to_save = config['triangulation']['reconstruction_output_path']
# #cams = ['exp00001.avi','exp00002.avi','exp00003.avi','exp00004.avi']
# cams = ['exp00001.avi','exp00002.avi','exp00003.avi']
# frame_counts = np.arange(1100,1101)
# paths_to_save = [os.path.join(path_to_save, cam.split('.')[0]) for cam in cams]
# vidfolder = 'C:/Users/dongq/DeepLabCut/Han-Qiwei-2020-02-21/videos/'
# vidpaths = [os.path.join(vidfolder, cam) for cam in cams]
# xyzs = []
# for joint in joints:
#     x = data_3d[joint+'_x']
#     y = data_3d[joint+'_y']
#     z = data_3d[joint+'_z']
#     c = np.stack([x,y,z], axis=1)
#     xyzs.append(c)
# xyzs = np.array(xyzs)
# xyzs = np.transpose(xyzs, [1,0,2])
# xyzs = np.reshape(xyzs, (xyzs.shape[0], -1))
# xyzs_3d = xyzs[frame_counts]
# 
# DLC_x = [inferred_cam1_x,inferred_cam2_x,inferred_cam3_x]
# DLC_y = [inferred_cam1_y,inferred_cam2_y,inferred_cam3_y]
# 
# 
# def extract_specific_frames(xyzs_3d, vidpath, frame_counts, path_to_save, DLCx, DLCy):
#     if not os.path.exists(vidpath):
#         print('Video does not exist.')
#         return
#     cap = cv2.VideoCapture(vidpath)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration = frame_num/fps
#     count = len(frame_counts)
#     colorclass = plt.cm.ScalarMappable(cmap='jet')
#     C = colorclass.to_rgba(np.linspace(0, 1, len(joints)))
#     colors = C[:, :3]
#     with tqdm(total=count) as pbar:
#         for f, frame_count in enumerate(frame_counts):
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
#             ret, frame = cap.read()
#             plt.figure()
#             plt.imshow(frame)
#             for i, color in enumerate(colors):
#                 xyz_p = np.expand_dims(xyzs_3d[f, 3*i:3*i+3], axis=0)
#                 xyz_p = xyz_p.dot((np.linalg.inv(recovery['registration_mat'].T))) + recovery['center']
#                 coord_2d = np.squeeze(cv2.projectPoints(xyz_p, rvec, tvec, cameraMatrix, distCoeffs)[0], axis=1)
#                 x = coord_2d[0][0]
#                 #print(x)
#                 y = coord_2d[0][1]
#                 #print(y)
#                 #print(i)
#                 #print(" ")
#                 plt.scatter(x, y, s=2, color=color, marker='o')
#                 plt.scatter(DLCx[i], DLCy[i], s=2, color=color, marker='_',alpha=0.5)
#             plt.savefig(os.path.join(path_to_save, 'img' + str(frame_count).zfill(6) + '.png'),
#                         bbox_inches='tight', pad_inches=0, dpi=600)
#             plt.close()
#             pbar.update(1)
#     print('\n{} frames were extracted.'.format(count))
#     
# #for vid_idx, vidpath, path_to_save in zip(vid_indices, vidpaths, paths_to_save):
# #    #print(i)
# #    if not os.path.exists(path_to_save):
# #        os.mkdir(path_to_save)
# #    extract_specific_frames(xyzs_3d, vidpath, frame_counts, path_to_save,DLC_x[i],DLC_y[i])
#     
# for i in range(3):
#     if not os.path.exists(paths_to_save[i]):
#         os.mkdir(paths_to_save[i])
#     extract_specific_frames(xyzs_3d, vidpaths[i], frame_counts, paths_to_save[i],DLC_x[i],DLC_y[i])    
# =============================================================================
    











