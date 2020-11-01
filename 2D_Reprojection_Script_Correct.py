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

config_path = 'C:/Users/dongq/DeepLabCut/Han-Qiwei-2020-02-21/config_20200221_static.toml'
config = load_config(config_path)

Recovery_3D_path = "C:/Users/dongq/DeepLabCut/Han-Qiwei-2020-02-21/Han-20200221.json"

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
snapshot = 'DLC_resnet50_HanFeb21shuffle1_1030000'

# Frame numbers you want to reproject.
# If you want to use the full data, put frame_counts = []
frame_counts = np.array([1000,2000,3000])
# frame_counts = []

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

#%% 2D error
paths_to_2d_data = config['paths_to_2d_data']
for vid_idx, path in zip(vid_indices, paths_to_2d_data):
    df_dlc = pd.read_csv(path, header=[1,2], index_col=0)
    df_rep = data_2d[vid_idx][0]
    for joint in joints:
        x = np.array(df_dlc[joint]['x'][frame_counts] - 
                     df_rep[snapshot][joint]['x'][frame_counts])
        y = np.array(df_dlc[joint]['y'][frame_counts] - 
                     df_rep[snapshot][joint]['y'][frame_counts])
        coord = np.stack([x, y], axis=1)
        err = np.linalg.norm(coord, axis=1)
        print(joint+': '+str(np.mean(err)))
    
    
#%%
cams = ['exp00001.avi','exp00002.avi','exp00003.avi']
paths_to_save = [os.path.join(path_to_save, cam.split('.')[0]) for cam in cams]
vidfolder = 'C:/Users/dongq/DeepLabCut/Han-Qiwei-2020-02-21/videos/'
vidpaths = [os.path.join(vidfolder, cam) for cam in cams]

def extract_specific_frames(df_dlc, df_rep, vidpath, frame_counts, path_to_save): 
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
                for joint, color in zip(joints, colors):
                    plt.scatter(df_dlc[joint]['x'][frame_count],
                                df_dlc[joint]['y'][frame_count], 
                                alpha=0.1, s=20, color=color, marker="o")
                    
                    plt.scatter(df_rep[snapshot][joint]['x'][frame_count],
                                df_rep[snapshot][joint]['y'][frame_count], 
                                alpha=0.1,s=80, color=color, marker="+")

            plt.savefig(os.path.join(path_to_save, 'img' + str(frame_count).zfill(6) + '.png'),
                        bbox_inches='tight', pad_inches=0)
            plt.close()
            pbar.update(1)
            
    print('\n{} frames were extracted.'.format(count))


for vid_idx, vidpath, dlc_path, img_path in zip(vid_indices, vidpaths, paths_to_2d_data, paths_to_save):
    if not os.path.exists(img_path):
        os.mkdir(img_path)
        
    df_dlc = pd.read_csv(dlc_path, header=[1,2], index_col=0)
    df_rep = data_2d[vid_idx][0]
    
    extract_specific_frames(df_dlc, df_rep, vidpath, frame_counts, img_path)
    