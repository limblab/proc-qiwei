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

#config_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/config_Crackle_20201203_RT3D_static.toml'

#config_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/Iteration_2_results/config_Crackle_20201203_RT3D_static_Iter2.toml'
#config_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/Iteration_3_results/config_Crackle_20201203_RT3D_static_Iter3.toml'
#config_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/Iteration_3_results/config_Crackle_20201203_RT2D_static_Iter3.toml'
#config_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/recon_config.toml'
#config_path = 'C:/Users/dongq/DeepLabCut/Han_20201204_rwFreeReach/recon_config_RT3D.toml'
config_path = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/recon_config_RT2D.toml'
config = load_config(config_path)




#Recovery_3D_path = "C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/Iteration_3_results/Crackle_20201203_RT2D.json"
#Recovery_3D_path = "C:/Users/dongq/DeepLabCut/Han_20201204_rwFreeReach/reconstructed-3d-data-1/output_3d_data_RT3D.json"
Recovery_3D_path = "C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/reconstructed-3d-data/output_3d_data_RT2D.json"

reprojected_file_prefix = 'output_3d_data' #The file name of Recovery_3D_path

#%%
from triangulation.triangulate import reconstruct_3d
from calibration.intrinsic import calibrate_intrinsic
from calibration.extrinsic import calibrate_extrinsic
calibrate_intrinsic(config)
calibrate_extrinsic(config)
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
f = open(r"C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\Han_20201204_RT2D_groundTruth.txt", "r") 

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
#%% 2D reprojection
# Frame numbers you want to reproject.
# If you want to use the full data, put frame_counts = []
#frame_counts = np.array([1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000])
#frame_counts = []
frame_counts = np.array(f_frame_list)

data_3d = pd.read_csv(path_to_3d)

if len(frame_counts) == 0:
    frame_counts = np.arange(len(data_3d))
else:
    data_3d = data_3d.iloc[frame_counts, :]
    
l = len(data_3d)

#%%
#Record the frame number of the frames that have low likelihood
def get_low_likelihood_and_nans(df):
    subs = 'score'
    df_cols = df.columns.tolist()
    df_likelihood_names = [i for i in df if subs in i] 
    df_likelihood_cols = df[df_likelihood_names]
    
    #Get values < threshold (0.8?) in df_likelihood_cols
    df_likelihood_low = df_likelihood_cols.copy(deep=True)
    df_likelihood_low[df_likelihood_low > 0.8] = 0
    df_likelihood_low[df_likelihood_low > 0.00000000000001] = 1
    df_likelihood_low[df_likelihood_low.isnull()] = 1
    #df_likelihood_cols[df_likelihood_cols < 0.8] = 1
    
    df_likelihood_low_sum = df_likelihood_low.sum(axis=1)
    
    
    #return df_likelihood_isnull
    return df_likelihood_low, df_likelihood_low_sum
    #return df_likelihood_bad_frames
    
data_3d_badframes, data_3d_badframes_sum = get_low_likelihood_and_nans(data_3d)

data_3d_badframes_sum[data_3d_badframes_sum >= 1] = 1

data_3d_goodframes_sum = ~data_3d_badframes_sum.astype(bool)




#%%

data_3d_shrink = data_3d[data_3d_goodframes_sum == 1]

#I'm not sure if I should actually do this (deleting a BIG BUNCH (1/3) of the
#data to delete any data that's probably bad in the slightest way, so... I'm
#writing this wried code. If one (or I) want(s) to use the original data,
#just comment this line below.)
data_3d = data_3d_shrink

l = data_3d.shape[0]

#%%
iterables = [[snapshot], joints, ['x', 'y']]
header = pd.MultiIndex.from_product(iterables, names=['scorer', 'bodyparts', 'coords'])

data_2d = {ind: [] for ind in vid_indices}

for video, vid_idx in zip(videos, vid_indices):
    df = pd.DataFrame(np.zeros((l, len(joints)*2)), index=data_3d.index, columns=header)
    
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
        #print(err)
        print(joint+' mean: '+str(np.mean(err)))
        print(joint+' std: '+str(np.std(err)))
        #cam_err.append(np.mean(err))
        cam_err.append(err)
        #plt.figure()
        #plt.boxplot(err)
        #print(joint+': '+str(np.mean(err_non_nan)))
    print("\n")
    avg_err.append(cam_err)
    
    
    



#%%
    
from matplotlib import rcParams
import seaborn as sns
rcParams['font.family'] = 'Arial'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

cam_num = 3
upper_err_limit = 10


xTicks_names = ['shoulder','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']

plt.figure(figsize=(8,5))
#plt.hist(avg_err[1][0],bins=50)
#plt.boxplot(avg_err[0][0])
#plt.boxplot(cam_err[0:8])
#part_to_plot = 
plt.boxplot(avg_err[cam_num][0:8])
#plt.boxplot(avg_err[1][0:8],widths = 0.65,positions = 9)

#sns.boxplot(data=avg_err[cam_num][0:8])
#sns.swarmplot(data=avg_err[cam_num][0:8])



#for i in range(len(part_to_plot)):
#    plt.boxplot(part_to_plot[i])

plt.xticks(np.arange(1,9,step=1),xTicks_names,fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(-0.1,40)
plt.ylabel("error (in number of frames)",fontsize=22)
plt.title("RT3D task reprojection difference for camera " + str(cam_num+1),fontsize=22)



#path, videos, vid_indices = get_video_path(config)
#%%
"""
#cams = ['Crackle_20201203_00007.avi','Crackle_20201203_00008.avi','Crackle_20201203_00009.avi','Crackle_20201203_00010.avi']
cams = ['Crackle_20201203_00001.avi','Crackle_20201203_00002.avi','Crackle_20201203_00003.avi','Crackle_20201203_00004.avi']

#cams = ['Crackle_20201203_00008.avi']

paths_to_save = [os.path.join(path_to_save, cam.split('.')[0]) for cam in cams]
#vidfolder = 'C:/Users/dongq/DeepLabCut/Han-Qiwei-2020-02-21/videos/'
vidfolder = 'C:/Users/dongq/DeepLabCut/Crackle-Qiwei-2020-12-03/videos/'
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

#vid_indices = [vid_indices[1]]
#paths_to_2d_data = [paths_to_2d_data[1]]
#frame_counts = np.array([1000,2000,3000,4000,5000,10000,20000,30000])

for vid_idx, vidpath, dlc_path, img_path in zip(vid_indices, vidpaths, paths_to_2d_data, paths_to_save):
    if not os.path.exists(img_path):
        os.mkdir(img_path)
        
    df_dlc = pd.read_csv(dlc_path, header=[1,2], index_col=0)
    df_rep = data_2d[vid_idx][0]
    
    extract_specific_frames(df_dlc, df_rep, vidpath, frame_counts, img_path)
"""