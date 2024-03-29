# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 19:03:42 2020

@author: dongq
"""

#%% Header?


# =============================================================================
# The script is used to check the performance of 3D reconstruction between two
# cameras with different distances. The distances between the two cameras are:
# 5 inches, 6 inches, 7 inches, 9.5 inches, and 15 inches. 5 inches is the
# closest distance we can get with the ximea cameras, the stands that hold the
# cameras, and the frame that holds these stands. 15 inches on the other hand,
# is the furtherest distance we can get between two cameras without the cameras
# too off from the optimal points.
# 
# Datasets to compare between:
#     3D marker data of the top of the robot arm traced by DLC:
#     folder: C:\Users\dongq\DeepLabCut\Test-Qiwei-2020-11-23\reconstructed-3d-data-in-meters
#     file names:
#         output_3d_data_5in.csv
#         output_3d_data_6in.csv
#         output_3d_data_7in.csv
#         output_3d_data_9_5in.csv
#         output_3d_data_15in.csv
#     
#     Robot arm position data tracked by the robot itself:
#     folder: C:\Users\dongq\DeepLabCut\Test-Qiwei-2020-11-23\neural-data
#     file names:
#         CameraDistances_20201123_RW_5_inches_004_cds_kin.csv
#         CameraDistances_20201123_RW_6_inches_006_cds_kin.csv
#         CameraDistances_20201123_RW_7_inches_005_cds_kin.csv
#         CameraDistances_20201123_RW_9_5inches_002_cds_kin.csv
#         CameraDistances_20201123_RW_15_inches_008_cds_kin.csv
# =============================================================================


#%% Import Packages
import pandas as pd 
import numpy as np
#import os
import math
import matplotlib.pyplot as plt
from peakutils.plot import plot as pplot
import peakutils
from moviepy.editor import *
import copy
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import cage_data
from matplotlib.ticker import PercentFormatter
import decimal
from numpy import savetxt
import scipy
from scipy import signal
from scipy import stats
from scipy.signal import find_peaks
from scipy.signal import butter
from scipy.interpolate import interp1d

from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


#%% Read in csvs
#DLC_folder = r'C:\Users\dongq\DeepLabCut\Test-Qiwei-2020-12-02\reconsturcted_3d_data_12in'
#DLC_name = '\output_3d_data.csv'

#robot_folder = r'C:\Users\dongq\DeepLabCut\Test-Qiwei-2020-12-02\neural_data_20201207'
#robot_name = '\CameraDistances_20201207_RW_staggeredDistances003_cds_kin.csv'
#robot_sync = '\CameraDistances_20201207_RW_staggeredDistances003_cds_analog.csv'

DLC_folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03-handle\reconstructed-3d-data'
DLC_name = '\output_3d_data.csv'

robot_folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03-handle\neural-data'
robot_name = '\Crackle_20201203_RT_LeftS1_4cameras_joe_002_cds_kin.csv'
robot_sync = '\Crackle_20201203_RT_LeftS1_4cameras_joe_002_cds_analog.csv'

rbt = pd.read_csv (robot_folder + robot_name)

rbt_sync = pd.read_csv (robot_folder +robot_sync)

DLC = pd.read_csv (DLC_folder + DLC_name)


#frames_per_second = 25
frames_per_second = 25
frames_per_second_neural = 25 #It's gonna be 25 anyways unless we change it specifically during the experiment


#%% Set plotting parameters
font = {'family' : 'normal',
        'size'   : 16}

plt.rc('font', **font)


#%% function to calculate speed for DLC markers
"""
NOTE: The 3D Scatter plots are plotted in m/frame, but the speed is plotted in m/s.
Does not change the units
"""
def speed_calc_3D(X,Y,Z,fps):
    temp_df = np.empty((X.shape[0]))
    temp_df[:] = np.nan
    for i in range(X.shape[0]-1): #NOT SURE IF THIS IS GOING TO WORK
        if not math.isnan(X[i]) and not math.isnan(X[i+1]): #if one of the three coordinates are not NaN, the other two will not be NaN
            temp_speed = np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2) #cm per second, BUT the numbers aren't right.
            #temp_speed = np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)
            temp_df[i] = temp_speed
    #return temp_df * fps
    return temp_df
#*1000/1e6

#%% function to calculate speed for robot markers
"""
Does not change the units
"""
def speed_calc_2D(X,Y,fps):
    temp_df = np.empty((X.shape[0]))
    temp_df[:] = np.nan
    for i in range(X.shape[0]-1):
        if not math.isnan(X[i]) and not math.isnan(X[i+1]): #if one of the three coordinates are not NaN, the other two will not be NaN
            temp_speed = np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2) #cm per second, BUT the numbers aren't right.
            #temp_speed = np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)
            temp_df[i] = temp_speed
    #return temp_df * fps
    return temp_df

#%% Pre process DLC data (delete unecessary columns)
"""
Does not change the units
"""
def pre_process_DLC_data(df, list_to_delete):
    try:
        df_processed = df.drop(columns = list_to_delete)
    except:
        df_processed = df
    df_processed_np = df_processed.to_numpy()
    return df_processed_np

#%% Calculate the handle marker's speed based on DLC tracking data

#nframes_2D = len(DLC_15in)

#DLC_15in_position_only = DLC_15in.drop(columns = list_to_delete)
#DLC_15in_np = DLC_15in_position_only.to_numpy()
#DLC_15in_speed = np.zeros((DLC_15in_np.shape[0],math.floor(DLC_15in_np.shape[1]/3)))
#DLC_15in_speed[:,0] = speed_3D_2D
#where_are_NaNs = np.isnan(DLC_15in_np)
#DLC_15in_np[where_are_NaNs] = 0

list_to_delete = ['objectA_error','objectA_ncams','objectA_score','fnum','pointX_x','pointX_y','pointX_z','pointX_error','pointX_ncams','pointX_score','pointY_x','pointY_y','pointY_z','pointY_error','pointY_ncams','pointY_score','pointZ_x','pointZ_y','pointZ_z','pointZ_error','pointZ_ncams','pointZ_score']

DLC_position_only = pre_process_DLC_data(DLC, list_to_delete)

#X = 42
#Y = 43
#Z = 44

X = 0
Y = 1
Z = 2

DLC_speed = speed_calc_3D(DLC_position_only[:,X],DLC_position_only[:,Y],DLC_position_only[:,Z],frames_per_second)

where_are_NaNs = np.isnan(DLC_speed)
DLC_speed[where_are_NaNs] = 0

#%% (TEMP) Plot DLC speed data 
#X = np.linspace(0,average_norm_c1.shape[0]-1,average_norm_c1.shape[0])/frames_per_second
#X_DLC = np.linspace(0, DLC['fnum'].shape[0]-1, DLC['fnum'].shape[0])/frames_per_second

#plt.figure()
#plt.plot(X_DLC, DLC_speed)


#%% Calculate handle marker speed based on robot recorded data

#X_rbt = np.linspace(0,rbt_15in['t'].shape[0]-1,) 
"""
Does not change the units
"""
def pre_process_robot_data(df):
    df_time = df['t']
    df_np = df.to_numpy()
    df_x = df_np[:,3]
    df_y = df_np[:,4]
    return df_time, df_x, df_y    

rbt_time, rbt_x, rbt_y = pre_process_robot_data(rbt)
#rbt_speed = speed_calc_2D(rbt_x, rbt_y, frames_per_second)
rbt_speed = speed_calc_2D(rbt_x, rbt_y, frames_per_second_neural)

#%% (TEMP) Plot robot data

#plt.plot(rbt_time, rbt_speed)



#%% function to choose the first uprising edge for each camera frame among the 100~ uprising edges blackrock has sent out for this 1 frame
"""
This function takes in a blackrock cds analog dataset that contains a 30K
(or maybe less) Hz channel, recording each timepoint and the corresponding
amplitude of the synchronization channel. In most cases, when the amplitude
of the synchronization channel is 1, it means that blackrock is sending a 
synchronization signal to the Ximea cameras, asking the cameras to take a
picture, which is one frame in the video. The problem is that, for each frame
Ximea took, Blackrock actually has around 100 data points that were labeled "1"
, which might be for the reason of stability, but still... we will need to 
deal with it. 
The function takes in the dataset, calculates the distance between each two
data points. If x (the current frame) is 0.03 (or whatever) seconds different
from the previous data point, that basically means that it is a data point
for the next frame, instead of being one of the 100 data points for the prev
frame.
0.03 is a number chosen with... experience. Currently the videos we are
recording has got a FPS not more than 30. And 30 fps means that between two 
frames there is a 0.03333 second difference. Also, "not more than 30" is an
extreme case, because most of our recordings are around or lower than 25fps,
which means that the time difference between two frames is around 0.04s.
So 0.03 s is a safe choice.  
"""
def frame_picking(df, fps):
    df_copy = copy.deepcopy(df)
    df_diff = [j-i for i, j in zip(df['t'][:-1],df['t'][1:])]
    df_diff = np.array(df_diff)
    
    #print(df_diff)
    
    df_diff_separate = np.where(df_diff >= 0.02,1,0) #Get only 1 uprising edge time point from the 100 points from blackrock for 1 frame
    df_diff_separate = np.append(df_diff_separate,0)
    df_picked = df_copy[df_diff_separate == 1]
    #plt.plot(df_copy['t'])
    
    return df_picked

#%% function to align the rbt_"x"in dataset to the time points in the picked_frames_"x"in dataset
"""
The whole story is this. After frame_picking() function, we have a dataset,
recording the time point of each frame in levels of prcise to 0.00001s. The
channel that records this time point of each frame is at 30000Hz, which is 
way too more precise than the data in the robot data, which we named 
"rbt_"x"in", which x is a number. The time points in the robot position dataset
are at the precision level of 0.001s, which, we can downsample the
picked_frames dataset that we get from frame_picking(), and align the robot
dataset to the picked_frames dataset per each time point in the downsampled
picked_frames dataset.
"""
def downsample_robot_data(rbt_df, frames_df):
    downsampled_timepts = frames_df['t']
    dsp_rd_timepts = round(downsampled_timepts,3)
    """
    There are, weird behaviors for this round() function. Even though it seems
    that the function has round the downsampled_timepts dataset to 3 decimals,
    comparing the results in downsampled_timepts to the numbers we actually
    want to see suggests that, the round function rounds this number to a
    number SLIGHTLY less than what we want it to be. Thus, there are 400 frames
    among the 1326 frames for this 15in dataset that cannot be found by
    the isin() function used later in the downsampled_timepts dataset because
    they are, different.
    And yes I just round rbt_df again. And that helps. I am stoopeed
    """
    rbt_df['t'] = round( rbt_df['t'], 3)
    dsp_rd_timepts = dsp_rd_timepts.tolist()
    dsp_rbt = rbt_df[rbt_df['t'].isin(dsp_rd_timepts)]
    
    return dsp_rbt



#%% Process the robot's synchronization stimulation data from float amplitude to 0 and 1 with a proper threshold
"""
NOTE: Currently only testing with 15in data. Need to apply these code on all
the data.
"""
#Deep copy robot data for further changes
rbt_sync_bool = copy.deepcopy(rbt_sync)
#Set all stimulation amplitudes larger than 300 as 1 (blackrock stimulating Ximea to record), others 0
rbt_sync_bool['videosync'] = np.where(rbt_sync_bool['videosync'] <= 4000, 0, 1)
#Take the time points for all the frames with uprising edge out.
"""
Note: Uprising edge for one time point does not necessarily mean that the
corresponding frame is recorded specifically at this time point. For each
frame recorded on the camera side, there are 100 corresponding uprising edge
time points. So...We are going to deal with this problem: Which time point in
these 100 time points are we choosing as the representative time point for this
frame?
"""
rbt_sync_bool_1_only = rbt_sync_bool.loc[rbt_sync_bool['videosync'] == 1]
#Pick the first uprising time point as the "representative" time point for this camera frame
rbt_analog_timepoints = frame_picking(rbt_sync_bool_1_only,frames_per_second_neural)
#take the time points recorded in the robot out, which is 1000Hz.
rbt_first_analog_time_sec = rbt_analog_timepoints['t'].iloc[0]
#Change them to frame numbers, so it' seasier to calculate between the analog data (30000Hz camera sync) and the robot data (1000Hz robot handle position data)
rbt_first_analog_time_frames = int(rbt_first_analog_time_sec * 1000)
rbt_first_DLC_frame = rbt_first_analog_time_sec * frames_per_second_neural
rbt_f1 = int(rbt_first_analog_time_frames)
#Pick the frames in the robot data with the same time points calculated/stored by the downsampled analog data
rbt_downsampled = downsample_robot_data(rbt, rbt_analog_timepoints)




#%% If robot frame numbers don't fit with DLC frame numbers, change it so they align

#rbt_downsampled
#DLC
"""
def downsample_dataframe(df, df_fr, df_expect_fr, df_expect_frames):
    df_c = df.copy(deep=True)
    #print(df_c)
    df_np = df_c.to_numpy()
    #print(df_np)
    
    #Uncomment this if you want to calculate pre fps but not per frames
    df_expect_frames = int(df_np.shape[0] * df_expect_fr / df_fr)
    
    where_are_NaNs = np.isnan(df_np)
    df_np[where_are_NaNs] = 0
    
    df_np_resample = scipy.signal.resample(df_np, df_expect_frames,axis=0)
    
    df_np_resample[:,-1] = df_np[0:df_np_resample.shape[0],-1]
    
    df_resample = pd.DataFrame(df_np_resample, columns = df.columns)
    
    #print(df_np_resample)
    #print(df_c)
    #return df_np_resample
    return df_resample
    
    

DLC = downsample_dataframe(DLC, 25, 24, rbt_downsampled.shape[0])

"""

#%% Update DLC speed after possible down_sampling
"""
DLC_position_only = pre_process_DLC_data(DLC, list_to_delete)

DLC_speed = speed_calc_3D(DLC_position_only[:,X],DLC_position_only[:,Y],DLC_position_only[:,Z],frames_per_second)

where_are_NaNs = np.isnan(DLC_speed)
DLC_speed[where_are_NaNs] = 0
"""


#%% Update robot speed after downsampling


rbt_time_ds, rbt_x_ds, rbt_y_ds = pre_process_robot_data(rbt_downsampled)
rbt_speed_ds = speed_calc_2D(rbt_x_ds, rbt_y_ds, frames_per_second_neural)



#%% plot both robot speed and DLC speed, for comparison, aligned the start of robot frame to the start of DLC frames
X_DLC = np.linspace(0, DLC['fnum'].shape[0]-1, DLC['fnum'].shape[0])/frames_per_second

plt.figure()
#plt.ylim(0,5)
#plt.plot(X_DLC, DLC_speed/100,label='DLC')
"""
Changed units, from mm to cm.
"""
plt.plot(X_DLC, DLC_speed/10,label='DLC')
#plt.plot(X_rbt, rbt_15in_speed,label='Robot')
#plt.plot(rbt_time[rbt_f1:], rbt_speed[rbt_f1:],label='Robot',alpha=0.5)
#plt.plot(rbt_time_ds[int(rbt_first_DLC_frame):],rbt_speed_ds[:rbt_speed_ds.shape[0]-int(rbt_first_DLC_frame)],label='Robot',alpha=0.6)
plt.plot(rbt_time_ds - rbt_first_analog_time_sec,rbt_speed_ds,label='Robot',alpha=0.6)
#rbt_speed_ds.shape[0]-int(rbt_first_DLC_frame)



#plt.plot(X_DLC, rbt_speed_ds,label='Robot',alpha=0.7)

#plt.plot(X_DLC[:-1], np.diff(rbt_downsampled['t']),label='Robot')

plt.legend()
#plt.ylim(0,0.1)
plt.title("robot recorded speed and DLC recorded speed")
plt.xlabel("time (in seconds)")
plt.ylabel("speed (in m/s?)")



#%% Align datapoints between robot and 3D data

def align_datapoints(df1, df2):
    
    def delete_end_markers(dfLarge,dfSmall,diff):
        return dfLarge.iloc[:-abs(diff)],dfSmall
    
    diff = df1.shape[0] - df2.shape[0]
    if diff > 0: #df1 > df2
        return delete_end_markers(df1,df2,diff)
    else:
        return delete_end_markers(df2,df1,diff)


#%% Offsetting the values to 0+, if there exists negative values (since lstsq can't handle those)
    
def offsetting_lstsq_values(df):
    columns = list(df)
    tmp_big_arr = np.zeros(df.shape)
    
    for i in range(len(columns)):
        min_val = df[columns[i]].min()
        tmp_arr = df[columns[i]].to_numpy()
        tmp_arr += abs(min_val)
        tmp_big_arr[:,i] = tmp_arr
    
    return_df = pd.DataFrame(tmp_big_arr, columns = df.columns)
    return return_df
        

#%% make all the NaN values in the DataFrame to 0
    
def zeroing_NaNs(df):
    df_np = df.to_numpy()
    where_are_NaNs = np.isnan(df_np)
    df_np[where_are_NaNs] = 0
    return_df = pd.DataFrame(df_np, columns = df.columns)
    return return_df


#%% Compare the position data between DLC data and robot data
"""
So from the figure generated by the previous block, we can see that even though
the speed is different (a question to be discussed later), the "shape" of the
speed is the same for the DLC data and the robot data. So now we have two
questions:
    1. Why the speed "amplitude" is different?
    2. How does it look like for the x,y positions comparing between DLC and
    robot data?
    

So you can do one of two things. You can either use the frame to get the same
 (x,y,z) axes as the handle, then just find your mean error and subtract that
 off to move your coordinate system.

Option 2 is to fit your (x,y,z) data to the handles (x,y,0) data
    You have a n x 3 matrix from your 3d reconstruction. There's a n x 3 
    matrix from the encoders (x,y and a column of 0's). You fit a 3x3 matrix
    mapping one to the other

    Idk how to do in python, maybe a backslash operation. I know how to do it
    in matlab
 
    ok. So the quick and dirty option is to literally do linear regression. 
    In matlab, this is the backslash operator. I'm not sure what it is in 
    python, maybe the same thing. There's likely a numpy/scipy function too. 
    The problem with fitting data is that it can scale the data as well as 
    rotate it, so after fitting you would need to normalize the resulting 
    matrix.
    
    a better way, one that I need to learn more about, might be canonical 
    correlation analysis. I have to see if that will scale the data at all 
    though
    
    lets go with fitting 
    Make sure to subtract the mean before fitting
    In matlab? Literally use the \ operator. So A\b.
    
"""

"""
Data to compare between

robot data:
    rbt_15in_downsampled['x'] and rbt_15in_downsampled['y'] (1326,1)
    DLC_15in['objectA_x'], DLC_15in['objectA_y'] and DLC_15in['objectA_z'] (1326,1)
    
"""
"""
numpy equivalent of "\" matlab operator, solving systems of linear equations,
Ax=B for x.
#https://stackoverflow.com/questions/33559946/numpy-vs-mldivide-matlab-operator

np.linalg.lstsq(a, b, rcond='warn')

Returns
x{(N,), (N, K)} ndarray
Least-squares solution. If b is two-dimensional, the solutions are in the K columns of x.

residuals{(1,), (K,), (0,)} ndarray
Sums of residuals; squared Euclidean 2-norm for each column in b - a*x. If the rank of a is < N or M <= N, this is an empty array. If b is 1-dimensional, this is a (1,) shape array. Otherwise the shape is (K,).

rankint
Rank of matrix a.

s(min(M, N),) ndarray
Singular values of a.
"""

lstsq_rbt_fillin = np.zeros(rbt_downsampled['x'].shape[0])
lstsq_rbt = rbt_downsampled[['x','y']]
lstsq_rbt['z'] = lstsq_rbt_fillin
lstsq_rbt_norm = lstsq_rbt - np.mean(lstsq_rbt)

#Version of DLC data that... has the z axis (in 3D)

#lstsq_dlc = DLC[['objectA_x','objectA_y','objectA_z']]
#lstsq_dlc = DLC[['hand3_x','hand3_y','hand3_z']]
lstsq_dlc = DLC[['handle_x','handle_y','handle_z']]
lstsq_dlc_norm = lstsq_dlc - np.mean(lstsq_dlc)
lstsq_dlc_norm.drop(lstsq_dlc_norm.head(1).index,inplace=True) #dlc has one more row than robot

print(np.std(lstsq_dlc))
#%%

"""
#Version of DLC data that... doesn't have the z axis (in 2D)
#lstsq_dlc = DLC[['objectA_x','objectA_y','objectA_z']]
lstsq_dlc = DLC[['hand3_x','hand3_y']]
lstsq_dlc_fillin = np.zeros(DLC['hand3_x'].shape[0])
lstsq_dlc['hand3_z'] = lstsq_dlc_fillin
lstsq_dlc_norm = lstsq_dlc - np.mean(lstsq_dlc)
lstsq_dlc_norm.drop(lstsq_dlc_norm.head(1).index,inplace=True) #dlc has one more row than robot
"""
#print(lstsq_dlc_norm.shape)
#print(lstsq_rbt_norm.shape)






lstsq_dlc_norm_aligned, lstsq_rbt_norm_aligned = align_datapoints(lstsq_dlc_norm,lstsq_rbt_norm)

#Offset values to 0+
lstsq_dlc_norm_aligned_offset = offsetting_lstsq_values(lstsq_dlc_norm_aligned)
lstsq_rbt_norm_aligned_offset = offsetting_lstsq_values(lstsq_rbt_norm_aligned)

lstsq_dlc_norm_aligned_offset_zeroed = zeroing_NaNs(lstsq_dlc_norm_aligned_offset)
lstsq_rbt_norm_aligned_offset_zeroed = zeroing_NaNs(lstsq_rbt_norm_aligned_offset)



"""
What if I ONLY use the data from 100s to 300s?
"""
start_sec = 100
end_sec = 300

start_fr = start_sec * frames_per_second
end_fr = end_sec * frames_per_second

#%%

#LFP DLC data (lstsq_dlc_norm)
#cutoff freq at 8 Hz (or maybe 10Hz, but use 8Hz for now)
order = 4
sampling_freq = 25
cutoff_freq = 4
normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
numerator_coeffs, denominator_coeffs = signal.butter(order,normalized_cutoff_freq)
#lstsq_dlc_norm_aligned_offset_zeroed_filt_test = signal.filtfilt(numerator_coeffs, denominator_coeffs,lstsq_dlc_norm_aligned_offset_zeroed.loc[start_fr:end_fr,:],axis=0)
lstsq_dlc_norm_aligned_offset_zeroed_filt_test = signal.filtfilt(numerator_coeffs, denominator_coeffs,lstsq_dlc_norm_aligned_offset_zeroed,axis=0)

#lstsq_dlc_norm_aligned_offset_zeroed_filt_test = lstsq_dlc_norm_aligned_offset_zeroed
#z,p,k = butter(order,normalized_cutoff_freq,output='zpk')
#print(p)


#%% check the differences between filtered and unfiltered data
"""
filtered = lstsq_dlc_norm_aligned_offset_zeroed_filt_test
unfiltered = lstsq_dlc_norm_aligned_offset_zeroed.to_numpy()

#plt.hist(filtered - unfiltered)

plt.figure()
plt.plot((filtered - unfiltered)[:,0],label='diff btwn filt & unfilt,x')
plt.plot((filtered - unfiltered)[:,1],label='diff btwn filt & unfilt,y')
plt.plot((filtered - unfiltered)[:,2],label='diff btwn filt & unfilt,z')

plt.plot(lstsq_dlc_norm_aligned_offset_zeroed.sum(axis=1),label='unfiltered data (sum of xyz)')

plt.title("difference between filtered and unfiltered data")
plt.xlabel("frames")
plt.ylabel("difference (in mm)")
plt.legend()
"""



#%%

#lstsq = np.linalg.lstsq(lstsq_dlc_norm,lstsq_rbt_norm)
#lstsq = np.linalg.lstsq(lstsq_dlc_norm_aligned_offset_zeroed,lstsq_rbt_norm_aligned_offset_zeroed)



#lstsq = np.linalg.lstsq(lstsq_dlc_norm_aligned_offset_zeroed.loc[start_fr:end_fr,:],lstsq_rbt_norm_aligned_offset_zeroed.loc[start_fr:end_fr,:])
lstsq = np.linalg.lstsq(lstsq_dlc_norm_aligned_offset_zeroed_filt_test[start_fr:end_fr,:],lstsq_rbt_norm_aligned_offset_zeroed.loc[start_fr:end_fr-1,:])



lstsq_dlc_prediction = np.matmul(lstsq_dlc_norm_aligned_offset_zeroed, lstsq[0]) 

lstsq_dlc_prediction_filt = np.matmul(lstsq_dlc_norm_aligned_offset_zeroed_filt_test,lstsq[0])

lstsq_dlc_prediction_filt_df = pd.DataFrame(lstsq_dlc_prediction_filt, columns = lstsq_dlc_prediction.columns)



#%% least-squares the data with only raw data
lstsq_rawData = np.linalg.lstsq(lstsq_dlc.loc[start_fr:end_fr,:]/10,lstsq_rbt.loc[start_fr:end_fr-1,:])



#%% Plot the prediction results

start_f = 16000
end_f = 16500

#start_f = 15000
#end_f = 15500

#start_f = 10000
#end_f = 12000

plt.figure(figsize = (10,10))#figsize=(8,6)
#plt.scatter(lstsq_dlc_prediction['objectA_x'],lstsq_dlc_prediction['objectA_y'],label="DLC")
#plt.scatter(lstsq_rbt_norm['x'],lstsq_rbt_norm['y'],label='Robot')
#plt.plot(lstsq_dlc_prediction['objectA_x'],lstsq_dlc_prediction['objectA_y'],label="DLC")

#plt.plot(lstsq_dlc_prediction['hand3_x'][start_f:end_f],lstsq_dlc_prediction['hand3_y'][start_f:end_f],label="DLC")
#plt.plot(lstsq_rbt_norm_aligned_offset_zeroed['x'][start_f:end_f],lstsq_rbt_norm_aligned_offset_zeroed['y'][start_f:end_f],label='Robot')

#plt.plot(lstsq_dlc_norm['hand3_x'][start_f:end_f],lstsq_dlc_norm['hand3_y'][start_f:end_f],label="DLC")
#plt.plot(lstsq_dlc_norm_aligned['hand3_x'][start_f:end_f],lstsq_dlc_norm_aligned['hand3_y'][start_f:end_f],label="DLC")






#plt.plot((lstsq_dlc_norm_aligned_offset_zeroed['handle_y'][start_f:end_f]/10),lstsq_dlc_norm_aligned_offset_zeroed['handle_x'][start_f:end_f]/10,label='DLC_original')

#plt.plot(lstsq_dlc_norm_aligned_offset_zeroed_filt_test[start_f:end_f,1]/10,lstsq_dlc_norm_aligned_offset_zeroed_filt_test[start_f:end_f,0]/10,label='DLC_filt')

#plt.plot(lstsq_dlc_prediction['handle_x'][start_f:end_f],lstsq_dlc_prediction['handle_y'][start_f:end_f],label="DLC lstsq")

plt.plot(lstsq_dlc_prediction_filt[start_f:end_f,0],lstsq_dlc_prediction_filt[start_f:end_f,1],label="DLC filt lstsq", linewidth = 2)



plt.plot(lstsq_rbt_norm_aligned_offset_zeroed['x'][start_f:end_f],lstsq_rbt_norm_aligned_offset_zeroed['y'][start_f:end_f],label='Robot')





#plt.plot(lstsq_rbt_norm_aligned_offset_zeroed['x'][start_f:end_f]*10,lstsq_rbt_norm_aligned_offset_zeroed['y'][start_f:end_f]*10,label='Robot')

#plt.plot(lstsq_dlc_norm['hand3_x'][start_f:end_f])
#plt.plot(lstsq_rbt_norm_aligned_offset_zeroed['y'][start_f:end_f]*10)

#plt.plot(lstsq_dlc_norm['handle_y'][start_f:end_f])
#plt.plot(lstsq_rbt_norm_aligned_offset_zeroed['x'][start_f:end_f]*10)

font_size = 22

plt.xlabel("x axis",fontsize = font_size)
plt.ylabel("y axis",fontsize = font_size)
plt.title("marker position comparison from f" + str(start_f) + " to f" + str(end_f),fontsize = font_size)
plt.title("marker position comparison from f" + str(start_f) + " to f" + str(end_f) + " cutoff at " + str(cutoff_freq) + "Hz",fontsize = font_size)
#plt.ylim(-20,20)
#plt.xlim(-30,30)
#plt.ylim(-15,15)
#plt.xlim(-20,20)
plt.legend()




#%% function to calc the difference between dlc_norm data and rbt_norm data
"""
Calculate the 3D euc distance between the predicted DLC data and the recorded
robot data
"""
def pos_diff_calculation(dlc_data, rbt_data):
    dlc_arr = dlc_data.to_numpy()
    rbt_arr = rbt_data.to_numpy()
    squared_dist = np.sum((dlc_arr - rbt_arr)**2,axis=1)
    dist = np.sqrt(squared_dist)
    return dist
    
def pos_diff_calculation_dlc_arr(dlc_data, rbt_data):
    dlc_arr = dlc_data
    rbt_arr = rbt_data.to_numpy()
    squared_dist = np.sum((dlc_arr - rbt_arr)**2,axis=1)
    dist = np.sqrt(squared_dist)
    return dist

#%% Calculate the difference between dlc_norm data and rbt_norm data

selected_columns_dlc = lstsq_dlc_norm_aligned_offset_zeroed.columns[0:2]
selected_columns_rbt = lstsq_rbt_norm_aligned_offset_zeroed.columns[0:2]

col1 = lstsq_dlc_norm_aligned_offset_zeroed_filt_test[:,1]
col2 = lstsq_dlc_norm_aligned_offset_zeroed_filt_test[:,0]
lstsq_dlc_norm_aligned_offset_zeroed_flipped = np.transpose(np.array((col1,col2)))/10


pos_diff = pos_diff_calculation_dlc_arr(lstsq_dlc_norm_aligned_offset_zeroed_flipped,lstsq_rbt_norm_aligned_offset_zeroed[selected_columns_rbt])
mean_pos_diff = np.mean(pos_diff)
std_pos_diff = np.std(pos_diff)

pos_diff_filt = pos_diff_calculation(lstsq_dlc_prediction[selected_columns_dlc],lstsq_rbt_norm_aligned_offset_zeroed[selected_columns_rbt])
mean_pos_diff_filt = np.mean(pos_diff_filt)
std_pos_diff_filt = np.std(pos_diff_filt)

#pos_diff_dlc_and_dlcFilt= pos_diff_calculation(lstsq_dlc_prediction[selected_columns_dlc],lstsq_dlc_prediction_filt_df[selected_columns_dlc])
pos_diff_dlc_and_dlcFilt= pos_diff_calculation(lstsq_dlc_prediction_filt_df[selected_columns_dlc],lstsq_rbt_norm_aligned_offset_zeroed[selected_columns_rbt])
mean_pos_diff_dlc_and_dlcFilt = np.mean(pos_diff_dlc_and_dlcFilt)
std_pos_diff_dlc_and_dlcFilt = np.std(pos_diff_dlc_and_dlcFilt)


print("mean position difference " + str(mean_pos_diff))
print("std position difference " + str(std_pos_diff))
print("mean position difference (filt) " + str(mean_pos_diff_filt))
print("std position difference (filt) " + str(std_pos_diff_filt))
print("mean diff btwn dlc and dlc(filt)" + str(mean_pos_diff_dlc_and_dlcFilt))
print("std diff btwn dlc and dlc(filt)" + str(std_pos_diff_dlc_and_dlcFilt))

plt.figure(figsize=(10,6))
plt.boxplot([pos_diff[pos_diff < 5],pos_diff_filt[pos_diff_filt < 2],pos_diff_dlc_and_dlcFilt[pos_diff_dlc_and_dlcFilt < 2]])
plt.xticks([1,2,3],["rbt <> no align","rbt <> align only", "rbt <> align + filter"])
plt.title("position difference between robot data and DLC data")
plt.ylabel("difference (in cm)")
#plt.boxplot()

# =============================================================================
# 
# name = ['5in','6in','7in','9.5in','15in']
# mean_pos_diff = [mean_pos_diff_5in,mean_pos_diff_6in,mean_pos_diff_7in,mean_pos_diff_9_5in,mean_pos_diff_15in]
# =============================================================================

#%% t-test pos_diff and pos_diff_filt

t_test = scipy.stats.ttest_ind(pos_diff,pos_diff_filt)



#%%
plt.figure()
plt.hist(pos_diff[pos_diff < 1.5])
plt.xlabel("difference btwn rbt and dlc data (in cm)")
plt.ylabel("number of frames")
plt.title("error distribution between robot handle and DLC tracked handle")


#%% Plot the difference between dlc_norm data and rbt_norm data
# =============================================================================
# 
# plt.figure()
# plt.bar(x = name, height = mean_pos_diff)
# for i, v in enumerate(mean_pos_diff):
#     plt.text(i-0.25, v, round(mean_pos_diff[i],2))
# plt.ylabel("error")
# plt.title("DLC estimation error")
# 
# =============================================================================


# =============================================================================
# 
# name_list = ['3in','4in','5in','7in','9in','12in']
# mean_pos_diff_list = [1.109,1.023,1.042,0.918,0.895,0.877]
# 
# plt.figure()
# plt.bar(x = name_list, height = mean_pos_diff_list)
# for i, v in enumerate(mean_pos_diff_list):
#     plt.text(i-0.25, v, round(mean_pos_diff_list[i],2))
# plt.ylabel("error")
# plt.title("DLC estimation error")
# 
# =============================================================================

#%% Make the filtered DLC data as csv


#%% [TEMP] Save as csv

dir_handle = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\handle-data'
lstsq_dlc_prediction.to_csv(dir_handle + "\handle_dlc.csv")
lstsq_dlc_prediction_filt_df.to_csv(dir_handle + "\handle_dlc_filt.csv")
lstsq_rbt_norm_aligned_offset_zeroed.to_csv(dir_handle + "\handle_rbt.csv")







