# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 12:11:31 2020

@author: dongq
"""

#%% Import Packages
import pandas as pd 
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from peakutils.plot import plot as pplot
import peakutils
from scipy import signal
from scipy.interpolate import interp1d
from moviepy.editor import *
from scipy import stats
#%% Read in the file
df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\3D-data\output_3d_data_rotate4.csv')

#%% Pre-process the data, use the body makers that we only need
nframes = len(df)

#Previously used to determine which section of the video to take out from.
#cfg["start"] should be something between 0 - 1, representing the starting (in percentage) of the video
#cfg["stop"] might also be the case, representing the ending (in percentage) of the video
"""
startindex = max([int(np.floor(nframes * cfg["start"])), 0])
stopindex = min([int(np.ceil(nframes * cfg["stop"])), nframes])
Index = np.arange(stopindex - startindex) + startindex
"""

#The code is from outlier_frames.py in deeplabcut. It's from a function inside so there are
#native parameters inside the function. the parameter "bodyparts" is a native parameter in the
#function, taking in a list of strings describing what bodyparts are needed as reference to
#determine whether a whole frame is an outlier frame or not.
"""
df = df.iloc[Index]
mask = df.columns.get_level_values("bodyparts").isin(bodyparts)
df_temp = df.loc[:, mask]
Indices = []
"""

#%% calculate the speed of each marker throughout the dataframe

#Reference from outlier_frames.py in the Deeplabcut project
"""
temp_dt = df_temp.diff(axis=0) ** 2
temp_dt.drop("likelihood", axis=1, level=-1, inplace=True)
#print(df_temp) #to delete, Qiwei
#print(temp_dt) #to delete, Qiwei
sum_ = temp_dt.sum(axis=1, level=1)
#print(sum_) #to delete, Qiwei
ind = df_temp.index[(sum_ > epsilon ** 2).any(axis=1)].tolist()
Indices.extend(ind)
"""
list_to_delete = ['pointX_x','pointX_y','pointX_z','pointX_error','pointX_ncams','pointX_score','pointY_x','pointY_y','pointY_z','pointY_error','pointY_ncams','pointY_score','pointZ_x','pointZ_y','pointZ_z','pointZ_error','pointZ_ncams','pointZ_score','shoulder1_error','shoulder1_ncams','shoulder1_score','arm1_error','arm1_ncams','arm1_score','arm2_error','arm2_ncams','arm2_score','shoulder1_error','elbow1_ncams','elbow1_score','elbow1_error','elbow2_error','elbow2_ncams','elbow2_score','wrist1_error','wrist1_ncams','wrist1_score','wrist2_error','wrist2_ncams','wrist2_score','hand1_error','hand1_ncams','hand1_score','hand2_error','hand2_ncams','hand2_score','hand3_error','hand3_ncams','hand3_score']
df = df.drop(columns = list_to_delete)

#%% initialize the arrays to put the speed parameters

df_np = df.to_numpy()
df_speed = np.zeros((df_np.shape[0],math.floor(df_np.shape[1]/3)))


#%% function to calculate speed marker by marker

def speed_calc_3D(X,Y,Z):
    temp_df = np.empty((X.shape[0]))
    temp_df[:] = np.nan
    for i in range(X.shape[0]-1): #NOT SURE IF THIS IS GOING TO WORK
        if not math.isnan(X[i]) and not math.isnan(X[i+1]): #if one of the three coordinates are not NaN, the other two will not be NaN
            #print("HERE")
            temp_speed = np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)
            temp_df[i] = temp_speed
    return temp_df

#%% use the function to calculate the distance
    
for i in range(df_speed.shape[1]):
    X = i*3 + 0
    Y = i*3 + 1
    Z = i*3 + 2
    speed_3D = speed_calc_3D(df_np[:,X],df_np[:,Y],df_np[:,Z])
    print(speed_3D)
    df_speed[:,i] = speed_3D

#%% plot the data first to see if it makes sense or not
#plt.plot(df_speed[:,1])

T = range(df_speed.shape[0])
for i in range(df_speed.shape[1]):
    plt.plot(T,df_speed[:,i])
    
plt.show()

#%% Use some sort of find peak function?
"""
#actually use only the last 5 columns, or maybe even less, because those are the
#hand and wrist points that matter
height5 = 20
dist5 = 15
thres5 = 5

peaks5, properties5 = find_peaks(df_speed[:,5],distance=dist5,height=height5)
#or maybe pick the peaks found by all 5 points? or maybe 3 of the 5, or some of the 5 hand/wrist points?
#plt.plot(T,peaks)
#plt.show()

#Reference: https://peakutils.readthedocs.io/en/latest/tutorial_a.html#importing-the-libraries
X = np.linspace(0,df_speed.shape[0]-1,df_speed.shape[0])
plt.figure()
#plt.plot(T,df_speed[:,5])
pplot(X,df_speed[:,5],peaks5)
plt.show()
"""



#%% Marker 5: Plot the peak height distribution to see the extreme peaks of monkey's hand speed

height5 = 20
dist5 = 15
peaks5, properties5 = find_peaks(df_speed[:,5],distance=dist5,height=height5)
marker5_peak_heights = properties5["peak_heights"]

plt.hist(marker5_peak_heights,bins=90)
plt.xlabel("peak height")
plt.ylabel("peak numbers")
plt.title("peak height distribution")

#%% Marker 5: Calculate statistical scores for these peaks
marker5_zscore = stats.zscore(marker5_peak_heights)
plt.hist(marker5_zscore,bins=50)
plt.xlabel("z-score",**font_medium)
plt.ylabel("number of peaks",**font_medium)
plt.title("z-score for peaks",**font_medium)

#%% Take all the points with zscore larger than 1.5 out, and see what they are
#marker5_zscore_potential_outliers = marker5_zscore[marker5_zscore>1.5]
marker5_zscore_outlier_positions = [j for (i,j) in zip(marker5_zscore,peaks5) if i>1]

#%% Plot the outlier points with the speed map, see if they match
X = np.linspace(0,df_speed.shape[0]-1,df_speed.shape[0])/30
plt.figure()
pplot(X,df_speed[:,5],marker5_zscore_outlier_positions)
plt.show()
plt.xlabel("time (in seconds)")
plt.ylabel("speed (in mm)")
plt.title("OUTLIER peaks for marker 5")

#%% Filter and then find peaks for filter 5
"""
From the previous grid we can see that there are certain parts that are not
considered peaks by the find_peaks() algorithm due to frame dropping. In that
case, I think it might be good to add 
"""
#%% Interpolate marker 5 before filtering


#general_x_axis = np.floor(np.linspace(0,df.shape[0],num=df.shape[0],endpoint=True))
#####interp_5 = interp1d(general_x_axis, df_col5,kind='cubic')
#Error: interpolate, line 537, call nan_spline, returns nan

#From interpolate.py:
# Quadratic or cubic spline. If input contains even a single
# nan, then the output is all nans. We cannot just feed data
# with nans to make_interp_spline because it calls LAPACK.
# So, we make up a bogus x and y with no nans and use it
# to get the correct shape of the output, which we then fill
# with nans.
# For slinear or zero order spline, we just pass nans through.

#So I can't use interp1d because of the NaNs

#What else can I use?

#per "https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array"
df_col5 = df_speed[:,5]
interp_5 = pd.DataFrame(df_col5).interpolate().values.ravel().tolist() #interpolated marker 5


#%% Filtfilt marker 5
###############################################################
# Define the sampling frequency in Hz
fs = 30 
# Define the corner frequency of the filter you want to use in Hz
f_c = 5
# Design a 4-pole low pass filter with butter
blow, alow = signal.butter(4,f_c/(fs/2), 'low')
# Get your filtered data
filt_5 = signal.filtfilt(blow, alow, interp_5)
############################################################### 

plt.figure()
plt.plot(X,filt_5)
plt.xlabel("time (in seconds)")
plt.ylabel('speed?')
plt.title("filtered marker 5")
plt.show()


#%% Now do the find outlier thing again, since we have interpolated data.


height5 = 20
dist5 = 15
thres5 = 5

peaks5_filt, properties5_filt = find_peaks(filt_5,distance=dist5,height=height5)

marker5_peak_heights_filt = properties5_filt["peak_heights"]

marker5_zscore_filt = stats.zscore(marker5_peak_heights_filt)

marker5_zscore_outlier_positions_filt = [j for (i,j) in zip(marker5_zscore_filt,peaks5_filt) if i>1]

X = np.linspace(0,df_speed.shape[0]-1,df_speed.shape[0])/30
plt.figure()
#pplot(X,df_speed[:,5],marker5_zscore_outlier_positions_filt)
#plt.plot(X,filt_5)
pplot(X,filt_5,marker5_zscore_outlier_positions_filt)
plt.show()
plt.xlabel("time (in seconds)",**font_medium)
plt.ylabel("speed (in mm)",**font_medium)
plt.title("OUTLIER peaks for marker 5",**font_medium)

"""
From result, we can see that find_peaks() function works better after
the data is filtered. Thus we use this method.
"""

# Actually the speed looks pretty good for thresholding.
#Three steps.
#1. Find and delete absolute outliers (peak values >=80? Or maybe peak values that are statistically significant?)
#2. Use average method to find experiment period. (The average num of points is larger than a value)
#3. Check if the experiment period separation makes sense, if not, tweak it.

#%% Try using np.clip() method to press the outliers down


#%% Find a threshold value for the clip() method.

filt_peak_heights_5 = properties5_filt["peak_heights"]
filt_peak_heights_5.sort()
half_array_length = int(filt_peak_heights_5.shape[0]*4/5)
full_array_length = filt_peak_heights_5.shape[0]
limit_height_5 = np.mean(filt_peak_heights_5[half_array_length:full_array_length])

"""
NOTE: I'm using the average height of the peaks from 80% of the dataset to
100% of the dataset. Or say, the height of the tallest 20% peaks among all the 
peaks. I just... feel like, this is something doable (and the calculated
threshold, around 60, is also what I wanted to have.).
"""

#%% clip the dataset.

clip_5 = np.clip(filt_5, 0, limit_height_5)
plt.figure()
plt.plot(X,clip_5)
plt.xlabel("time (in seconds)",**font)
plt.ylabel("speed (in mm)")
plt.title("clipped speed dataset for marker 5")


#%% Calculate the average speed value during reaching trials
partial_mean_5 = np.mean(filt_5[13200:18000])
partial_mean_5_2 = np.mean(filt_5[660*30:830*30])
partial_mean_5_3 = np.mean(filt_5[220*30:380*30])
partial_mean_5_4 = np.mean(filt_5[50*30:190*30])
partial_mean_5_5 = np.mean(filt_5[900*30:1175*30])
partial_mean_5_6 = np.mean(filt_5[1250*30:1550*30])

"""
And from here we can see that the average speed for marker 5 during reaching
trials is around 16. I would like to find something back from the zscore,
or any statistical values, to relatively separate this reaching period speed
from the resting period speed.
"""

#%% Try to find this "16" through the dataset itself



#%%About the algorithm
"""
record_array = np.zeros(np.ceil(length of frame-by-frame array / 60))
for loop:(having a length 1/60 of the original frame-by-frame array) [I think this can be done in one line, like that i for i in array > threshold or something]
    in_experiment_trial = 0
    if i > threshold (something like 14?)
        record_array[i] = 1
And then use a frame-by-frame array to record all the frame numbers, as a reference to get the marker position data and neural data.
"""

#%%Apply the function to all the peaks
"""
NOTE:
The order of the speed parameters is:    
    df_speed
    interp_speed
    filt_speed
    clip_speed
    peak_speed
    threshold_speed
"""
#wrist_hand_speed 

#Interpolate
interp_speed = np.zeros((df_speed.shape[0],df_speed.shape[1]))
for i in range(df_speed.shape[1]):
    interp_speed[:,i] = pd.DataFrame(df_speed[:,i]).interpolate().values.ravel().tolist()
    
#Before filter, change NaNs to 0s, since NaNs don't work in filtfilt
interp_speed[np.isnan(interp_speed)] = 0

#Filter
filt_speed = np.zeros((df_speed.shape[0],df_speed.shape[1]))
fs = 30 # Define the sampling frequency in Hz
f_c = 5 # Define the corner frequency of the filter you want to use in Hz
blow, alow = signal.butter(4,f_c/(fs/2), 'low') # Design a 4-pole low pass filter with butter
for i in range(df_speed.shape[1]): # Get your filtered data
    filt_speed[:,i] = signal.filtfilt(blow,alow,interp_speed[:,i])

#From what I can see, the threshold is 70.
#I still don't have a full-auto way to calculate it. Maybe average of peaks + 10, but the 10 is also manual
#Clip
#clip_speed
#clip_5 = np.clip(filt_5, 0, limit_height_5)
clip_speed = np.zeros((filt_speed.shape[0],filt_speed.shape[1]))
for i in range(filt_speed.shape[1]):
    clip_speed[:,i] = np.clip(filt_speed[:,i],0,70)

#find max in a certain time region. If max crosses a threshold for multiple markers, this region is considered "experiment region"
length_of_summed_array = int(np.floor(df.shape[0]/60)+1)
width_of_summed_array = df_speed.shape[1]
#summed_array = np.zeros((int(length_of_summed_array),int(width_of_summed_array)))

peak_speed = np.zeros((int(length_of_summed_array),clip_speed.shape[1]))
for i in range(filt_speed.shape[1]):
    for j in range(length_of_summed_array):
        peak_speed[j,i] = np.max(clip_speed[j*60:j*60+59,i])
    
#Find those above threshold
#threshold_speed = peak_speed[peak_speed>15]
threshold_speed = np.zeros((peak_speed.shape[0],peak_speed.shape[1]))
for i in range(peak_speed.shape[0]):
    for j in range(peak_speed.shape[1]):
        if peak_speed[i,j] > 15:
            threshold_speed[i,j] = 1

#So the threshold_speed is roughly accurate, we now need to... low pass it a bit
            
#Two things:
#1. If four (or maybe less?) of five markers on the wrist and arms are 1 (in experimental trial), we say that is
#2. If left and right are both 1 (in experiment trial), the one in the middle should also be. If left and right are both 0, set that to 0
            
total_speed = np.zeros(threshold_speed.shape[0])
for i in range(threshold_speed.shape[0]):
    i_total = sum(threshold_speed[i,5:9])
    if i_total > 3: #if three of the 5 markers passed threshold, set the corresponding 60 frames to "experimental trial"
        total_speed[i] = 1

FILT_speed = np.zeros(total_speed.shape[0])
for i in range(FILT_speed.shape[0]-4):
    if FILT_speed[i+2+1] == 1 and FILT_speed[i+2+2] == 1 and FILT_speed[i+2-1] == 1 and FILT_speed[i+2-2] == 1 and FILT_speed[i] == 0:
        FILT_speed[i] = 1
    if FILT_speed[i+2+1] == 0 and FILT_speed[i+2+2] == 0 and FILT_speed[i+2-1] == 0 and FILT_speed[i+2-2] == 0 and FILT_speed[i] == 1:
        FILT_speed[i] = 0
    

#plot section    
#for i in range(threshold_speed.shape[1]-5):
#    plt.plot(clip_speed[:,i+5],label=str(i+5)+"_origin_data")
#for i in range(threshold_speed.shape[1]-5):
#   plt.plot(threshold_speed[:,i+5],label=str(i+5))
#plt.plot(threshold_speed[:,5])
#plt.plot(peak_speed[:,5])
plt.plot(total_speed)
plt.xlabel("time (in seconds)")
plt.ylabel("speed (in mm / 1/30s)")
plt.title("Speed throughout time for Wrist1,2 and Hand 1,2,3")
plt.legend()
plt.show()






















#%% Marker 6 findpeaks (tentative, didn't filter, interpolate or anything)
height6 = 20
dist6 = 15
thres6 = 5
peaks6, properties6 = find_peaks(df_speed[:,6],height=height6,distance=dist6)
plt.figure()
pplot(X,df_speed[:,6],peaks6)
plt.show()

#%% Marker 7 findpeaks (tentative, didn't filter, interpolate or anything)
height7 = 20
dist7 = 15
thres7 = 2
peaks7, properties7 = find_peaks(df_speed[:,7],distance=dist7,height = height7,threshold = thres7)
plt.figure()
pplot(X,df_speed[:,7],peaks7)
plt.show()
#%% Marker 8 findpeaks (tentative, didn't filter, interpolate or anything)
dist8 = 15
thres8 = 5
peaks8, properties8 = find_peaks(df_speed[:,8],distance=dist8,threshold=thres8)
plt.figure()
pplot(X,df_speed[:,8],peaks8)
plt.show()
#%% Marker 9 findpeaks (tentative, didn't filter, interpolate or anything)
dist9 = 15
thres9 = 5
peaks9, properties9 = find_peaks(df_speed[:,9],distance=dist9,threshold=thres9)
plt.figure()
pplot(X,df_speed[:,9],peaks9)
plt.show()





#%% Try findpeaks after filtering for marker 5
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

font_medium = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

peaks5_filt, properties5_filt = find_peaks(filt_5,distance=15,height=10)
X = np.linspace(0,df_speed.shape[0]-1,df_speed.shape[0])/30
plt.figure()
pplot(X,filt_5,peaks5_filt)
plt.show()
plt.xlabel("time (in seconds)",**font)
plt.ylabel("speed (in mm)",**font)
plt.title("Filtered Speed for Marker 5 (wrist marker)",**font)

#%% (deprecated) Try acceleration on marker 5
"""
At first, I (we) thought that using acceleration may be easier for us to
recognize when a trial starts. But then we found out that, instead of finding
peaks, finding 0s, and separating experiment periods from idle periods is
a bit harder. So no more accelerations.
"""
#acc_5 = np.diff(filt_5)
#plt.plot(acc_5)

#%% (PPT) Take one peak, plot the acc, speed and video out, see what's this about
#for example ,frame 17809 (300th speed peak)

#temp_seconds = 17809/30
#temp_time_minutes = math.floor(temp_seconds/60)
#temp_time_seconds = temp_seconds - (temp_time_minutes*60)
#27155
#13559
#10576
#9528
#15233 #USED

#17000? TRY THIS
wrist_1 = 15 #where wrist 1 is in "df".

temp_frame = 17000
time_radius = 300
temp_start_frame = temp_frame - time_radius
temp_end_frame = temp_frame + time_radius

temp_start_seconds = temp_start_frame / 30
temp_end_seconds = temp_end_frame / 30

temp_start_time_minutes = math.floor(temp_start_seconds/60)
temp_start_time_seconds = temp_start_seconds - (temp_start_time_minutes*60)

temp_end_time_minutes = math.floor(temp_end_seconds/60)
temp_end_time_seconds = temp_end_seconds - (temp_end_time_minutes*60)

X_one_reach = np.linspace(-time_radius,time_radius,time_radius*2)/30

font_small = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

# plot X,Y,Z's position
plt.figure()
plt.plot(X_one_reach,df_np[temp_start_frame:temp_end_frame, wrist_1+0]) #X
plt.xlabel("time (in seconds)",**font_small)
plt.ylabel("position (in mm)",**font_small)
plt.title("Position of Marker 5 (wrist marker) Point X",**font_small)

plt.figure()
plt.plot(X_one_reach,df_np[temp_start_frame:temp_end_frame, wrist_1+1]) #Y
plt.xlabel("time (in seconds)",**font_small)
plt.ylabel("position (in mm)",**font_small)
plt.title("Position of Marker 5 (wrist marker) Point Y",**font_small)

plt.figure()
plt.plot(X_one_reach,df_np[temp_start_frame:temp_end_frame, wrist_1+2]) #Z
plt.xlabel("time (in seconds)",**font_small)
plt.ylabel("position (in mm)",**font_small)
plt.title("Position of Marker 5 (wrist marker) Point Z",**font_small)

#%% (PPT) Plot speed (diff once)

# plot X,Y,Z's position, speed, acc
wrist_1 = 15 #where wrist 1 is in "df".
plt.figure()
plt.plot(X_one_reach,np.diff(df_np[temp_start_frame:temp_end_frame+1, wrist_1+0])) #X
plt.xlabel("time (in seconds)",**font_small)
plt.ylabel("Speed (in mm)",**font_small)
plt.title("Speed of Marker 5 (wrist marker) Point X",**font_small)

plt.figure()
plt.plot(X_one_reach,np.diff(df_np[temp_start_frame:temp_end_frame+1, wrist_1+1])) #Y
plt.xlabel("time (in seconds)",**font_small)
plt.ylabel("Speed (in mm)",**font_small)
plt.title("Speed of Marker 5 (wrist marker) Point Y",**font_small)

plt.figure()
plt.plot(X_one_reach,np.diff(df_np[temp_start_frame:temp_end_frame+1, wrist_1+2])) #Z
plt.xlabel("time (in seconds)",**font_small)
plt.ylabel("Speed (in mm)",**font_small)
plt.title("Speed of Marker 5 (wrist marker) Point Z",**font_small)

#%% (PPT) Plot acceleration (diff twice)

# plot X,Y,Z's position, speed, acc
wrist_1 = 15 #where wrist 1 is in "df".
plt.figure()
plt.plot(X_one_reach,np.diff(np.diff(df_np[temp_start_frame:temp_end_frame+2, wrist_1+0]))) #X
plt.xlabel("time (in seconds)",**font_small)
plt.ylabel("Acceleration (in mm)",**font_small)
plt.title("Acceleration of Marker 5 (wrist marker) Point X",**font_small)

plt.figure()
plt.plot(X_one_reach,np.diff(np.diff(df_np[temp_start_frame:temp_end_frame+2, wrist_1+1]))) #Y
plt.xlabel("time (in seconds)",**font_small)
plt.ylabel("Acceleration (in mm)",**font_small)
plt.title("Acceleration of Marker 5 (wrist marker) Point Y",**font_small)

plt.figure()
plt.plot(X_one_reach,np.diff(np.diff(df_np[temp_start_frame:temp_end_frame+2, wrist_1+2]))) #Z
plt.xlabel("time (in seconds)",**font_small)
plt.ylabel("Acceleration (in mm)",**font_small)
plt.title("Acceleration of Marker 5 (wrist marker) Point Z",**font_small)

#%% (PPT) Plot gif video clips in videos. Parameters are decided 4 blocks above
clip_1 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\exp00001.avi")
        .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
        .resize(0.9))
clip_1.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam1.gif")

clip_2 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\exp00002.avi")
        .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
        .resize(0.9))
clip_2.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam2.gif")

clip_3 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\exp00003.avi")
        .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
        .resize(0.9))
clip_3.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam3.gif")
#%% (PPT) Plot the filtered speed (and acceleration (no more)) as comparisons
plt.figure()
plt.plot(X_one_reach,filt_5[temp_start_frame:temp_end_frame],label='Speed')
plt.xlabel("time (in seconds)",**font_small)
plt.ylabel("Speed (in mm)",**font_small)
plt.title("Speed of Marker 5 (wrist marker) in 3D, filtered",**font_small)
plt.legend()

#plt.figure()
#plt.plot(X_one_reach,acc_5[temp_start_frame:temp_end_frame],label='Acc')
#plt.xlabel("time (in seconds)",**font_small)
#plt.ylabel("Acceleration (in mm)",**font_small)
#plt.title("Acceleration of Marker 5 (wrist marker) in 3D, filtered",**font_small)

#%%(PPT) 4 cams gif clipping

"""
clip_1 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-07-02\videos\exp00001.avi")
        .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
        .resize(0.9))
clip_1.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam1.gif")

clip_2 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-07-02\videos\exp00002.avi")
        .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
        .resize(0.9))
clip_2.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam2.gif")

clip_3 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-07-02\videos\exp00003.avi")
        .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
        .resize(0.9))
clip_3.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam3.gif")

clip_4 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-07-02\videos\exp00004.avi")
        .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
        .resize(0.9))
clip_4.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam4.gif")
"""

#%% Separate trials and non-trials
"""
In here, I am (was) trying to low pass filter / downsample the speed dataset
for one of the markers (marker 5 for now), and check if there are any significant
property differences between the experiment part and the idle part. For now...
no, using the pure speed itself might be a better idea. If speed itself doesn't
work, I'll come back to here.
"""

length_of_summed_array = int(np.floor(df.shape[0]/60)+1)
width_of_summed_array = df_speed.shape[1]
#summed_array = np.zeros((int(length_of_summed_array),int(width_of_summed_array)))

summed_array_5 = np.zeros((int(length_of_summed_array)))
for i in range(length_of_summed_array):
    summed_array_5[i] = sum(clip_5[i*60:i*60+59]) /60

#plt.figure()

summed_DIFF_array_5 = np.zeros((length_of_summed_array-1))
for i in range(summed_array_5.shape[0]-1):
    summed_DIFF_array_5[i] = (np.absolute(summed_array_5[i+1] - summed_array_5[i]))**2

plt.figure()
plt.plot(summed_array_5)
#plt.plot(summed_DIFF_array_5)    
    
#normalized_v = v / np.sqrt(np.sum(v**2))
 
normalized_sum_array_5 = summed_array_5 / np.sqrt(np.sum(summed_array_5**2))
normalized_sum_DIFF_array_5 = summed_DIFF_array_5 / np.sqrt(np.sum(summed_DIFF_array_5**2))
        
b_5, a_5 = signal.butter(3,0.4)
filt_normalized_sum_DIFF_array_5 = signal.filtfilt(b_5,a_5,normalized_sum_DIFF_array_5)
  
plt.figure()
plt.plot(normalized_sum_array_5,label='sum_array')
plt.plot(filt_normalized_sum_DIFF_array_5,label='sum_diff_array')
plt.legend()   
  
#check if 100s and 500s are outliers or not
#use speed for multiple markers
#look at N timepoints and ask if they're all above threshold or not, to determine if it is in experiment phase or not
 
    
    
    
