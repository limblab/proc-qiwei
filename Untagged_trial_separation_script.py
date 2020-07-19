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



#%%
height6 = 20
dist6 = 15
thres6 = 5
peaks6, properties6 = find_peaks(df_speed[:,6],height=height6,distance=dist6)
plt.figure()
pplot(X,df_speed[:,6],peaks6)
plt.show()

#%%
height7 = 20
dist7 = 15
thres7 = 2
peaks7, properties7 = find_peaks(df_speed[:,7],distance=dist7,height = height7,threshold = thres7)
plt.figure()
pplot(X,df_speed[:,7],peaks7)
plt.show()
#%%
dist8 = 15
thres8 = 5
peaks8, properties8 = find_peaks(df_speed[:,8],distance=dist8,threshold=thres8)
plt.figure()
pplot(X,df_speed[:,8],peaks8)
plt.show()
#%%
dist9 = 15
thres9 = 5
peaks9, properties9 = find_peaks(df_speed[:,9],distance=dist9,threshold=thres9)
plt.figure()
pplot(X,df_speed[:,9],peaks9)
plt.show()


#%% Interpolation before filtfilt?
general_x_axis = np.floor(np.linspace(0,df.shape[0],num=df.shape[0],endpoint=True))
#%%
df_col5 = df_speed[:,5]

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
interp_5 = pd.DataFrame(df_col5).interpolate().values.ravel().tolist()
#%% Try filtfilt
b, a = signal.butter(3,0.05)
filt_5 = signal.filtfilt(b,a,interp_5)

plt.plot(filt_5)
plt.ylabel('filtered marker 5')
plt.show()

#%% Try findpeaks after filtering for marker 5

peaks5_filt, properties5_filt = find_peaks(filt_5,distance=15,height=10)
X = np.linspace(0,df_speed.shape[0]-1,df_speed.shape[0])
plt.figure()
pplot(X,filt_5,peaks5_filt)
plt.show()