# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 16:27:11 2020

@author: dongq

Does a 3D scatterplot of the wrist2 marker in the space based on the input dataset
Does a speed historgram plot of the wrist2 marker, comparing between 2D and 3D
 reaching tasks (gets another dataset, not the one mentioned in the previous line)
Does a speed plot of all the markers based on the input dataset

"""

#%% Import Packages
import pandas as pd 
import numpy as np
#import os
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from peakutils.plot import plot as pplot
import peakutils
from scipy import signal
from scipy.interpolate import interp1d
from moviepy.editor import *
from scipy import stats
import copy
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
#%% Read in the file
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\3D-data\output_3d_data_rotate4.csv')
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\3D-data\output_3d_data_rotate7_copy.csv')
df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\reconstructed-3d-data\output_3d_data.csv')

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

#%% delete the unwanted parametres like the scores and static reference points
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

#%%Drop some rows that are useless
df = df.drop(df.index[[0,1,2,3,4]])

#%% initialize the arrays to put the speed parameters
df_np = df.to_numpy()*0.001

df_speed = np.zeros((df_np.shape[0],math.floor(df_np.shape[1]/3)))


#%% function to calculate speed marker by marker
"""
NOTE: The 3D Scatter plots are plotted in m/frame, but the speed is plotted in m/s.
"""
def speed_calc_3D(X,Y,Z,fps):
    temp_df = np.empty((X.shape[0]))
    temp_df[:] = np.nan
    for i in range(X.shape[0]-1): #NOT SURE IF THIS IS GOING TO WORK
        if not math.isnan(X[i]) and not math.isnan(X[i+1]): #if one of the three coordinates are not NaN, the other two will not be NaN
            temp_speed = np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2) #cm per second, BUT the numbers aren't right.
            #temp_speed = np.sqrt((X[i+1]-X[i])**2 + (Y[i+1]-Y[i])**2 + (Z[i+1]-Z[i])**2)
            temp_df[i] = temp_speed
    #return temp_df/0.03333
    return temp_df*fps#*1000/1e6

#%% use the function to calculate the distance
for i in range(df_speed.shape[1]):
    X = i*3 + 0
    Y = i*3 + 1
    Z = i*3 + 2
    speed_3D = speed_calc_3D(df_np[:,X],df_np[:,Y],df_np[:,Z],25)
    print(speed_3D)
    df_speed[:,i] = speed_3D
    
#%% Scatter plot hand speed in 3D space
"""
So we need x,y,z,value for the point in the 3D space, and the speed value for the color
for wrist2:
x:df_np[:,18]
y:df_np[:,19]
z:df_np[:,20]
speed: df_speed[:,6]
"""
#where_are_NaNs = np.isnan(all_points)
#all_points[where_are_NaNs] = 0
where_are_NaNs = np.isnan(df_np)
df_np[where_are_NaNs] = 0

where_are_NaNs = np.isnan(df_speed)
df_speed[where_are_NaNs] = 0

#%% Plot wrist2 speed distribution heatmap with monkey shoulder as reference
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
cmap = plt.get_cmap("plasma")
cax = ax.scatter(df_np[:,18],df_np[:,19],df_np[:,20],c=df_speed[:,6],s=1,cmap='plasma')
ax.scatter(0,0,0,'rp',s=500,c='r')
ax.plot([0,0.2],[0,0],[0,0],linewidth=3,c='r')

#If I really want to plot arrows in 3D: https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot

#cb = plt.colorbar(cmap=cmap)
#cb.set_array([])
#fig.colorbar(cb, ticks=np.linspace(0,2,N), 
#             boundaries=np.arange(-0.05,2.1,.1))

fig.colorbar(cax)
plt.xlabel("X Axis (in meters)")
plt.ylabel("Y Axis (in meters)")
plt.title("Wrist2 Movement Speed Heatmap")
plt.show()

"""
As per the pyplot.scatter documentation, the points specified to be plotted 
must be in the form of an array of floats for cmap to apply, otherwise the 
default colour (in this case, jet) will continue to apply.
"""

#%% Read in the 2D reaching dataset to compare the hand speed with 3D dataset (TEMP)
df_2D = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\reconstructed-3d-data\output_3d_data.csv')
nframes_2D = len(df_2D)
df_2D = df_2D.drop(columns = list_to_delete)
df_2D = df_2D.drop(df.index[[0,1,2,3,4]])
df_np_2D = df_2D.to_numpy()*0.001
df_speed_2D = np.zeros((df_np_2D.shape[0],math.floor(df_np_2D.shape[1]/3)))
for i in range(df_speed.shape[1]):
    X = i*3 + 0
    Y = i*3 + 1
    Z = i*3 + 2
    speed_3D_2D = speed_calc_3D(df_np_2D[:,X],df_np_2D[:,Y],df_np_2D[:,Z],25)
    print(speed_3D_2D)
    df_speed_2D[:,i] = speed_3D_2D

where_are_NaNs = np.isnan(df_np_2D)
df_np_2D[where_are_NaNs] = 0
where_are_NaNs = np.isnan(df_speed_2D)
df_speed_2D[where_are_NaNs] = 0


#%% Plot a histogram distribution of the hand speed for both 2D and 3D datasets
"""
2D markers' speed: df_speed_2D
3D markers' speed: df_speed

2D wrist2 speed: df_speed_2D[:,6]
3D wrist2 speed: df_speed[:,6]
"""

x1 = df_speed_2D[:,6]
x2 = df_speed[:,6]
plt.figure()
plt.hist(x1,alpha=0.5,label='2D')
plt.hist(x2,alpha=0.5,label='3D')
plt.xlabel("wirst2 marker speed")
plt.ylabel("number of frames with such speed")
plt.title("Comparing wirst2 speed between 2D and 3D dataset")
plt.legend()
plt.show()

print(np.mean(x1))
print(np.mean(x2))

#%% plot the speed data
X = np.linspace(0,df_speed.shape[0]-1,df_speed.shape[0])/30
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

font_medium = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

T = range(df_speed.shape[0])

plt.figure()
plt.ylim(0,10)
for i in range(df_speed.shape[1]):
    plt.plot(X,df_speed[:,i])

plt.xlabel("time (in frames)",**font_medium)
plt.ylabel("speed (in m/s)",**font_medium)
plt.title("Raw data plot for all markers",**font_medium)
plt.show()
