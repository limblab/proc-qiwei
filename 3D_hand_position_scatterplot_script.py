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
import cage_data
from matplotlib.ticker import PercentFormatter
import decimal
#%% Read in the file
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\3D-data\output_3d_data_rotate4.csv')
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\3D-data\output_3d_data_rotate7_copy.csv')
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\reconstructed-3d-data\output_3d_data.csv')

#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')
#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\Ground_truth_segments_20200804_FR.txt", "r") 

#df_2D = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\reconstructed-3d-data\output_3d_data.csv')
#f_2D = open(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\Ground_truth_segments_20200804_RT.txt',"r")

#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-07-RT2D\reconsturcted-3d-data\output_3d_data.csv')
#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-07-RT2D\videos\Ground_truth_segments_2020-08-07-RT2D.txt", "r")

#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-07-RT3D\reconsturcted-3d-data\output_3d_data.csv')
#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-07-RT3D\videos\Ground_truth_segments_2020-08-07-RT3D.txt", "r")

#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\reconsturcted-3d-data\output_3d_data.csv')
#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\videos\Ground_truth_segments_2020-09-22-RT2D.txt", "r")


df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT3D\reconsturcted-3d-data\output_3d_data.csv')
f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT3D\videos\Ground_truth_segments_2020-09-22-RT3D.txt", "r")

df_2D = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\reconsturcted-3d-data\output_3d_data.csv')
f_2D = open(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\videos\Ground_truth_segments_2020-09-22-RT2D.txt',"r")
#%%Get the ground truth array for experiment trial segmentation

frames_per_second = 25
seconds_per_minute = 60

def ground_truth_array_extraction(f,df):

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
    ground_truth_segment = np.zeros((df.shape[0]))
    
    f_frame_list = list()
    
    for i in range(len(f_frame)):
        #f_frame_list.append(list(range(int(f_frame[i,0]),int(f_frame[i,1]+1))))
        #print(list(range(int(f_frame[i,0]),int(f_frame[i,1]+1))))
        f_frame_list = f_frame_list + list(range(int(f_frame[i,0]),int(f_frame[i,1]+1)))
    
    return f_frame_list
        
    #for i in range(len(f_frame_list)):
    #    #print(i)
    #    ground_truth_segment[f_frame_list[i]] = 1
    


#%%Define a trial segmentation function
def experiment_trial_segment(df,ground_truth_list):
    df2 = pd.DataFrame(np.zeros((0,df.shape[1])),columns = df.columns)
    print(df2.shape)
    for i in range(len(ground_truth_list)):
        df2.loc[i] = df.iloc[ground_truth_list[i]]
    #print(df2)
    return df2

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

#%%Segment/Separate the experiment trials out from this dataset, and convert the digits from mm to m
f_frame_list_3D = ground_truth_array_extraction(f,df)
df_exp_only = experiment_trial_segment(df, f_frame_list_3D)
#df_exp_only = experiment_trial_segment(df, f_frame_list_3D)*1000/1e6
df = df_exp_only

f_frame_list_2D = ground_truth_array_extraction(f_2D,df_2D)
df_exp_only_2D = experiment_trial_segment(df_2D, f_frame_list_2D)
#df_exp_only_2D = experiment_trial_segment(df, f_frame_list_3D)*1000/1e6
df_2D = df_exp_only_2D


#%% Read in the 2D reaching dataset to compare the hand speed with 3D dataset (TEMP)

nframes_2D = len(df_2D)
try:
    list_to_delete = ['pointX_x','pointX_y','pointX_z','pointX_error','pointX_ncams','pointX_score','pointY_x','pointY_y','pointY_z','pointY_error','pointY_ncams','pointY_score','pointZ_x','pointZ_y','pointZ_z','pointZ_error','pointZ_ncams','pointZ_score','shoulder1_error','shoulder1_ncams','shoulder1_score','arm1_error','arm1_ncams','arm1_score','arm2_error','arm2_ncams','arm2_score','shoulder1_error','elbow1_ncams','elbow1_score','elbow1_error','elbow2_error','elbow2_ncams','elbow2_score','wrist1_error','wrist1_ncams','wrist1_score','wrist2_error','wrist2_ncams','wrist2_score','hand1_error','hand1_ncams','hand1_score','hand2_error','hand2_ncams','hand2_score','hand3_error','hand3_ncams','hand3_score']
    df_2D = df_2D.drop(columns = list_to_delete)
    df_2D = df_2D.drop(df.index[[0,1,2,3,4]])
except:
    print("error line 128")
    
df_np_2D = df_2D.to_numpy()*0.001

df_speed_2D = np.zeros((df_np_2D.shape[0],math.floor(df_np_2D.shape[1]/3)))
for i in range(df_speed_2D.shape[1]):
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





nframes = len(df)
try:
    list_to_delete = ['pointX_x','pointX_y','pointX_z','pointX_error','pointX_ncams','pointX_score','pointY_x','pointY_y','pointY_z','pointY_error','pointY_ncams','pointY_score','pointZ_x','pointZ_y','pointZ_z','pointZ_error','pointZ_ncams','pointZ_score','shoulder1_error','shoulder1_ncams','shoulder1_score','arm1_error','arm1_ncams','arm1_score','arm2_error','arm2_ncams','arm2_score','shoulder1_error','elbow1_ncams','elbow1_score','elbow1_error','elbow2_error','elbow2_ncams','elbow2_score','wrist1_error','wrist1_ncams','wrist1_score','wrist2_error','wrist2_ncams','wrist2_score','hand1_error','hand1_ncams','hand1_score','hand2_error','hand2_ncams','hand2_score','hand3_error','hand3_ncams','hand3_score']
    df = df.drop(columns = list_to_delete)
    df = df.drop(df.index[[0,1,2,3,4]])
except:
    print("error line 154")
#df_np = df.to_numpy()*0.001 #in meters?
    
df_np = df.to_numpy()*0.001 #in meters?

df_speed = np.zeros((df_np.shape[0],math.floor(df_np.shape[1]/3)))
for i in range(df_speed.shape[1]):
    X = i*3 + 0
    Y = i*3 + 1
    Z = i*3 + 2
    speed_3D = speed_calc_3D(df_np[:,X],df_np[:,Y],df_np[:,Z],25)
    print(speed_3D)
    df_speed[:,i] = speed_3D
    
where_are_NaNs = np.isnan(df_np)
df_np[where_are_NaNs] = 0
where_are_NaNs = np.isnan(df_speed)
df_speed[where_are_NaNs] = 0


    
#%% Plot wrist2 speed distribution heatmap with monkey shoulder as reference
"""
So we need x,y,z,value for the point in the 3D space, and the speed value for the color
for wrist2:
x:df_np[:,18]
y:df_np[:,19]
z:df_np[:,20]
speed: df_speed[:,6]
"""
#%% Get neural data from Pickle code

#pkl_3D = r'C:\Users\dongq\Desktop\Han\20200804\Neural_Data\Han_20200804_freeReach_leftS1_4cameras_joe_002.pkl'


#%% Pre-set plotting values


sample_session_start = 300 #in seconds
sample_session_end = 320 #in seconds
sample_start_frame = sample_session_start * frames_per_second
sample_end_frame = sample_session_end * frames_per_second

sample_session_start_2D = 400 #in seconds
sample_session_end_2D = 420 #in seconds
sample_start_frame_2D = sample_session_start_2D * frames_per_second
sample_end_frame_2D = sample_session_end_2D * frames_per_second


font = {'family' : 'normal',
#        'weight' : 'bold',
        'size'   : 22}

font_medium = {'family' : 'normal',
 #       'weight' : 'bold',
        'size'   : 16}

X = np.linspace(0,df_speed.shape[0]-1,df_speed.shape[0])/frames_per_second
small_X = list(np.linspace(sample_session_start,sample_session_end,(sample_session_end-sample_session_start)*frames_per_second))

X_2D = np.linspace(0,df_speed_2D.shape[0]-1,df_speed_2D.shape[0])/frames_per_second
small_X_2D = list(np.linspace(sample_session_start_2D,sample_session_end_2D,(sample_session_end_2D-sample_session_start_2D)*frames_per_second))



#%% 20200912 TEMP Get Speed Data for 20 seconds for a plot
"""
This section is temporary. We're just taking 20 seconds of speed data from the 3D dataset
The 20 seconds of data come from 1:00 - 1:20
In frames (25fps) ir would be from frame 1500 - 2000
The speed dataset is called "df_speed"
the position dataset is called "df_np"
wrist 2 is the 7th marker, if the markes are arranged in order starting from shoulder
for wrist 2, the speed is df_speed[6]
for wrist 2, the position is df_np[:,18:20]

So for that specific section
"""
sec_speed = df_speed[1500:2000,6]
sec_pos = df_np[1500:2000,18:21]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
cmap = plt.get_cmap('plasma')
cax = ax.scatter(sec_pos[:,0],sec_pos[:,1],sec_pos[:,2],c=sec_speed,s=100,cmap='plasma')
#cax = ax.scatter(sec_pos[:,0],sec_pos[:,1],sec_pos[:,2],c=list(range(0,500)),s=100,cmap='plasma')
fig.colorbar(cax)
plt.show()


#%% 3D Scatter speed & location plot
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
cmap = plt.get_cmap("jet")

#cax = ax.scatter(df_np_2D[:,18],df_np_2D[:,19],df_np_2D[:,20],c=df_speed_2D[:,6],s=5,cmap=cmap,vmax=0.5)
cax = ax.scatter(df_np[:,18],df_np[:,19],df_np[:,20],c=df_speed[:,6],s=5,cmap=cmap,vmax=1.2)

plt.xlim(-0.05,0.23)
plt.ylim(-0.05,0.35)
ax.scatter(0,0,0,'rp',s=500,c='r')
ax.plot([0,0.2],[0,0],[0,0],linewidth=3,c='r')

#If I really want to plot arrows in 3D: https://stackoverflow.com/questions/22867620/putting-arrowheads-on-vectors-in-matplotlibs-3d-plot

#cb = plt.colorbar(cmap=cmap)
#cb.set_array([])
#fig.colorbar(cb, ticks=np.linspace(0,2,N), 
#             boundaries=np.arange(-0.05,2.1,.1))

cbar = plt.colorbar(cax)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel("m/s",rotation=270,**font_medium)

plt.xlabel("X Axis (in meters)",**font_medium,labelpad=15)
plt.ylabel("Y Axis (in meters)",**font_medium,labelpad=15)
plt.title("Wrist2 Movement Speed Heatmap",**font_medium)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.rc('font', size=16)
plt.show()

"""
As per the pyplot.scatter documentation, the points specified to be plotted 
must be in the form of an array of floats for cmap to apply, otherwise the 
default colour (in this case, jet) will continue to apply.
"""




#%% Plot a histogram distribution of the hand speed for both 2D and 3D datasets
"""
2D markers' speed: df_speed_2D
3D markers' speed: df_speed

2D wrist2 speed: df_speed_2D[:,6]
3D wrist2 speed: df_speed[:,6]
"""


plt.figure(figsize=(9,6))

"""
Uncomment line 283 and 284 if you have 2D data
"""

bin_size = 50
x1 = df_speed_2D[:,6]
x1_limited = x1[x1<3]
#plt.hist(x1,alpha=0.5,label='2D',bins=50,histtype=u'step',linewidth = 3)
#plt.hist(x1_limited,alpha=0.5,label='2D ' + str(x1_limited.shape[0]) + ' frames',bins=50,histtype=u'step',linewidth = 3)
#plt.hist(x1_limited,density=True,bins=100,alpha=0.5,linewidth=3,label='2D ' + str(x1_limited.shape[0]) + ' frames',histtype=u'step')

#D = decimal.Decimal
#N = 100
#data_2D = [D(str(item)) for item in np.random.random(N)]


plt.hist(x1_limited,density=True,stacked=True,range=(0,1.5),bins=bin_size,alpha=0.5,label='2D ',histtype='step',linewidth = 3)

x2 = df_speed[:,6]
x2_limited = x2[x2<3]
#x2 = x2[x2<3]
#plt.hist(x2,alpha=0.5,label='3D',bins=15,histtype=u'step',linewidth = 3)
#plt.hist(x2_limited,density=True,alpha=0.5,label='3D ' + str(x2_limited.shape[0]) + ' frames',bins=50,histtype=u'step',linewidth = 3)

plt.hist(x2_limited,density=True,stacked=True,range=(0,1.5),bins=bin_size,alpha=0.5,label='3D ',histtype='step',linewidth = 3)


plt.xlabel("wirst2 marker speed (in m/s)",**font_medium)
plt.ylabel("number of frames",**font_medium)
plt.title("Comparing wirst2 speed between 2D and 3D dataset",**font_medium)
#plt.title("Wrist2 speed distribution",**font_medium)


plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.gca().yaxis.set_major_formatter(PercentFormatter(bin_size))
plt.legend()
plt.show()

print(np.mean(x1))
print(np.mean(x2))

#%% plot the speed data





T = range(df_speed.shape[0])


plt.figure(figsize=(12,5))
#plt.ylim(0,10)
#for i in range(df_speed.shape[1]):
#    plt.plot(X,df_speed[:,i])
plt.subplot(211)
plt.ylim(0,2)
plt.plot(small_X_2D,df_speed_2D[sample_start_frame_2D:sample_end_frame_2D,6],label='2D')
plt.xlabel("time (in seconds)",**font_medium)
plt.ylabel("speed (in m/s)",**font_medium)
plt.title("Section of raw speed plot for marker wrist2 in RT2D and RT3D task",**font_medium)
plt.legend()

plt.subplot(212)
plt.ylim(0,2)
plt.plot(small_X,df_speed[sample_start_frame:sample_end_frame,6],label='3D',color='r')
plt.xlabel("time (in seconds)",**font_medium)
plt.ylabel("speed (in m/s)",**font_medium)
plt.legend()
#plt.title("Raw speed plot for marker wrist2",**font_medium)
#plt.plot(df_speed_2D[:,6])
#plt.plot(X,df_speed[:,6])


plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()
