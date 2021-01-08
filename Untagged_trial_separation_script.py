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
import copy
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns



"""
TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
FIX 30 to parameter FPS, don't use 30 again.


"""
#%% Read in the file
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\3D-data\output_3d_data_rotate4.csv')
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\3D-data\output_3d_data_rotate7_copy.csv')
df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\reconstructed-3d-data\output_3d_data.csv')

fps = 25

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
# =============================================================================
# """
# df_2D = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\reconstructed-3d-data\output_3d_data.csv')
# nframes_2D = len(df_2D)
# df_2D = df_2D.drop(columns = list_to_delete)
# df_2D = df_2D.drop(df.index[[0,1,2,3,4]])
# df_np_2D = df_2D.to_numpy()*0.001
# df_speed_2D = np.zeros((df_np_2D.shape[0],math.floor(df_np_2D.shape[1]/3)))
# for i in range(df_speed.shape[1]):
#     X = i*3 + 0
#     Y = i*3 + 1
#     Z = i*3 + 2
#     speed_3D_2D = speed_calc_3D(df_np_2D[:,X],df_np_2D[:,Y],df_np_2D[:,Z],25)
#     print(speed_3D_2D)
#     df_speed_2D[:,i] = speed_3D_2D
# 
# where_are_NaNs = np.isnan(df_np_2D)
# df_np_2D[where_are_NaNs] = 0
# where_are_NaNs = np.isnan(df_speed_2D)
# df_speed_2D[where_are_NaNs] = 0
# """
# =============================================================================

#%% Plot a histogram distribution of the hand speed for both 2D and 3D datasets (TEMP)

"""
2D markers' speed: df_speed_2D
3D markers' speed: df_speed

2D wrist2 speed: df_speed_2D[:,6]
3D wrist2 speed: df_speed[:,6]
"""
"""
x1 = df_speed_2D[:,6]
x2 = df_speed[:,6]

plt.hist(x1,alpha=0.5,label='2D')
plt.hist(x2,alpha=0.5,label='3D')
plt.xlabel("wirst2 marker speed")
plt.ylabel("number of frames with such speed")
plt.title("Comparing wirst2 speed between 2D and 3D dataset")
plt.legend()
plt.show()

print(np.mean(x1))
print(np.mean(x2))
"""


#%% plot the data first to see if it makes sense or not
#plt.plot(df_speed[:,1])
    
X = np.linspace(0,df_speed.shape[0]-1,df_speed.shape[0])/30

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

font_medium = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}


T = range(df_speed.shape[0])
plt.ylim(0,10)
for i in range(df_speed.shape[1]):
    plt.plot(X,df_speed[:,i])

plt.xlabel("time (in frames)",**font_medium)
plt.ylabel("speed (in m/s)",**font_medium)
plt.title("Raw data plot for all markers",**font_medium)
plt.show()

#%%Apply the function to all the peaks
"""
NOTE:
The order of the speed parameters is:    
    df_speed        (46610,10)
    interp_speed    (46610,10)
    filt_speed      (46610,10)
    clip_speed      (46610,10)
    peak_speed      (777,10)
    threshold_speed (777,10)
    total_speed     (777,10)
    FILT_speed      (777,10)
    unfolded_speed  (46610,1)
"""
#wrist_hand_speed 

"""
#Copied on box
TODO: change the speed unit to cm/s from mm/1/30s
TODO: check what's happening around 600s in the dataset, if Han's actually doing reaching or not
TODO: check what the speed actually is
"""

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

#clip_threshold = 70
clip_threshold = 2.5 #m/s, speed of the marker
cluster_threshold = 60 #frames, like, clustering 60 frames into 1

#From what I can see, the threshold is around 70.
#I still don't have a full-auto way to calculate it. Maybe average of peaks + 10, but the 10 is also manual
#Clip
#clip_speed
#clip_5 = np.clip(filt_5, 0, limit_height_5)
clip_speed = np.zeros((filt_speed.shape[0],filt_speed.shape[1]))
for i in range(filt_speed.shape[1]):
    clip_speed[:,i] = np.clip(filt_speed[:,i],0,clip_threshold)

#find max in a certain time region. If max crosses a threshold for multiple markers, this region is considered "experiment region"
length_of_summed_array = int(np.floor(df.shape[0]/cluster_threshold)+1)
width_of_summed_array = df_speed.shape[1]
#speed_threshold = 15 #for dataset in milimeters
speed_threshold = 0.45 #for dataset in meters, milimeters/0.0333333
#summed_array = np.zeros((int(length_of_summed_array),int(width_of_summed_array)))

peak_speed = np.zeros((int(length_of_summed_array),clip_speed.shape[1]))
for i in range(filt_speed.shape[1]):
    for j in range(length_of_summed_array):
        peak_speed[j,i] = np.max(clip_speed[j*cluster_threshold:j*cluster_threshold+cluster_threshold-1,i])
    
#Find those above threshold
#threshold_speed = peak_speed[peak_speed>15]
threshold_speed = np.zeros((peak_speed.shape[0],peak_speed.shape[1]))
for i in range(peak_speed.shape[0]):
    for j in range(peak_speed.shape[1]):
        if peak_speed[i,j] > speed_threshold:
            threshold_speed[i,j] = 1

#So the threshold_speed is roughly accurate, we now need to... low pass it a bit
            
#Two things:
#1. If four (or maybe less?) of five markers on the wrist and arms are 1 (in experimental trial), we say that is
#2. If left and right are both 1 (in experiment trial), the one in the middle should also be. If left and right are both 0, set that to 0
            
total_speed = np.zeros(threshold_speed.shape[0])
for i in range(threshold_speed.shape[0]):
    i_total = sum(threshold_speed[i,5:9]) +threshold_speed[i,5]
    if i_total > 3: #if three of the 5 markers passed threshold, set the corresponding 60 frames to "experimental trial"
        total_speed[i] = 1

FILT_speed = copy.deepcopy(total_speed)
decision_range = 4
side_range = int(decision_range/2)
for i in range(FILT_speed.shape[0]-decision_range):
    #print(i)
    #if FILT_speed[i+2+1] == 1 and FILT_speed[i+2+2] == 1 and FILT_speed[i+2-1] == 1 and FILT_speed[i+2-2] == 1 and FILT_speed[i] == 0:
    if sum(total_speed[i+side_range-side_range:i+side_range+side_range])+total_speed[i+side_range-side_range] == decision_range and total_speed[i+side_range] == 0:
        print(i)
        FILT_speed[i+side_range] = 1
    #if FILT_speed[i+2+1] == 0 and FILT_speed[i+2+2] == 0 and FILT_speed[i+2-1] == 0 and FILT_speed[i+2-2] == 0 and FILT_speed[i] == 1:
    if sum(total_speed[i+side_range-side_range:i+side_range+side_range])+total_speed[i+side_range-side_range] == 1 and total_speed[i+side_range] == 1:
        #print(i)
        FILT_speed[i+side_range] = 0
    
"""
#plot section  

#for i in range(threshold_speed.shape[1]-5):
#    plt.plot(clip_speed[:,i+5],label=str(i+5)+"_origin_data")
#for i in range(threshold_speed.shape[1]-5):
#   plt.plot(threshold_speed[:,i+5],label=str(i+5))
#plt.plot(threshold_speed[:,5])
#plt.plot(peak_speed[:,5])
plt.figure()
plt.plot(total_speed)
plt.xlabel("time (in 2 seconds)")
plt.ylabel("determining whether it should be experiment period or not")
plt.title("Using threshold to determine whether a time range is in experiment period or not")
plt.legend()
plt.show()
"""  



#Since thresholding is sort of done, we now "unfold" the dataset to frame-by-frame
unfolded_speed = np.zeros((df_speed.shape[0]))
length_limit = df_speed.shape[0]
for i in range(FILT_speed.shape[0]): #777
#    for j in range(FILT_speed.shape[1]): #10
        #cluster_threshold
    for j in range(cluster_threshold):
        if i*cluster_threshold + j < length_limit:
            unfolded_speed[i*cluster_threshold + j] = FILT_speed[i]
#%%

"""
#plot section
"""
#Plot the "final result"(0 or 1) and the "low-passed" data together to see if things match

plt.figure()
#plt.plot(peak_speed/np.sqrt(np.sum(peak_speed))*100)
plt.plot(X,clip_speed)
plt.xlabel("time (in seconds)",**font_medium)
plt.ylabel("speed (in m/s)",**font_medium)
plt.title("Clipped data plot for all markers",**font_medium)
#normalized_sum_DIFF_array_5 = summed_DIFF_array_5 / np.sqrt(np.sum(summed_DIFF_array_5**2))



#%% Plot a graph for all the points' speed in m/seconds
length_of_summed_array = int(np.floor(df.shape[0]/30)+1)
width_of_summed_array = df_speed.shape[1]
peak_speed = np.zeros((int(length_of_summed_array),clip_speed.shape[1]))

for i in range(filt_speed.shape[1]):
    for j in range(length_of_summed_array):
#        peak_speed[j,i] = np.sum(clip_speed[j*30:j*30+29,i])/1000 #Original dataset in milimeters
        peak_speed[j,i] = np.sum(clip_speed[j*30:j*30+29,i]) #Original dataset in meters

plt.figure()
plt.plot(peak_speed)

#%% (Calcuate Figure) Compare the segments from the code and the ground truth
f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\videos\Ground_truth_segments_20200804_FR.txt", "r") 
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
    f_second[i,0] = f_seg[i,0]*60 + f_seg[i,1]
    f_second[i,1] = f_seg[i,2]*60 + f_seg[i,3]
    
f_frame = f_second*25

#ground_truth_segment = np.zeros((len(df_speed)))
ground_truth_segment = np.zeros((28926))

f_frame_list = list()

for i in range(len(f_frame)):
    #f_frame_list.append(list(range(int(f_frame[i,0]),int(f_frame[i,1]+1))))
    #print(list(range(int(f_frame[i,0]),int(f_frame[i,1]+1))))
    f_frame_list = f_frame_list + list(range(int(f_frame[i,0]),int(f_frame[i,1]+1)))
    
for i in range(len(f_frame_list)):
    ground_truth_segment[f_frame_list[i]] = 1

#%% (Figure) Plot the ground truth segment


X = np.linspace(0,ground_truth_segment.shape[0],ground_truth_segment.shape[0])/25 
X_temp = np.linspace(0,ground_truth_segment.shape[0] - 4,ground_truth_segment.shape[0] - 4)/25 

plt.figure()
plt.subplot(211)
plt.plot(X,ground_truth_segment,label="ground_truth",linewidth=5)
plt.plot(X_temp,clip_speed/4,label="speed data")
plt.xlabel("Time (in s)",**font_medium)
plt.ylabel("speed (in m/s)",**font_medium)
plt.title("Ground Truth",**font_medium)
plt.subplot(212)
plt.plot(X_temp,unfolded_speed,'r',label="predicting result",linewidth=5)
#plt.legend()
plt.xlabel("Time (in s)",**font_medium)
plt.ylabel("speed (in m/s)",**font_medium)
plt.title("Predicted Result",**font_medium)

#%% Calculate the percentage the estimated result is similar to the ground truth
true_positive_nums = 0
false_positive_nums = 0
true_neg_nums = 0
false_neg_nums = 0
for i in range(len(unfolded_speed)):
#    if unfolded_speed[i] == ground_truth_segment[i]:
#        true_positive_nums = true_positive_nums + 1
    if ground_truth_segment[i] == 1 and unfolded_speed[i] == 1:
         true_positive_nums = true_positive_nums + 1
    if ground_truth_segment[i] == 0 and unfolded_speed[i] == 1:
        false_positive_nums = false_positive_nums + 1
    if ground_truth_segment[i] == 0 and unfolded_speed[i] == 0:
        true_neg_nums = true_neg_nums + 1
    if ground_truth_segment[i] == 1 and unfolded_speed[i] == 0:    
        false_neg_nums = false_neg_nums + 1
     
#f_frame_list
true_positive_percent = true_positive_nums / len(ground_truth_segment)   
    
    
#%% Deal with the parts that the monkey is moving but not doing reaching

#Either a k-means (which is a bit complicated)
#Or set a threshold (circle) for the hand position
#historesis?
    
#%% Plot directly where the hand and wrist markers are around 1200s, see where they are
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

#%% (Not useful for now) Use some sort of find peak function?
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

#%% (Not useful for now) Marker 5: Plot the peak height distribution to see the extreme peaks of monkey's hand speed

"""
height5 = 20
dist5 = 15
peaks5, properties5 = find_peaks(df_speed[:,5],distance=dist5,height=height5)
marker5_peak_heights = properties5["peak_heights"]

plt.hist(marker5_peak_heights,bins=90)
plt.xlabel("peak height")
plt.ylabel("peak numbers")
plt.title("peak height distribution")
"""

#%% (Not useful for now) Marker 5: Calculate statistical scores for these peaks
"""
marker5_zscore = stats.zscore(marker5_peak_heights)
plt.hist(marker5_zscore,bins=50)
plt.xlabel("z-score",**font_medium)
plt.ylabel("number of peaks",**font_medium)
plt.title("z-score for peaks",**font_medium)
"""

#%% (Not useful for now) Take all the points with zscore larger than 1.5 out, and see what they are
"""
#marker5_zscore_potential_outliers = marker5_zscore[marker5_zscore>1.5]
marker5_zscore_outlier_positions = [j for (i,j) in zip(marker5_zscore,peaks5) if i>1]
"""

#%% (Not useful for now) Plot the outlier points with the speed map, see if they match
"""
X = np.linspace(0,df_speed.shape[0]-1,df_speed.shape[0])/30
plt.figure()
pplot(X,df_speed[:,5],marker5_zscore_outlier_positions)
plt.show()
plt.xlabel("time (in seconds)")
plt.ylabel("speed (in cm)")
plt.title("OUTLIER peaks for marker 5")
"""

#%% (Not useful for now) Filter and then find peaks for filter 5
"""
From the previous grid we can see that there are certain parts that are not
considered peaks by the find_peaks() algorithm due to frame dropping. In that
case, I think it might be good to add 
"""

#%% (Not useful for now) Interpolate marker 5 before filtering


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
"""
df_col5 = df_speed[:,5]
interp_5 = pd.DataFrame(df_col5).interpolate().values.ravel().tolist() #interpolated marker 5
"""

#%% (Not useful for now) Filtfilt marker 5
"""
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
"""

#%% (Not useful for now) Now do the find outlier thing again, since we have interpolated data (for marker 5)

"""
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
"""
From result, we can see that find_peaks() function works better after
the data is filtered. Thus we use this method.
"""

# Actually the speed looks pretty good for thresholding.
#Three steps.
#1. Find and delete absolute outliers (peak values >=80? Or maybe peak values that are statistically significant?)
#2. Use average method to find experiment period. (The average num of points is larger than a value)
#3. Check if the experiment period separation makes sense, if not, tweak it.

# Try using np.clip() method to press the outliers down

#%% (Not useful for now) Find a threshold value for the clip() method. (for marker 5)
"""
filt_peak_heights_5 = properties5_filt["peak_heights"]
filt_peak_heights_5.sort()
half_array_length = int(filt_peak_heights_5.shape[0]*4/5)
full_array_length = filt_peak_heights_5.shape[0]
limit_height_5 = np.mean(filt_peak_heights_5[half_array_length:full_array_length])
"""
"""
NOTE: I'm using the average height of the peaks from 80% of the dataset to
100% of the dataset. Or say, the height of the tallest 20% peaks among all the 
peaks. I just... feel like, this is something doable (and the calculated
threshold, around 60, is also what I wanted to have.).
"""

#%% (Not useful for now) clip the dataset. (for marker 5 only)
"""
clip_5 = np.clip(filt_5, 0, limit_height_5)
plt.figure()
plt.plot(X,clip_5)
plt.xlabel("time (in seconds)",**font)
plt.ylabel("speed (in mm)")
plt.title("clipped speed dataset for marker 5")
"""

#%% (Not useful for now) Calculate the average speed value during reaching trials
"""
partial_mean_5 = np.mean(filt_5[13200:18000])
partial_mean_5_2 = np.mean(filt_5[660*30:830*30])
partial_mean_5_3 = np.mean(filt_5[220*30:380*30])
partial_mean_5_4 = np.mean(filt_5[50*30:190*30])
partial_mean_5_5 = np.mean(filt_5[900*30:1175*30])
partial_mean_5_6 = np.mean(filt_5[1250*30:1550*30])
"""
"""
And from here we can see that the average speed for marker 5 during reaching
trials is around 16. I would like to find something back from the zscore,
or any statistical values, to relatively separate this reaching period speed
from the resting period speed.
"""

#%% (Not useful for now) About the algorithm
"""
record_array = np.zeros(np.ceil(length of frame-by-frame array / 60))
for loop:(having a length 1/60 of the original frame-by-frame array) [I think this can be done in one line, like that i for i in array > threshold or something]
    in_experiment_trial = 0
    if i > threshold (something like 14?)
        record_array[i] = 1
And then use a frame-by-frame array to record all the frame numbers, as a reference to get the marker position data and neural data.
"""


        


"""
num540-num640
1080s - 1280s
18min - 21min20s

In the video:
18:00-18:08 idle
18:09-18:15 reach, twice
18:16-19:21 idle, hand randomly moving around
19:22-19:38 reach, 12 times
19:39-21:24 idle

Guessed by the code:
18:00-18:04 idle
18:05-18:20 reach
18:21-18:25 idle
18:26-18:40 reach
18:41-18:54 reach
18:54-19:00 idle
19:01-19:12 reach
19:12-19:16 idle
19:17-19:40 reach
19:41-20:16 idle
20:17-20:23 reach
20:24-20:54 idle
20:55-21:00 reach
21:01-21:20 idle
"""













#%% (deprecated) Marker 6 findpeaks (tentative, didn't filter, interpolate or anything)
"""
height6 = 20
dist6 = 15
thres6 = 5
peaks6, properties6 = find_peaks(df_speed[:,6],height=height6,distance=dist6)
plt.figure()
pplot(X,df_speed[:,6],peaks6)
plt.show()
"""

#%% (deprecated) Marker 7 findpeaks (tentative, didn't filter, interpolate or anything)
"""
height7 = 20
dist7 = 15
thres7 = 2
peaks7, properties7 = find_peaks(df_speed[:,7],distance=dist7,height = height7,threshold = thres7)
plt.figure()
pplot(X,df_speed[:,7],peaks7)
plt.show()
"""
#%% (deprecated) Marker 8 findpeaks (tentative, didn't filter, interpolate or anything)
"""
dist8 = 15
thres8 = 5
peaks8, properties8 = find_peaks(df_speed[:,8],distance=dist8,threshold=thres8)
plt.figure()
pplot(X,df_speed[:,8],peaks8)
plt.show()
"""
#%% (deprecated) Marker 9 findpeaks (tentative, didn't filter, interpolate or anything)
"""
dist9 = 15
thres9 = 5
peaks9, properties9 = find_peaks(df_speed[:,9],distance=dist9,threshold=thres9)
plt.figure()
pplot(X,df_speed[:,9],peaks9)
plt.show()
"""
#%% (deprecated) Try findpeaks after filtering for marker 5
"""
#per "https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array"
interp_5 = pd.DataFrame(df_col5).interpolate().values.ravel().tolist()
"""
#%% (deprecated) Try filtfilt on marker 5
"""
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


#b, a = signal.butter(3,0.05)
#filt_5 = signal.filtfilt(b,a,interp_5)

peaks5_filt, properties5_filt = find_peaks(filt_5,distance=15,height=10)
X = np.linspace(0,df_speed.shape[0]-1,df_speed.shape[0])/30
plt.figure()
pplot(X,filt_5,peaks5_filt)
plt.show()
plt.xlabel("time (in seconds)",**font)
plt.ylabel("speed (in mm)",**font)
plt.title("Filtered Speed for Marker 5 (wrist marker)",**font)
"""
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
# =============================================================================
# 
# """
# clip_1 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-07-02\videos\exp00001.avi")
#         .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
#         .resize(0.9))
# clip_1.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam1.gif")
# 
# clip_2 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-07-02\videos\exp00002.avi")
#         .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
#         .resize(0.9))
# clip_2.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam2.gif")
# 
# clip_3 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-07-02\videos\exp00003.avi")
#         .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
#         .resize(0.9))
# clip_3.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam3.gif")
# 
# clip_4 = (VideoFileClip(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-07-02\videos\exp00004.avi")
#         .subclip((temp_start_time_minutes, temp_start_time_seconds),(temp_end_time_minutes, temp_end_time_seconds))
#         .resize(0.9))
# clip_4.write_gif(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\single_reach_cam4.gif")
# """
# =============================================================================

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
 
    
pplot(X,filt_5,peaks5_filt)
plt.show()
