# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 09:53:00 2020

@author: dongq
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.image as mpimg

import cv2
import os
import math
import pandas as pd 
from matplotlib.ticker import PercentFormatter
from scipy import stats
import seaborn as sns
import copy
#%% Read in file

#file_dir = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\videos'

#cam1 = r'\exp_han_00017DLC_resnet50_HanSep22shuffle1_1030000.csv'
#cam2 = r'\exp_han_00018DLC_resnet50_HanSep22shuffle1_1030000.csv'
#cam3 = r'\exp_han_00019DLC_resnet50_HanSep22shuffle1_1030000.csv'
#cam4 = r'\exp_han_00020DLC_resnet50_HanSep22shuffle1_1030000.csv'
#g_truth = r'\Ground_truth_segments_2020-09-22-RT2D.txt'

file_dir = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\videos\results_Qiwei_New_Iter3'
cam1 = r'\Crackle_20201203_00007DLC_resnet50_TestDec3shuffle1_1030000filtered.csv'
cam2 = r'\Crackle_20201203_00008DLC_resnet50_TestDec3shuffle1_1030000filtered.csv'
cam3 = r'\Crackle_20201203_00009DLC_resnet50_TestDec3shuffle1_1030000filtered.csv'
cam4 = r'\Crackle_20201203_00010DLC_resnet50_TestDec3shuffle1_1030000filtered.csv'
g_truth = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\Ground_truth_segments_2020-12-03-RT3D-2.txt'

df_c1 = pd.read_csv(file_dir + cam1)
df_c2 = pd.read_csv(file_dir + cam2)
df_c3 = pd.read_csv(file_dir + cam3)
df_c4 = pd.read_csv(file_dir + cam4)

df_c1 = df_c1.to_numpy()
df_c2 = df_c2.to_numpy()
df_c3 = df_c3.to_numpy()
df_c4 = df_c4.to_numpy()

df_c1 = np.delete(df_c1,[0,1],0)
df_c2 = np.delete(df_c2,[0,1],0)
df_c3 = np.delete(df_c3,[0,1],0)
df_c4 = np.delete(df_c4,[0,1],0)

df_c1_copy = copy.deepcopy(df_c1).astype(np.float)
df_c2_copy = copy.deepcopy(df_c2).astype(np.float)
df_c3_copy = copy.deepcopy(df_c3).astype(np.float)
df_c4_copy = copy.deepcopy(df_c4).astype(np.float)

df_c1 = df_c1.astype(np.float)*25/1000
df_c2 = df_c2.astype(np.float)*25/1000
df_c3 = df_c3.astype(np.float)*25/1000
df_c4 = df_c4.astype(np.float)*25/1000

#%% check diff / speed
diff_c1 = np.diff(df_c1,axis=0)
diff_c2 = np.diff(df_c2,axis=0)
diff_c3 = np.diff(df_c3,axis=0)
diff_c4 = np.diff(df_c4,axis=0)

#%% get experiment phase only

frames_per_second = 25
seconds_per_minute = 60

f = open(g_truth, 'r')
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

f_frame_list = list()

for i in range(len(f_frame)):
    f_frame_list = f_frame_list + list(range(int(f_frame[i,0]),int(f_frame[i,1]+1)))
        
ground_truth_segment = np.zeros((df_c1.shape[0]))        
for i in range(len(f_frame_list)):
    #print(i)
    ground_truth_segment[f_frame_list[i]] = 1
    
"""
f_frame_list: list of frame numbers in which monkey is doing experiment
ground_truth_segment: list that shows whether in each frame monkey is doing experiment or not
"""

#%%
"""
ax1 = plt.subplot(211)
plt.plot(diff_c1)

ax2 = plt.subplot(212)
plt.plot(ground_truth_segment)
"""

#%% Get only the speed (not the likelihood) in experiment phase out
"""
NOTE: At this section, the units of these arrays are: pixels/frame
*frame as in each frame the cam records, based on frames per second
"""

speed_section = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29]

speed_c1 = diff_c1[f_frame_list]
speed_c1 = speed_c1[:,speed_section]
speed_c2 = diff_c2[f_frame_list]
speed_c2 = speed_c2[:,speed_section]
speed_c3 = diff_c3[f_frame_list]
speed_c3 = speed_c3[:,speed_section]
speed_c4 = diff_c4[f_frame_list]
speed_c4 = speed_c4[:,speed_section]

df_c1_copy = df_c1_copy[f_frame_list]
df_c2_copy = df_c2_copy[f_frame_list]
df_c3_copy = df_c3_copy[f_frame_list]
df_c4_copy = df_c4_copy[f_frame_list]

#%% Find the l2 norm of dx and dy for each point
def l2_norm(x,y):
    return math.sqrt(abs(x)**2 + abs(y)**2) 

def l1_norm(x,y):
    return (abs(x) + abs(y))

shape_x = int(speed_c1.shape[0])
shape_y = int(speed_c1.shape[1]/2)

norm_c1 = np.zeros((shape_x,shape_y))
norm_c2 = np.zeros((shape_x,shape_y))
norm_c3 = np.zeros((shape_x,shape_y))
norm_c4 = np.zeros((shape_x,shape_y))

for i in range(int(speed_c1.shape[0])):
    for j in range(int(speed_c1.shape[1]/2)):
        norm_c1[i,j] = l2_norm(speed_c1[i,j*2],speed_c1[i,j*2+1])
        norm_c2[i,j] = l2_norm(speed_c2[i,j*2],speed_c2[i,j*2+1])
        norm_c3[i,j] = l2_norm(speed_c3[i,j*2],speed_c3[i,j*2+1])
        norm_c4[i,j] = l2_norm(speed_c4[i,j*2],speed_c4[i,j*2+1])
        
"""
Change the units in norm_c1 from pixels/frame to pixels/second
"""
norm_c1 = norm_c1 * frames_per_second
norm_c2 = norm_c2 * frames_per_second
norm_c3 = norm_c3 * frames_per_second
norm_c4 = norm_c4 * frames_per_second
        

#%%
font = {'family' : 'normal',
#        'weight' : 'bold',
        'size'   : 22}

font_medium = {'family' : 'normal',
 #       'weight' : 'bold',
        'size'   : 16}

font_small = {'family' : 'normal',
 #       'weight' : 'bold',
        'size'   : 12}
#%% Try histogram

bin_size = 12
#reference: plt.hist(x1_limited,density=True,stacked=True,range=(0,1.5),bins=bin_size,alpha=0.5,label='2D ' + str(x1_limited.shape[0]) + ' frames',histtype='step',linewidth = 3)
plt.figure(figsize=(12,9))
marker_names = ['shoulder1','arm1','arm2','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']

for i in range(norm_c1.shape[1]):
    kde = stats.gaussian_kde(norm_c1[:,i])  
    xx = np.linspace(0,2,500)
    plt.hist(norm_c1[:,i],stacked=True,alpha=0.9,range=(0,2),bins=bin_size, label=marker_names[i],histtype='step',linewidth = 3)
    #plt.plot(xx,kde(xx),label=marker_names[i])
#plt.hist(norm_c1,alpha=0.5,range=(0,40),bins=bin_size, label='2D ',histtype='step',linewidth = 3)
plt.yscale('log')

#plt.hist(norm_c1,density=True,stacked=True,alpha=0.5,range=(0,50),bins=bin_size, label='2D ',histtype='step',linewidth = 3)

plt.xlabel("",**font_medium)
plt.ylabel("number of frames",**font_medium)
plt.title("Marker dx dy",**font_medium)
#plt.title("Wrist2 speed distribution",**font_medium)


plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.ylim(0.001,20000)

#plt.gca().yaxis.set_major_formatter(PercentFormatter(bin_size))
plt.legend(fontsize=12)
plt.show()

#%% Same speed plot, try average of hand, wrist, elbow, arm and shoulder

def average_markers(cam, num_markers):
    average_cam = np.zeros((cam.shape[0],num_markers))
    average_cam[:,0] = cam[:,0]
    average_cam[:,1] = (cam[:,1] + cam[:,2])/2
    #average_cam[:,1] = cam[:,2]
    average_cam[:,2] = (cam[:,3] + cam[:,4])/2
    average_cam[:,3] = (cam[:,5] + cam[:,6])/2
    average_cam[:,4] = (cam[:,7] + cam[:,8] + cam[:,9])/3
    return average_cam



#reference: plt.hist(x1_limited,density=True,stacked=True,range=(0,1.5),bins=bin_size,alpha=0.5,label='2D ' + str(x1_limited.shape[0]) + ' frames',histtype='step',linewidth = 3)

marker_names_averaged = ['shoulder','arm','elbow','wrist','hand']

average_norm_c1 = average_markers(norm_c1,len(marker_names_averaged))
average_norm_c2 = average_markers(norm_c2,len(marker_names_averaged))
average_norm_c3 = average_markers(norm_c3,len(marker_names_averaged))
average_norm_c4 = average_markers(norm_c4,len(marker_names_averaged))

bin_size = 100
fig, ax = plt.subplots(figsize=(12,6))

for i in range(average_norm_c1.shape[1]):
    #kde = stats.gaussian_kde(average_norm_c1[:,i])  
    #xx = np.linspace(0,2,500)
    #plt.hist(average_norm_c1[:,i],stacked=True,alpha=0.9,range=(0,1),bins=bin_size, label=marker_names_averaged[i],histtype='step',linewidth = 3)
    #plt.hist(average_norm_c1[:,i],stacked=True,alpha=0.9,range=(0,1),bins=bin_size, label=marker_names_averaged[i],histtype='step',linewidth = 3,density=True,cumulative='True')
    #plt.hist(average_norm_c1[:,i][average_norm_c1[:,i]>0.001],stacked=True,range=(0,25),alpha=0.9,bins=bin_size, label=marker_names_averaged[i],histtype='step',linewidth = 3)
    plt.hist(average_norm_c1[:,i][average_norm_c1[:,i]>0.001],stacked=True,alpha=0.9,bins=bin_size, label=marker_names_averaged[i],histtype='step',linewidth = 3)
    #plt.plot(xx,kde(xx),label=marker_names_averaged[i],linewidth=4)
#plt.hist(norm_c1,alpha=0.5,range=(0,40),bins=bin_size, label='2D ',histtype='step',linewidth = 3)
#plt.yscale('log')

#plt.hist(norm_c1,density=True,stacked=True,alpha=0.5,range=(0,50),bins=bin_size, label='2D ',histtype='step',linewidth = 3)

plt.xlabel("Speed (pixels/second)",**font_medium)
plt.ylabel("Number of frames",**font_medium)
plt.title("Marker speed (l2 norm of dx dy)",**font_medium)
#plt.title("Wrist2 speed distribution",**font_medium)

#y_ticks = np.arange(0,1.00000000001,0.05)


plt.xticks(fontsize = 16)

#ax.set_yticks(y_ticks)
plt.yticks(fontsize = 16)
#ax.get_xticklabels()[-2].set_color("red") (https://stackoverflow.com/questions/41924963/formatting-only-selected-tick-labels)
#ax.get_yticklabels()[-2].set_color('red')
#ax.get_yticklabels()[-2].set_weight('bold')
#plt.ylim(0,400)
#plt.yscale('log')
ax.grid(True)
#plt.gca().yaxis.set_major_formatter(PercentFormatter(bin_size))
plt.legend(fontsize=12)
plt.show()


#%% Try seaborn KDE (Same as the stats.gaussian_kde method)
"""
plt.figure(figsize=(12,6))
for i in range(average_norm_c1.shape[1]):
    sns.kdeplot(data = average_norm_c1[:,i],gridsize=1000,label=marker_names_averaged[i],linewidth=4)


plt.xlabel("Speed",fontsize=16)
plt.ylabel("Density?",fontsize=16)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.legend(fontsize=16)
plt.xlim(0,2)
#plt.ylim
"""

#%% Plot section of speed data (30 seconds)



marker_names = ['shoulder1','arm1','arm2','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
marker_names_averaged = ['shoulder','arm','elbow','wrist','hand']

num_rows = len(marker_names_averaged)

average_norm_c1 = average_markers(norm_c1,len(marker_names_averaged))
average_norm_c2 = average_markers(norm_c2,len(marker_names_averaged))
average_norm_c3 = average_markers(norm_c3,len(marker_names_averaged))
average_norm_c4 = average_markers(norm_c4,len(marker_names_averaged))

#Experiment phase
start_frame = 5000
end_frame = 5000 + 30 * frames_per_second

#Non-Experiment phase
#start_frame = 18500
#end_frame = 18500 + 30 * frames_per_second


X = np.linspace(0,average_norm_c1.shape[0]-1,average_norm_c1.shape[0])/frames_per_second

plt.figure(figsize=(7,7))
for i in range(num_rows):
    plt.subplot(num_rows,1,i+1)
    
    plt.plot(average_norm_c1[start_frame:end_frame,i])
    #plt.plot(average_norm_c1[:,i])
    
    plt.ylabel(marker_names_averaged[i],**font_medium)
    
    plt.ylim(0,30)
    #plt.ylim(0,700)
    
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False, # labels along the bottom edge are off
        labelsize=16) 
    
plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True, # labels along the bottom edge are off
        labelsize=16) 
plt.xlabel("time (in seconds)",**font_medium)


#%% Plot stick figures for a section using several chosen markers
"""
Returns one marker on shoulder, one on elbow, and one on hand

based on directly read in df_c1_copy files, not from the cleaned (deleted likelihood) files
"""
def choose_markers_position(cam, num_markers):
    chosen_markers_x = np.zeros((cam.shape[0],num_markers)) #for both x and y positions
    chosen_markers_y = np.zeros((cam.shape[0],num_markers)) #for both x and y positions
    chosen_markers_x[:,0] = cam[:,1] #shoulder x
    chosen_markers_y[:,0] = cam[:,2] #shoulder y
    #average_cam[:,1] = (cam[:,1] + cam[:,2])/2
    #average_cam[:,1] = cam[:,2]
    chosen_markers_x[:,1] = cam[:,10] #elbow x
    chosen_markers_y[:,1] = cam[:,11] #elbow y
    
    chosen_markers_x[:,2] = cam[:,19] #wrist & hand x
    chosen_markers_y[:,2] = cam[:,20] #wrist & hand y
    #average_cam[:,4] = (cam[:,7] + cam[:,8] + cam[:,9])/3
    return chosen_markers_x, chosen_markers_y


marker_names_averaged_2 = ['shoulder','elbow','wrist']

chosen_pos_c1_x, chosen_pos_c1_y = choose_markers_position(df_c1_copy,len(marker_names_averaged_2))
chosen_pos_c2_x, chosen_pos_c2_y = choose_markers_position(df_c2_copy,len(marker_names_averaged_2))
chosen_pos_c3_x, chosen_pos_c3_y = choose_markers_position(df_c3_copy,len(marker_names_averaged_2))
chosen_pos_c4_x, chosen_pos_c4_y = choose_markers_position(df_c4_copy,len(marker_names_averaged_2))



#Experiment phase
plot_length = 3 * frames_per_second #number of frames
start_frame = 6000
end_frame = start_frame + plot_length

#Non-Experiment phase
#start_frame = 18500
#end_frame = 18500 + 30 * frames_per_second

X = np.linspace(0,chosen_pos_c1_x.shape[0]-1,chosen_pos_c1_x.shape[0])/frames_per_second

fig = plt.figure()
#ax1 = fig.add_subplot(111, projection = '3d')
ax1 = fig.add_subplot(111)
#ax1.scatter(chosen_pos_c1_x[start_frame,0],chosen_pos_c1[start_frame,1],0)
for i in range(plot_length):
    plt.plot(chosen_pos_c1_x[start_frame + i,:],chosen_pos_c1_y[start_frame+i,:])

plt.title("stick figures of camera 1 in 2D")

