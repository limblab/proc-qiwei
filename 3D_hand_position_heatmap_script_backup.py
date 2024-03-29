# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:43:27 2020

@author: dongq
"""


#%% Import Packages
import pandas as pd 
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#from peakutils.plot import plot as pplot
#import peakutils
from scipy import signal
from scipy.interpolate import interp1d
#from moviepy.editor import *
from scipy import stats
import copy
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm
import matplotlib.colorbar
import matplotlib.colors

from matplotlib import pyplot

import seaborn as sns

#%% Step 0: read in CSV

#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\reconstructed-3d-data\output_3d_data.csv')
#f = open(r"C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-RandomTarget\videos\Ground_truth_segments_20200804_RT.txt", "r") 

df = pd.read_csv (r'D:\DLC_Folders_Currently_In_Use\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')
f = open(r"D:\DLC_Folders_Currently_In_Use\Han-Qiwei-2020-08-04-FreeReaching\videos\Ground_truth_segments_20200804_FR.txt", "r") 

"""
NOTICE:
    In this dataset output_3d_data.csv, 
    XY side looks from up to down
    XZ side looks from left to right
    YZ side looks from front to back
    FRONT means the side the monkey is looking towards when sitting in the chair
"""

#%% Step1-1: Get the ground truth array for experiment trial segmentation

frames_per_second = 25
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
ground_truth_segment = np.zeros((df.shape[0]))

f_frame_list = list()

for i in range(len(f_frame)):
    #f_frame_list.append(list(range(int(f_frame[i,0]),int(f_frame[i,1]+1))))
    #print(list(range(int(f_frame[i,0]),int(f_frame[i,1]+1))))
    f_frame_list = f_frame_list + list(range(int(f_frame[i,0]),int(f_frame[i,1]+1)))
    
#for i in range(len(f_frame_list)):
#    #print(i)
#    ground_truth_segment[f_frame_list[i]] = 1
    
#%% Step1-2: Define a trial segmentation function
def experiment_trial_segment(df,ground_truth_list):
    df2 = pd.DataFrame(np.zeros((0,df.shape[1])),columns = df.columns)
    for i in range(len(ground_truth_list)):
        df2.loc[i] = df.iloc[ground_truth_list[i]]
    #print(df2)
    return df2


#%% Step1-3: Segment/Separate the experiment trials out from this dataset 
df_exp_only = experiment_trial_segment(df, f_frame_list)*1000/1e6
#df_exp_only = experiment_trial_segment(df, f_frame_list)*frames_per_second*1000/1e6
#df_exp_only = experiment_trial_segment(df, f_frame_list)*0.025
#Previously it was ms/frames?
#We want it meters/second

#%%Step1-3-1: Separate all the useful columns (x,y,z; not the scores) out, and take the max and min values

whole_plot_limit = 10


bp_interested = ['shoulder1', 'arm1', 'arm2', 'elbow1', 'elbow2', 'wrist1', 
          'wrist2', 'hand1', 'hand2', 'hand3',
          'pointX', 'pointY', 'pointZ']

all_points_non_shoulder_centered = np.array([np.array(df_exp_only.loc[:, (bp+'_x', bp+'_y', bp+'_z')])
                        for bp in bp_interested])

all_points = all_points_non_shoulder_centered - all_points_non_shoulder_centered[0,:,:]
"""
SIGN OF STUPIDITY

where_are_NaNs = np.isnan(all_points)
all_points[where_are_NaNs] = 0

nan_arg_pos = np.argwhere(np.isnan(all_points))

for i in range(nan_arg_pos.shape[0]):
    x = nan_arg_pos[i,0]
    y = nan_arg_pos[i,1]
    z = nan_arg_pos[i,2]
    all_points[x,y,z] = 0
    
print(all_points)
"""

all_errors = np.array([np.array(df_exp_only.loc[:, bp+'_error'])
                        for bp in bp_interested])

all_errors[np.isnan(all_errors)] = 10000
    
#good = (all_errors < 150)
#all_points[~good] = np.nan

all_points_flat = all_points.reshape(-1, 3)
check = ~np.isnan(all_points_flat[:, 0])
low, high = np.percentile(all_points_flat[check], [5, 95], axis=0)

#Even though I didn't understand what the previous 3 lines were doing,
#I still set all the NaN values to 0, because it would bother me
#When calculating.
where_are_NaNs = np.isnan(all_points)
all_points[where_are_NaNs] = 0

all_x_points = all_points[:,:,0]
#print(sum(sum(np.isnan(all_x_points))))
all_y_points = all_points[:,:,1]
all_z_points = all_points[:,:,2]

#zero in the points with reference of the shoulder
all_x_points = all_x_points - all_x_points[0,:]
all_y_points = all_y_points - all_y_points[0,:]
all_z_points = all_z_points - all_z_points[0,:]

x_low, x_high = np.percentile(all_x_points[np.isfinite(all_x_points)], [3, 97])
y_low, y_high = np.percentile(all_y_points[np.isfinite(all_y_points)], [3, 97])
z_low, z_high = np.percentile(all_z_points[np.isfinite(all_z_points)], [3, 97])

x_lim_low = x_low-whole_plot_limit
x_lim_high = x_high+whole_plot_limit
y_lim_low = y_low-whole_plot_limit
y_lim_high = y_high+whole_plot_limit
z_lim_low = z_low-whole_plot_limit
z_lim_high = z_high+whole_plot_limit

#%%
"""
Front should be the XZ plane
Side should be the YZ? Or Y.-Z plane?
"""

#fig = plt.figure()
#ax = Axes3D(fig)

#all_points = np.transpose(all_points, (1,0,2))
#ax.view_init(azim=-48, elev=28)

#ax.set_xlim([x_low-100, x_high+100])
#ax.set_ylim([y_low-100, y_high+100])
#ax.set_zlim([z_low-100, z_high+100])
#ax.set_xlabel('x (mm)')
#ax.set_ylabel('y (mm)')
#ax.set_zlabel('z (mm)')    
#ax.scatter(points[:, 0], points[:, 1], points[:, 2], 'bo', alpha=0.5)

#%% Step1-4: Take only wrist_2 out
#https://stackoverflow.com/questions/43643506/select-columns-based-on-columns-names-containing-a-specific-string-in-pandas?noredirect=1&lq=1
#df_exp_wrist2 = df_exp_only.loc[:,df_exp_only.columns.str.contains('wrist2')]

#And delete the scoring columns that we don't need for this script, only the x,y,z
#df_exp_wrist2 = df_exp_wrist2.drop(labels=['wrist2_error','wrist2_ncams','wrist2_score'],axis=1)

#And make it from milimeters/frame to meters/second
#df_exp_wrist2 = df_exp_wrist2

#%% Step1-4 Again: Take wrist_2 out but with arrays instead of dataframes
df_exp_wrist2_x = all_x_points[6,:].T
df_exp_wrist2_y = all_y_points[6,:].T
df_exp_wrist2_z = all_z_points[6,:].T

df_exp_wrist2 = np.vstack((df_exp_wrist2_x, df_exp_wrist2_y, df_exp_wrist2_z)).T


#df_exp_wrist2 = df_exp_wrist2.reshape(int(df_exp_wrist2.shape[0]/3),3)

#%% Step2: Find the min and max for x,y,z values for wrist2/hand3 marker

"""
TODO: We shouldn't use the min and max values for wrist marker only as the upper
and lower limits to create a 3D plot, but the values of the whole experiment space
recorded by the video. Try to use Min's plotting method as a reference of plotting
such things
"""

wrist2_mins = np.nanmin(df_exp_wrist2)
wrist2_maxs = np.nanmax(df_exp_wrist2)

#/0.03333*1000/1e6
wrist2_min_of_mins = wrist2_mins.min()
wrist2_max_of_maxs = wrist2_maxs.max()

axis_limit_min = wrist2_min_of_mins - 2
axis_limit_max = wrist2_max_of_maxs + 2


    
#%% Step3-1: Count, for each block (x,y,z), how many markers are there throughout the whole experiment phase

#Current unit should be in meters (m)
decimal_nums = 0

df_exp_wrist2_rounded = df_exp_wrist2.round(decimals=decimal_nums)

wrist2_rounded_x_min = np.nanmin(df_exp_wrist2_rounded[:,0])
wrist2_rounded_y_min = np.nanmin(df_exp_wrist2_rounded[:,1])
wrist2_rounded_z_min = np.nanmin(df_exp_wrist2_rounded[:,2])

df_exp_wrist2_rounded_zeroed_x = (df_exp_wrist2_rounded[:,0] - wrist2_rounded_x_min)*10
df_exp_wrist2_rounded_zeroed_y = (df_exp_wrist2_rounded[:,1] - wrist2_rounded_y_min)*10
df_exp_wrist2_rounded_zeroed_z = (df_exp_wrist2_rounded[:,2] - wrist2_rounded_z_min)*10

heatmap_x_axis_length = int(np.nanmax(df_exp_wrist2_rounded_zeroed_x) - np.nanmin(df_exp_wrist2_rounded_zeroed_x) + 1)
heatmap_y_axis_length = int(np.nanmax(df_exp_wrist2_rounded_zeroed_y) - np.nanmin(df_exp_wrist2_rounded_zeroed_y) + 1)
heatmap_z_axis_length = int(np.nanmax(df_exp_wrist2_rounded_zeroed_z) - np.nanmin(df_exp_wrist2_rounded_zeroed_z) + 1)

wrist2_position_heatmap = np.zeros((heatmap_x_axis_length,heatmap_y_axis_length,heatmap_z_axis_length))

for i in range(df_exp_wrist2_rounded_zeroed_x.shape[0]):
    wrist2_position_heatmap[int(df_exp_wrist2_rounded_zeroed_x[i]),int(df_exp_wrist2_rounded_zeroed_y[i]),int(df_exp_wrist2_rounded_zeroed_z[i])] += 1

    
    
    
#%% Step3-2: Count in 2D how many markers are there in the right (XZ) plane

df_exp_wrist2_rounded_right_XZ = np.delete(df_exp_wrist2_rounded,1,1)
df_exp_wrist2_rounded_up_XY = np.delete(df_exp_wrist2_rounded,2,1)
df_exp_wrist2_rounded_front_YZ = np.delete(df_exp_wrist2_rounded,0,1)

lim_percentage = 1

#x_lim = int((x_lim_high - x_lim_low)*lim_percentage)
#y_lim = int((y_lim_high - y_lim_low)*lim_percentage)
#z_lim = int((z_lim_high - z_lim_low)*lim_percentage)

if decimal_nums != 0:
    step = 1/(decimal_nums*10)
else:
    step = 1

x_lim = np.arange(int(x_lim_low*lim_percentage), int(x_lim_high*lim_percentage), step)
y_lim = np.arange(int(y_lim_low*lim_percentage), int(y_lim_high*lim_percentage), step)
z_lim = np.arange(int(z_lim_low*lim_percentage), int(z_lim_high*lim_percentage), step)

wrist2_right_XZ_axis_heatmap = np.zeros((int((x_lim_high-x_lim_low)*lim_percentage/step),int((z_lim_high-z_lim_low)*lim_percentage/step)))
wrist2_up_XY_axis_heatmap = np.zeros((int((x_lim_high-x_lim_low)*lim_percentage/step),int((y_lim_high-y_lim_low)*lim_percentage/step)))
wrist2_front_YZ_axis_heatmap = np.zeros((int((y_lim_high-y_lim_low)*lim_percentage/step),int((z_lim_high-z_lim_low)*lim_percentage/step))) 

for i in range(df_exp_wrist2_rounded_right_XZ.shape[0]):
    temp_x = int((df_exp_wrist2_rounded_right_XZ[i,0] + whole_plot_limit) / step)
    temp_z = int((df_exp_wrist2_rounded_right_XZ[i,1] + whole_plot_limit) / step)
    wrist2_right_XZ_axis_heatmap[temp_x,temp_z] += 1

for i in range(df_exp_wrist2_rounded_up_XY.shape[0]):
    temp_x = int((df_exp_wrist2_rounded_up_XY[i,0] + whole_plot_limit) / step)
    temp_y = int((df_exp_wrist2_rounded_up_XY[i,1] + whole_plot_limit) / step)
    wrist2_up_XY_axis_heatmap[temp_x,temp_y] += 1
    #wrist2_up_XY_axis_heatmap[temp_y,temp_x] += 1
    
for i in range(df_exp_wrist2_rounded_front_YZ.shape[0]):
    temp_y = int((df_exp_wrist2_rounded_front_YZ[i,0] + whole_plot_limit) / step)
    temp_z = int((df_exp_wrist2_rounded_front_YZ[i,1] + whole_plot_limit) / step)
    wrist2_front_YZ_axis_heatmap[temp_y,temp_z] += 1
    
#wrist2_up_XY_axis_heatmap_reverse = (wrist2_up_XY_axis_heatmap * step) - whole_plot_limit

"""
HOW-TO SNS HEATMAP: https://zhuanlan.zhihu.com/p/35494575
"""

x_ticklabels = np.linspace(-50,80,27)
y_ticklabels = np.linspace(70,-75,30)

font_medium = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.figure()
#ax = sns.heatmap(np.flipud(wrist2_right_XZ_axis_heatmap.T), linewidth=0.5,annot=True,fmt ='.0f',cmap='gist_gray_r',square='True',xticklabels=x_ticklabels,yticklabels=y_ticklabels)
#ax = sns.heatmap(np.flipud(wrist2_right_XZ_axis_heatmap.T), linewidth=0.5,annot=True,fmt ='.0f',cmap='gist_gray_r',square='True',xticklabels=x_ticklabels)
#plt.xlabel('X axis (in cm)',**font_medium)
#plt.ylabel('Y axis (in cm)',**font_medium)
#plt.scatter(10,15,1000)
#plt.title("Wrist2 Hand Position Heatmap Right->Left View",**font_medium)

#ax = sns.heatmap(np.flipud(wrist2_up_XY_axis_heatmap.T), linewidth=0.5,annot=True,fmt ='.0f',cmap='gist_gray_r',square='True',xticklabels=x_ticklabels,yticklabels=y_ticklabels)
#ax = sns.heatmap(np.flipud(wrist2_up_XY_axis_heatmap.T), linewidth=0.5,annot=True,fmt ='.0f',cmap='gist_gray_r',square='True',xticklabels=x_ticklabels)
#plt.xlabel('X axis (in cm)',**font_medium)
#plt.ylabel('Y axis (in cm)',**font_medium)
#plt.scatter(10,15,1000)
#plt.title("Wrist2 Hand Position Heatmap Up->Down View",**font_medium)

ax = sns.heatmap(np.flipud(wrist2_front_YZ_axis_heatmap.T), linewidth=0.5,annot=True,fmt ='.0f',cmap='gist_gray_r',square='True',xticklabels=x_ticklabels,yticklabels=y_ticklabels)
#ax = sns.heatmap(np.flipud(wrist2_front_YZ_axis_heatmap.T), linewidth=0.5,annot=True,fmt ='.0f',cmap='gist_gray_r',square='True',xticklabels=x_ticklabels)
plt.xlabel('X axis (in cm)',**font_medium)
plt.ylabel('Y axis (in cm)',**font_medium)
plt.scatter(10,15,1000)
plt.title("Wrist2 Hand Position Heatmap Front->Back View",**font_medium)

#ax2 = ax.twiny()

plt.show()

"""
If I assume the dataset "df_exp_wrist2_rounded"'s first column is x, second
column y and third column z, here's the problem I will have when plotting
the heatmap.


wrist2_up_XY_axis_heatmap: is actually FRONT but UPSIDE-DOWN... OR TOP, but 90 degrees rotated clockwise
wrist2_front_YZ_axis_heatmap: If it's front, then it is 90 degrees rotated clockwise

So transpose doesn't work. After transpose, It's still up-down reversed.


"""


#plt.figure()
#ax.set_xlabel("left bottom")
#ax.set_ylabel("left side")
#ax.set_title("left<->right, XZ side")

#ax = plt.scatter(df_exp_wrist2[:,0],df_exp_wrist2[:,2],s=0.1)
#plt.xlabel("left bottom")
#plt.ylabel("left side")
#plt.title("left<->right, XZ side")
#ax = plt.scatter(df_exp_wrist2[:,0],df_exp_wrist2[:,1],s=0.5)
#plt.xlim((x_lim_low,x_lim_high))
#plt.ylim((z_lim_low,z_lim_high))
#plt.show()


#%% Plot heatmap with scatterplot
"""
fig = plt.figure()
ax = fig.add_subplot(111,projection="3d")
cmap = plt.get_cmap("plasma")
cax = ax.scatter(df_np[:,18],df_np[:,19],df_np[:,20],c=df_speed[:,6],s=1,cmap='plasma')
plt.show()
"""




#%% 
"""
df_huge_array = df_exp_only.to_numpy()
point_X = [df_huge_array[0,60],df_huge_array[0,61],df_huge_array[0,62]]
point_Y = [df_huge_array[0,66],df_huge_array[0,67],df_huge_array[0,68]]
point_Z = [df_huge_array[0,72],df_huge_array[0,73],df_huge_array[0,74]]
"""

#point_X = [all_x_points[10,0],all_x_points[11,0],all_x_points[12,0]]
#point_Y = [all_y_points[10,0],all_y_points[11,0],all_y_points[12,0]]
#point_Z = [all_z_points[10,0],all_z_points[11,0],all_z_points[12,0]]


#%% 3D Scatterplot figure of wirst 2 movement in experiment phase
#( View angle set to from top to down)
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_array[:,0],df_array[:,1],df_array[:,2],s=0.1)
ax.scatter(point_X[0],point_X[1],point_X[2])
ax.scatter(point_Y[0],point_Y[1],point_Y[2])
ax.scatter(point_Z[0],point_Z[1],point_Z[2])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_zlim(z_lim_low,z_lim_high)
plt.xlim((x_lim_low,x_lim_high))
plt.ylim((y_lim_low,y_lim_high))
ax.view_init(azim=-90, elev=90) #up-down
ax.view_init(azim=-90, elev=0) #left-right
#ax.view_init(azim=-48, elev=28)
plt.title("Wrist 2 Position Right left 3D Reaching")
#plt.zlim((z_lim_low,z_lim_high))
plt.show()
"""    

#%%
#uniform_data = np.random.rand(10, 12)
#ax = sns.heatmap(uniform_data, linewidth=0.5)
#plt.show()

#%% Step3-2: Construct a 3D plotting function
#copied from: https://stackoverflow.com/questions/40853556/3d-discrete-heatmap-in-matplotlib
"""
def cuboid_data(center, size=(1,1,1)):
    # code taken from
    # http://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid?noredirect=1&lq=1
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    return x, y, z

def plotCubeAt(pos=(0,0,0), c="b", alpha=0.1, ax=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( (pos[0],pos[1],pos[2]) )
        ax.plot_surface(X, Y, Z, color=c, rstride=1, cstride=1, alpha=0.1)

def plotMatrix(ax, x, y, z, data, cmap="jet", cax=None, alpha=0.1):
    # plot a Matrix 
    norm = matplotlib.colors.Normalize(vmin=data.min(), vmax=data.max())
    colors = lambda i,j,k : matplotlib.cm.ScalarMappable(norm=norm,cmap = cmap).to_rgba(data[i,j,k]) 
    for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                for k, zi, in enumerate(z):
                    plotCubeAt(pos=(xi, yi, zi), c=colors(i,j,k), alpha=alpha,  ax=ax)



    if cax !=None:
        cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')  
        cbar.set_ticks(np.unique(data))
        # set the colorbar transparent as well
        cbar.solids.set(alpha=alpha)              
"""



#%% Step4-1: Plot this data into the 3D 
"""
if __name__ == '__main__':

    # x and y and z coordinates
    x = np.array(range(10))
    y = np.array(range(10,15))
    z = np.array(range(15,20))
    data_value = np.random.randint(1,4, size=(len(x), len(y), len(z)) )
    print data_value.shape

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection='3d')
    ax_cb = fig.add_axes([0.8, 0.3, 0.05, 0.45])
    ax.set_aspect('equal')

    plotMatrix(ax, x, y, z, data_value, cmap="jet", cax = ax_cb)

    plt.savefig(__file__+".png")
    plt.show()
"""

"""
def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()  
"""

#%%
"""
decimals = 1
#wrist2_rounded_x_min
#https://stackoverflow.com/questions/477486/how-to-use-a-decimal-range-step-value
x = np.arange((0/10+wrist2_rounded_x_min),(heatmap_x_axis_length/10+wrist2_rounded_x_min),0.1) #??? +1 ???
y = np.arange((0/10+wrist2_rounded_y_min),(heatmap_y_axis_length/10+wrist2_rounded_y_min),0.1) #??? +1 ???
z = np.arange((0/10+wrist2_rounded_z_min),(heatmap_z_axis_length/10+wrist2_rounded_z_min),0.1) #??? +1 ???

ref_x = np.array(range(0,heatmap_x_axis_length,1)) #??? +1 ???
ref_y = np.array(range(0,heatmap_y_axis_length,1)) #??? +1 ???
ref_z = np.array(range(0,heatmap_z_axis_length,1)) #??? +1 ???
data_value = wrist2_position_heatmap

#fig = plt.figure()
#scatter3d(ref_x,ref_y,ref_z,data_value)
"""

#%% Step4-2: Construct a 3D plot

#fig = plt.figure() 
#ax = fig.add_subplot(111,projection='3d')

#ax.axes.set_xlim3d(left=axis_limit_min, right=axis_limit_max) 
#ax.axes.set_ylim3d(bottom=axis_limit_min, top=axis_limit_max) 
#ax.axes.set_zlim3d(bottom=axis_limit_min, top=axis_limit_max)

#%% LEVEL1 RESULT: JUST PLOTTING THE WRIST2 POINTS
"""
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(df_exp_wrist2_rounded['wrist2_x'],df_exp_wrist2_rounded['wrist2_y'],df_exp_wrist2_rounded['wrist2_z'])
pyplot.show()
"""

#%% LEVEL2: SET X,Y,Z AXIS TO THE RANGE OF THE ARM ACTIVITY RANGE