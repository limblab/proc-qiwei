# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:20:02 2020

@author: dongq
"""

#%% import packages

import scipy
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import cv2
import os
import math
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
from scipy.interpolate import interp1d
import pandas as pd
import csv
import random
import sys
import shelve

#%% read in files

main_folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03'

vid_folder = r'\neural-data'

#RT3D
file_name_cam1 = r'\Crackle_20201203_RT2D_RobotData'

#RT2D
#file_name_cam1 = r'\Crackle_20201203_00001DLC_resnet50_TestDec3shuffle1_1030000filtered'
#file_name_cam2 = r'\Crackle_20201203_00002DLC_resnet50_TestDec3shuffle1_1030000filtered'
#file_name_cam3 = r'\Crackle_20201203_00003DLC_resnet50_TestDec3shuffle1_1030000filtered'
#file_name_cam4 = r'\Crackle_20201203_00004DLC_resnet50_TestDec3shuffle1_1030000filtered'

file_type = '.csv'

#RT3D
groundTruth_file_name = r'\Ground_truth_segments_2020-12-03-RT2D-ForHandleData.txt'

#RT2D
#groundTruth_file_name = r'\Ground_truth_segments_2020-12-03-RT2D.txt'

cam1 = pd.read_csv(main_folder + vid_folder + file_name_cam1 + file_type)
#cam2 = pd.read_csv(main_folder + vid_folder + file_name_cam2 + file_type)
#cam3 = pd.read_csv(main_folder + vid_folder + file_name_cam3 + file_type)
#cam4 = pd.read_csv(main_folder + vid_folder + file_name_cam4 + file_type)


f = open(main_folder + groundTruth_file_name, "r") 


"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Remember to check this for EACH VIDEO cuz they'll vary
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
#RT3D
frames_per_second = 24

#RT2D
#frames_per_second = 24

#%% Define Trial segmenting function
"""
input:
    df: dataframe for one camera
    ground_truth_list: array, showing that for each frame in the camera, is 
    this frame in experiment phase (1) or not. In other words, is the monkey
    doing experiment in this frame or not.
    
output:
    df2: dataframe with experiment phase included only.

The function uses the ground_truth_list to segment the experiment phase in the
input dataframe, and then returns the result.
"""
def experiment_trial_segment(df,ground_truth_list):
    df2 = pd.DataFrame(np.zeros((0,df.shape[1])),columns = df.columns)
    for i in range(len(ground_truth_list)):
        df2.loc[i] = df.iloc[ground_truth_list[i]-1]
    #print(df2)
    return df2


#def experiment_trial_segment(array, ground_truth_list):

#%% Get the array for trial segmentation
#if experiment_phase_only == 1:
    #df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')

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
        
#%% get ground truth segment 
        
ground_truth_segment = np.zeros((cam1.shape[0]))        
for i in range(len(f_frame_list)):
    #print(i)
    ground_truth_segment[f_frame_list[i]-1] = 1

         
#%% Use the experiment_trial_segment() for each camera, to get experiment_only dataframe

#cam1_exp_only = experiment_trial_segment(cam1,f_frame_list)
#cam2_exp_only = experiment_trial_segment(cam2,f_frame_list)
#cam3_exp_only = experiment_trial_segment(cam3,f_frame_list)
#cam4_exp_only = experiment_trial_segment(cam4,f_frame_list)
    
cams = [cam1]
cams_arr = []
cams_arr_exp = []

for i in range(len(cams)):
    cams[i] = cams[i].drop([0,1],axis=0)
    cams_arr.append(cams[i].to_numpy().astype(np.float))
    cams_arr_exp.append(cams[i].to_numpy().astype(np.float)[f_frame_list])
         
#cam1_arr = cam1.to_numpy()
#cam2_arr = cam2.to_numpy()
#cam3_arr = cam3.to_numpy()
#cam4_arr = cam4.to_numpy()

#cam1_exp_only =         
    
#%% Extract sections of good data
    
"""
This section extracts a bunch (segment_nums) of good tracking results of size
(segment_size), drop a number (dropout_size) of frames in the middle of each
of these sections, and use interpolate function to fix them back, see how well
these interpolation functions perform, in other words, how large is the
interpolation error (in terms of how far the interpolation data is from the
actual tracked data)

segment_nums: how many segments per camera per marker to extract
segment_size: how long each segment should be
dropout_size: how many frames to drop for each segment
likelihood_lim: Only take a segment of data into account if all the tracking
                results in this segment has likelihood values larger than likelihood_lim 
cams_list: which cams to use
"""

#segment_nums = 100 #number of segments chosen per camera per marker
segment_size = 30    #frames
#dropout_size = 2 #frames in each segment
dropout_size = 5 #frames in each segment
likelihood_lim = 0.9 #0~1, 1 being the most likely, 0 being the least likely
cams_list = [1,2,3,4] #which cams to use

markers_x = [4]
markers_y = [5]
markers_l = [3] #likelihood
markers_f = [0] #frames

"""

cam_arr_exp



for i in each camera: #4
    for j in each marker: #8
        from k in all the frames #30000+
            if k >= likelihood_lim
                if the likelihood values from k to k + "segment_size"(30 frames) are all larger than "likelihood_lim"
                    append this segment (frame nums, x, y, likelihood) that is "from k to k + "segment_size"" to the result array
                    
"""
camera_lists = []

for i in range(len(cams_arr_exp)): # 4 cameras
    #print(i)
    
    camera_marker = []
    
    for j in range(len(markers_x)): #8 markers
        #use the markers_x, markers_y, markers_l, markers_f for j
        
        marker_segment = []
        
        k = 0
        #for k in range(cams_arr_exp[i].shape[0]): #33647 frames
        while k < cams_arr_exp[i].shape[0]:
            
            tmp_likelihood = cams_arr_exp[i][k , markers_l[j]]
            #print(tmp_likelihood)
            #If the likelihood is in good shape, we check the subsequent 30 frames,
            #see if they're also in good shape or not
            
            if tmp_likelihood >= likelihood_lim:
                
                tmp_likelihood_list = cams_arr_exp[i][k:k+segment_size , markers_l[j]]
                
                if sum(tmp_likelihood_list > likelihood_lim) == segment_size:
                    
                    tmp_segment = np.zeros((segment_size, 4)) #4 for frame numbers, x value, y value, likelihood value
                    tmp_segment[:,0] = cams_arr_exp[i][k:k+segment_size , markers_f[0]]
                    tmp_segment[:,1] = cams_arr_exp[i][k:k+segment_size , markers_x[j]]
                    tmp_segment[:,2] = cams_arr_exp[i][k:k+segment_size , markers_y[j]]
                    tmp_segment[:,3] = cams_arr_exp[i][k:k+segment_size , markers_l[j]]
                    
                    marker_segment.append(tmp_segment)
                    
                    #print(tmp_segment)
                    
                    k = k + segment_size
                    #print(k)
                #print(sum(tmp_likelihood_list > likelihood_lim))
            k = k + 1
        camera_marker.append(marker_segment)
    
    #print(i)
    camera_lists.append(camera_marker)
    
#%% If segment_nums parameter exists, randomly choose the segment_nums numbers of
    #segments, and cover the camera_lists list with that.

if 'segment_nums' in globals():
    #print("1")
    camera_lists_limit = []
    
    for i in range(len(camera_lists)):
        
        marker_lists_limit = []
        
        for j in range(len(camera_lists[i])):
            
            if camera_lists[i][j] == []:
                marker_lists_limit.append([])
                
            else:
                randomList = random.sample(range(0, len(camera_lists[i][j])),segment_nums)
                #print(randomList)
                #print("\n")
                
                #print(len(camera_lists[i][j]))
                #print("\n")
                markers_limited = []
                
                for k in range(segment_nums):
                    #print(randomList[i])
                    markers_limited.append(camera_lists[i][j][randomList[k]])
                #markers_limit = camera_lists[i][j]randomList
                #print(markers_limit)
                marker_lists_limit.append(markers_limited)
        camera_lists_limit.append(marker_lists_limit)
    
    #Replace camera_lists with camera_lists_limit
    camera_lists = camera_lists_limit

            
#%% Drop the middle frames according to dropout_size
camera_lists_dropped = copy.deepcopy(camera_lists)

#drop column 1 and 2 (X axis value and Y axis value)'s center n values
#segment_size 30
#dropout_size 2

dropout_start_pos = int((segment_size/2) - int(dropout_size/2))
dropout_end_pos = dropout_start_pos + dropout_size
            

#%% Use interpolation function on the extracted data

camera_lists_dropped = copy.deepcopy(camera_lists)

camera_lists_interpolated = []

for i in range(len(camera_lists_dropped)): #4 for each camera
    
    camera_marker_dropped = []
    
    for j in range(len(camera_lists_dropped[i])): #8 for each marker
        
        marker_segment_dropped = []
        
        #if camera_lists_dropped[i][j] == []: #if it's empty, for cam1 cam3 shoulder
        #    camera_marker_dropped.append(marker_segment_dropped)
        #else:

        for k in range(len(camera_lists_dropped[i][j])): #for each segment, around 1000
            
            tmp_segment = camera_lists_dropped[i][j][k]
            tmp_segment_dropped = copy.deepcopy(tmp_segment)
            
            tmp_seg_x = tmp_segment[:,1]
            tmp_seg_y = tmp_segment[:,2]
            
            tmp_seg_x[dropout_start_pos:dropout_end_pos] = np.nan
            tmp_seg_y[dropout_start_pos:dropout_end_pos] = np.nan
            
            tmp_seg_x_s = pd.Series(tmp_seg_x)
            tmp_seg_y_s = pd.Series(tmp_seg_y)
            
            #order 3 polynomial
            new_x_s = tmp_seg_x_s.interpolate(method = 'polynomial', order = 3)
            new_y_s = tmp_seg_y_s.interpolate(method = 'polynomial', order = 3)
            
            #linear
            #new_x_s = tmp_seg_x_s.interpolate(method = 'linear')
            #new_y_s = tmp_seg_y_s.interpolate(method = 'linear')
            
            #order 2 polynomial
            #new_x_s = tmp_seg_x_s.interpolate(method = 'polynomial', order = 2)
            #new_y_s = tmp_seg_y_s.interpolate(method = 'polynomial', order = 2)
            
            #order 3 spline
            #new_x_s = tmp_seg_x_s.interpolate(method = 'spline', order = 3)
            #new_y_s = tmp_seg_y_s.interpolate(method = 'spline', order = 3)
            
            #order 2 spline
            #new_x_s = tmp_seg_x_s.interpolate(method = 'spline', order = 2)
            #new_y_s = tmp_seg_y_s.interpolate(method = 'spline', order = 2)
            
            #order 2 spline
            #new_x_s = tmp_seg_x_s.interpolate(method = 'spline', order = 1)
            #new_y_s = tmp_seg_y_s.interpolate(method = 'spline', order = 1)
            
            #cubic spline
            #new_x_s = tmp_seg_x_s.interpolate(method = 'cubicspline')
            #new_y_s = tmp_seg_y_s.interpolate(method = 'cubicspline')
            
            new_x_np = new_x_s.to_numpy()
            new_y_np = new_y_s.to_numpy()
            
            tmp_segment_dropped[:,1] = new_x_np
            tmp_segment_dropped[:,2] = new_y_np
            
            
            marker_segment_dropped.append(tmp_segment_dropped)
        
        camera_marker_dropped.append(marker_segment_dropped)
        
    camera_lists_interpolated.append(camera_marker_dropped)
               

#fq = interp1d(test_x_drop, test_y_drop, kind = 'quadratic') #not useful

#import pylab as pl
#pl.plot(xint,fl(xint), color="green", label = "Linear")
#pl.plot(test_x_drop,fq(test_x_drop), color="yellow", label ="Quadratic")
#pl.legend(loc = "best")
#pl.show()

#%% Calculate diff between acutal recorded results and interpolated results
"""
#diff between
#camera_lists
#camera_lists_interpolated

in between
dropout_start_pos
dropout_end_pos

for x and y values separately [:,1] and [:,2]

In the marker_segment_diff there will be a few hundred of lists with the
structure of n*4. n representing the number of dropped out points we set 
by the aprameter "dropout_size".
Column 1: frame number
Column 2: absolute value of the difference bewteen actual x marker position and interpolated x marker position
Column 3: absolute value of the difference bewteen actual y marker position and interpolated y marker position
Column 4: Euclidean distance between actual marker position and interpolated marker position

"""

camera_diff = []

for i in range(len(camera_lists)): #4, per camera
    
    camera_marker_diff = []
    
    for j in range(len(camera_lists[i])): #8, per each marker
        
        marker_segment_diff = []
        
        for k in range(len(camera_lists[i][j])): #a few hundreds, per each segment of interpolated data
            
            tmp_truth = copy.deepcopy(camera_lists[i][j][k][dropout_start_pos:dropout_end_pos,:])
            tmp_interp = copy.deepcopy(camera_lists_interpolated[i][j][k][dropout_start_pos:dropout_end_pos,:])
            
            tmp_segment = copy.deepcopy(tmp_truth) #I just want this 2*4 list structure without reconstructing it from 0
            
            #tmp_segment = 
            
            tmp_diff = tmp_truth[:,1:3] - tmp_interp[:,1:3]
            
            tmp_dist = np.sqrt((tmp_diff[:,0] ** 2) + (tmp_diff[:,1] ** 2))
            
            tmp_segment[:,1:3] = abs(tmp_diff)
            tmp_segment[:,3] = tmp_dist
            
            #print(tmp_segment)
            #print(tmp_dist)
            
            marker_segment_diff.append(tmp_segment)
            
        camera_marker_diff.append(marker_segment_diff)
    
    camera_diff.append(camera_marker_diff)




#%% Calculate the stats of these interpolation results

#total average (actually present)

#average per marker (calculate, but don't need to present)

#average per camera (calculate, but don't need to present)

avg_cam = []
stderr_cam = []
wholeArray_diff = []
wholeArray_diff_with_frame_num = []

for i in range(len(camera_diff)): #4, per camera
    
    avg_marker = []
    stderr_marker = []
    wholeArray_marker = []
    wholeArray_marker_with_frame_num = []
    
    for j in range(len(camera_diff[i])): #8, per marker
        
        if camera_diff[i][j] == []:
            avg_marker.append(0)
            stderr_marker.append(0)
        else:
            
            whole_array = np.zeros(len(camera_diff[i][j]) * dropout_size)
            
            wholeArray_marker_section_frame_num = np.zeros(shape=(len(camera_diff[i][j]) * dropout_size , 2))
            
            for k in range(len(camera_diff[i][j])): #number of segments
                
                start = k * dropout_size 
                end = k * dropout_size + dropout_size
                
                whole_array[start:end] = camera_diff[i][j][k][:,3]
                wholeArray_marker_section_frame_num[start:end,0] = camera_diff[i][j][k][:,0]
                wholeArray_marker_section_frame_num[start:end,1] = camera_diff[i][j][k][:,3]
                
            #print(whole_array)

            #avg_diff = ttl_diff / (len(camera_diff[i][j]) *2)
            #print(np.mean(whole_array))
            #print(avg_diff)
            #print("\n")
            
            avg_marker.append(np.mean(whole_array))
            stderr_marker.append(np.std(whole_array))
            wholeArray_marker.append(whole_array)
            wholeArray_marker_with_frame_num.append(wholeArray_marker_section_frame_num)
            
            #print(np.std(whole_array))
            #print('\n')
            
            
    avg_cam.append(avg_marker)
    stderr_cam.append(stderr_marker)
    wholeArray_diff.append(wholeArray_marker)
    wholeArray_diff_with_frame_num.append(wholeArray_marker_with_frame_num)


#%% save as csv

# os.path.dirname(main_folder) 'C:\\Users\\dongq\\DeepLabCut'
# os.path.basename(main_folder) 'Crackle-Qiwei-2020-12-03'
    
save_file_avg = avg_cam
save_file_name_avg = 'avg'

save_file_stderr = stderr_cam
save_file_name_stderr = 'stderr'

#my_var_name = [ k for k,v in locals().iteritems() if v == my_var][0]#%%
np.savetxt(os.path.basename(main_folder) + '_dropout_size_' + str(dropout_size) + '_' + save_file_name_avg + '.csv', save_file_avg, delimiter = ',', fmt='%1.3f')
np.savetxt(os.path.basename(main_folder) + '_dropout_size_' + str(dropout_size) + '_' + save_file_name_stderr + '.csv', save_file_stderr, delimiter = ',', fmt='%1.3f')


#%% Combine Total average and standard deviation for this entire dropout length

#Get all the diff values for all 8 markers into one array for computation easiness

#First, get how long this whole thing is. Since each array has different lengths,
#it is utterly impossible to get all these single arrays to one 2D array so anyways
#I SUCK AT CODING FUCK

total_len = 0
for i in range(len(wholeArray_diff)):
    marker_len = 0
    for j in range(len(wholeArray_diff[i])):
        marker_len = marker_len + wholeArray_diff[i][j].shape[0]
    total_len  = total_len + marker_len
    
wholeArray = np.zeros((total_len,))
total_len_count = 0

for i in range(len(wholeArray_diff)):
    for j in range(len(wholeArray_diff[i])):
        start = total_len_count
        end = total_len_count + wholeArray_diff[i][j].shape[0]
        wholeArray[start:end] = wholeArray_diff[i][j]
        total_len_count = total_len_count + wholeArray_diff[i][j].shape[0]
        
#%% Save the wholeArray as one single array, for boxplot

save_file_wholeArr = wholeArray
save_file_name_wholeArr = 'wholeArr'

#my_var_name = [ k for k,v in locals().iteritems() if v == my_var][0]#%%
np.savetxt(os.path.basename(main_folder) + '_dropout_size_' + str(dropout_size) + '_' + save_file_name_wholeArr + '.csv', save_file_wholeArr, delimiter = ',', fmt='%1.3f')


#%% Calculate the overall mean and std dev
        

outlier_lim = 0.9

overall_mean = np.mean(wholeArray)
print("overall mean")
print(overall_mean)
print("\n")

overall_stddev = np.std(wholeArray)
print("overall standard deviation")
print(overall_stddev)
print("\n")

overall_outliers = sum(wholeArray > outlier_lim)/wholeArray.shape[0]
print("overall outliers")
print(overall_outliers)



#plt.hist(wholeArray)

#plt.boxplot(wholeArray)

#%% Calculate the outlier percentage of each marker and each camera

#wholeArray_diff_with_frame_num


outlier_per_cam = []

for i in range(len(wholeArray_diff_with_frame_num)):
    
    outlier_per_marker = []
    
    if len(wholeArray_diff_with_frame_num[i]) == 7: #shoulder is so bad that it's not counted
        outlier_per_marker.append(0)
    
    for j in range(len(wholeArray_diff_with_frame_num[i])):
        
        marker_diff_arr = wholeArray_diff_with_frame_num[i][j][:,1]
        
        marker_diff_per = sum(marker_diff_arr > 20) / marker_diff_arr.shape[0]
        
        outlier_per_marker.append(marker_diff_per)
    
    outlier_per_cam.append(outlier_per_marker)


#%% Plot the outlier percentage of each maker and of each camera
    
#outlier_per_cam
"""
xLabel_list = ['shoulder','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']

for i in range(len(outlier_per_cam)):
    
    x = range(len(outlier_per_cam[i]))
    
    plt.figure(figsize = (10,5.5))
    plt.bar(x, height = outlier_per_cam[i])
    
    plt.xlabel("marker",fontsize=16)
    plt.ylabel("outlier ratio",fontsize=16)
    plt.title("outlier ratio for each marker on cam " + str(i+1) + " with " + str(dropout_size) + " dropout(s) per section",fontsize=16)
    
    plt.xticks(x, xLabel_list,fontsize = 16)
    
    plt_name = "dropout_" +  str(dropout_size) + "_cam_" + str(i+1) + ".png"
    plt.savefig(plt_name)
"""
    



#%% Take the ones in camera_lists and camera_lists_interpolated that have cam_diff larger than 20

cam_lists_outliers_original = []
cam_lists_outliers_interpolated = []

cam_lists_outlier_original_wholeSection = []
cam_lists_outliers_interpolated_wholeSection = []

for i in range(len(camera_diff)):
    
    marker_lists_outliers_original = []
    marker_lists_outliers_interpolated = []
    
    marker_lists_outliers_original_wholeSection = []
    marker_lists_outliers_interpolated_wholeSection = []
    
    for j in range(len(camera_diff[i])):
        
        segments_list_outliers_original = []
        segments_list_outliers_interpolated = []
        
        segments_list_outliers_original_wholeSection = []
        segments_list_outliers_interpolated_wholeSection = []
        
        for k in range(len(camera_diff[i][j])):
            
            diffs = camera_diff[i][j][k][:,3]
            outlier_diffs = diffs > outlier_lim
            
            tmp_original = camera_lists[i][j][k][dropout_start_pos:dropout_end_pos,:]
            tmp_interpolated = camera_lists_interpolated[i][j][k][dropout_start_pos:dropout_end_pos,:]
            
            tmp_original_wholeSection = camera_lists[i][j][k][:,:]
            tmp_interpolated_wholeSection = camera_lists_interpolated[i][j][k][:,:]
            
            if tmp_original[outlier_diffs,:].size != 0:
                segments_list_outliers_original.append(tmp_original[outlier_diffs,:])
                segments_list_outliers_interpolated.append(tmp_interpolated[outlier_diffs,:])
                
                segments_list_outliers_original_wholeSection.append(tmp_original_wholeSection)
                segments_list_outliers_interpolated_wholeSection.append(tmp_interpolated_wholeSection)
                #print(tmp_original[outlier_diffs,:])
        
        marker_lists_outliers_original.append(segments_list_outliers_original)
        marker_lists_outliers_interpolated.append(segments_list_outliers_interpolated)
        
        marker_lists_outliers_original_wholeSection.append(segments_list_outliers_original_wholeSection)
        marker_lists_outliers_interpolated_wholeSection.append(segments_list_outliers_interpolated_wholeSection)
    
    cam_lists_outliers_original.append(marker_lists_outliers_original)
    cam_lists_outliers_interpolated.append(marker_lists_outliers_interpolated)
    
    cam_lists_outlier_original_wholeSection.append(marker_lists_outliers_original_wholeSection)
    cam_lists_outliers_interpolated_wholeSection.append(marker_lists_outliers_interpolated_wholeSection)
    
#%% Combine all the sections in each marker into one array, simply just for easiness for later plotting

cam_lists_outliers_original_combined = []
cam_lists_outliers_interpolated_combined = []


for i in range(len(cam_lists_outliers_original)):
    
    marker_lists_outliers_original_combined = []
    marker_lists_outliers_interpolated_combined = []
    
    for j in range(len(cam_lists_outliers_original[i])):
        
        if (len(cam_lists_outliers_original[i][j]) != 0):
            total_rows = 0
            total_cols = cam_lists_outliers_original[i][j][0].shape[1]
            for k in range(len(cam_lists_outliers_original[i][j])):
                total_rows += cam_lists_outliers_original[i][j][k].shape[0]
                
            segments_list_outliers_original_combined = np.zeros((total_rows, total_cols))
            segments_list_outliers_interpolated_combined = np.zeros((total_rows, total_cols))
            #print(segments_list_outliers_interpolated_combined)
            
            row_count = 0
            for k in range(len(cam_lists_outliers_original[i][j])):
                row_end = row_count + cam_lists_outliers_original[i][j][k].shape[0]
                #print(row_count)
                #print(row_end)
                #print("\n")
                segments_list_outliers_original_combined[row_count:row_end,:] = cam_lists_outliers_original[i][j][k]
                segments_list_outliers_interpolated_combined[row_count:row_end,:] = cam_lists_outliers_interpolated[i][j][k] 
                
                row_count = row_end

            marker_lists_outliers_original_combined.append(segments_list_outliers_original_combined)
            marker_lists_outliers_interpolated_combined.append(segments_list_outliers_interpolated_combined)
        else:
            marker_lists_outliers_original_combined.append([])
            marker_lists_outliers_interpolated_combined.append([])
    cam_lists_outliers_original_combined.append(marker_lists_outliers_original_combined)
    cam_lists_outliers_interpolated_combined.append(marker_lists_outliers_interpolated_combined)
    
    
#%%
save_file_original = cam_lists_outliers_original_combined
save_file_name_original = 'original'

save_file_interpolated = cam_lists_outliers_interpolated_combined
save_file_name_interpolated = 'interpolated'

#my_var_name = [ k for k,v in locals().iteritems() if v == my_var][0]#%%
#np.savetxt(os.path.basename(main_folder) + '_dropout_size_' + str(dropout_size) + '_' + save_file_name_original + '.csv', save_file_original, delimiter = ',', fmt='%1.3f')
#np.savetxt(os.path.basename(main_folder) + '_dropout_size_' + str(dropout_size) + '_' + save_file_name_interpolated + '.csv', save_file_interpolated, delimiter = ',', fmt='%1.3f')


    
#%% Plot cam_lists

#cam_lists_outlier_original_wholeSection
#cam_lists_outliers_interpolated_wholeSection

#dropout_start_pos
#dropout_end_pos


x_column = 1
y_column = 2

cams_columns = len(cam_lists_outlier_original_wholeSection)
markers_rows = len(cam_lists_outlier_original_wholeSection[0])

cams_columns_names = ['3D']
cams_count = 0
markers_rows_names = ['Handle']
markers_rows_nums = [0]
#markers_rows_names = ['shoulder','elbow1','wrist1','hand1']
#markers_rows_nums = [0,1,3,5]

markers_count = 0

current_plot_num = 1

frame_num_to_plot = 1

#plt.plot()

#plt.title("Outlier Examples of Interpolation Results")

for i in range(len(cam_lists_outlier_original_wholeSection)): #camreas, columns
    
    #for j in range(len(cam_lists_outlier_original_wholeSection[i])): #markers, rows
    for j in range(len(markers_rows_nums)): #markers, rows
        
        plt.subplot(cams_columns, markers_rows, current_plot_num)

        if i == 0:
            plt.title(markers_rows_names[j],fontsize = 16)
        if j == 0:
            plt.ylabel(cams_columns_names[i],fontsize = 16)
        
        if len(cam_lists_outlier_original_wholeSection[i][j]) == 0:
            plt.plot(0,0)
        else:
            x_val_orig = cam_lists_outlier_original_wholeSection[i][markers_rows_nums[j]][frame_num_to_plot][:,x_column:x_column+1]
            y_val_orig = cam_lists_outlier_original_wholeSection[i][markers_rows_nums[j]][frame_num_to_plot][:,y_column:y_column+1]
            
            x_val_interp = cam_lists_outliers_interpolated_wholeSection[i][markers_rows_nums[j]][frame_num_to_plot][:,x_column:x_column+1]
            y_val_interp = cam_lists_outliers_interpolated_wholeSection[i][markers_rows_nums[j]][frame_num_to_plot][:,y_column:y_column+1]
            
            plt.plot(x_val_interp, y_val_interp,'red')
            plt.plot(x_val_orig,y_val_orig,'black')
        
            #plt.plot(x_val_orig[dropout_start_pos:dropout_end_pos],y_val_orig[dropout_start_pos:dropout_end_pos],'blue')
            #plt.plot(x_val_interp[dropout_start_pos:dropout_end_pos], y_val_interp[dropout_start_pos:dropout_end_pos],'green')
            
            
            
        current_plot_num += 1
    
#plt.xlabel("Cameras")
#plt.ylabel("Markers")

    
    
    
    