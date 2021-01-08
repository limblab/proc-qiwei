# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:59:18 2020

@author: dongq
"""

#%% Import packages

import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import cv2
import os
import math
import pandas as pd 
import copy
import seaborn as sns
import ffmpeg

#%%record file names

folder_1 = r'C:\Users\dongq\Desktop\Qiwei-Labeled-Data-2'
folder_2 = r'C:\Users\dongq\Desktop\Joe-Labeled-Data-2'

cam_folders = [r'\Crackle_20201203_00001',
               r'\Crackle_20201203_00002',
               r'\Crackle_20201203_00003',
               r'\Crackle_20201203_00004',
               r'\Crackle_20201203_00007',
               r'\Crackle_20201203_00008',
               r'\Crackle_20201203_00009',
               r'\Crackle_20201203_00010',]

file_name = r'\CollectedData_Qiwei.csv'


#%%calculate diff

list_1 = []
list_2 = []

list_1_arr = []
list_2_arr = []

diff = []

for i in range(len(cam_folders)):
    file_1_dir = folder_1 + cam_folders[i] + file_name
    file_2_dir = folder_2 + cam_folders[i] + file_name
    
    list_1.append(pd.read_csv(file_1_dir))
    list_2.append(pd.read_csv(file_2_dir))
    
    #list_1_arr.append
    
    list_1_arr.append(pd.read_csv(file_1_dir))
    list_2_arr.append(pd.read_csv(file_2_dir))
    
    list_1_arr[i] = list_1_arr[i].drop(['scorer','Qiwei.2','Qiwei.3','Qiwei.4','Qiwei.5','Qiwei.20','Qiwei.21','Qiwei.22','Qiwei.23','Qiwei.24','Qiwei.25'],axis=1)
    list_1_arr[i] = list_1_arr[i].drop([0,1],axis=0)
    
    list_2_arr[i] = list_2_arr[i].drop(['scorer','Qiwei.2','Qiwei.3','Qiwei.4','Qiwei.5','Qiwei.20','Qiwei.21','Qiwei.22','Qiwei.23','Qiwei.24','Qiwei.25'],axis=1)
    list_2_arr[i] = list_2_arr[i].drop([0,1],axis=0)
    
    list_1_arr[i] = list_1_arr[i].to_numpy()
    list_2_arr[i] = list_2_arr[i].to_numpy()
    
    where_are_nans_list_1_arr = pd.isnull(list_1_arr[i])
    tmp_list_1 = list_1_arr[i].astype(np.float)
    tmp_list_1[where_are_nans_list_1_arr] = 0
    
    where_are_nans_list_2_arr = pd.isnull(list_2_arr[i])
    tmp_list_2 = list_2_arr[i].astype(np.float)
    tmp_list_2[where_are_nans_list_2_arr] = 0
    
    tmp_diff = tmp_list_1 - tmp_list_2
    
    tmp_diff_sqr = tmp_diff ** 2
    
    tmp_diff_sqrt = np.zeros((tmp_diff_sqr.shape[0],int(tmp_diff_sqr.shape[1]/2)))
        
    for i in range(int(tmp_diff_sqr.shape[1]/2)): #10 iterations for 10 points, each column is a marker in 2D
        print(i)
        x = i*2
        y = i*2+1
        tmp_tmp_diff_sqrt = np.sqrt(tmp_diff_sqr[:,x] + tmp_diff_sqr[:,y])
        #tmp_diff_sqrt = np.hstack((tmp_diff_sqrt,tmp_tmp_diff_sqrt))
        #tmp_diff_sqrt = np.append(tmp_diff_sqrt,tmp_tmp_diff_sqrt,axis=1)
        tmp_diff_sqrt[:,i] = tmp_tmp_diff_sqrt
        #tmp_diff_sqrt
        print(tmp_diff_sqrt.shape)
    #tmp_diff_sqrt_copy = np.zeros((tmp_diff_sqrt.shape[0],tmp_diff_sqrt.shape[1]))
    
    diff.append(tmp_diff_sqrt)
    
    


#%%average over frames

"""
: * meaning difference larger than 100

Large differences:
    2D cam1: no
    2D cam2: shoulder1 [frame 4,5]
    2D cam3: elbow1 [frame 1,2]
    2D cam4: shoulder1 [frame 3,4] 
    
    3D cam1: elbow1 [frame *1,*4,*5], elbow2 [frame *4,*5]
    3D cam2: elbow1 [frame 3]
    3D cam3: elbow1 [frame *5], elbow2 [frame *3,*5], hand2 [frame *5], hand3 [frame *5]
    3D cam4: shoulder1 [frame *1]
    
"""




#diff_avg = np.zeros((len(diff),diff[1].shape[1]))

diff_avg = []

for i in range(len(diff)):
    
    tmp_diff = diff[i]
    
    sum_diff = sum(tmp_diff >= 20)
    
    tmp_diff_in_range = (tmp_diff >= 20)
    
    tmp_diff[tmp_diff_in_range] = 0
    
    #tmp_diff_avg = sum(tmp_diff)/(tmp_diff.shape[0]
    
    tmp_diff_avg = []
    for i in range(tmp_diff.shape[1]):
        result = sum(tmp_diff[:,i])/(tmp_diff.shape[0]-sum_diff[i])
        tmp_diff_avg.append(result)
    
    print(tmp_diff_avg)
    diff_avg.append(tmp_diff_avg)
    #diff_avg[:,i] = tmp_diff_avg




#%% average over cams
#diff_avg_avg = []
#for i in range(len(diff_avg)):
#    sum_over_cams = np.zeros((len(diff_avg[0])))
#    non_zero_cams = np.zeros((len(diff_avg[0])))
#    
#    sum_over_cams = sum_over_cams + diff_

diff_avg_arr = np.zeros((len(diff_avg),len(diff_avg[0])))

for i in range(len(diff_avg)):
    for j in range(len(diff_avg[i])):
        diff_avg_arr[i,j] = diff_avg[i][j]
    

diff_avg_arr_zero_counter = sum(diff_avg_arr != 0)

diff_avg_over_cams = sum(diff_avg_arr)/diff_avg_arr_zero_counter

    
    
#%% Plot the results

markers = ['shoulder','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
x_pos = [1,2,3,4,5,6,7,8]

plt.figure()
plt.bar(x=markers,height=diff_avg_over_cams)
plt.xlabel("markers (over all cams in 2 experiments)")
plt.ylabel("difference (in pixels")
plt.title("difference between Qiwei anc Joe's markers")
#ax.set_xticks(diff_avg_over_cams)
for i in range(diff_avg_over_cams.shape[0]):
    #plt.text(x = r4[i]-0.5 , y = bars4[i]+0.1, s = label[i], size = 6)
    plt.text(x=x_pos[i]-1.4,y=diff_avg_over_cams[i]+0.1,s = str(round(diff_avg_over_cams[i],3)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    