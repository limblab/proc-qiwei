# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 19:26:03 2021

@author: dongq
"""



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
import h5py
import glob
import seaborn as sns
import matplotlib.patches as mpatches
from itertools import repeat

from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


#%%



# =============================================================================
# Project_folder = r'D:\DLCdata\!Exp - Model Perf w Diff num of Training Iters\Crackle-Qiwei-2020-12-14-backup'
# DLC_train_test_set_dir = Project_folder + r'\training-datasets\iteration-0\UnaugmentedDataSet_TestDec14\Documentation_data-Test_95shuffle1.pickle'
# DLC_train_test_set = pd.read_pickle(DLC_train_test_set_dir)
# DLC_train_set = DLC_train_test_set[1]
# DLC_test_set = DLC_train_test_set[2]
# 
# DLC_estimated_dir = Project_folder + r'\evaluation-results\iteration-0\TestDec14-trainset95shuffle1'
# DLC_estimated_name = '\\DLC_resnet50_TestDec14shuffle1_1030000-snapshot-1030000.h5'
# DLC_estimated_filename = DLC_estimated_dir + DLC_estimated_name
# DLC_estimated = pd.read_hdf(DLC_estimated_filename)
# DLC_estimated.to_csv(DLC_estimated_dir + '\DLC_estimated.csv')
# #And then I deleted X,Y,Z columns by hand, just because I don't want to search for solutions when meeting multi-layer dataframes
# =============================================================================



Project_folder_1 = r'C:\Users\dongq\DeepLabCut\Datasets for 2D tracking evaluation\Han_202012-Joe-2020-12-04'
DLC_train_test_set_dir_1 = Project_folder_1 + r'\training-datasets\iteration-0\UnaugmentedDataSet_Han_202012Dec14\Documentation_data-Han_202012_75shuffle1.pickle'
DLC_train_test_set_1 = pd.read_pickle(DLC_train_test_set_dir_1)
DLC_train_set_1 = DLC_train_test_set_1[1]
DLC_test_set_1 = DLC_train_test_set_1[2]
DLC_estimated_dir_1 = Project_folder_1 + r'\evaluation-results\iteration-0\Han_202012Dec14-trainset75shuffle1'
DLC_estimated_name_1 = '\\DLC_resnet50_Han_202012Dec14shuffle1_400000-snapshot-400000.h5'
DLC_estimated_filename_1 = DLC_estimated_dir_1 + DLC_estimated_name_1
DLC_estimated_1 = pd.read_hdf(DLC_estimated_filename_1)
DLC_estimated_1.to_csv(DLC_estimated_dir_1 + '\DLC_estimated.csv')
#And then I deleted X,Y,Z columns by hand, just because I don't want to search for solutions when meeting multi-layer dataframes

Project_folder_2 = r'C:\Users\dongq\DeepLabCut\Datasets for 2D tracking evaluation\Crackle-Qiwei-2020-12-14'
DLC_train_test_set_dir_2 = Project_folder_2 + r'\training-datasets\iteration-0\UnaugmentedDataSet_TestDec14\Documentation_data-Test_95shuffle1.pickle'
DLC_train_test_set_2 = pd.read_pickle(DLC_train_test_set_dir_2)
DLC_train_set_2 = DLC_train_test_set_2[1]
DLC_test_set_2 = DLC_train_test_set_2[2]
DLC_estimated_dir_2 = Project_folder_2 + r'\evaluation-results\iteration-0\TestDec14-trainset95shuffle1'
DLC_estimated_name_2 = '\\DLC_resnet50_TestDec14shuffle1_1030000-snapshot-1030000.h5'
DLC_estimated_filename_2 = DLC_estimated_dir_2 + DLC_estimated_name_2
DLC_estimated_2 = pd.read_hdf(DLC_estimated_filename_2)
DLC_estimated_2.to_csv(DLC_estimated_dir_2 + '\DLC_estimated.csv')

Project_folder_3 = r'C:\Users\dongq\DeepLabCut\Datasets for 2D tracking evaluation\Rocket-Joe-2021-06-29'
DLC_train_test_set_dir_3 = Project_folder_3 + r'\training-datasets\iteration-3\UnaugmentedDataSet_RocketJun29\Documentation_data-Rocket_95shuffle1.pickle'
DLC_train_test_set_3 = pd.read_pickle(DLC_train_test_set_dir_3)
DLC_train_set_3 = DLC_train_test_set_3[1]
DLC_test_set_3 = DLC_train_test_set_3[2]
DLC_estimated_dir_3 = Project_folder_3 + r'\evaluation-results\iteration-3\RocketJun29-trainset95shuffle1'
DLC_estimated_name_3 = '\\DLC_resnet50_RocketJun29shuffle1_800000-snapshot-800000.h5'
DLC_estimated_filename_3 = DLC_estimated_dir_3 + DLC_estimated_name_3
DLC_estimated_3 = pd.read_hdf(DLC_estimated_filename_3)
DLC_estimated_3.to_csv(DLC_estimated_dir_3 + '\DLC_estimated.csv')









#%% read in the labeled data (ground truth labeled by human) from the labeled-data folder

def read_in_labeled_data(project_folder):
    DLC_labeled_dir = project_folder + r'\labeled-data'
    DLC_labeled_sub_dirs = os.listdir(DLC_labeled_dir)
    DLC_labeled_csv_file_names = []
    for i in range(len(DLC_labeled_sub_dirs)):
        DLC_labeled_csv_file_names.append(glob.glob(DLC_labeled_dir + "/" + DLC_labeled_sub_dirs[i] + "/*.csv"))
    
    DLC_labeled_csv_files = []
    for i in range(len(DLC_labeled_csv_file_names)):
        DLC_labeled_csv_files.append(pd.read_csv(DLC_labeled_csv_file_names[i][0]))
    
    DLC_labeled_csv_files_combined = pd.concat(DLC_labeled_csv_files)
    #DLC_labeled_name = r'\CollectedData_Qiwei.csv'
    
    DLC_frame_names_labeled = np.array(DLC_labeled_csv_files_combined['scorer'])
    return DLC_frame_names_labeled, DLC_labeled_csv_files_combined



DLC_frame_names_labeled_1, DLC_labeled_csv_files_combined_1 = read_in_labeled_data(Project_folder_1)
DLC_frame_names_estimated_1 = DLC_estimated_1.index.values

DLC_frame_names_labeled_2, DLC_labeled_csv_files_combined_2 = read_in_labeled_data(Project_folder_2)
DLC_frame_names_estimated_2 = DLC_estimated_2.index.values

DLC_frame_names_labeled_3, DLC_labeled_csv_files_combined_3 = read_in_labeled_data(Project_folder_3)
DLC_frame_names_estimated_3 = DLC_estimated_3.index.values

#%%  clean the DLC_frame_names_labeled
def delete_unnecessary_names(DLC_frame_names_labeled):
    i = 0
    while i < (len(DLC_frame_names_labeled)):
        #print(i)
        if DLC_frame_names_labeled[i] == 'bodyparts' or DLC_frame_names_labeled[i] == 'coords':
            DLC_frame_names_labeled = np.delete(DLC_frame_names_labeled,i)
        if DLC_frame_names_labeled[i] == 'bodyparts' or DLC_frame_names_labeled[i] == 'coords':
            DLC_frame_names_labeled = np.delete(DLC_frame_names_labeled,i)
        i+=1
    return DLC_frame_names_labeled

DLC_frame_names_labeled_1 = delete_unnecessary_names(DLC_frame_names_labeled_1)
DLC_frame_names_labeled_2 = delete_unnecessary_names(DLC_frame_names_labeled_2)
DLC_frame_names_labeled_3 = delete_unnecessary_names(DLC_frame_names_labeled_3)

    


#%% make a deep copy of the labeled dataset, and delete the unnecessary data columns from the labeled dataset
def delete_and_rename_hand_labeled_dataset(DLC_labeled_csv_files_combined, columns_to_keep):
    
    DLC_labeled_csv_files_combined_copied = DLC_labeled_csv_files_combined.copy(deep=True)
    
    DLC_labeled_csv_files_combined_copied.index = DLC_labeled_csv_files_combined_copied['scorer']
    DLC_labeled_csv_files_combined_copied_cols = [c for c in DLC_labeled_csv_files_combined_copied.columns if c != 'scorer'] #For whatever reason we have arm labeles here, and we don't use that anymore, so for this dataset I deleted them by hand, but we don't need to do so for other datasets later
    DLC_labeled_csv_files_combined_copied = DLC_labeled_csv_files_combined_copied[DLC_labeled_csv_files_combined_copied_cols]
    DLC_labeled_csv_files_combined_copied.index.name = 'index'
    
    #Add bodyparts and coords into the DLC_labeled_csv_files_combined_copied dataset
    DLC_labeled_csv_files_combined_copied_bodyparts = DLC_labeled_csv_files_combined_copied.iloc[0]
    DLC_labeled_csv_files_combined_copied_coords = DLC_labeled_csv_files_combined_copied.iloc[1]
    
    DLC_labeled_csv_files_combined_copied = DLC_labeled_csv_files_combined_copied.drop(['bodyparts','coords'])
    DLC_labeled_csv_files_combined_copied.index = DLC_labeled_csv_files_combined_copied.index.str.replace('\\','/')
    
    #Fix the column names so that it's at least readable
    DLC_marker_names_labeled = ['shoulder1_x','shoulder1_y',
                        'elbow1_x','elbow1_y',
                        'elbow2_x','elbow2_y',
                        'wrist1_x','wrist1_y',
                        'wrist2_x','wrist2_y',
                        'hand1_x','hand1_y',
                        'hand2_x','hand2_y',
                        'hand3_x','hand3_y']
    DLC_labeled_csv_files_combined_copied = DLC_labeled_csv_files_combined_copied[columns_to_keep]
    DLC_labeled_csv_files_combined_copied.columns = DLC_marker_names_labeled
    return DLC_labeled_csv_files_combined_copied

columns_to_keep_1 = ['Joe','Joe.1','Joe.2','Joe.3','Joe.4','Joe.5','Joe.6','Joe.7','Joe.8','Joe.9','Joe.10','Joe.11','Joe.12','Joe.13','Joe.14','Joe.15']
columns_to_keep_2 = ['Qiwei','Qiwei.1','Qiwei.2','Qiwei.3','Qiwei.4','Qiwei.5','Qiwei.6','Qiwei.7','Qiwei.8','Qiwei.9','Qiwei.10','Qiwei.11','Qiwei.12','Qiwei.13','Qiwei.14','Qiwei.15']
columns_to_keep_3 = ['Joe.2','Joe.3','Joe.4','Joe.5','Joe.6','Joe.7','Joe.8','Joe.9','Joe.10','Joe.11','Joe.12','Joe.13','Joe.14','Joe.15','Joe.16','Joe.17']

DLC_labeled_csv_files_combined_copied_1 = delete_and_rename_hand_labeled_dataset(DLC_labeled_csv_files_combined_1,columns_to_keep_1)
DLC_labeled_csv_files_combined_copied_2 = delete_and_rename_hand_labeled_dataset(DLC_labeled_csv_files_combined_2,columns_to_keep_2)
DLC_labeled_csv_files_combined_copied_3 = delete_and_rename_hand_labeled_dataset(DLC_labeled_csv_files_combined_3,columns_to_keep_3)


#%%
#Need to delete X,Y,Z columns, and arm columns?
#delete_XYZ_cols = True
#delete_arm_markers = False
delete_estimate_XYZ_cols = True

#if delete_XYZ_cols == True:
#    #DLC_labeled_csv_files_combined_copied = DLC_labeled_csv_files_combined_copied[DLC_labeled_csv_files_combined_copied.columns[:-6]]
#    DLC_labeled_csv_files_combined_copied = DLC_labeled_csv_files_combined_copied.drop(['pointX','pointY','pointZ'],level="bodyparts",axis=1)
#if delete_arm_markers == True:
#    DLC_labeled_csv_files_combined_copied = DLC_labeled_csv_files_combined_copied.drop(columns=['Qiwei.16','Qiwei.17','Qiwei.18','Qiwei.19']) #? I'm confused but
if delete_estimate_XYZ_cols == True:
    try:
        DLC_estimated_1 = DLC_estimated_1.drop(['pointX','pointY','pointZ'],level="bodyparts",axis=1)
    except:
        print("reference points not found in dataset 1")
    try:
        DLC_estimated_2 = DLC_estimated_2.drop(['pointX','pointY','pointZ'],level="bodyparts",axis=1)
    except:
        print("reference points not found in dataset 2")
    try:
        DLC_estimated_3 = DLC_estimated_3.drop(['pointX','pointY','pointZ'],level="bodyparts",axis=1)
    except:
        print("reference points not found in dataset 3")
#Fix the column names so that it's at least readable



#%% If DLC_estimated_1 isn't exactly 24 columns (8 markers * 3 columns (x,y,likelihood)),
#       then we need to delete some of the markers that are not in the list

included_bodyparts = ['shoulder1','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
scorer_1 = DLC_estimated_1.columns.levels[0].to_numpy()[0]
scorer_2 = DLC_estimated_2.columns.levels[0].to_numpy()[0]
scorer_3 = DLC_estimated_3.columns.levels[0].to_numpy()[0]

DLC_estimated_1 = DLC_estimated_1[scorer_1][included_bodyparts]
DLC_estimated_2 = DLC_estimated_2[scorer_2][included_bodyparts]
DLC_estimated_3 = DLC_estimated_3[scorer_3][included_bodyparts]

#%% Change the names of the column indexes for the estimated markers, so that
# they can be aligned with the labeled (ground truth) markers, and it would be 
# easier for us to call them later.

DLC_marker_names_estimated = ['shoulder1_x','shoulder1_y','shoulder1_likelihood',
                    'elbow1_x','elbow1_y','elbow1_likelihood',
                    'elbow2_x','elbow2_y','elbow2_likelihood',
                    'wrist1_x','wrist1_y','wrist1_likelihood',
                    'wrist2_x','wrist2_y','wrist2_likelihood',
                    'hand1_x','hand1_y','hand1_likelihood',
                    'hand2_x','hand2_y','hand2_likelihood',
                    'hand3_x','hand3_y','hand3_likelihood']


#Drop the likelihood columns for DLC_estimated datasets
DLC_estimated_1.columns = DLC_marker_names_estimated
DLC_estimated_1 = DLC_estimated_1.drop(['shoulder1_likelihood','elbow1_likelihood','elbow2_likelihood','wrist1_likelihood','wrist2_likelihood','hand1_likelihood','hand2_likelihood','hand3_likelihood'],axis=1)

DLC_estimated_2.columns = DLC_marker_names_estimated
DLC_estimated_2 = DLC_estimated_2.drop(['shoulder1_likelihood','elbow1_likelihood','elbow2_likelihood','wrist1_likelihood','wrist2_likelihood','hand1_likelihood','hand2_likelihood','hand3_likelihood'],axis=1)

DLC_estimated_3.columns = DLC_marker_names_estimated
DLC_estimated_3 = DLC_estimated_3.drop(['shoulder1_likelihood','elbow1_likelihood','elbow2_likelihood','wrist1_likelihood','wrist2_likelihood','hand1_likelihood','hand2_likelihood','hand3_likelihood'],axis=1)


DLC_estimated_train_set_1 = DLC_estimated_1.iloc[DLC_train_set_1]
DLC_estimated_test_set_1 = DLC_estimated_1.iloc[DLC_test_set_1]
DLC_labeled_train_set_1 = DLC_labeled_csv_files_combined_copied_1.loc[DLC_estimated_train_set_1.index & DLC_labeled_csv_files_combined_copied_1.index].astype(float)
DLC_labeled_test_set_1 = DLC_labeled_csv_files_combined_copied_1.loc[DLC_estimated_test_set_1.index & DLC_labeled_csv_files_combined_copied_1.index].astype(float)

DLC_estimated_train_set_2 = DLC_estimated_2.iloc[DLC_train_set_2]
DLC_estimated_test_set_2 = DLC_estimated_2.iloc[DLC_test_set_2]
DLC_labeled_train_set_2 = DLC_labeled_csv_files_combined_copied_2.loc[DLC_estimated_train_set_2.index & DLC_labeled_csv_files_combined_copied_2.index].astype(float)
DLC_labeled_test_set_2 = DLC_labeled_csv_files_combined_copied_2.loc[DLC_estimated_test_set_2.index & DLC_labeled_csv_files_combined_copied_2.index].astype(float)

DLC_estimated_train_set_3 = DLC_estimated_3.iloc[DLC_train_set_3]
DLC_estimated_test_set_3 = DLC_estimated_3.iloc[DLC_test_set_3]
DLC_labeled_train_set_3 = DLC_labeled_csv_files_combined_copied_3.loc[DLC_estimated_train_set_3.index & DLC_labeled_csv_files_combined_copied_3.index].astype(float)
DLC_labeled_test_set_3 = DLC_labeled_csv_files_combined_copied_3.loc[DLC_estimated_test_set_3.index & DLC_labeled_csv_files_combined_copied_3.index].astype(float)

#%% Calculate difference between ground truth and DLC estimated results
def calculate_2D_estimation_difference(DLC_estimated_train_set,DLC_estimated_test_set,DLC_labeled_train_set,DLC_labeled_test_set):
    #Construct a new test and train dataframe
    DLC_diff_train_set = pd.DataFrame(np.nan, index = DLC_estimated_train_set.index, columns = DLC_estimated_train_set.columns)
    DLC_diff_test_set = pd.DataFrame(np.nan, index = DLC_estimated_test_set.index, columns = DLC_estimated_test_set.columns)
    
    dist_columns = ['shouder1','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3']
    
    for i in range(len(DLC_diff_train_set.index)):
        for j in range(len(DLC_diff_train_set.columns)):
            DLC_diff_train_set.iloc[i,j] = DLC_estimated_train_set.iloc[i,j] - DLC_labeled_train_set.iloc[i,j]
    
    for i in range(len(DLC_diff_test_set.index)):
        for j in range(len(DLC_diff_test_set.columns)):
            DLC_diff_test_set.iloc[i,j] = DLC_estimated_test_set.iloc[i,j] - DLC_labeled_test_set.iloc[i,j]
    
    DLC_diff_distance_train_set = pd.DataFrame(np.nan, index = DLC_estimated_train_set.index, columns = dist_columns)
    DLC_diff_distance_test_set = pd.DataFrame(np.nan, index = DLC_estimated_test_set.index, columns = dist_columns)
    
    for i in range(len(DLC_diff_distance_train_set.index)):
        for j in range(len(DLC_diff_distance_train_set.columns)):
            #print(str(i) + " " + str(j))
            DLC_diff_distance_train_set.iloc[i,j] = np.sqrt(DLC_diff_train_set.iloc[i,j*2]**2 + DLC_diff_train_set.iloc[i,j*2+1]**2)
    
    for i in range(len(DLC_diff_distance_test_set.index)):
        for j in range(len(DLC_diff_distance_test_set.columns)):
            #print(str(i) + " " + str(j))
            DLC_diff_distance_test_set.iloc[i,j] = np.sqrt(DLC_diff_test_set.iloc[i,j*2]**2 + DLC_diff_test_set.iloc[i,j*2+1]**2)
    return DLC_diff_train_set,DLC_diff_test_set

DLC_diff_train_set_1, DLC_diff_test_set_1 = calculate_2D_estimation_difference(DLC_estimated_train_set_1,DLC_estimated_test_set_1,DLC_labeled_train_set_1,DLC_labeled_test_set_1)
DLC_diff_train_set_2, DLC_diff_test_set_2 = calculate_2D_estimation_difference(DLC_estimated_train_set_2,DLC_estimated_test_set_2,DLC_labeled_train_set_2,DLC_labeled_test_set_2)
DLC_diff_train_set_3, DLC_diff_test_set_3 = calculate_2D_estimation_difference(DLC_estimated_train_set_3,DLC_estimated_test_set_3,DLC_labeled_train_set_3,DLC_labeled_test_set_3)


#%% Combine the 3 datasets together

DLC_diff_distance_train_set = pd.concat([DLC_diff_train_set_1,DLC_diff_train_set_2,DLC_diff_train_set_3])
DLC_diff_distance_test_set = pd.concat([DLC_diff_test_set_1,DLC_diff_test_set_2,DLC_diff_test_set_3])

#%% Since there're a bunch of outliers in the test set, we need to get rid of them
DLC_diff_distance_test_set = DLC_diff_distance_test_set.apply(lambda x: [y if abs(y) <=25 else np.NaN for y in x])
DLC_diff_distance_train_set = DLC_diff_distance_train_set.apply(lambda x: [y if abs(y) <=25 else np.NaN for y in x])

#Also, change both diff datasets to absolute values, because we don't care about positive or negative
DLC_diff_distance_test_set = abs(DLC_diff_distance_test_set)
DLC_diff_distance_train_set = abs(DLC_diff_distance_train_set)
#df1['A'] = df1['A'].apply(lambda x: [y if y <= 9 else 11 for y in x]) (example)




#%% Plot the distribution of estimation error (difference) for all the markers (not practically useful) by violinplots
        
#Try: change dataframe to np
DLC_diff_distance_test_set_np = DLC_diff_distance_test_set.to_numpy()
DLC_diff_distance_train_set_np = DLC_diff_distance_train_set.to_numpy()

DLC_diff_dist_test_set_T = DLC_diff_distance_test_set.transpose()
DLC_diff_dist_train_set_T = DLC_diff_distance_train_set.transpose()

#plt.plot()
plt.figure(figsize=(14,8))
#plt.plot(DLC_diff_test_set)
#ax = sns.violinplot(DLC_diff_distance_test_set_np)
#plt.violinplot(DLC_diff_distance_test_set)

plt.violinplot([DLC_diff_distance_train_set_np[:,0][~np.isnan(DLC_diff_distance_train_set_np[:,0])],
                DLC_diff_distance_train_set_np[:,1][~np.isnan(DLC_diff_distance_train_set_np[:,1])],
                DLC_diff_distance_train_set_np[:,2][~np.isnan(DLC_diff_distance_train_set_np[:,2])],
                DLC_diff_distance_train_set_np[:,3][~np.isnan(DLC_diff_distance_train_set_np[:,3])],
                DLC_diff_distance_train_set_np[:,4][~np.isnan(DLC_diff_distance_train_set_np[:,4])],
                DLC_diff_distance_train_set_np[:,5][~np.isnan(DLC_diff_distance_train_set_np[:,5])],
                DLC_diff_distance_train_set_np[:,6][~np.isnan(DLC_diff_distance_train_set_np[:,6])],
                DLC_diff_distance_train_set_np[:,7][~np.isnan(DLC_diff_distance_train_set_np[:,7])]],
               positions = [0.8,1.8,2.8,3.8,4.8,5.8,6.8,7.8])

plt.violinplot([DLC_diff_distance_test_set_np[:,0][~np.isnan(DLC_diff_distance_test_set_np[:,0])],
                DLC_diff_distance_test_set_np[:,1][~np.isnan(DLC_diff_distance_test_set_np[:,1])],
                DLC_diff_distance_test_set_np[:,2][~np.isnan(DLC_diff_distance_test_set_np[:,2])],
                DLC_diff_distance_test_set_np[:,3][~np.isnan(DLC_diff_distance_test_set_np[:,3])],
                DLC_diff_distance_test_set_np[:,4][~np.isnan(DLC_diff_distance_test_set_np[:,4])],
                DLC_diff_distance_test_set_np[:,5][~np.isnan(DLC_diff_distance_test_set_np[:,5])],
                DLC_diff_distance_test_set_np[:,6][~np.isnan(DLC_diff_distance_test_set_np[:,6])],
                DLC_diff_distance_test_set_np[:,7][~np.isnan(DLC_diff_distance_test_set_np[:,7])]],
               positions = [1.2,2.2,3.2,4.2,5.2,6.2,7.2,8.2])

plt.xlabel("markers (train in blue, test in orange)")
plt.ylabel("RMSE error (in pixels)")
plt.title("Tracking error on a non-tattooed monkey")
plt.yticks(np.arange(0, int(np.nanmax(DLC_diff_distance_test_set_np)), 1))
#ax.set_xticks([1,2,3,4,5,6,7,8])
#ax.set_xticklabels(['shoulder1','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3'])
plt.xticks([1,2,3,4,5,6,7,8],['shoulder1','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3'])
font = {'family' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

#%% Plot the distribution of error (difference) for 4 TYPES of representative markers (not all of them)

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)
       
#plt.plot()
plt.figure(figsize=(14,8))
#plt.plot(DLC_diff_test_set)
#ax = sns.violinplot(DLC_diff_distance_test_set_np)
#plt.violinplot(DLC_diff_distance_test_set)

plt.violinplot([DLC_diff_distance_train_set_np[:,0][~np.isnan(DLC_diff_distance_train_set_np[:,0])],
                DLC_diff_distance_train_set_np[:,2][~np.isnan(DLC_diff_distance_train_set_np[:,2])],
                DLC_diff_distance_train_set_np[:,3][~np.isnan(DLC_diff_distance_train_set_np[:,3])],
                DLC_diff_distance_train_set_np[:,6][~np.isnan(DLC_diff_distance_train_set_np[:,6])]],
                positions = [0.8,1.8,2.8,3.8])

plt.violinplot([DLC_diff_distance_test_set_np[:,0][~np.isnan(DLC_diff_distance_test_set_np[:,0])],
                DLC_diff_distance_test_set_np[:,2][~np.isnan(DLC_diff_distance_test_set_np[:,2])],
                DLC_diff_distance_test_set_np[:,3][~np.isnan(DLC_diff_distance_test_set_np[:,3])],
                DLC_diff_distance_test_set_np[:,6][~np.isnan(DLC_diff_distance_test_set_np[:,6])]],
                positions = [1.2,2.2,3.2,4.2])

plt.xlabel("markers (train in blue, test in orange)")
plt.ylabel("RMSE error (in pixels)")
plt.title("2D Tracking error")
#plt.title("2D Tracking error on a tattooed monkey")
#plt.yticks(np.arange(0, int(np.nanmax(DLC_diff_distance_test_set_np)), 1))
plt.yticks(np.arange(0, 30, 2))
#ax.set_xticks([1,2,3,4,5,6,7,8])
#ax.set_xticklabels(['shoulder1','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3'])
plt.xticks([1,2,3,4],['shoulder','elbow','wrist','hand'])
font = {'family' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

#And also calculate and print the mean and standard deviation of the distribution of errors
print("mean test err shoulder: " + str(np.mean(DLC_diff_distance_test_set_np[:,0][~np.isnan(DLC_diff_distance_test_set_np[:,0])])))
print("std test err shoulder: " + str(np.std(DLC_diff_distance_test_set_np[:,0][~np.isnan(DLC_diff_distance_test_set_np[:,0])])))
print("\n")
print("mean test err elbow: " + str(np.mean(DLC_diff_distance_test_set_np[:,2][~np.isnan(DLC_diff_distance_test_set_np[:,2])])))
print("std test err elbow: " + str(np.std(DLC_diff_distance_test_set_np[:,2][~np.isnan(DLC_diff_distance_test_set_np[:,2])])))
print("\n")
#print("mean test err wrist: " + str(np.mean(DLC_diff_distance_test_set_np[:,3][~np.isnan(DLC_diff_distance_test_set_np[:,3])])))
#print("std test err wrist: " + str(np.std(DLC_diff_distance_test_set_np[:,3][~np.isnan(DLC_diff_distance_test_set_np[:,3])])))
print("mean test err wrist: " + str(np.mean(DLC_diff_distance_test_set_np[:,4][~np.isnan(DLC_diff_distance_test_set_np[:,4])])))
print("std test err wrist: " + str(np.std(DLC_diff_distance_test_set_np[:,4][~np.isnan(DLC_diff_distance_test_set_np[:,4])])))
print("\n")
print("mean test err hand: " + str(np.mean(DLC_diff_distance_test_set_np[:,6][~np.isnan(DLC_diff_distance_test_set_np[:,6])])))
print("std test err hand: " + str(np.std(DLC_diff_distance_test_set_np[:,6][~np.isnan(DLC_diff_distance_test_set_np[:,6])])))

shoulder_noNan = DLC_diff_distance_test_set_np[:,0][~np.isnan(DLC_diff_distance_test_set_np[:,0])]
elbow_noNan = DLC_diff_distance_test_set_np[:,2][~np.isnan(DLC_diff_distance_test_set_np[:,2])]
wrist_noNan = DLC_diff_distance_test_set_np[:,3][~np.isnan(DLC_diff_distance_test_set_np[:,3])]
wrist_noNan = DLC_diff_distance_test_set_np[:,6][~np.isnan(DLC_diff_distance_test_set_np[:,6])]

#stats.ttest_ind()

#%% function if a variable in the indexList contains any variable mentioned in the videoList

def containsVideos(videoList, indexList):
    contains_bool_list = np.zeros(len(indexList))
    
    for i in range(len(indexList)):
        for j in videoList:
            if j in indexList[i]:
                contains_bool_list[i] = 1
    
    return contains_bool_list

#%% Plot the test error between RT2D tasks and RT3D tasks

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 26}

plt.rc('font', **font)
       

RT2D_experimentVideos = ['Han_20210628_00001',
                         'Han_20210628_00002',
                         'Han_20210628_00003',
                         'Han_20210628_00004',
                         'Han_20210630_00006',
                         'Han_20210630_00007',
                         'Han_20210630_00008',
                         'Han_20210630_00009',
                         'Han_2021062300001',
                         'Han_2021062300002',
                         'Han_2021062300003',
                         'Han_2021062300004',
                         'Crackle_20201203_00001_trimmed',
                         'Crackle_20201203_00002_trimmed',
                         'Crackle_20201203_00003_trimmed',
                         'Crackle_20201203_00004_trimmed',
                         'Crackle_20201215_00005',
                         'Crackle_20201215_00006',
                         'Crackle_20201215_00007',
                         'Crackle_20201215_00008',
                         'Crackle_2020121600005',
                         'Crackle_2020121600006',
                         'Crackle_2020121600007',
                         'Crackle_2020121600008',
                         'Rocket_20210707_00001',
                         'Rocket_20210707_00002',
                         'Rocket_20210707_00003',
                         'Rocket_20210707_00004',
                         'Rocket_20210723_00009',
                         'Rocket_20210723_00010',
                         'Rocket_20210723_00011',
                         'Rocket_20210723_00012']

RT3D_experimentVideos = ['Han_20210628_00005',
                         'Han_20210628_00006',
                         'Han_20210628_00007',
                         'Han_20210628_00008',
                         'Han_20210630_00002',
                         'Han_20210630_00003',
                         'Han_20210630_00004',
                         'Han_20210630_00005',
                         'Han_2021062300005',
                         'Han_2021062300006',
                         'Han_2021062300007',
                         'Han_2021062300008',
                         'Crackle_20201203_00007_trimmed',
                         'Crackle_20201203_00008_trimmed',
                         'Crackle_20201203_00009_trimmed',
                         'Crackle_20201203_00010_trimmed',
                         'Crackle_20201215_00001',
                         'Crackle_20201215_00002',
                         'Crackle_20201215_00003',
                         'Crackle_20201215_00004',
                         'Crackle_2020121600001',
                         'Crackle_2020121600002',
                         'Crackle_2020121600003',
                         'Crackle_2020121600004',
                         'Rocket_20210707_00005',
                         'Rocket_20210707_00006',
                         'Rocket_20210707_00007',
                         'Rocket_20210707_00008',
                         'Rocket_20210723_00001',
                         'Rocket_20210723_00002',
                         'Rocket_20210723_00003',
                         'Rocket_20210723_00004']

RT2D_experimentFrameNums = containsVideos(RT2D_experimentVideos, DLC_diff_distance_test_set.index)
RT3D_experimentFrameNums = containsVideos(RT3D_experimentVideos, DLC_diff_distance_test_set.index)

RT2D_experimentFrames = DLC_diff_distance_test_set_np[RT2D_experimentFrameNums == 1]
RT3D_experimentFrames = DLC_diff_distance_test_set_np[RT3D_experimentFrameNums == 1]





#plt.plot()
plt.figure(figsize=(14,8))
#plt.plot(DLC_diff_test_set)
#ax = sns.violinplot(DLC_diff_distance_test_set_np)
#plt.violinplot(DLC_diff_distance_test_set)


blue_patch = mpatches.Patch(color='blue')
orange_patch = mpatches.Patch(color = 'orange')
# 'fake' invisible object

labels = ['RT2D','RT3D']
pos_RT2D = [0.8,1.8,2.8,3.8]
pos_RT3D = [1.2,2.2,3.2,4.2]

plt.violinplot([RT2D_experimentFrames[:,0][~np.isnan(RT2D_experimentFrames[:,0])],
                RT2D_experimentFrames[:,2][~np.isnan(RT2D_experimentFrames[:,2])],
                RT2D_experimentFrames[:,3][~np.isnan(RT2D_experimentFrames[:,3])],
                RT2D_experimentFrames[:,6][~np.isnan(RT2D_experimentFrames[:,6])]],
                positions = pos_RT2D)

plt.violinplot([RT3D_experimentFrames[:,0][~np.isnan(RT3D_experimentFrames[:,0])],
                RT3D_experimentFrames[:,2][~np.isnan(RT3D_experimentFrames[:,2])],
                RT3D_experimentFrames[:,3][~np.isnan(RT3D_experimentFrames[:,3])],
                RT3D_experimentFrames[:,6][~np.isnan(RT3D_experimentFrames[:,6])]],
                positions = pos_RT3D)



plt.xlabel("landmarks (RT2D in blue, RT3D in orange)")
plt.ylabel("RMSE (in pixels)")
plt.title("2D (DLC) Tracking Error for RT2D and RT3D tasks")

fake_handles = repeat([blue_patch,orange_patch])
plt.legend(fake_handles, labels)
#plt.title("2D Tracking error on a tattooed monkey")
#plt.yticks(np.arange(0, int(np.nanmax(DLC_diff_distance_test_set_np)), 1))
plt.yticks(np.arange(0, 30, 2))
#ax.set_xticks([1,2,3,4,5,6,7,8])
#ax.set_xticklabels(['shoulder1','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3'])
plt.xticks([1,2,3,4],['shoulder','elbow','wrist','hand'])
font = {'family' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

