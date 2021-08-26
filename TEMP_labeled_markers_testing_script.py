# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:13:55 2021

@author: dongq
"""

import pandas as pd
import numpy as np
labeled_data_folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-16\labeled-data'
camera_list = ['\Crackle_2020121600001',
               '\Crackle_2020121600002',
               '\Crackle_2020121600003',
               '\Crackle_2020121600004',
               '\Crackle_2020121600005',
               '\Crackle_2020121600006',
               '\Crackle_2020121600007',
               '\Crackle_2020121600008',
               '\Crackle_2020121600009',
               '\Crackle_2020121600010',
               '\Crackle_2020121600011',
               '\Crackle_2020121600012',
               ]
file_name = r'\CollectedData_Qiwei.csv'

cam_data_list = []
cam_data_list_np = []

for i in range(len(camera_list)):
    cam_data_list.append(pd.read_csv(labeled_data_folder + camera_list[i] + file_name))
    cam_data_list_np.append(pd.read_csv(labeled_data_folder + camera_list[i] + file_name).to_numpy())
    print(camera_list[i] +  " " + str(cam_data_list[i].shape))
    
#%% Try concatenate these pd arrarys
    
concatenated_array = pd.DataFrame() 

for i in cam_data_list:
    #print(i)
    concatenated_array = pd.concat([concatenated_array,i])

















