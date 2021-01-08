# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 18:06:40 2021

@author: dongq
"""

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
import itertools


"""
This is a one-time script (highly likely.)
"""


#%% read in whole array (without separating per cam and pre marker)

folder_name = r'C:\Users\dongq\Desktop\proc-qiwei'

#file_name_1 = 'Crackle-Qiwei-2020-12-03_dropout_size_1_wholeArr.csv'

file_name = '\Crackle-Qiwei-2020-12-03_dropout_size_'
file_suffix = '_wholeArr.csv'

file_num = [1,2,3,4,5,6,7,8,9,10]

file_name_list = []
file_dir_list = []

for i in range(len(file_num)):
    file_name_list.append(file_name + str(file_num[i]) + file_suffix)
    file_dir_list.append(folder_name + file_name + str(file_num[i]) + file_suffix)
    

file_list = []

for i in range(len(file_dir_list)):
    file_list.append(pd.read_csv(file_dir_list[i]))
    
file_list_np = []
#Convert DataFrame from 2 dimentional to 1 dimentional
for i in range(len(file_list)):
    #file_list_1dim.append(pd.DataFrame(list(itertools.product(file_list[i].index, file_list[i].columns.values))).set_index([0]))
    file_list_np.append(file_list[i].to_numpy())
#%% 
    
plt.boxplot(file_list_np)
plt.xlabel("consecutive dropout frames in each section",fontsize = 16)
plt.ylabel("difference",fontsize = 16)
plt.title("difference between interpolated results and DLC tracked data",fontsize = 16)















