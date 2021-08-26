# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 09:55:12 2021

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
import glob

#%%

#path = r'C:\Users\dongq\Desktop\proc-qiwei\20210117-Interpolation-errors'   
path = r'C:\Users\dongq\Desktop\proc-qiwei\20210117-Interpolation_errors-filt'                    # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

df_from_each_file = (pd.read_csv(f) for f in all_files)
concatenated_df   = pd.concat(df_from_each_file,axis=1)



poly = 'poly3'
linear = 'linear'

handlehandle = 'handle'
DLChandle = 'DLC'

concatenated_df.columns

#%%

"""
Hardcode warning
"""

polynomial_handlehandle = []
polynomial_DLChandle = []

linear_handlehandle = []
linear_DLChandle = []


polynomial_df = []
linear_df = []

handlehandle_df = []
DLChandle_df = []

for column in concatenated_df.columns:
    if poly in column and handlehandle in column:
        polynomial_handlehandle.append(concatenated_df[column])
    if poly in column and DLChandle in column:
        polynomial_DLChandle.append(concatenated_df[column])
    if linear in column and handlehandle in column:
        linear_handlehandle.append(concatenated_df[column])
    if linear in column and DLChandle in column:
        linear_DLChandle.append(concatenated_df[column])
        
    if handlehandle in column:
        handlehandle_df.append(concatenated_df[column])
    if DLChandle in column:
        DLChandle_df.append(concatenated_df[column])
    #print(column)


#Change the list of series to dataframes
polynomial_handlehandle = pd.DataFrame(polynomial_handlehandle).transpose()
polynomial_DLChandle = pd.DataFrame(polynomial_DLChandle).transpose()

linear_handlehandle = pd.DataFrame(linear_handlehandle).transpose()
linear_DLChandle = pd.DataFrame(linear_DLChandle).transpose()

handlehandle_df = pd.DataFrame(handlehandle_df).transpose()
DLChandle_df= pd.DataFrame(DLChandle_df).transpose()

#%% Plot using box plots? (all 4 types together)


#linear_DLChandle[linear_DLChandle < 5].boxplot()
plt.figure()
concatenated_df[concatenated_df < 1.25].boxplot(figsize=(30,5))
plt.xlabel("dataset types")
plt.ylabel("error (in cm)")
plt.title("Error differences between datasets and interpolation types")

#%% Plot using box plots (only handle)

handlehandle_df[handlehandle_df < 1.25].boxplot(fontsize=17)
plt.xlabel("dataset types",fontsize=20)
plt.ylabel("error (in cm)",fontsize=20)
plt.title("Error differences between datasets and interpolation types when using ground truth data only",fontsize=20)
#%% Plot using box plots (only DLC)
plt.figure()
DLChandle_df[DLChandle_df < 1.25].boxplot(fontsize=20)
plt.xlabel("dataset types",fontsize=20)
plt.ylabel("error (in cm)",fontsize=20)
plt.title("Error differences between datasets and interpolation types when using DLC tracked data",fontsize=20)

#%%Plot using box plots (only filtered DLC)
plt.figure()
DLChandle_df[DLChandle_df < 1.25].boxplot(fontsize=20)
plt.xlabel("dataset types",fontsize=20)
plt.ylabel("error (in cm)",fontsize=20)
plt.title("Error differences between datasets and interpolation types when using DLC data to interpolate after filtering",fontsize=20)



#%% do a t test









#%%

np.random.seed(1234)
df = pd.DataFrame(np.random.randn(10, 4),
                  columns=['Col1', 'Col2', 'Col3', 'Col4'])
boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])








