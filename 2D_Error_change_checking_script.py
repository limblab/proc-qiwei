# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 10:48:06 2020

@author: dongq
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%

data = pd.read_csv(r'C:\Users\dongq\DeepLabCut\Rocket-Chris-2020-07-29\dlc-models\iteration-0\RocketJul29-trainset95shuffle1\train\learning_stats.csv')

#%%

data_arr = data.to_numpy()

#%%


font_small = {'family' : 'normal',
 #       'weight' : 'bold',
        'size'   : 18}

plt.figure(figsize=(8,6))

#plt.plot(data_arr[:,0],data_arr[:,1],linewidth=3)

#step_array = [0.005, 0.02, 0.002, 0.001]

#for i in range(4): #This is hardcording, there are 4 sizes of iteration steps, 0.005, 0.02, 0.002, 0.001
#    temp_array = np.where(data_arr[:,2]==step_array[i])
#    tempX = data_arr[temp_array,0]
#    tempY = data_arr[temp_array,1]
#    plt.plot(tempX[0,:], tempY[0,:], linewidth = 3,label=str(step_array[i]))
plt.plot(data_arr[:,0], data_arr[:,1], linewidth = 3)

plt.legend()
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

plt.ylabel('loss',**font_small)
plt.xlabel('iterations',**font_small)

plt.show()    