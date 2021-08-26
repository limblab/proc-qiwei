# -*- coding: utf-8 -*-
"""
Created on Fri May 28 17:39:13 2021

@author: dongq
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
file1_RT2D_name = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\videos\2D_tracking_likelihood_RT2D.csv'
file1_RT3D_name = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\videos\2D_tracking_likelihood_RT3D.csv'
file2_RT2D_name = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-15\videos\2D_tracking_likelihood_RT2D.csv'
file2_RT3D_name = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-15\videos\2D_tracking_likelihood_RT3D.csv'
file3_RT2D_name = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-16\videos\2D_tracking_likelihood_RT2D.csv'
file3_RT3D_name = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-16\videos\2D_tracking_likelihood_RT3D.csv'
file4_RT2D_name = r'C:\Users\dongq\DeepLabCut\Han_20201203_rwFreeReach\videos\2D_tracking_likelihood_RT2D.csv'
file4_RT3D_name = r'C:\Users\dongq\DeepLabCut\Han_20201203_rwFreeReach\videos\2D_tracking_likelihood_RT3D.csv'
file5_RT2D_name = r'C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\videos\2D_tracking_likelihood_RT2D.csv'
file5_RT3D_name = r'C:\Users\dongq\DeepLabCut\Han_20201204_rwFreeReach\videos\2D_tracking_likelihood_RT3D.csv'
file6_RT2D_name = r'C:\Users\dongq\DeepLabCut\Han_20201217_rwFreeReach\videos\2D_tracking_likelihood_RT2D.csv'
file6_RT3D_name = r'C:\Users\dongq\DeepLabCut\Han_20201217_rwFreeReach\videos\2D_tracking_likelihood_RT3D.csv'

file1_RT2D = pd.read_csv(file1_RT2D_name).to_numpy()[:,1:9].astype(np.float)
file1_RT3D = pd.read_csv(file1_RT3D_name).to_numpy()[:,1:9].astype(np.float)
file2_RT2D = pd.read_csv(file2_RT2D_name).to_numpy()[:,1:9]
file2_RT3D = pd.read_csv(file2_RT3D_name).to_numpy()[:,1:9]
file3_RT2D = pd.read_csv(file3_RT2D_name).to_numpy()[:,1:9]
file3_RT3D = pd.read_csv(file3_RT3D_name).to_numpy()[:,1:9]
file4_RT2D = pd.read_csv(file4_RT2D_name).to_numpy()[:,1:9]
file4_RT3D = pd.read_csv(file4_RT3D_name).to_numpy()[:,1:9]
file5_RT2D = pd.read_csv(file5_RT2D_name).to_numpy()[:,1:9]
file5_RT3D = pd.read_csv(file5_RT3D_name).to_numpy()[:,1:9]
file6_RT2D = pd.read_csv(file6_RT2D_name).to_numpy()[:,1:9]
file6_RT3D = pd.read_csv(file6_RT3D_name).to_numpy()[:,1:9]

origin_X = list(range(0,8))
min3_X = [x-0.3 for x in origin_X]
min2_X = [x-0.2 for x in origin_X]
min1_X = [x-0.1 for x in origin_X]
centr_X = origin_X
pls1_X = [x+0.1 for x in origin_X]
pls2_X = [x+0.2 for x in origin_X]
#plx3_X = [x+0.3 for x in origin_X]

fig = plt.figure()
plt.ylim([0.7,1.001])
plt.errorbar(min3_X, file1_RT2D[0,:],yerr = file1_RT2D[1,:],label='C-1203',fmt='o',elinewidth=2,capsize=4)
plt.errorbar(min2_X, file2_RT2D[0,:],yerr = file2_RT2D[1,:],label='C-1215',fmt='o',elinewidth=2,capsize=4)
plt.errorbar(min1_X, file3_RT2D[0,:],yerr = file3_RT2D[1,:],label='C-1216',fmt='o',elinewidth=2,capsize=4)
plt.errorbar(centr_X, file4_RT2D[0,:],yerr = file4_RT2D[1,:],label='H-1203',fmt='o',elinewidth=2,capsize=4)
plt.errorbar(pls1_X, file5_RT2D[0,:],yerr = file5_RT2D[1,:],label='H-1204',fmt='o',elinewidth=2,capsize=4)
plt.errorbar(pls2_X, file6_RT2D[0,:],yerr = file6_RT2D[1,:],label='H-1217',fmt='o',elinewidth=2,capsize=4)
#plt.errorbar(X_2, file1_RT3D[0,:],yerr = file1_RT3D[1,:],fmt='o',elinewidth=2,capsize=4)

plt.xticks(np.arange(8),['shoulder','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3'])
plt.legend()

fig = plt.figure()
plt.ylim([0.5,1.001])
plt.errorbar(min3_X, file1_RT3D[0,:],yerr = file1_RT3D[1,:],label='C-1203',fmt='o',elinewidth=2,capsize=4)
plt.errorbar(min2_X, file2_RT3D[0,:],yerr = file2_RT3D[1,:],label='C-1215',fmt='o',elinewidth=2,capsize=4)
plt.errorbar(min1_X, file3_RT3D[0,:],yerr = file3_RT3D[1,:],label='C-1216',fmt='o',elinewidth=2,capsize=4)
plt.errorbar(centr_X, file4_RT3D[0,:],yerr = file4_RT3D[1,:],label='H-1203',fmt='o',elinewidth=2,capsize=4)
plt.errorbar(pls1_X, file5_RT3D[0,:],yerr = file5_RT3D[1,:],label='H-1204',fmt='o',elinewidth=2,capsize=4)
plt.errorbar(pls2_X, file6_RT3D[0,:],yerr = file6_RT3D[1,:],label='H-1217',fmt='o',elinewidth=2,capsize=4)
#plt.errorbar(X_2, file1_RT3D[0,:],yerr = file1_RT3D[1,:],fmt='o',elinewidth=2,capsize=4)

plt.xticks(np.arange(8),['shoulder','elbow1','elbow2','wrist1','wrist2','hand1','hand2','hand3'])
plt.legend()
