# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:24:12 2021

@author: dongq
"""

import xlrd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import signal
#%%
rcParams['font.family'] = 'Arial'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
#%%
def plot_spectral(data, names, fs, nFFT):
    """
    This function is used to calculate and plot the power spectral of multi-ch EMG signals.
    It calls welch in scipy.signal to do the computation
    
    data: the EMG signals you want to analyze, a T*n numpy array, T is the number of
          samples, n is the number of channels
    EMG_names: a list for the names of EMG channels or labels for forces
    fs: the sampling frequency
    plt_start_time: the start time for plotting, a float number
    plt_end_time: the end time for plotting, a float number
    """
    N = data.shape[1]
    grid = plt.GridSpec(N, 1, wspace=0.5,hspace=0.4)
    for i in range(N):
        ax = plt.subplot(grid[i,0])
        plt.tick_params(labelsize = 14)
        plt.ylabel('PSD', fontsize = 14)
        #plt.ylim([0.5e-3, 100])
        sns.despine()
        f, Pxx_den = signal.welch(data[:, i], fs, nperseg=nFFT)
        ax.text(8, ax.get_ylim()[1], names[i], fontsize=14)
        #plt.grid(which='both', axis='both')
        if i<N-1:
            plt.semilogy(f, Pxx_den)
            plt.setp(ax.get_xticklabels(),visible=False)
        elif i == N-1:
            plt.semilogy(f, Pxx_den)
            plt.setp(ax.get_xticklabels(),visible=True)
            plt.xlabel('Frequency (Hz)', fontsize = 18)
#%%
path = 'C:\\Users\\dongq\\DeepLabCut\\Crackle-Qiwei-2020-12-03\\neural-data\\'
xls_name = 'Crackle_20201203_RT2D_RobotData.xlsx'
#%%
data=xlrd.open_workbook(path+xls_name)
table=data.sheets()[0]
#%%
markers_frame_no = np.array(table.col_values(0)[1:])
#%%
N = 20000

handle = [np.array(table.col_values(i)[1:N]) for i in (4, 5)]

#shoulder = [np.array(table.col_values(i)[1:N]) for i in (1, 2, 3)]
#elbow1 = [np.array(table.col_values(i)[1:N]) for i in (7, 8, 9)]
#wrist1 = [np.array(table.col_values(i)[1:N]) for i in (19, 20, 21)]
#hand1 = [np.array(table.col_values(i)[1:N]) for i in (31, 32, 33)]
#%%
# This line asembles an array for the 'plot_spectral' function
# If wanting to use speed rather than the velocities, calculate the speeds first accordingly, and then put them in this array
data = np.asarray([np.diff(handle[0])]).T
#%%
#plt.figure('Velosity power spectral density', figsize = (6, 8))
plt.figure('Velocity power spectral density')
plt.subplots_adjust(left = 0.2)
plot_spectral(data, ['Handle'], 25, 256)