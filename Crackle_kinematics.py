
import xlrd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import signal

rcParams['font.family'] = 'Arial'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

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

#path = 'C:\\Users\\dongq\\DeepLabCut\\Crackle-Qiwei-2020-12-03\\'
#xls_name = 'Crackle_20201203_RT3D_3D_expOnly.xlsx'

path = 'C:\\Users\\dongq\\DeepLabCut\\Crackle-Qiwei-2020-12-03\\Iteration_3_results\\reconstructed-3d-data-RT2D\\'
#path = 'C:\\Users\\dongq\\DeepLabCut\\Crackle-Qiwei-2020-12-03\\Iteration_3_results\\reconstructed-3d-data-RT3D\\'
xls_name = 'output_3d_data.xlsx'

data=xlrd.open_workbook(path+xls_name)
table=data.sheets()[0]

markers_frame_no = np.array(table.col_values(0)[1:])


#def deal_with_NaN(arr):
N_start = 17000
N_end = 18000
shoulder = [np.array(table.col_values(i)[N_start:N_end]) for i in (1, 2, 3)]
elbow1 = [np.array(table.col_values(i)[N_start:N_end]) for i in (7, 8, 9)]
wrist1 = [np.array(table.col_values(i)[N_start:N_end]) for i in (19, 20, 21)]
hand1 = [np.array(table.col_values(i)[N_start:N_end]) for i in (31, 32, 33)]


#where_are_NaNs = np.isnan(DLC_speed)
#DLC_speed[where_are_NaNs] = 0


# This line asembles an array for the 'plot_spectral' function
# If wanting to use speed rather than the velocities, calculate the speeds first accordingly, and then put them in this array
data = np.asarray([np.diff(shoulder[0]), np.diff(elbow1[0]), np.diff(wrist1[0]), np.diff(hand1[0])]).T

#x = data[0:len(data)-1:]
#y = data[1::]
#data_diff = y-x #diff per frame
#data_diff = data_diff * 25 #diff per second

#for x,y in zip(, data[1::]):
#    data_diff = y-x

print(sum(sum(data == 0)))

plt.figure('Velosity power spectral density', figsize = (6, 8))
plt.subplots_adjust(left = 0.2)
#plot_spectral(data, ['Shoulder', 'Elbow 1', 'Wrist 1', 'Hand 1'], 25, 256)
plot_spectral(data, ['Shoulder', 'Elbow 1', 'Wrist 1', 'Hand 1'], 25, 256)