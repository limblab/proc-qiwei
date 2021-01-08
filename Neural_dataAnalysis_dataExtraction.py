# -*- coding: utf-8 -*-

"""
Change MATLAB data to PICKLE data, only need to run ONCE for ONE MATLAB file
"""
from cage_data import cage_data
# ---------- Specify the path and files ---------- #
data_path = 'C:/Users/dongq/Desktop/'
# -------- The first file is a MATLAB version of .nev file ---------- #
nev_file = 'Han_20200115_freeReach_leftS1_joe_qiwei002.mat'

# --------- Create a cage_data instance ---------- #
my_cage_data = cage_data()
my_cage_data.create(data_path, nev_file, '', has_analog = 1)
# ---------- Simply removed large amplitude artefacts. ---------- #
# ---------- Waveforms with a maximum larger than K1 times of the threshold, or the first sample larger than K2 times of
# ---------- the threshold will be removed. ---------- #
my_cage_data.clean_cortical_data()
# ---------- Filter EMG with LPF 10 Hz ---------- #
my_cage_data.pre_processing_summary()
# ---------- Save ---------- #
save_path = data_path
my_cage_data.save_to_pickle(data_path, nev_file[:-4])
#%%
import pickle
"""
-------- Specify path and file name --------
"""
path = 'C:/Users/dongq/Desktop/'
pickle_filename = 'Han_20200115_freeReach_leftS1_joe_qiwei002.pkl'
with open ( ''.join((path, pickle_filename)), 'rb' ) as fp:
    my_cage_data = pickle.load(fp)

"""
-------- Bin data with this function: --------
"""
bin_size = 0.02
my_cage_data.bin_data(bin_size, mode = 'center')
# Here the 'mode' parameter spcifies the way of binning.
# The default setting is 'center', meaning each bin is centered on the sampling time point
# If the mode is set to 'left', then the spike counts are calculated on the left side of the sampling time point
"""
-------- Binned data can be visited like this --------
"""

#Time for each BINNED_SPIKES
timeframe = my_cage_data.binned['timeframe']
#Spikes AFTER BINNING (not the original dataset)
binned_spikes = my_cage_data.binned['spikes']

my_cage_data.pre_processing_summary()
#Pulse amplitude for each data point, sampled at 30k.
sync_pulse = my_cage_data.analog['video_sync']

#Get a timeframe that has a time unit in seconds(s) for sync_pulse, sampled at 30k
import numpy as np
t=np.arange(len(sync_pulse))
analog_time_frame = t/my_cage_data.analog['analog_fs']

#%% temp 
#NEXT STEP TODO: find the uprising edge for each synchronization pulse, put that into an array
temp_sync_pulse_time_frame = np.where(sync_pulse > 32741/2)
#if left is smaller and right is larger


#%% temp
#All the SPIKES detected in all the 96 channels, each spike has 48 data points, sampled at 30k
waveform = my_cage_data.waveforms
#Plot the transpose of any row of the matrix (any channel) to see how the SPIKES look like
import matplotlib.pyplot as plt
plt.plot(waveform[11].T)

#Appearing time for each SPIKE, correlating to the waveform variable, sampled at 30k
spike_times = my_cage_data.spikes