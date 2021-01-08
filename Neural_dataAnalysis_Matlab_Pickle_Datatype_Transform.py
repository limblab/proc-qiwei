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


timeframe = my_cage_data.binned['timeframe']
binned_spikes = my_cage_data.binned['spikes']
my_cage_data.pre_processing_summary()
sync_pulse = my_cage_data.analog['video_sync']
