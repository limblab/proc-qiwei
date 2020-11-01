"""
Basic settings
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams['font.family'] = 'Arial'
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
def plot_behave_dict(behave_dict, plot_len, bin_size, video_bin_size = 0.0333):
    #fig = plt.figure(behave_dict['label']+str(np.random.randint(0,1000)), figsize=(8, 10))
    if plot_len == 0:
        plot_len = behave_dict['spike'].shape[0]
    else:
        plot_len = int(plot_len/bin_size)
    N = 1
    spike_grid = 2
    p_spike, p_cont = behave_dict['spike'], behave_dict['cont']
    plot_start = 0
    grid = plt.GridSpec(N+spike_grid,1,wspace=0.5,hspace=0.2)
    main_ax = plt.subplot(grid[0:spike_grid,0])
    p=p_spike[plot_start:plot_start+plot_len,:]
    x=np.arange(plot_len)*bin_size
    x_cont = np.arange(len(p_cont))*video_bin_size
    y=np.arange(np.size(p,1))
    cmap=plt.cm.binary
    im = main_ax.pcolormesh(x, y, p.T/bin_size, cmap=cmap)
    main_ax.axis('off')
    plt.xticks(color = 'w')
    plt.yticks([])
    #fig.colorbar(im, ax = main_ax)
    #plot through each row
    for i in range(N):
        ax0 = plt.subplot(grid[i+spike_grid,0], sharex = main_ax)
        #plt.yticks([])
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(False)
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['left'].set_visible(False)
        if i<N-1:
            plt.plot(x_cont, p_cont, 'k')
            ax0.spines['bottom'].set_visible(False)
            plt.setp(ax0.get_xticklabels(),visible=False)
            ax0.tick_params(axis=u'both', which=u'both',length=0)
        if i == N-1:
            ax0.tick_params(axis=u'both', which=u'both',length=4)
            plt.setp(ax0.get_xticklabels(),visible=True)
            plt.plot(x_cont, p_cont, 'k')
            plt.xticks(color='k')
            #plt.yticks(color='k')
            ax0.set_xlabel('t (s)', fontsize = 22)
            plt.tick_params(labelsize=22)
            # labels = ax0.get_xticklabels() + ax0.get_yticklabels()
            # [label.set_fontname('Arial') for label in labels]
            #ax0.set_xticks(np.arange(0, len(p1), 500))           
        #plt.ylim(0, 200)
#%%

"""
-------- Specify path and file name --------
"""
path = 'C:/Users/dongq/DeepLabCut/Neural_Data/'
pickle_filename = 'Han_20200804_freeReach_leftS1_4cameras_joe_002.pkl'
with open ( ''.join((path, pickle_filename)), 'rb' ) as fp:
    my_cage_data = pickle.load(fp)

"""
-------- Bin data with this function: --------
"""
bin_size = 0.001
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
video_timeframe = my_cage_data.ximea_video_sync()
print('The time when the video recording was started is at the %.3f th second' % (video_timeframe[1]))
#%%
behave1 = [1500,	2000] # Provide the number of frames here
time_behave1 = video_timeframe[behave1] # The exact time for those frames will be returned

# figure out the time interval you want to plot and put them here, in seconds
t_start, t_end = 60+video_timeframe[1], 60+20+video_timeframe[1]
#t_start, t_end = 60,80

idx = np.where( (my_cage_data.binned['timeframe']>t_start) & (my_cage_data.binned['timeframe']<t_end) )[0]

behave_dict = dict()
# Here is the neural data (unsorted)
behave_dict['spike'] = np.asarray(my_cage_data.binned['spikes']).T[idx, :]

# Here is your hand speed data. In this example I just put a series of 1
behave_dict['cont'] = sec_speed #sec_speed is from 3D_hand_position_scatterplot_script.py's last section/block of code

fig = plt.figure('rasters', figsize=(11, 10))
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plot_behave_dict(behave_dict, 0, 0.001,1/25)
# If you need to save the figure as pdf, uncomment the next line
#plt.savefig('rasters.pdf')