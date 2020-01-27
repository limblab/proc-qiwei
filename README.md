# proc-qiwei
Qiwei's analysis code

The thing is still... a mess. The actual thing to run is the dropout_analysis_cleaned.py file. It's in /Dropout_Interpolation_Analysis/xds.

Everything before line 195 are functions. 
Line 195-211 is an example of interpolation. 
Line 212-364 is the main loop. dropout() function call is on line 254, and the function itself is on line 87. 

dropout_timepoints: timepoints for simulated dropouts.
dropout_timepoints_length: how long is it for each dropout_timepoints. These two arrays are 1 to 1 correlated.
spike_counts_dropout: the spike_counts array that has the points on dropout_timepoints cleared to 0.
