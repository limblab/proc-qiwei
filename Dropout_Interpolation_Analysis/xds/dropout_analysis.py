# -*- coding: utf-8 -*-
"""
Spyder Editor

Using code from basic_use_of_sds_Python.py to READIN datasets
"""

import numpy as np
from xds import lab_data, list_to_nparray, smooth_binned_spikes
import matplotlib.pyplot as plt
import random
import math
import pandas
from wiener_filter import form_decoder_dataset, wiener_only_train, \
w_filter_test, vaf
import sys
import os
import scipy.stats as stats
import scipy as sp
from scipy.interpolate import griddata, splrep, splev, interp1d

print(sys.path)

"""
Dropout file read in, shows the probability of dropout lengths in the file
from 0.01s to 10s, each block is 0.01s
"""
dropout_probability_list = pandas.read_csv('dropout_probability.csv')
print(dropout_probability_list)


#base_path = '../dataset/'
#file_name = 'Jango2016/' + 'Jango_20160630_001.mat' 
#There are 9 datasets in the /dataset folder, use other datasets later
#dataset = lab_data(base_path, file_name)

#%%
"""
Then, we can grab data from the 'lab_data' object. If you are using Spyder, you 
may directly see the variables in 'Variable explorer' on the right side.
"""

"""
We can do without trial information, and everything is NumPy array.
"""
def dataset_breakdown(dataset):
    time_frame = dataset.time_frame
    bin_width = dataset.bin_width
    
    # spike counts, using the bin width specific by bin_width
    # each row is a sample, each colum is an electrode
    spike_counts = dataset.spike_counts
    
    # kinematics : position
    kin_p = dataset.kin_p
    
    # kinematics : velocity
    kin_v = dataset.kin_v
    """
    If there are EMG signals, they can be got using:
    EMG = dataset.EMG
    EMG_names = dataset.EMG_names
    """
    if dataset.has_EMG == 1:
        EMG = dataset.EMG
        EMG_names = dataset.EMG_names
    else:
        print('This file does not contrain EMG')  
        
    """
    In Spyder, figures can be shown direcly inside the console window
    
    1. Plot spikes
    2. Plot kinematics position and velocity
    3. Plot EMG data
    """
    plt.figure()
    plt.plot(time_frame[:1000], spike_counts[:1000, 23])
    plt.figure()
    plt.plot(time_frame[:1000], kin_p[:1000, 0])
    plt.plot(time_frame[:1000], kin_v[:1000, 0])
    if dataset.has_EMG == 1:
        plt.figure()
        plt.plot(time_frame[:1000], EMG[:1000, 4])
    return time_frame, bin_width, spike_counts, kin_p, kin_v, EMG, EMG_names

#%%
"""
Make the dropout section a function
file: which .mat file you want to dropout some data out of
percentage: write in 0.01 for 1%
dropout_list: put "dropout_probability_list" without double quotes, it's a
              magic parameter and I put the parameter out just because I don't
              know what to do and I want to keep it safe.

returns: float64 np array?
"""
def dropout(file,percentage,dropout_list):
    
    dropout_timepoints_temp = np.array([])
    #print(dropout_timepoints_temp)
    dropout_timepoints_length_temp = np.array([])
    
    if percentage == 0:
        return dropout_timepoints_temp,dropout_timepoints_length_temp,file
    #0.0334
    dropout_timepoints_temp = np.array([])
    #print(dropout_timepoints_temp)
    dropout_timepoints_length_temp = np.array([])
    
    total_dropout_percentage = percentage
    #dropout_probability_list
    total_dropout_time = file.shape[0] * total_dropout_percentage#in 0.001s
    dropout_time_list = dropout_list * total_dropout_time #in 0.001s
    
    #Magic numbers, works in this case because the percentage of each dropout
    #time length is statistically stable in a kind of way
    temp_list = np.arange(10,10001,10)
    temp_list = temp_list.reshape((1000,1))
    
    #calculate how many times of dropouts there might occur according to the
    #previously calculated total_dropout_percentage, per each element. The first
    #element means the estimated number of dropouts in the 0.00s to 0.01s range, 
    #and the 1000th element means the number of dropouts in the 9.99s to 10.00s 
    #range. 
    dropout_numbers_list = np.divide(dropout_time_list,temp_list)
    dropout_numbers_np = dropout_numbers_list.to_numpy().copy()
    

    #round up all the numbers in dropout_numbers_np
    for j in range(dropout_numbers_np.shape[0]):
        if dropout_numbers_np[j] != 0:
            dropout_numbers_np[j] = int(dropout_numbers_np[j]) + 1


            
    
    #copy these two elements for safety and further calculation
    dropout_numbers_check = dropout_numbers_np.copy()
    spike_counts_dropout = file.copy()
    
    #put all the dropouts randomly into the dataset
    for k in range(dropout_numbers_check.shape[0]):
        if dropout_numbers_check[k] != 0:
            while dropout_numbers_check[k] > 0:
                rand_position = random.randint(1,file.shape[0])
                #Magic solution, to prevent any random numbers go less than 10000, so 
                #that... the further calculation (use the 1000 points before the 
                #dropout timepoint as interpolation reference) will not mess up
                if(rand_position) < 20000:
                    rand_position = rand_position + 20000
                dropout_timepoints_temp = np.append(dropout_timepoints_temp,rand_position)
                dropout_timepoints_length_temp= np.append(dropout_timepoints_length_temp,k)
                
                #print(rand_position)
                for l in range((k+1)*10):
                    if (rand_position+l) < spike_counts_dropout.shape[0]:
                        spike_counts_dropout[rand_position + l,:] = 0
                dropout_numbers_check[k] = dropout_numbers_check[k] - 1
                
    #print(dropout_timepoints_temp)
    #print(dropout_timepoints_length_temp)
    #print("")
    print("Dropout Finished")
    # dropout_timepoints_temp, dropout_timepoints_length_temp,
    return dropout_timepoints_temp, dropout_timepoints_length_temp,spike_counts_dropout
    #return 1,2,3

#%%
"""
Rebin the spike_counts and EMG data to 50ms bins, assuming that they're both
1000Hz sampled

sampleData: which file you want to rebin
binSize: 10,20,30,40,50, in int, milliseconds

returns: float64 np array?
""" 

def rebin(sampleData, binSize):
    
    binned_data = np.zeros([int(sampleData.shape[0]/binSize)+1,sampleData.shape[1]])
    
    for i in range(sampleData.shape[0]):
        curr_bin = int(i/binSize)
        binned_data[curr_bin] = binned_data[curr_bin] + sampleData[i]
        
    return binned_data

#%%
"""
separates training and testing datasets

x: dataset to split
train_percentage: how much of the data you want to set as training set

returns: separated training and testing set, should be np array?
"""
def separate(x,train_percentage):
    separate_point = int(x.shape[0] * train_percentage)
    x_train = x[:separate_point,].copy()
    x_test = x[separate_point+1:,].copy()
    return x_train, x_test

#%%
from scipy.interpolate import interp1d
x = np.linspace(0, 10, num=11, endpoint=True)
y = np.cos(-x**2/9.0)
f = interp1d(x, y)
print(f)
f2 = interp1d(x, y, kind='cubic')
xnew = np.linspace(0, 10, num=41, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()

#%%

#i = 1000
#%%
#Test loading multiple files
"""
用这个loop作为主循环？
步骤：
0. call dataset_breakdown()
1. call dropout(),分dropout1,2,3,4,5..........
2. 给以上的每一项call rebin(),分rebin10，20，30，40，50，这东西能用for loop吗？
3. 给以上每一项call smooth_binned_spikes()
4. 给所有项call separate(),分training和testing计算结果
四层for loop?
"""



file_address_list = list()

"""
Or matrix, file numbers * 10 * 5 * 2
"""
VAF_list = list()

#https://stackoverflow.com/questions/10377998/how-can-i-iterate-over-files-in-a-given-directory
for subdir, dirs, files in os.walk("..\dataset_small"):
    for file in files:
        print(" ")
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".mat"):
            file_address_list.append(filepath)
            dataset = lab_data('',filepath)
            time_frame, bin_width, spike_counts, kin_p, kin_v, EMG, EMG_names = dataset_breakdown(dataset)
                        
            #dropout_percentage = [0, 0.0334, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            dropout_percentage = [ 0.0334, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
            for a in range(len(dropout_percentage)):
                dropout_timepoints = np.array([])
                dropout_timepoints_length = np.array([])

                dropout_timepoints,dropout_timepoints_length,spike_counts_dropout = dropout(spike_counts, dropout_percentage[a], dropout_probability_list)
                dropout_timepoints = dropout_timepoints.astype(np.int64)
                dropout_timepoints_length = dropout_timepoints_length.astype(np.int64)
                
                print(" ")
                print("Length of Each Dropout Timepoint ")
                print(dropout_timepoints_length)

                
                """
                !!!TODO is this 0 or 1, check after run
                """
                #Count time points of each dropout, since sample rate is 1000Hz
                #(in 0.001s), but the dropout is in 0.01s. *10 will be the number
                #of datapoints for 1ms.
                for i in range(dropout_timepoints_length.shape[0]):
                    dropout_timepoints_length[i] = int((dropout_timepoints_length[i]+1)*10)
                
                print(" ")
                print("Length (in terms of how many datapoints) of Each Dropout Timepoint ")
                print(dropout_timepoints_length)

                #print(dropout_timepoints)
                #print(dropout_timepoints_length)
                
                bin_array = [10, 20, 30, 40, 50]
                for b in range(len(bin_array)): #5
                    spike_counts_rebin = rebin(spike_counts_dropout,bin_array[b])
                    spike_counts_rebin_smoothed = smooth_binned_spikes(spike_counts_rebin,0.05,'gaussian',0.1)
                    spike_counts_rebin_dropout = rebin(spike_counts_dropout, bin_array[b])
                    
                    spike_counts_spline_origin = spike_counts_dropout.copy()
                    
                    spike_counts_rebin_dropout_spline = spike_counts_rebin_dropout.copy()
                    #加上Spline语句，for每个dropout point做个spline补足对应的长度,然后删掉这句
                    
                    #print(" ")
                    #print("Number of Dropout Timepoints ")
                    #print(dropout_timepoints.shape[0], dropout_timepoints_length.shape[0])

                    if a != 0: 
                        
                        for i in range(dropout_timepoints.shape[0]): #for each recorded timepoints that I dropped the data over there out #3925
                            #print("1")
                            
                            for j in range(spike_counts_spline_origin.shape[1]): #for each single channel,96
                                #x_temp = list(range(spike_counts_spline_origin[dropout_timepoints[i]-1000:dropout_timepoints[i],j].size))
                                
                                #print(dropout_timepoints[i] + reference_num > spike_counts_spline_origin.shape[0])
                                
                                reference_num = dropout_timepoints_length[i] * 1000
                                if dropout_timepoints[i] - reference_num <= 0:
                                    reference_num = reference_num
                                elif dropout_timepoints[i] + reference_num > spike_counts_spline_origin.shape[0]:
                                    reference_num = 0 - reference_num
                                else:
                                    reference_num = 0 - reference_num
                                
                                if reference_num*2 > spike_counts_spline_origin[:,1].size:
                                    reference_num = spike_counts_spline_origin[:,1].size

                                if reference_num < 0:
                                    x_temp = list(range(dropout_timepoints[i]+reference_num,dropout_timepoints[i]))
                                    y_temp = spike_counts_spline_origin[dropout_timepoints[i]+reference_num:dropout_timepoints[i],j]
                                else: 
                                    x_temp = list(range(dropout_timepoints[i],dropout_timepoints[i]+reference_num))
                                    y_temp = spike_counts_spline_origin[dropout_timepoints[i]:dropout_timepoints[i]+reference_num,j]

                                if reference_num == spike_counts_spline_origin[:,1].size:
                                    x_temp = list(range(spike_counts_spline_origin[:,1].size))
                                    y_temp = spike_counts_spline_origin[:,j]
                                
                                #get the spline from x and y
                                spl_temp = splrep(x_temp,y_temp)
                                
                                #x_spline = list(range(int(dropout_timepoints[i]+500),int(dropout_timepoints[i]+dropout_timepoints_length[i]+500)))
                                
                                #print(type(dropout_timepoints[i]))
                                #print(type(dropout_timepoints_length[i]))
                                x_spline = list(range(dropout_timepoints[i],dropout_timepoints[i]+dropout_timepoints_length[i]))
                                
                                y_spline = splev(x_spline, spl_temp)
                                y_spline[y_spline < 0.005] = 0 #regularize extremely small values to 0
                                #print("2")
                                
                                #print(dropout_timepoints[i])
                                #print(y_spline.size)
                                #print(np.mean(y_temp))
                                #print(np.mean(y_spline))
                                #print(" ")
                                
                                """
                                if np.mean(y_spline) > 1: #error
                                    #np.vstack(())
                                else:
                                    #np.vstack(())
                                """
                                
                                for k in range(len(y_spline)):#10,or a little bit more
                                    spike_counts_spline_origin[dropout_timepoints[i]+k,j] = y_spline[k]
                                    
                                    #print("3")
                        #print("1 round")    


                    #Debugging Spline Function Random Shooting-up errors through previous statistical(?) data


                    #rebin after spline
                    spike_counts_rebin_dropout_spline = rebin(spike_counts_spline_origin,bin_array[b])
                    
                    EMG_rebin = rebin(EMG, bin_array[b])
                    
                    train_percent = 95/100
                    spike_counts_rebin_train, spike_counts_rebin_test = separate(spike_counts_rebin,train_percent)
                    spike_counts_rebin_smoothed_train, spike_counts_rebin_smoothed_test = separate(spike_counts_rebin_smoothed, train_percent)
                    spike_counts_rebin_dropout_train, spike_counts_rebin_dropout_test = separate(spike_counts_rebin_dropout, train_percent)
                    spike_counts_rebin_dropout_spline_train, spike_counts_rebin_dropout_spline_test = separate(spike_counts_rebin_dropout_spline, train_percent) #!!
                    EMG_rebin_train, EMG_rebin_test = separate(EMG_rebin, train_percent)
                    
                    (X_smooth_train,Y_smooth_train) = form_decoder_dataset(spike_counts_rebin_smoothed_train, EMG_rebin_train,10)
                    (X_smooth_test,Y_smooth_test) = form_decoder_dataset(spike_counts_rebin_smoothed_test, EMG_rebin_test,10)
                    H_reg_smooth_train = wiener_only_train(X_smooth_train,Y_smooth_train)
                    Y_pred_smooth_test = w_filter_test(X_smooth_test, H_reg_smooth_train)
                    vaf1 = vaf(Y_smooth_test, Y_pred_smooth_test)
                    #print(vaf1)
                    VAF_list.append(vaf1)
                    
                    (X_dropout_train,Y_dropout_train) = form_decoder_dataset(spike_counts_rebin_dropout_train, EMG_rebin_train,10)
                    (X_dropout_test,Y_dropout_test) = form_decoder_dataset(spike_counts_rebin_dropout_test, EMG_rebin_test,10)
                    H_reg_dropout_train = wiener_only_train(X_dropout_train,Y_dropout_train)
                    Y_pred_dropout_test = w_filter_test(X_dropout_test, H_reg_dropout_train)
                    vaf2 = vaf(Y_dropout_test,Y_pred_dropout_test)
                    #print(vaf2)
                    VAF_list.append(vaf2)
                    
                    (X_spline_train, Y_spline_train) = form_decoder_dataset(spike_counts_rebin_dropout_spline_train, EMG_rebin_train,10)
                    (X_spline_test, Y_spline_test) = form_decoder_dataset(spike_counts_rebin_dropout_spline_test,EMG_rebin_test,10)
                    H_reg_spline_train = wiener_only_train(X_spline_train,Y_spline_train)
                    Y_pred_spline_test = w_filter_test(X_spline_test,H_reg_spline_train)
                    vaf3 = vaf(Y_spline_test,Y_pred_spline_test)
                    print(vaf1, vaf2, vaf3)
                    #VAF_list.append(vaf3)
                    
                    
            print("one file")
"""
虽然这最里面的一个for loop里只有smooth和dropout,但因为dropout_percentage
里第一个数值是0，实际上等于（0，1）这个值是没有dropout的（我也不知道为什么我要把
smoothed的放在前面去）
"""
            

#print(dataset)
#print(file_address_list[0])

#%% Store results after running the previous block
"""
#先把这段comment掉以防万一
with open('results_run_2.txt', 'a') as f:
    for item in VAF_list:
        f.write("%s\n" % item)
f.close()
"""     
#%% Open results for further analysis
f = open('results_run_2.txt','r')
contents = f.read()
f.close()
           
import re
readin_content = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", contents)
for i in range(len(readin_content)):
    readin_content[i] = float(readin_content[i])

#%%
"""
Reconstruct the VAF_list to make it readable
"""

VAF_reconstructed = np.asarray(readin_content).reshape((55,90))

#%% calcualte standard error
VAF_reconstructed_stderr = np.zeros(90)
for i in range(VAF_reconstructed.shape[1]): #columns, 90
    curr_col = VAF_reconstructed[:,i]
    curr_col_improved = [x for x in curr_col if x > 0]
    #print(len(curr_col_improved))
    #print(stats.sem(curr_col_improved))
    temp = stats.sem(curr_col_improved)
    VAF_reconstructed_stderr[i] = temp
print(VAF_reconstructed_stderr)

#%%

VAF_average = np.zeros(90)

#Take average of each column, but delete all the wrongly calculated VAF data
for i in range(VAF_reconstructed.shape[1]): #columns, 90
    VAF_column_total = 0
    VAF_available_numbers = 0
    for j in range(VAF_reconstructed.shape[0]): #rows, 55
        if VAF_reconstructed[j,i] > 0:
            VAF_column_total = VAF_column_total + VAF_reconstructed[j,i]
            VAF_available_numbers = VAF_available_numbers + 1
    VAF_column_average = VAF_column_total/VAF_available_numbers
    VAF_average[i] = VAF_column_average

#Separate Smoothed and unsmoothed data to two sections
VAF_average_smooth = np.zeros(45)
VAF_average_noSmooth = np.zeros(45)
for i in range(VAF_average.shape[0]): #shape 0? shape 1?
    if i%2 == 0:
        VAF_average_smooth[int(i/2)] = VAF_average[i] #could be wrong
    else:
        VAF_average_noSmooth[int(i/2)] = VAF_average[i]

#(different types of dropouts, different types of binnings)
#Reshape them to the right groupings
VAF_average_smooth_int = VAF_average_smooth.astype(int)
VAF_average_noSmooth_int = VAF_average_noSmooth.astype(int)
VAF_average_smooth_int =VAF_average_smooth_int.reshape((9,5))
VAF_average_noSmooth_int = VAF_average_noSmooth_int.reshape((9,5))

#Plot the resultsa
"""
X axis: 9 different levels of dropout
Y axis: VAF value
Five lines: different binning values (10,20,30,40,50)
"""
plt.figure()
plt.ylabel("VAF value")
plt.xlabel("dropout levels")
plt.title("Smoothed")
plt.ylim(30,70)
x_axis = [0, 0.0334, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
bin_array_string = ['10', "20", "30", "40", "50"]

for column in range(VAF_average_smooth_int.shape[1]):
    plt.plot(x_axis,VAF_average_smooth_int[:,column],label = bin_array_string[column])
plt.legend()
plt.show()

plt.figure()
plt.ylabel("VAF value")
plt.xlabel("dropout levels")
plt.title("Not Smoothed")
plt.ylim(30,70)
for column in range(VAF_average_noSmooth_int.shape[1]):
    plt.plot(x_axis,VAF_average_noSmooth_int[:,column],label = bin_array_string[column])
plt.legend()
plt.show()

#%% Interpolation

