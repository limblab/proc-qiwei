# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 14:52:55 2020

@author: dongq
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import cv2
import os
import math
import pandas as pd 
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%% Open video file
#vid_folder = r'C:\Users\dongq\DeepLabCut\Rocket-Chris-2020-07-29\videos'
#vid_name = r'\exp00001.avi'
#vid_num = r'1'

vid_folder = r'D:\DLC_Folders_Currently_In_Use\Han-Qiwei-2020-09-22-RT2D\videos'
vid_name = r'\exp_han_00018DLC_resnet50_HanSep22shuffle1_1030000_filtered_labeled.mp4'
vid_num = r'2'

vidDir1 = vid_folder + vid_name

vidcap1 = cv2.VideoCapture(vidDir1)



#%% Read in the extraced video, and the csv files

#Colormap reference: https://matplotlib.org/3.3.2/tutorials/colors/colormaps.html

#Read in DLC marker estimation files
#In csv format
marker_csv1 = r'\exp_han_00018DLC_resnet50_HanSep22shuffle1_1030000filtered.csv'
df_c1 = pd.read_csv(vid_folder + marker_csv1)
arr_c1 = df_c1.to_numpy()

#%% Extract frames

length1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_COUNT))
print(length1)


success, image = vidcap1.read()

#%% read 1 frame
#https://web.archive.org/web/20161010175545/https://tobilehman.com/blog/2013/01/20/extract-array-of-frames-from-mp4-using-python-opencv-bindings/
"""
count = 0;
while success:
  success,image = vidcap1.read()
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1

"""
fps = 25
def count_frame_no(minutes, seconds,fps):

    #time_length = vid_frames / fps
    frame_seq_min = minutes
    frame_seq_sec = seconds
    frame_seq = (frame_seq_min * 60 + frame_seq_sec) * fps
    #frame_no = frame_seq / vid_frames
    return frame_seq

r"""
C:\Users\dongq\DeepLabCut\Rocket-Chris-2020-07-29\videos\exp00001.avi
"""
#frame_no1 = count_frame_no(16, 0,fps) #Example of a good tracking frame in experiment phase
#frame_no1 = count_frame_no(30, 49, fps) #Example of a bad tracking frame out of experiment phase
#frame_no1 = count_frame_no(31, 10, fps) #Example of a bad tracking frame out of experiment phase
frame_no1 = count_frame_no(16, 24,fps) #Example of a good tracking frame in experiment phase

vidcap1.set(1, frame_no1)

ret,frame = vidcap1.read()

frame_name = r'\frame_' + str(frame_no1) + r'_cam_' + vid_num + r'.png'

cv2.imwrite(vid_folder + frame_name, frame)

vidcap1.release()
cv2.destroyAllWindows()



#%% Prepare the data to plot likelihood values for all markers in 1 second
#marker_list = ['.','.','.','.','.','.','.','.','.','.',]
marker_list = ["o","v","s","p","P","*","X","D","<","d"]
#marker_list = ['$1$','$2$','$3$','$4$','$5$','$6$','$7$','$8$','$9$','$0$',]
#marker_list = ['$Shoulder1$','$Arm1$','$Arm2$','$Elbow1$','$Elbow2$','$Wrist1$','$Wrist2$','$Hand1$','$Hand2$','$Hand3$',]
#marker_list = ['$S1$','$A1$','$A2$','$E1$','$E2$','$W1$','$W2$','$H1$','$H2$','$H3$',]

#Read in extracted video frames
#In png format
img1 = mpimg.imread(vid_folder + frame_name)
#print(img1)
#imgplot = plt.imshow(img1)

#X seconds * 24 fps
#section_length = int(2*fps) #frames #Take only one second (24 frames) of data, can change
section_length = 1 #frames #Take only one second (24 frames) of data, can change

one_in_X = 1
markers_section1 = arr_c1[frame_no1-section_length: frame_no1,:]
markers_section1_X = list(range(0,section_length,one_in_X))
markers_section1 = markers_section1[markers_section1_X,:]

speed_section = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29]
likelihood_section = [3,6,9,12,15,18,21,24,27,30]

markers_speed_section1 = markers_section1[:,speed_section]
markers_likelihood_section1 = markers_section1[:,likelihood_section]

float_markers_speed_section1 = np.vstack(markers_speed_section1[:,:]).astype(np.float)
float_markers_likelihood_section1 = np.vstack(markers_likelihood_section1[:,:]).astype(np.float)
float_markers_section1 = np.vstack(markers_section1[:,:]).astype(np.float)
float_markers_section1 = np.delete(float_markers_section1,0,axis=1)


#%% Prepare to plot the demonstration of speed calculation in 2D
"""
2 markers, one line
"""

img1 = mpimg.imread(vid_folder + frame_name)
section_length = 2
one_in_X = 2

ttl_markers_in_section = section_length * one_in_X
markers_section1_Y = arr_c1[frame_no1-ttl_markers_in_section: frame_no1,:]
markers_section1_X = list(range(0,ttl_markers_in_section,one_in_X)) #range(start, stop, step) #if n*step >= stop, doesn't count that last step
markers_section1_Y = markers_section1_Y[markers_section1_X,:]

speed_section = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29]
likelihood_section = [3,6,9,12,15,18,21,24,27,30]
marker_list_full_name = ['$Shoulder1$','$Arm1$','$Arm2$','$Elbow1$','$Elbow2$','$Wrist1$','$Wrist2$','$Hand1$','$Hand2$','$Hand3$',]


markers_speed_section1 = markers_section1_Y[:,speed_section]
markers_likelihood_section1 = markers_section1_Y[:,likelihood_section]

float_markers_speed_section1 = np.vstack(markers_speed_section1[:,:]).astype(np.float)
float_markers_likelihood_section1 = np.vstack(markers_likelihood_section1[:,:]).astype(np.float)

float_markers_section1 = np.vstack(markers_section1_Y[:,:]).astype(np.float)
float_markers_section1 = np.delete(float_markers_section1,0,axis=1) #The first col is the frame number, which we don't need

#Plot the demonstration of speed calculation in 2D

fig, ax = plt.subplots(figsize=(14, 10))
#plt.figure()
imgplot = plt.imshow(img1)

#for i in range(int(float_markers_speed_section1.shape[1]/2)):
#    #print(i)
#    plt.scatter(float_markers_speed_section1[:,i*2],float_markers_speed_section1[:,i*2+1], s=70,marker=marker_list[i])
    
for i in range(int(float_markers_section1.shape[1]/3)):
    #print(i)
    plt.scatter(float_markers_section1[:,i*3],float_markers_section1[:,i*3+1],s=100,label=marker_list_full_name[i])
    plt.plot(float_markers_section1[:,i*3],float_markers_section1[:,i*3+1],linewidth=5)
    print(float_markers_section1[:,i*3+2])
    
plt.legend()
plt.show()
#plt.clim(0,1.00000001)
#plt.clim(0,2)
#cbar = plt.colorbar()

#plt.savefig(r'C:\Users\dongq\Desktop\dxdy_demonstration.png')