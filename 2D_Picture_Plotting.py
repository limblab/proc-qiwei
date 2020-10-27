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
vid_folder = r'C:\Users\dongq\DeepLabCut\Rocket-Chris-2020-07-29\videos'
vid_name = r'\exp00001.avi'
vid_num = r'1'

vidDir1 = vid_folder + vid_name

vidcap1 = cv2.VideoCapture(vidDir1)

#%% Extract frames

length1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_COUNT))
print(length1)


success, image = vidcap1.read()

#%% convert the whole video to frames and save each one
#https://web.archive.org/web/20161010175545/https://tobilehman.com/blog/2013/01/20/extract-array-of-frames-from-mp4-using-python-opencv-bindings/
"""
count = 0;
while success:
  success,image = vidcap1.read()
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1

"""
fps = 24
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
frame_no1 = count_frame_no(16, 24,fps) #Example of a good tracking frame in experiment phase

vidcap1.set(1, frame_no1)

ret,frame = vidcap1.read()

frame_name = r'\frame_' + str(frame_no1) + r'_cam_' + vid_num + r'.png'

cv2.imwrite(vid_folder + frame_name, frame)

vidcap1.release()
cv2.destroyAllWindows()



#%% Read in the extraced video, and the csv files

#Read in DLC marker estimation files
#In csv format
marker_csv1 = r'\exp00001DLC_resnet50_RocketJul29shuffle1_1030000filtered.csv'
df_c1 = pd.read_csv(vid_folder + marker_csv1)
arr_c1 = df_c1.to_numpy()

#Read in extracted video frames
#In png format
img1 = mpimg.imread(vid_folder + frame_name)
#print(img1)
#imgplot = plt.imshow(img1)

section_length = fps #frames #Take only one second (24 frames) of data, can change
markers_section1 = arr_c1[frame_no1-section_length: frame_no1,:]

speed_section = [1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29]
markers_speed_section1 = markers_section1[:,speed_section]
float_markers_speed_section1 = np.vstack(markers_speed_section1[:,:]).astype(np.float)

imgplot = plt.imshow(img1)

for i in range(int(float_markers_speed_section1.shape[1]/2)):
    #print(i)
    plt.scatter(float_markers_speed_section1[:,i*2],float_markers_speed_section1[:,i*2+1], s=70)