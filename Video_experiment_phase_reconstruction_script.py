# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:57:20 2020

@author: dongq
"""

#%% Import packages

import pandas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import cv2
import os
import math
import pandas as pd 
import copy
import seaborn as sns
import ffmpeg

#%% ONE: Add frame numbers on the video
#%% 1. read in videos && maybe add the frame numbers as well?

vid_path = r'C:\Users\dongq\Desktop\Crackle_20201203_rwFreeReach\Crackle_20201203'

for filename in os.listdir(vid_path):
    if (filename.endswith(".avi")): #or .avi, .mpeg, whatever.
        #os.system("ffmpeg -i {0} -f image2 -vf fps=fps=1 output%d.png".format(filename))
        out = ffmpeg.input(filename).drawtext(text="%{n}",start_number=0,
                 fontfile=r"C:\Windows\Fonts\arial.ttf",
                 fontcolor="red",x=40,y=100,
                 fontsize="64").output(os.path.join(vid_path,r"\ff_outputz",r'\\',filename))
# =============================================================================
#             
#             .output(os.path.join(vid_path,"ff_outputz",filename))
#             .run(overwrite_output=True)
# =============================================================================
        print(filename)
    else:
        continue

#Reference: https://stackoverflow.com/questions/42438380/ffmpeg-in-python-script
#Reference: https://github.com/kkroening/ffmpeg-python
#ffmpeg -i input -vf "drawtext=fontfile=Arial.ttf: text='%{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh): fontcolor=black: fontsize=20: box=1: boxcolor=white: boxborderw=5" -c:a copy output




#%% TWO: Segment and reconstruct the video
#%% Read in video data and ground truth file
