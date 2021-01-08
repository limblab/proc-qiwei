# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:52:55 2020

@author: dongq
"""

import ffmpeg
import subprocess

#%%

vid_folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\videos'
vid_name = r'\Crackle_20201203_00009DLC_resnet50_TestDec3shuffle1_1030000_filtered_labeled'
vid_type = '.mp4'

video_input_path = vid_folder + vid_name + vid_type
video_output_path = vid_folder + vid_name + "_fpsFixed" + vid_type


#c = 'ffmpeg -y -i ' + video_input_path + ' -r 30 -s 112x112 -c:v libx264 -b:v 3M -strict -2 -movflags faststart '+video_output_path
#c = 'ffmpeg -y -i ' + video_input_path + ' -r 30 -s 112x112 -c:v mp4v -b:v 3M -strict -2 -movflags faststart '+video_output_path
c = 'ffmpeg -i ' +  video_input_path + ' -r 25 ' + video_output_path
subprocess.call(c, shell=True)

#%%

import cv2
cap = cv2.VideoCapture(video_input_path)
cap.set 