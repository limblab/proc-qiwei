# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:43:38 2020

@author: dongq
"""

import os
import cv2

#%%
main_folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\low_likelihood_extracted_frames'

folder_1 = r'\cam1'
folder_2 = r'\cam2'
folder_3 = r'\cam3'
folder_4 = r'\cam4'

sub_folder = r'\extract_for_model_retraining'
sub_output_folder = r'\extract_for_model_retraining_markerless'

folder_lists = [main_folder + folder_1 + sub_folder,
                main_folder + folder_2 + sub_folder,
                main_folder + folder_3 + sub_folder,
                main_folder + folder_4 + sub_folder]

output_folder_lists = [main_folder + folder_1 + sub_output_folder,
                       main_folder + folder_2 + sub_output_folder,
                       main_folder + folder_3 + sub_output_folder,
                       main_folder + folder_4 + sub_output_folder]

frame_names = []

for i in range(len(folder_lists)):
    frame_names.append(os.listdir(folder_lists[i]))
    
frame_nums = [[] for i in range(len(folder_lists))]
for i in range(len(frame_names)):
    for j in range(len(frame_names[i])):
        frame_name = frame_names[i][j]
        if frame_name.endswith('.png'):
            frame_num = frame_name[:-4]
        frame_num = int(frame_num)
        frame_nums[i].append(frame_num)
        
               
#%% use cv2 to extract frames
        
vid_folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\videos'

vid_1 = r'\Crackle_20201203_00007'
vid_2 = r'\Crackle_20201203_00008'
vid_3 = r'\Crackle_20201203_00009'
vid_4 = r'\Crackle_20201203_00010'

vid_type = r'.avi'

vid_dirs = [vid_folder + vid_1 + vid_type,
            vid_folder + vid_2 + vid_type,
            vid_folder + vid_3 + vid_type,
            vid_folder + vid_4 + vid_type]
#%%
for i in range(len(frame_nums)):
    
    vid_dirs[i]
    vidcap = cv2.VideoCapture(vid_dirs[i])
    
    for j in range(len(frame_nums[i])): #frame nums
        
        #arr_end = len(cam_arr_exp_ll[i][k,:])-1
        frame_num = frame_nums[i][j]
        cam_name = vid_dirs[i]

        #read in the video
        vidcap.set(1, frame_num)
        ret, frame = vidcap.read()
        cv2.imwrite(output_folder_lists[i] + '\\' + str(frame_names[i][j]), frame)
        
        print(cam_name + ' ' + frame_names[i][j])





