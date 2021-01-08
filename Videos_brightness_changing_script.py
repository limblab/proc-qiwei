# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:46:06 2020

@author: dongq
"""
#referring to https://www.programmersought.com/article/96421442570/ for changing contrast and brightness
#referrring to https://blog.csdn.net/tuzixini/article/details/78847942 for reading and writing videos

#%% Import packages
import numpy as np
import cv2

#%% define function to change brightness and contrast
def Contrast_and_Brightness(alpha, beta, img):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + (1-alpha) * blank + beta
    dst = cv2.addWeighted(img, alpha, blank, 1-alpha, beta)
    return dst

#%% Read in videos
# =============================================================================
# vid_folder = r'C:\Users\dongq\Desktop\proc-qiwei'
# vid_name = r'\test_20200922_RT3D_cam2'
# vid_fmt = r'.mp4'
# =============================================================================

#vid_folder = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\videos'

#vid_name = r'\exp_han_00017_section'
#vid_name = r'\exp_han_00018'
#vid_name = r'\exp_han_00018_section'
#vid_name = r'\exp_han_00019'
#vid_name = r'\exp_han_00019_section'
#vid_name = r'\exp_han_00020'
vid_name = r'\exp_han_00020_section'


vid_folder = r'C:\Users\dongq\DeepLabCut\Crackle-Qiwei-2020-12-03\videos'
vid_name = r'\Crackle_20201203_00007_trimmed'
#vid_name = r'\Crackle_20201203_00008_trimmed'
#vid_name = r'\Crackle_20201203_00009_trimmed'
#vid_name = r'\Crackle_20201203_00010_trimmed'
#vid_name = r'\Crackle_20201203_00009'






vid_fmt = r'.avi'
vid_out_fmt = r'.mp4'

cap = cv2.VideoCapture( vid_folder + vid_name + vid_fmt )
 
#just some extra stuff
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
Ret, frame = cap.read()
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

#%% Change brightness, and write videos out

out_vid_name = vid_folder + vid_name + '_higher_lighting' + vid_out_fmt

videoWriter = cv2.VideoWriter(out_vid_name,cv2.VideoWriter_fourcc(*'MP4V'),fps,size)

Ret, frame = cap.read()

while Ret:
    #cv2.waitKey(int(fps))       #only for showing the videos as reference, takes time, can comment out
    #cv2.imshow('frame', frame)  #only for showing the videos as reference, takes time, can comment out
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #was originally commented
    frame1=Contrast_and_Brightness(1.8, 1.6, frame)
    #cv2.imshow('frame1', frame1) #only for showing the videos as reference, takes time, can comment out
    videoWriter.write(frame1)
    Ret, frame = cap.read()

# =============================================================================
# 
# while (cap.isOpened()):
#     Ret, frame = cap.read() ##retreturn boolean
#     cv2.imshow('frame', frame)
#  
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame1=Contrast_and_Brightness(2.0, 1.6, frame)
#  
#     cv2.imshow('frame1', frame1)
#     
#     videoWriter.write(frame1)
#     
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# =============================================================================
 
cap.release()
videoWriter.release()
cv2.destroyAllWindows()
