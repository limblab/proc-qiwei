# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 19:27:41 2020

@author: dongq
"""

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

#vid_dir = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT2D\videos'
vid_dir = r'C:\Users\dongq\Desktop\Crackle_20201202_rwFreeReach\Crackle_20201202'
#vid_name = r'\exp_han_00019DLC_resnet50_HanSep22shuffle1_1030000_filtered_labeled.mp4'
#vid_name = r'\exp_han_00017.avi'
#vid_out_name = r'\exp_han_00017_section.avi'

#vid_name = r'\exp_han_00018.avi'
#vid_out_name = r'\exp_han_00018_section.avi'

#vid_name = r'\exp_han_00019.avi'
#vid_out_name = r'\exp_han_00019_section.avi'

#vid_name = r'\exp_han_00020.avi'
#vid_out_name = r'\exp_han_00020_section.avi'

vid_name = r'\Crackle_20201202_00008.avi'
vid_out_name = r'\Crackle_20201202_00008_section.avi'

#start_time = 0
#end_time = 20


#vid_dir = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-09-22-RT3D\videos'
#vid_name = r'\exp_han_00010DLC_resnet50_HanSep22shuffle1_1030000_filtered_labeled.mp4'
#vid_name = r'\exp_han_00011DLC_resnet50_HanSep22shuffle1_1030000_filtered_labeled.mp4'
#vid_name = r'\exp_han_00012DLC_resnet50_HanSep22shuffle1_1030000_filtered_labeled.mp4'

#In seconds
start_time = 40
end_time = 60

ffmpeg_extract_subclip(vid_dir + vid_name, start_time, end_time, targetname=vid_dir + vid_out_name)