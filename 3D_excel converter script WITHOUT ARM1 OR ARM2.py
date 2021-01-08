# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:13:41 2019

@author: benja
"""

#NOTE: USE THIS SCRIPT WHEN OMITTING MARKER_8 AND MARKER_7 AKA ARM_1 AND ARM_2
#NOTE: USE THIS SCRIPT WHEN OMITTING MARKER_8 AND MARKER_7 AKA ARM_1 AND ARM_2
#NOTE: USE THIS SCRIPT WHEN OMITTING MARKER_8 AND MARKER_7 AKA ARM_1 AND ARM_2

import pandas as pd 
import numpy as np
#import DLC output file
df = pd.read_csv (Load the CSV file here, use r'C:\whatever' formatting)


#set up Excel output data table header with correct formatting
data = pd.DataFrame(np.zeros([1, 26])*np.nan)
row2 = ['Frame#',	'Time',	'Shoulder_JC', np.nan, np.nan, 	 	"Marker_6", np.nan, np.nan, 	"Pronation_Pt1", np.nan, np.nan, "Marker_5", np.nan, np.nan, 	"Marker_4", np.nan, np.nan,	"Marker_3", np.nan, np.nan, 	"Marker_2", np.nan, np.nan, 	"Marker_1", np.nan, np.nan]
row3 = [np.nan, np.nan, 'X1', 'Y1', 'Z1', 'X4','Y4','Z4','X5','Y5',	'Z5',	'X6',	'Y6',	'Z6',	'X7',	'Y7'	,'Z7',	'X8',	'Y8',	'Z8',	'X9',	'Y9',	'Z9',	'X10','Y10',	'Z10']
data.loc[1]=row2
data.loc[2]=row3
row4 = pd.DataFrame (np.zeros([1, 26])*np.nan)
data = data.append(row4, ignore_index = True)


#put data into second DF that will be appended to existing header DF

transfer = pd.DataFrame(np.zeros([len(df.index),1]))
transfer[0] = df['fnum']+1
transfer[1] = df['fnum']*1/30

#Some notes: the first two columns are just time and frame number markers.
#However, in order to convert from Min's formatting to Chris's, there's some axis and reference point
#conversion that's going on while loading the data, so here's a quick overview:
#Min's y -> Chris's x
#Min's x -> Chris's z
#Min's z -> Chris's -y
#Additionally, Chris's marker values are all referenced from the shoulder, so shoulder location
#data is subtracted from all corresponding columns. 

#These first three columns are the shoulder values, which are just set to 0, so I'm just loading
#some column from Min's excel sheet and multiplying it by 0 for all three columns

transfer[2] = df['fnum']*0
transfer[3] = df['fnum']*0
transfer[4] = df['fnum']*0 

#Load shoulder values into series so it's easier to reference when loading columns

shoulderx = df['Shoulder_x']
shouldery = df['Shoulder_y']
shoulderz = df['Shoulder_z']


#Load Marker_6 values, which is Elbow2

transfer[5] = df['Elbow2_y'] - shouldery
transfer[6] = shoulderz - df['Elbow2_z']
transfer[7] = df['Elbow2_x'] - shoulderx

#Load Pronation_Pt1 values, which is Elbow1

transfer[8] = df['Elbow1_y'] - shouldery
transfer[9] = shoulderz - df['Elbow1_z']
transfer[10] = df['Elbow2_x'] - shoulderx

#Load Marker_5 values, which is Wrist1

transfer[11] = df['Wrist1_y'] - shouldery
transfer[12] = shoulderz - df['Wrist1_z'] 
transfer [13] = df['Wrist1_x'] - shoulderx

#Load Marker_4 values, which is Wrist2

transfer[14] = df['Wrist2_y'] - shouldery
transfer[15] = shoulderz - df['Wrist2_z']
transfer[16] = df['Wrist2_x'] - shoulderx 

#Load Marker_3 values, which is Hand1

transfer[17] = df['Hand1_y'] - shouldery
transfer[18] = shoulderz - df['Hand1_z']
transfer[19] = df['Hand1_x'] - shoulderx

#Load Marker_2 values, which is Hand2

transfer[20] = df['Hand2_y'] - shouldery
transfer[21] = shoulderz - df['Hand2_z']
transfer[22] = df['Hand2_x'] - shoulderx

#Load Marker_1 values, which is MCP

transfer[23] = df['MCP3_y'] - shouldery
transfer[24] = shoulderz - df['MCP3_z']
transfer[25] = df['MCP3_x'] - shoulderx

#Divide all numbers by 1000

for x in range (5,26):
    transfer[x] = transfer[x]/1000
    

#Append numerical values to header

FinalOutput = data.append(transfer, ignore_index = True)

FinalOutput.to_excel(r'R:\Basic_Sciences\Phys\L_MillerLab\limblab\User_folders\ChrisV\TestExport2.xlsx', index = False, header = False)


print(transfer)