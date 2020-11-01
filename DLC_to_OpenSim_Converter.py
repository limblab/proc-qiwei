"""
Converts DLC (actually Min's 3D reconstruction Output File) (in .csv format)
to OpenSim (which is in .trc format)
"""
import pandas as pd 
import numpy as np
import os

#%%Import 3D reconstructed marker file from Min's 3D Reconstruction code
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\3D-data\output_3d_data_rotate4.csv')
#df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-08-04-FreeReaching\reconstructed-3d-data\output_3d_data.csv')
df = pd.read_csv (r'C:\Users\dongq\DeepLabCut\Rocket-Chris-2020-07-29\reconsturcted-3d-data\output_3d_data.csv')

#%% Set parameters for this dataset
"""
IMPORTANT: SPECIFY IF THIS DATASET HAS STATIC "WORLD GROUND TRUTH" POINTS OR NOT
"""
has_static = True

frameRate = 25
numFrames = df.shape[0]-1
numMarkers = 10

mm_to_m_conversion = 1000

#%% Delete the static points if this dataset has them
if has_static == True:
    list_to_delete = ['pointX_x','pointX_y','pointX_z','pointX_error','pointX_ncams','pointX_score','pointY_x','pointY_y','pointY_z','pointY_error','pointY_ncams','pointY_score','pointZ_x','pointZ_y','pointZ_z','pointZ_error','pointZ_ncams','pointZ_score',]
    df = df.drop(columns = list_to_delete)
    
#%% TEMP, check dropout rate of the 3D dataset
df_array = df.to_numpy()

num_nans = 0
for i in range(df_array.shape[0]):
    for j in range(df_array.shape[1]):
        if np.isnan(df_array[i,j]):
            num_nans += 1
actual_num_nans = num_nans/2 #x,y,z are needed, but the following 3 are not needed
total_points = df_array.size*30/61
dropout_percentage = actual_num_nans/total_points
print("Number of markers in all frames with NaNs", num_nans)
print("Total points in all frames", total_points)
print("Marker dropout percentage", dropout_percentage)

#%% A specific function adding blanks to the header of the dataset for OpenSim
def add_blanks(list_input, ref_dataframe):
    num_blanks = ref_dataframe.shape[1] - len(list_input)
    for i in range(num_blanks):
        list_input.append(np.nan)
    #print(list_input)
    return list_input

#%% Convert the markers from the structure in Min's 3D reconstruction code
    #to OpenSim's rules.
data = pd.DataFrame(np.zeros([1, 32])*np.nan)
row1 = ['PathFileType', '4', 'X/Y/Z', 'output_3d_data.trc']
add_blanks(row1, data)
row2 = ['DataRate', 'CameraRate', 'NumFrames', 'NumMarkers', 'Units', 'OrigDataRate', 'OrigDataStartFrame', 'OrigNumFrames']
add_blanks(row2, data)
row3 = [frameRate, frameRate, numFrames, numMarkers, 'm', frameRate, 1, numFrames]
add_blanks(row3, data)
row4 = ['fnum',	'Time',	'Shoulder_JC', np.nan, np.nan, 	'Marker_8', np.nan, np.nan, 	"Marker_7", np.nan, np.nan, 	"Marker_6", np.nan, np.nan, 	"Pronation_Pt1", np.nan, np.nan, "Marker_5", np.nan, np.nan, 	"Marker_4", np.nan, np.nan,	"Marker_3", np.nan, np.nan, 	"Marker_2", np.nan, np.nan, 	"Marker_1", np.nan, np.nan]
row5 = [np.nan, np.nan, 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2','X3','Y3','Z3','X4','Y4','Z4','X5','Y5',	'Z5',	'X6',	'Y6',	'Z6',	'X7',	'Y7'	,'Z7',	'X8',	'Y8',	'Z8',	'X9',	'Y9',	'Z9',	'X10','Y10',	'Z10']
row6 = pd.DataFrame (np.zeros([1, 32])*np.nan)

data.loc[0] = row1
data.loc[1] = row2
data.loc[2] = row3
data.loc[3] = row4
data.loc[4] = row5
data = data.append(row6, ignore_index = True)

#set up Excel output data table with correct formatting
"""
data = pd.DataFrame(np.zeros([1, 32])*np.nan)
row2 = ['fnum',	'Time',	'Shoulder_JC', np.nan, np.nan, 	'Marker_8', np.nan, np.nan, 	"Marker_7", np.nan, np.nan, 	"Marker_6", np.nan, np.nan, 	"Pronation_Pt1", np.nan, np.nan, "Marker_5", np.nan, np.nan, 	"Marker_4", np.nan, np.nan,	"Marker_3", np.nan, np.nan, 	"Marker_2", np.nan, np.nan, 	"Marker_1", np.nan, np.nan]
row3 = [np.nan, np.nan, 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2','X3','Y3','Z3','X4','Y4','Z4','X5','Y5',	'Z5',	'X6',	'Y6',	'Z6',	'X7',	'Y7'	,'Z7',	'X8',	'Y8',	'Z8',	'X9',	'Y9',	'Z9',	'X10','Y10',	'Z10']
data.loc[1]=row2
data.loc[2]=row3
row4 = pd.DataFrame (np.zeros([1, 32])*np.nan)
data = data.append(row4, ignore_index = True)
"""

#put data into second DF that will be appended to existing header DF

transfer = pd.DataFrame(np.zeros([len(df.index),1]))
transfer[0] = df['fnum']+1
transfer[1] = df['fnum']*1/25

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

# =============================================================================
# shoulderx = df['shoulder1_x']
# shouldery = df['shoulder1_y']
# shoulderz = df['shoulder1_z']
# =============================================================================

shoulderx = df['shoulder1_x']
shouldery = df['shoulder1_z']
shoulderz = df['shoulder1_y']

for i in range(len(shoulderx)):
    if np.isnan(shoulderx[i]):
        shoulderx[i] = 0.0
    if np.isnan(shouldery[i]):
        shouldery[i] = 0.0
    if np.isnan(shoulderz[i]):
        shoulderz[i] = 0.0

#Load Marker_8 values, which is Arm1 
# =============================================================================
# 
# transfer[5] = df['arm1_x'] - shoulderx
# transfer[6] = df['arm1_y'] - shouldery
# transfer[7] = df['arm1_z'] - shoulderz
# =============================================================================

transfer[5] = df['arm1_x'] - shoulderx
#transfer[5] = df['arm1_x'] - shoulderx*-1
transfer[6] = df['arm1_z'] - shoulderz
transfer[7] = (df['arm1_y'] - shouldery)*-1
#transfer[7] = (df['arm1_y'] - shouldery)

#Load Marker_7 values, which is Arm2
# =============================================================================
# 
# transfer[8] = df['arm2_x']-shoulderx
# transfer[9] = df['arm2_y']-shouldery
# transfer[10] = df['arm2_z'] - shoulderz
# =============================================================================

#transfer[8] = df['arm2_x']-shoulderx*-1
transfer[8] = df['arm2_x']-shoulderx
transfer[9] = df['arm2_z']-shoulderz
transfer[10] = (df['arm2_y'] - shouldery)*-1
#transfer[10] = (df['arm2_y'] - shouldery)
#Load Marker_6 values, which is Elbow2, (Qiwei) elbow2 in my case
# =============================================================================
# 
# transfer[11] = df['elbow2_x'] - shoulderx
# transfer[12] = df['elbow2_y'] - shouldery
# transfer[13] = df['elbow2_z'] - shoulderz
# =============================================================================

#transfer[11] = df['elbow2_x'] - shoulderx*-1
transfer[11] = df['elbow2_x'] - shoulderx
transfer[12] = df['elbow2_z'] - shoulderz
transfer[13] = (df['elbow2_y'] - shouldery)*-1
#transfer[13] = (df['elbow2_y'] - shouldery)
#Load Pronation_Pt1 values, which is Elbow1, (Qiwei) elbow1 in my case
# =============================================================================
# 
# transfer[14] = df['elbow1_x'] - shoulderx
# transfer[15] = df['elbow1_y'] - shouldery 
# transfer[16] = df['elbow1_z'] - shoulderz
# =============================================================================

#transfer[14] = df['elbow1_x'] - shoulderx*-1
transfer[14] = df['elbow1_x'] - shoulderx
transfer[15] = df['elbow1_z'] - shoulderz
transfer[16] = (df['elbow1_y'] - shouldery)*-1
#transfer[16] = (df['elbow1_y'] - shouldery)
#print(transfer[14])
#print(df['elbow2_y'])
#print(df['elbow2_y'] - shouldery)
#???

#Load Marker_5 values, which is Wrist1, (Qiwei) wrist2 in my case
# =============================================================================
# 
# transfer[17] = df['wrist2_x'] - shoulderx
# transfer[18] = df['wrist2_y'] - shouldery
# transfer[19] = df['wrist2_z'] - shoulderz
# =============================================================================

#transfer[17] = df['wrist2_x'] - shoulderx*-1
transfer[17] = df['wrist2_x'] - shoulderx
transfer[18] = df['wrist2_z'] - shoulderz
transfer[19] = (df['wrist2_y'] - shouldery)*-1
#transfer[19] = (df['wrist2_y'] - shouldery)
#Load Marker_4 values, which is Wrist2, (Qiwei) wrist1 in my case
# =============================================================================
# 
# transfer[20] = df['wrist1_x'] - shoulderx
# transfer[21] = df['wrist1_y'] - shouldery
# transfer[22] = df['wrist1_z'] - shoulderz
# =============================================================================

#transfer[20] = df['wrist1_x'] - shoulderx*-1
transfer[20] = df['wrist1_x'] - shoulderx
transfer[21] = df['wrist1_z'] - shoulderz
transfer[22] = (df['wrist1_y'] - shouldery)*-1
#transfer[22] = (df['wrist1_y'] - shouldery)
#Load Marker_3 values, which is Hand1, (Qiwei) hand3 in my case
# =============================================================================
# 
# transfer[23] = df['hand3_x'] - shoulderx 
# transfer[24] = df['hand3_y'] - shouldery
# transfer[25] = df['hand3_z'] - shoulderz
# =============================================================================

#transfer[23] = df['hand3_x'] - shoulderx*-1
transfer[23] = df['hand3_x'] - shoulderx 
transfer[24] = df['hand3_z'] - shoulderz
transfer[25] = (df['hand3_y'] - shouldery)*-1
#transfer[25] = (df['hand3_y'] - shouldery)
#Load Marker_2 values, which is Hand2, (Qiwei) hand1 in my case
# =============================================================================
# 
# transfer[26] = df['hand1_x'] - shoulderx 
# transfer[27] = df['hand1_y'] - shouldery
# transfer[28] = df['hand1_z'] - shoulderz
# =============================================================================

#transfer[26] = df['hand1_x'] - shoulderx*-1
transfer[26] = df['hand1_x'] - shoulderx 
transfer[27] = df['hand1_z'] - shoulderz
transfer[28] = (df['hand1_y'] - shouldery)*-1
#transfer[28] = (df['hand1_y'] - shouldery)
#Load Marker_1 values, which is MCP, (Qiwei) hand2 in my case
# =============================================================================
# 
# transfer[29] = df['hand2_x'] - shoulderx 
# transfer[30] = df['hand2_y'] - shouldery
# transfer[31] = df['hand2_z'] - shoulderz
# =============================================================================

#transfer[29] = df['hand2_x'] - shoulderx*-1
transfer[29] = df['hand2_x'] - shoulderx 
transfer[30] = df['hand2_z'] - shoulderz
transfer[31] = (df['hand2_y'] - shouldery)*-1
#transfer[31] = (df['hand2_y'] - shouldery)


#Divide all numbers by 1000, from mm to m
for x in range (5,32):
    transfer[x] = transfer[x]/mm_to_m_conversion

#Set all the points in a frame to 0 if that frame has more than 5 markers un-seenable
for i in range(transfer.shape[0]): #for each frame
    num_of_nans = 0
    #1. First step, show in OpenSim, that the bad points are actually bad by assigning them all to 0, so that it behaves in a wierd way during simulation
    for j in range(5,32): #for each tracking point
        if np.isnan(transfer[j][i]):
            num_of_nans += 1
    if num_of_nans >= 15: #if five(*3 for x,y,z) of ten points are unavailable, this point is basically useless
        for j in range(5,32): #clear all the points on this frame to 0
            transfer[j][i] = np.nan

#Append numerical values to header
FinalOutput = data.append(transfer, ignore_index = True)

#%% Store an .xlsx version of the 3D dataset
FinalOutput.to_excel(r'C:\Users\dongq\DeepLabCut\Rocket-Chris-2020-07-29\reconsturcted-3d-data\output_3d_data.xlsx', index = False, header = False)
file = pd.read_excel(r'C:\Users\dongq\DeepLabCut\Rocket-Chris-2020-07-29\reconsturcted-3d-data\output_3d_data.xlsx', header = None)

#%% Store an .trc version of the 3D dataset (the one we actually need for OpenSim)
#np.savetxt(r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-02-21\videos\output_3d_data.txt', transfer.values)
#https://stackoverflow.com/questions/41211619/how-to-convert-xlsx-to-tab-delimited-files
file.to_csv(r'C:\Users\dongq\DeepLabCut\Rocket-Chris-2020-07-29\reconsturcted-3d-data\output_3d_data.trc',sep = "\t",index = False, header = None)
