# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 21:09:14 2021

@author: dongq
"""

"""
This is a hardcode file using the reprojection error of the average of shoulder,
elbow, wrist and hand landmarks for Han-1217-RT2D, Han-1204-RT3D, Crackle-1214-RT2D,
and Crackle-1214-RT3D.

for each array, [0] is shoulder, [1] is elbow, [2] is wrist, [3] is hand
"""
import numpy as np

with open('ReprojectionError-Crackle-2D.npy', 'rb') as f:
    RepErr_C_2D = np.load(f,allow_pickle=True)

with open('ReprojectionError-Crackle-3D.npy', 'rb') as f:
    RepErr_C_3D = np.load(f,allow_pickle=True)
    
with open('ReprojectionError-Han-2D.npy', 'rb') as f:
    RepErr_H_2D = np.load(f,allow_pickle=True)
    
with open('ReprojectionError-Han-3D.npy', 'rb') as f:
    RepErr_H_3D = np.load(f,allow_pickle=True)
    
    
overall_repErr_C_2D = [*RepErr_C_2D[0],*RepErr_C_2D[1],*RepErr_C_2D[2],*RepErr_C_2D[3]]
overall_repErr_C_3D = [*RepErr_C_3D[0],*RepErr_C_3D[1],*RepErr_C_3D[2],*RepErr_C_3D[3]]
overall_repErr_H_2D = [*RepErr_H_2D[0],*RepErr_H_2D[1],*RepErr_H_2D[2],*RepErr_H_2D[3]]
overall_repErr_H_3D = [*RepErr_H_3D[0],*RepErr_H_3D[1],*RepErr_H_3D[2],*RepErr_H_3D[3]]
    
overall_repErr  = [*overall_repErr_C_2D, *overall_repErr_C_3D, *overall_repErr_H_2D, *overall_repErr_H_3D]     
overall_shoulder = [*RepErr_C_2D[0], *RepErr_C_3D[0], *RepErr_H_2D[0], *RepErr_H_3D[0]]
overall_elbow = [*RepErr_C_2D[1], *RepErr_C_3D[1], *RepErr_H_2D[1], *RepErr_H_3D[1]]
overall_wrist = [*RepErr_C_2D[2], *RepErr_C_3D[2], *RepErr_H_2D[2], *RepErr_H_3D[2]]
overall_hand = [*RepErr_C_2D[3], *RepErr_C_3D[3], *RepErr_H_2D[3], *RepErr_H_3D[3]]

overall_2D = [*overall_repErr_C_2D, *overall_repErr_H_2D]
overall_3D = [*overall_repErr_C_3D, *overall_repErr_H_3D]

overall_C = [*overall_repErr_C_2D, *overall_repErr_C_3D]
overall_H = [*overall_repErr_H_2D, *overall_repErr_H_3D]

print("overall overall mean reprojection error: " + str(np.mean (overall_repErr)))
print("overall overall std reprojection error: " + str(np.std (overall_repErr)))

print("overall shoulder mean reproj err: " + str(np.mean(overall_shoulder)))
print("overall shoulder std reproj err: " + str(np.std(overall_shoulder)))

print("overall elbow mean reproj err: " + str(np.mean(overall_elbow)))
print("overall elbow std reproj err: " + str(np.std(overall_elbow)))

print("overall wrist mean reproj err: " + str(np.mean(overall_wrist)))
print("overall wrist std reproj err: " + str(np.std(overall_wrist)))

print("overall hand mean reproj err: " + str(np.mean(overall_hand)))
print("overall hand std reproj err: " + str(np.std(overall_hand)))

print("overall RT2D mean reproj err: " + str(np.mean(overall_2D)))
print("overall RT2D std reproj err: " + str(np.std(overall_2D)))

print("overall RT3D mean reproj err: " + str(np.mean(overall_3D)))
print("overall RT3D std reproj err: " + str(np.std(overall_3D)))

print("overall Crackle mean reproj err: " + str(np.mean(overall_C)))
print("overall Crackle std reproj err: " + str(np.std(overall_C)))

print("overall Han mean reproj err: " + str(np.mean(overall_H)))
print("overall Han std reproj err: " + str(np.std(overall_H)))