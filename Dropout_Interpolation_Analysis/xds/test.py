# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:22:04 2019

@author: dongq
"""
import numpy as np
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2
#on a grid in [0, 1]x[0, 1]


grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
#but we only know its values at 1000 data points:


points = np.random.rand(1000, 2)
values = func(points[:,0], points[:,1])
#This can be done with griddata – below we try out all of the interpolation methods:


from scipy.interpolate import griddata
grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')