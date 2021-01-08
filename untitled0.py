# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:59:04 2020

@author: dongq
"""
import pickle
import ruamel
#%%
file_dir = r'C:\Users\dongq\DeepLabCut\Han-Qiwei-2020-07-02\training-datasets\iteration-0\UnaugmentedDataSet_HanJul2\Documentation_data-Han_95shuffle1.pickle'
objects = []
with (open(file_dir, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break