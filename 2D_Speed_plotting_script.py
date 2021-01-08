# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:18:05 2020

@author: dongq
"""

import pandas as pd 
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from peakutils.plot import plot as pplot
import peakutils
from scipy import signal
from scipy.interpolate import interp1d
from moviepy.editor import *
from scipy import stats
import copy
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns