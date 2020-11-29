import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Local
from floris.tools.optimization.scipy.yaw import YawOptimization
import floris.tools as wfct
from floris.utilities import Vec3
import pickle

data = pickle.load(open('sowfa_data_set/sowfa_data_set.p','rb'))

layouts = []
for i in range(data.shape[0]):
    temp = data.iloc[i]
    layoutx = len(set(temp['layout_x']))
    layouty = len(set(temp['layout_y']))
    layouts.append(str(layouty)+'x'+str(layoutx))

data['Layouts'] = layouts

data.to_csv('SowfaDataSet.csv')


# data1 = data.iloc[0]
# test = len(set(data1['layout_x']))
# test_y = len(set(data1['layout_y']))

# strtest = str(test)+'x'+str(test_y)
# print(strtest)

# print(data1)