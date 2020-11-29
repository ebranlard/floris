import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Local
from floris.tools.optimization.scipy.yaw import YawOptimization
import floris.tools as wfct
from floris.utilities import Vec3
import time
# -- Optimization
from scipy.optimize import minimize
from floris.tools.optimization.scipy import optimization

tstart = time.time()

"""
Optimizes wind speed for turbine powers for single row turbine fields to match SOWFA Powers
"""
#Specify SOWFA powers in order of increasing increasing y locations
sowfa_pow = np.array([1888.4,1818.6,1736.6])
# sowfa_pow = np.array([1879,793.1,1724.2,767.4,1809.1,798.3]) # 3x2


# --- Resolution Parameters
ny= 100
nx=ny*4
resolution=Vec3(nx, ny, 2)

input_file="../../example_induction_input.json"
# Initialize the floris interface
fi = wfct.floris_interface.FlorisInterface(input_file)

# Set paramters for iteration test
sep = 5 # streamwise separation for turbines (*D)
sepy = 3 # spanwise spearation between turbines (*D)
# Creates a turbine field with n rows and m columns
n = 1
m = 3

D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = []
layout_y = []
for i in range(m):
    for j in range(n):
        layout_x.append(j*sep*D)
        layout_y.append(i*sepy*D)

# Reinitialize flow field with new specified layout
fi.reinitialize_flow_field(layout_array=[layout_x,layout_y])

# Read in induction options from input file
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
Ind_Opts['induction'] = True
Ind_Opts['Model'] = 'VC'
Ind_Opts['nIter'] = 2
fi.IndOpts = Ind_Opts

# Calculate wake
fi.calculate_wake()

# Store value for total wind farm power
power_initial = fi.get_turbine_power()

print('Initial powers w/ TI = 0.09, U  = 8.38: ', power_initial)

minval = 5
maxval = 25
x0 = minval
opt_options = {'maxiter': 100,'disp': True,'iprint': 1,'ftol': 1e-7,'eps': 0.01}

def speed_function(x):
    fi.reinitialize_flow_field(wind_speed=x)
    # print(fi.floris.farm.wind_map.input_speed[0])
    fi.calculate_wake()
    # print('Turbine powers at %.2f m/s: ' %x, fi.get_turbine_power())
    pow = np.abs(np.sum(np.array(fi.get_turbine_power())/1000 - sowfa_pow))
    return pow

plant = minimize(
    speed_function, # the objective function to be minimized
    x0, # Initial guess. Array of real elements the size of n where n in the number of independent variables
    method = 'SLSQP', # SLSQP
    bounds = [(minval,maxval) for _ in range(1)], # (min,max) pairs for each element in x
    options = opt_options, # maxiter:int and disp (set to true to print convergence messages)
)

print('===================================================')
print('Optimal Wind Speed To Match SOWFA Data')
print('---------------------------------------------------')
print(round(plant.x[0],4))
print('---------------------------------------------------')

# fi.reinitialize_flow_field(wind_speed=round(plant.x[0],2))
# fi.calculate_wake()
# print('Turbine powers at %.2f m/s: ' %round(plant.x[0],2), np.array(fi.get_turbine_power())/1000)

print(time.strftime("%H:%M:%S", time.gmtime(time.time()-tstart)))