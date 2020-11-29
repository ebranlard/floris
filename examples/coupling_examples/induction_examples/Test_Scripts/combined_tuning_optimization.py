import numpy as np
import copy
import os
import pandas as pd
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
Optimizes wind speed for turbine powers for multiple turbine row turbine fields to match SOWFA Powers,
then optimizes turbulence intensity to match SOWFA Powers
"""
#Specify SOWFA powers in order of increasing increasing y locations
# sowfa_pow = np.array([1888.4,1818.6,1736.6])
sowfa_pow = np.array([1879,793.1,1724.2,767.4,1809.1,798.3]) # 3x2
# sowfa_pow = np.array([1877.1,786.4,784.1,1722,757.3,755.9,1807.2,791.1,787.6]) # 3x3
# sowfa_pow = np.array([1876.9,785,776.8,820,1721.8,755.6,745.6,791.9,1807,789.7,780.5,831.6]) # 3x4
# sowfa_pow = np.array([1808.2,778.8,766.7,807.9,817.6,1663.1,720,672,714,779.2,1906.3,802.2,800.6,857.2,900.7]) # 3x5
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
n = 2
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

speed_minval = 5
speed_maxval = 25
ti_multfactor = 10
ti_minval = 0.05 * ti_multfactor # Multiply ti parameter to be on same order as windspeed
ti_maxval = 0.2 * ti_multfactor #0.15
x0 = [speed_minval, ti_minval]
# opt_options = {'maxiter': 200,'disp': True,'iprint': 1,'ftol': 1e-8,'eps': 0.02}
opt_options = {'maxiter': 100,'disp': True,'iprint': 1,'ftol': 1e-7,'eps': 0.01}

sowfa_pow_frow = []
for i in range(m):
    sowfa_pow_frow.append(sowfa_pow[i*n])
sowfa_pow_frow = np.array(sowfa_pow_frow)

def opt_function(param):
    fi.reinitialize_flow_field(wind_speed=param[0],turbulence_intensity=param[1]/ti_multfactor)
    fi.calculate_wake()
    turb_powers = np.array(fi.get_turbine_power())/1000
    frow_pow = []
    for i in range(m):
        frow_pow.append(turb_powers[i*n])
    frow_pow = np.array(frow_pow)
    # pow = np.sum(np.abs((frow_pow - sowfa_pow_frow)/sowfa_pow_frow))*5 + np.sum(np.abs((turb_powers-sowfa_pow)/sowfa_pow)) * 100
    pow = np.sum(np.abs((turb_powers-sowfa_pow)/sowfa_pow))
    # pow = np.linalg.norm(turb_powers - sowfa_pow)/np.linalg.norm(sowfa_pow)
    return pow

plant = minimize(
    opt_function, # the objective function to be minimized
    x0, # Initial guess. Array of real elements the size of n where n in the number of independent variables
    method = 'SLSQP', # SLSQP
    bounds = [(speed_minval,speed_maxval),(ti_minval,ti_maxval)], # (min,max) pairs for each element in x
    options = opt_options, # maxiter:int and disp (set to true to print convergence messages)
)


fi.reinitialize_flow_field(wind_speed=plant.x[0],turbulence_intensity=plant.x[1]/ti_multfactor)
fi.calculate_wake()
print('===================================================')
print('U: ',plant.x[0])
print('TI: ',plant.x[1]/ti_multfactor)
print('===================================================')
print("Farm Power: ", fi.get_farm_power())
print('==============================================')
print("Turbine Powers:")
print('----------------------------------------------')
turbpow = fi.get_turbine_power()

turbine_powers = pd.DataFrame()
for i in range(m):
    temp = []
    for j in range(n):
        temp.append(turbpow[i*n+j]/1000)
    turbine_powers['Column_'+str(i)]=temp
print(turbine_powers)

print(time.strftime("%H:%M:%S", time.gmtime(time.time()-tstart)))