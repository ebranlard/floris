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

data = pickle.load(open('5x5_Layout.p','rb'))

hor_plane = data[0]
hor_plane_opt = data[1]
power_initial = data[2]
power_opt = data[3]
init_powers = data[4]
opt_powers = data[5]
turbine_vel = data[6]
optTurbine_vel = data[7]
yaw_angles = data[8]

minspeed = -0.4
maxspeed = 8.2

# --- Plot and show
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(12.0, 6.0))
wfct.visualization.visualize_cut_plane(hor_plane, ax=axs[0], minSpeed=minspeed, maxSpeed=maxspeed)
axs[0].set_title("Baseline Case", fontsize=16)
# axs[0].set_xlim(right = 2000)

wfct.visualization.visualize_cut_plane(hor_plane_opt, ax=axs[1], minSpeed=minspeed, maxSpeed=maxspeed,fig=fig,cbar=True)
axs[1].set_title("Optimized Case", fontsize=16)
# axs[1].set_xlim(right = 2000)

print("==========================================")
print("optimized yaw angles = ")
for i in range(len(yaw_angles)):
    print("Turbine ", i, "=", yaw_angles[i], " deg")

print("==========================================")
print('Original Power:', power_initial)
print('Optimized Power:', power_opt)

print("==========================================")
print("Total Power Gain = %.1f%%" %(100.0 * (power_opt - power_initial) / power_initial))

print("==========================================")
print('Turbine Powers:')
print("------------------------------------------")
print('\t\tBaseline\tOptimized')
print("------------------------------------------")
for i in range(len(init_powers)):
    print('Turbine %d: \t%.2f\t%.2f' %(i, init_powers[i],opt_powers[i]))

print("==========================================")
print('Velocities Seen By Each Turbine:')
print("------------------------------------------")
print('\t\tBaseline\tOptimized')
print("------------------------------------------")
for i in range(len(turbine_vel)):
    print('Turbine %d: \t%.2f\t\t%.2f' %(i,turbine_vel[i],optTurbine_vel[i]))

plt.show()