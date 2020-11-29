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
import time
tstart = time.time()
"""
Compares the optimization of a wind farm with and without modelling blockage effects in the 
induction zone of turbines.
"""

input_file="../OptLayout_2x3.json"
# Initialize the floris interface
fi = wfct.floris_interface.FlorisInterface(input_file)

# Set paramters for iteration test
sep = 5 # streamwise separation for turbines (*D)
sepy = 3 # spanwise spearation between turbines (*D)
# Creates a turbine field with n rows and m columns
n = 5
m = 1

D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = []
layout_y = []
for i in range(m):
    for j in range(n):
        layout_x.append(j*sep*D)
        layout_y.append(i*sepy*D)

# Reinitialize flow field with new specified layout
fi.reinitialize_flow_field(layout_array=[layout_x,layout_y])

# Make a copy for floris interface with induction
fi_ind = copy.deepcopy(fi)

# Calculate wake and get horizontal plane at turbine height for original farm field
fi.calculate_wake()

# Set bounds for allowable wake steering
min_yaw = 0.0
max_yaw = 25.0

# Instantiate the Optimization object
yaw_opt = YawOptimization(fi, minimum_yaw_angle=min_yaw, maximum_yaw_angle=max_yaw)

# Perform optimization
yaw_angles = yaw_opt.optimize()

# Calculate wake and get horizontal plane at turbine height for yaw optimized farm field
fi.calculate_wake(yaw_angles=yaw_angles)

# Initialize the horizontal cut
hor_plane = fi.get_hor_plane(x_resolution=400, y_resolution=100)

# Plot and show
fig, axs = plt.subplots(nrows = 2, ncols=1)
wfct.visualization.visualize_cut_plane(hor_plane, ax=axs[0], minSpeed=4, maxSpeed=8.5)
axs[0].set_title("Baseline")

base_power = fi.get_farm_power()
# =============================================================================
# Induction
# =============================================================================
# Read in induction options from flow field class
Ind_Opts = fi_ind.floris.farm.flow_field.Ind_Opts

# Sets induction to true
Ind_Opts['induction'] = True
Ind_Opts['Model'] = 'VC'
Ind_Opts['nIter'] = 2
# fi_ind.floris.farm.flow_field.Ind_Opts = Ind_Opts
fi_ind.IndOpts = Ind_Opts

fi_ind.calculate_wake()

# Instantiate the Optimization object
yaw_ind_opt = YawOptimization(fi_ind, minimum_yaw_angle=min_yaw, maximum_yaw_angle=max_yaw)

# Perform optimization
yaw_ind_angles = yaw_ind_opt.optimize()

# Calculate wake and get horizontal plane at turbine height for yaw optimized farm field
fi_ind.calculate_wake(yaw_angles=yaw_ind_angles)

# Initialize the horizontal cut
hor_plane2 = fi_ind.get_hor_plane(x_resolution=400, y_resolution=100)

# Plot and show
wfct.visualization.visualize_cut_plane(hor_plane2, ax=axs[1], minSpeed=4, maxSpeed=8.5)
axs[1].set_title("FLORIS With Blockage Effects")
fig.tight_layout()

ind_power = fi_ind.get_farm_power()

print("================================================================")
print("Farm Powers: ")
print("----------------------------------------------------------------")
print('No Induction Power:', base_power)
print('Induction Power:', ind_power)

print("================================================================")
print("Total Power Change = %.1f%%" %(100.0 * (ind_power - base_power) / base_power))

print("================================================================")
print('Turbine Powers:')
print("----------------------------------------------------------------")
print('\t\tNo Induction\tInduction\tDifference')
print("----------------------------------------------------------------")
init_powers = fi.get_turbine_power()
opt_powers = fi_ind.get_turbine_power()
for i in range(len(init_powers)):
    print('Turbine %d: \t%.2f\t%.2f\t%.2f' %(i, init_powers[i],opt_powers[i],(opt_powers[i] - init_powers[i])))

print("================================================================")
print('Velocities Seen By Each Turbine:')
print("----------------------------------------------------------------")
print('\t\tNo Induction\tInduction\tDifference')
print("----------------------------------------------------------------")
turbine_vel = []
indTurbine_vel = []
for turbine in fi.floris.farm.turbine_map.turbines:
    turbine_vel.append(turbine.average_velocity)
for turbine in fi_ind.floris.farm.turbine_map.turbines:
    indTurbine_vel.append(turbine.average_velocity)

for i in range(len(turbine_vel)):
    print('Turbine %d: \t%.2f\t\t%.2f\t\t%.2f' %(i,turbine_vel[i],indTurbine_vel[i],(indTurbine_vel[i]-turbine_vel[i])))

print("================================================================")
print('Optimized Yaw Angles (deg): ')
print("----------------------------------------------------------------")
print('\t\tNo Induction\tInduction\tDifference')
print("----------------------------------------------------------------")
for i in range(len(yaw_angles)):
    print("Turbine %d:\t%.2f\t\t%.2f\t\t%.2f" %(i,yaw_angles[i],yaw_ind_angles[i],(yaw_ind_angles[i]-yaw_angles[i])))

# # Save pickle file for longer simulations (i.e. 5x5)
# pickle.dump([hor_plane,hor_plane2,base_power,ind_power,init_powers,opt_powers,turbine_vel,indTurbine_vel,yaw_angles,yaw_ind_angles],open('Stored_Simulations/5x5_Layout_Comp2.p','wb'))

fig,ax = plt.subplots()
x = np.arange(len(yaw_angles))
ax.scatter(x,yaw_angles,zorder=3)
ax.scatter(x,yaw_ind_angles,marker='s',zorder=2)
ax.plot(x,yaw_angles,zorder=1)
ax.plot(x,yaw_ind_angles,zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(['T'+str(i) for i in x],fontsize=14)
ax.tick_params(axis='y',labelsize=14)
ax.set_ylabel('Yaw Angles (deg)',fontsize=16)
ax.set_xlabel('Turbine',fontsize=16)
ax.legend(['Baseline','Blockage'],loc = 'lower left',fontsize=14)
fig.suptitle('%dx%d Yaw Optimized Turbine Yaw Angles' %(m,n),fontsize=20)
fig.tight_layout(rect=(0,0,1,0.93))

print(time.strftime("%H:%M:%S", time.gmtime(time.time()-tstart)))
plt.show()