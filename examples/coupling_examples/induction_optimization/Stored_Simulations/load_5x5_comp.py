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

data = pickle.load(open('4x5_Layout_Comp.p','rb'))

hor_plane = data[0]
hor_plane2 = data[1]
base_power = data[2]
ind_power = data[3]
init_powers = data[4]
opt_powers = data[5]
turbine_vel = data[6]
indTurbine_vel = data[7]
yaw_angles = data[8]
yaw_ind_angles = data[9]

# Plot and show
fig, axs = plt.subplots(nrows = 2, ncols=1)
wfct.visualization.visualize_cut_plane(hor_plane, ax=axs[0], minSpeed=4, maxSpeed=8.5)
axs[0].set_title("Baseline FLORIS Yaw Optimization")

# Plot and show
wfct.visualization.visualize_cut_plane(hor_plane2, ax=axs[1], minSpeed=4, maxSpeed=8.5)
axs[1].set_title("FLORIS With Blockage Effect Yaw Optimization")
fig.tight_layout()

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
for i in range(len(init_powers)):
    print('Turbine %d: \t%.2f\t%.2f\t%.2f' %(i, init_powers[i],opt_powers[i],(opt_powers[i] - init_powers[i])))

print("================================================================")
print('Velocities Seen By Each Turbine:')
print("----------------------------------------------------------------")
print('\t\tNo Induction\tInduction\tDifference')
print("----------------------------------------------------------------")
for i in range(len(turbine_vel)):
    print('Turbine %d: \t%.2f\t\t%.2f\t\t%.2f' %(i,turbine_vel[i],indTurbine_vel[i],(indTurbine_vel[i]-turbine_vel[i])))

print("================================================================")
print('Optimized Yaw Angles (deg): ')
print("----------------------------------------------------------------")
print('\t\tNo Induction\tInduction\tDifference')
print("----------------------------------------------------------------")
for i in range(len(yaw_angles)):
    print("Turbine %d:\t%.2f\t\t%.2f\t\t%.2f" %(i,yaw_angles[i],yaw_ind_angles[i],(yaw_ind_angles[i]-yaw_angles[i])))

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
ax.legend(['Baseline','Blockage'], loc = 'lower left',fontsize=14)
fig.suptitle('Yaw Optimized Turbine Yaw Angles',fontsize=20)
fig.tight_layout(rect=(0,0,1,0.93))

plt.show()