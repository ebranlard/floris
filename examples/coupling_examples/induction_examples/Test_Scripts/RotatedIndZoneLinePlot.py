import numpy as np
import copy
import os
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Local
import floris.tools as wfct
from floris.utilities import Vec3
"""
Plots horizontal plane with blockage effects. Returns farm and turbine powers.
"""

input_file="../../example_induction_input.json"
# Initialize floris interface object
fi = wfct.floris_interface.FlorisInterface(input_file)
D = fi.floris.farm.turbines[0].rotor_diameter
U0 = fi.floris.farm.initwind[0]
yaw = fi.floris.farm.turbines[0].yaw_pos

print("Yaw: ", yaw)

# Read in induction options
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
# Set induction to true to model blockage effect
Ind_Opts['induction']=True
Ind_Opts["Model"] = 'VC'
Ind_Opts['nIter'] = 2
fi.IndOpts = Ind_Opts

# Set paramters for iteration test
sep = 5 # streamwise separation for turbines (*D)
sepy = 3 # spanwise spearation between turbines (*D)
# Creates a turbine field with n rows and m columns
n = [1,2,3,4,5]
m = [2,2,2,2,2]

Readouts = True
line = []
for i in range(len(n)):
    layout_x = []
    layout_y = []
    for j in range(m[i]):
        for k in range(n[i]):
            layout_x.append(k*sep*D)
            layout_y.append(j*sepy*D)

    fi.reinitialize_flow_field(layout_array=[layout_x,layout_y])
    fi.calculate_wake(Ind_Opts=Ind_Opts)

    if Readouts:
        df = fi.get_plane_of_points(
            # x1_resolution = 28,
            # x2_resolution = 1,
            # x3_value = 90.0,
            # x1_bounds = [-10*D,3.5*D],
            # x2_bounds = [0,0],
            # Ind_Opts=Ind_Opts
            # x1_resolution = 39,
            # x2_resolution = 1,
            # x3_value = 90.0,
            # x1_bounds = [-10*D,9*D],
            # x2_bounds = [0,0],
            # Ind_Opts=Ind_Opts
            x1_resolution = 67,#265,#67,
            x2_resolution = 1,
            x3_value = 90.0,
            x1_bounds = [-8*D,25*D],
            x2_bounds = [0,0],
            Ind_Opts=Ind_Opts
            )
    else:
        df = fi.get_plane_of_points(
            x1_resolution = 13,
            x2_resolution = 1,
            x3_value = 90.0,
            x1_bounds = [-8*D,-2*D],
            x2_bounds = [0,0],
            Ind_Opts=Ind_Opts
            # x1_resolution = 9,
            # x2_resolution = 1,
            # x3_value = 90.0,
            # x1_bounds = [0*D,4*D],
            # x2_bounds = [0,0],
            # Ind_Opts=Ind_Opts
            )
    line.append(df)

# print(df)

if Readouts:
    print('=================================================')
    print('u/U0')
    print('-------------------------------------------------')
    print('Loc | ', end='')
    for i in range(len(m)):
        print('\t%s' %(str(m[i])+'x'+str(n[i])), end='')
    print('')
    print('-------------------------------------------------')
    print('-8D: ', end='')
    for i in range(len(m)):
        print('\t%.2f%%' %((((line[i].loc[line[i].x1 == -8*D].u_ind+U0)*np.cos(yaw) + line[i].loc[line[i].x1 == -8*D].v_ind*np.sin(yaw))/(U0*np.cos(yaw)))*100),end='')
    print()
    print('-5D: ', end='')
    for i in range(len(m)):
        print('\t%.2f%%' %((((line[i].loc[line[i].x1 == -5*D].u_ind+U0)*np.cos(yaw) + line[i].loc[line[i].x1 == -5*D].v_ind*np.sin(yaw))/(U0*np.cos(yaw)))*100),end='')
    print()
    print('-3D: ', end='')
    for i in range(len(m)):
        print('\t%.2f%%' %((((line[i].loc[line[i].x1 == -3*D].u_ind+U0)*np.cos(yaw) + line[i].loc[line[i].x1 == -3*D].v_ind*np.sin(yaw))/(U0*np.cos(yaw)))*100),end='')
    print()
    print('-2D: ', end='')
    for i in range(len(m)):
        print('\t%.2f%%' %((((line[i].loc[line[i].x1 == -2*D].u_ind+U0)*np.cos(yaw) + line[i].loc[line[i].x1 == -2*D].v_ind*np.sin(yaw))/(U0*np.cos(yaw)))*100),end='')
    print()
    print('-1D: ', end='')
    for i in range(len(m)):
        print('\t%.2f%%' %((((line[i].loc[line[i].x1 == -1*D].u_ind+U0)*np.cos(yaw) + line[i].loc[line[i].x1 == -1*D].v_ind*np.sin(yaw))/(U0*np.cos(yaw)))*100),end='')
    print()
    print(' 1D: ', end='')
    for i in range(len(m)):
        print('\t%.2f%%' %((((line[i].loc[line[i].x1 == 1*D].u_ind+U0)*np.cos(yaw) + line[i].loc[line[i].x1 == 1*D].v_ind*np.sin(yaw))/(U0*np.cos(yaw)))*100),end='')
    print()
    print(' 3D: ', end='')
    for i in range(len(m)):
        print('\t%.2f%%' %((((line[i].loc[line[i].x1 == 3*D].u_ind+U0)*np.cos(yaw) + line[i].loc[line[i].x1 == 3*D].v_ind*np.sin(yaw))/(U0*np.cos(yaw)))*100),end='')
    print()
    
# Plot and Show
fig, ax = plt.subplots()
for i in range(len(n)):
    UInd = ((line[i].u_ind + U0)*np.cos(yaw) + (line[i].v_ind)*np.sin(yaw)) /(U0*np.cos(yaw))
    # ax.scatter(line[i].x1/D,UInd,zorder=2)
    ax.plot(line[i].x1/D,UInd,zorder=2)

ax.plot(line[i].x1/D,np.ones(len(line[i].x1))*0.98,linestyle='--')
# ax.grid(axis='both',zorder=1)
ax.set_title('Velocity Normal To the Rotor at Y=0', fontsize=18)
ax.set_xlabel('X/D [-]',fontsize=12)
ax.set_ylabel('u/U0 (normal to the rotor) [%]',fontsize=12)
ax.legend([str(m[i])+'x'+str(n[i]) for i in range(len(n))])
fig.tight_layout()
plt.show()