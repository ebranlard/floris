"""
Plots velocity plots for the flow field with and without induction and induction only.
"""

import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Local
import floris.tools as wfct
from floris.utilities import Vec3

input_file="../example_induction_input.json"
# --- Resolution Parameters
ny= 100
nx=ny*4
resolution=Vec3(nx, ny, 2)

crossplot = True
vertplot = False

minspeed = None #2
maxspeed = None #7
contours = np.sort([1.01,0.99,0.98,0.95,0.9,0.8,0.7,0.6,0.5])

# Initialize floris interface object
fi = wfct.floris_interface.FlorisInterface(input_file)

# Set paramters for iteration test
sep = 5 # streamwise separation for turbines (*D)
sepy = 3 # spanwise spearation between turbines (*D)
# Creates a turbine field with n rows and m columns
n = 3
m = 2

D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = []
layout_y = []
for i in range(m):
    for j in range(n):
        layout_x.append(j*sep*D)
        layout_y.append(i*sepy*D)

# Reinitialize flow field with new specified layout
fi.reinitialize_flow_field(layout_array=[layout_x,layout_y])

Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
Ind_Opts['Model'] ='VC'
Ind_Opts['nIter'] = 2
Ind_Opts['Ground'] = True
fi.IndOpts = Ind_Opts

# Calculate wake
fi.calculate_wake()
# print('Ind Opts:', fi.IndOpts)
# Initialize the horizontal cut
hor_plane = fi.get_hor_plane(x_resolution = resolution.x1, y_resolution = resolution.x2)

hor_plane2 = copy.deepcopy(hor_plane)
hor_plane2.df.u = hor_plane2.df.u_ind

hor_plane3 = copy.deepcopy(hor_plane)
hor_plane3.df.u = hor_plane3.df.u - hor_plane3.df.u_ind
# print(hor_plane3.df.head())
# U0 = fi.floris.farm.initwind[0]
# yaw = fi.floris.farm.turbines[0].yaw_pos
# hor_plane2.df.u = ( (hor_plane2.df.u_ind + U0)*np.cos(yaw) + (hor_plane2.df.v_ind)*np.sin(yaw) ) /U0*np.cos(yaw)

# # ===============================================================================================
# Vertical Plot
if vertplot:
    y_plane = fi.get_y_plane(y_loc=0,x_resolution = resolution.x1, z_resolution = resolution.x2)
    y_plane2 = copy.deepcopy(y_plane)
    y_plane2.df.u = y_plane2.df.u_ind

    fig, ax = plt.subplots()
    # Plot vertical plane of streamwise induction
    wfct.visualization.visualize_cut_plane(y_plane2, ax=ax)#, minSpeed = -0.25, maxSpeed = 0.25)
    ax.set_title('Induction Only Vertical Plot',fontsize=16)
    ax.set_xlabel('x [-]', fontsize = 14)
    ax.set_ylabel('z [-]', fontsize = 14)

# ===============================================================================================
# Cross Plot
x_loc = [1400,800,1500]
if crossplot:
    for i in range(len(x_loc)):
        x_plane = fi.get_cross_plane(x_loc=x_loc[i],y_resolution = resolution.x2, z_resolution = resolution.x2)
        x_plane2 = copy.deepcopy(x_plane)
        x_plane2.df.u = x_plane2.df.u_ind

        fig, ax = plt.subplots()
        # Plot cross plane of streamwise induction
        wfct.visualization.visualize_cut_plane(x_plane, ax=ax, fig=fig, cbar=True)#, minSpeed = -0.25, maxSpeed = 0.25)
        ax.set_title('Streamwise Velocity Cross Plane at X = '+str(x_loc[i]),fontsize=16)
        ax.set_xlabel('y [-]', fontsize = 14)
        ax.set_ylabel('z [-]', fontsize = 14)

        fig, ax = plt.subplots()
        # Plot cross plane of streamwise induction
        wfct.visualization.visualize_cut_plane(x_plane2, ax=ax, fig=fig, cbar=True)#, minSpeed = -0.25, maxSpeed = 0.25)
        ax.set_title('Streamwise Induction Velocity Cross Plane at X = '+str(x_loc[i]),fontsize=16)
        ax.set_xlabel('y [-]', fontsize = 14)
        ax.set_ylabel('z [-]', fontsize = 14)
# ===============================================================================================
fig, ax = plt.subplots()
# Plot horizontal plane of streamwise induction
wfct.visualization.visualize_cut_plane(hor_plane2, ax=ax, fig=fig, cbar=True)#, minSpeed = -0.25, maxSpeed = 0.25)
wfct.visualization.plot_turbines_with_fi(ax,fi)
ax.set_title('Streamwise Induction Only', fontsize=22)
ax.set_xlabel('x [-]', fontsize = 16)
ax.set_ylabel('y [-]', fontsize = 16)
ax.set_xlim([-250,200])
ax.set_ylim([-150,150])
# ===============================================================================================

fig, ax = plt.subplots(nrows = 3, sharex=True, sharey=True)
# Plot flow field without induction
wfct.visualization.visualize_cut_plane(hor_plane3, ax=ax[0], minSpeed = minspeed, maxSpeed = maxspeed, fig=fig, cbar=True)
wfct.visualization.plot_turbines_with_fi(ax[0],fi)
ax[0].set_title('Flow Field without Induction', fontsize=14)

# Plot induction
wfct.visualization.visualize_cut_plane(hor_plane2, ax=ax[1], minSpeed = minspeed, maxSpeed = maxspeed, fig=fig, cbar=True)
wfct.visualization.plot_turbines_with_fi(ax[1],fi)
ax[1].set_title('Induction Only', fontsize=14)
fig.tight_layout()

# Plot flow field with induction
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax[2], minSpeed = minspeed, maxSpeed = maxspeed, fig=fig, cbar=True)
wfct.visualization.plot_turbines_with_fi(ax[2],fi)
ax[2].set_title('Flow Field with Induction', fontsize=14)
if crossplot:
    for i in x_loc:
        ax[2].plot(np.ones(40)*i,np.linspace(hor_plane.df.x2.min(),hor_plane.df.x2.max(),40),linewidth = 0.5)
plt.show()