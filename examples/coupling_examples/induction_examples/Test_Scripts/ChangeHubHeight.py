import numpy as np
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Local
import floris.tools as wfct
from floris.utilities import Vec3
import time
tstart = time.time()
"""
Plots horizontal plane with blockage effects. Returns farm and turbine powers.
"""

input_file="../../example_induction_input.json"
# Initialize floris interface object
fi = wfct.floris_interface.FlorisInterface(input_file)

# Set paramters for iteration test
sep = 5 # streamwise separation for turbines (*D)
sepy = 3 # spanwise spearation between turbines (*D)
# Creates a turbine field with n rows and m columns
n = 4
m = 4

D = fi.floris.farm.turbines[0].rotor_diameter
layout_x = []
layout_y = []
for i in range(m):
    for j in range(n):
        layout_x.append(j*sep*D)
        layout_y.append(i*sepy*D)

# Reinitialize flow field with new specified layout
fi.reinitialize_flow_field(layout_array=[layout_x,layout_y])

# Read in induction options
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
# Set induction to true to model blockage effect
Ind_Opts['induction']=False
Ind_Opts["Model"] = 'VC'
fi.IndOpts = Ind_Opts

fi2 = copy.deepcopy(fi)
# Change turbine 1 to have a hub height of 100
fi2.change_turbine([5,6,9,10], {"hub_height": fi2.floris.farm.turbines[0].hub_height+50})

# Calculate wake
fi.calculate_wake(Ind_Opts=Ind_Opts)
fi2.calculate_wake(Ind_Opts=Ind_Opts)

# Initialize the horizontal cut
hor_plane = fi.get_hor_plane(Ind_Opts = Ind_Opts, x_resolution=400, y_resolution=100)

horplane = fi2.get_hor_plane(height = fi2.floris.farm.turbines[0].hub_height, Ind_Opts=Ind_Opts, x_resolution=400, y_resolution=100)
horplane2 = fi2.get_hor_plane(height = fi2.floris.farm.turbines[0].hub_height+50, Ind_Opts=Ind_Opts, x_resolution=400, y_resolution=100)

y_plane = fi.get_y_plane(y_loc=0,x_resolution = 400, z_resolution = 100,z_bounds=[0,250])
yplane = fi2.get_y_plane(y_loc=0,x_resolution = 400, z_resolution = 100,z_bounds=[0,250])

print('==============================================')
print("Farm Power:")
print('----------------------------------------------')
print('Original: ', fi.get_farm_power())
print('Altered: ', fi2.get_farm_power())
print('==============================================')
print("Turbine Powers: Original\tAltered")
print('----------------------------------------------')
turbpow = fi.get_turbine_power()
alt_turbpow = fi2.get_turbine_power()
for i in range(len(turbpow)):
    print('Turbine %d: \t%.2f\t\t%.2f'%(i, turbpow[i]/1000,alt_turbpow[i]/1000))
print('==============================================')

# Plot and Show Baseline Configuration
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax, fig=fig, cbar=True)
wfct.visualization.plot_turbines_with_fi(ax,fi)
ax.set_title('Original Configuration')

# Plot and Show Altered Configuration
fig, ax = plt.subplots(nrows = 2,sharey = True,sharex = True)
wfct.visualization.visualize_cut_plane(horplane, ax=ax[0], fig=fig, cbar=True)
wfct.visualization.plot_turbines_with_fi(ax[0],fi2)
ax[0].set_title('Hub Height of Turbine[0]')
wfct.visualization.visualize_cut_plane(horplane2, ax=ax[1], fig=fig, cbar=True)
wfct.visualization.plot_turbines_with_fi(ax[1],fi2)
ax[1].set_title('Hub Height at Increased Height')
fig.suptitle('Altered Configuration',fontsize = 14)
fig.tight_layout(rect = [0,0,1,0.95])

# Plot and Show Vertical Plots
fig, ax = plt.subplots(nrows = 2,sharey = True,sharex = True)
wfct.visualization.visualize_cut_plane(y_plane, ax=ax[0], fig=fig, cbar=True)
ax[0].set_title('Original Configuration')
wfct.visualization.visualize_cut_plane(yplane, ax=ax[1], fig=fig, cbar=True)
ax[1].set_title('Altered Configuration')
fig.suptitle('Vertical Plot (Y=0)',fontsize = 14)
fig.tight_layout(rect = [0,0,1,0.95])

print('Time Elapsed: ', (time.time()-tstart))
plt.show()