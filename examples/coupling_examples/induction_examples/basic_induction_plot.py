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

input_file="../example_induction_input.json"
# Initialize floris interface object
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

# Read in induction options
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
# Set induction to true to model blockage effect
Ind_Opts['induction']=True
Ind_Opts["Model"] = 'VC'

fi.IndOpts = Ind_Opts

# Calculate wake
fi.calculate_wake(Ind_Opts=Ind_Opts)

# Initialize the horizontal cut
hor_plane = fi.get_hor_plane(Ind_Opts = Ind_Opts, x_resolution=400, y_resolution=100)

print('==============================================')
print("Farm Power: ", fi.get_farm_power())
print('==============================================')
print("Turbine Powers:")
print('----------------------------------------------')
turbpow = fi.get_turbine_power()
# for i in range(len(turbpow)):
#     print('Turbine %d: '%i, turbpow[i]/1000)
# print('==============================================')
turbine_powers = pd.DataFrame()
for i in range(m):
    temp = []
    for j in range(n):
        temp.append(turbpow[i*n+j]/1000)
    turbine_powers['Column_'+str(i)]=temp
print(turbine_powers)

# Plot and Show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax, fig=fig, cbar=True)
wfct.visualization.plot_turbines_with_fi(ax,fi)
ax.set_title('Floris with Induction')

print('----------------------------------------------')
print('Time Elapsed: ', (time.time()-tstart))
plt.show()