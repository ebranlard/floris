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
Plots horizontal plane with blockage effects. Returns velocity reduction and velocity/freestream velocity at several upstream locations.
Returns plots of streamwise velocity and velocity normal to yawed rotor plane plots along x=0.
"""

input_file="../../example_induction_input.json"
# Initialize floris interface object
fi = wfct.floris_interface.FlorisInterface(input_file)
D = fi.floris.farm.turbines[0].rotor_diameter
U0 = fi.floris.farm.initwind[0]
yaw = fi.floris.farm.turbines[0].yaw_pos

# Reinitialize flow field with new specified layout
fi.reinitialize_flow_field(layout_array=[[0],[0]])

# Read in induction options
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
# Set induction to true to model blockage effect
Ind_Opts['induction']=True
Ind_Opts["Model"] = 'VC'
Ind_Opts['nIter'] = 2

fi.IndOpts = Ind_Opts

fi.calculate_wake(Ind_Opts=Ind_Opts)

# Initialize the horizontal cut
hor_plane = fi.get_hor_plane(Ind_Opts = Ind_Opts, x_resolution=400, y_resolution=100)

# Plot and Show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
wfct.visualization.plot_turbines_with_fi(ax,fi)
ax.set_title('Floris with Induction')

df = fi.get_plane_of_points(
    x1_resolution = 25,
    x2_resolution = 1,
    x3_value = 90.0,
    x1_bounds = [-10*D,2*D],
    x2_bounds = [0,0],
    Ind_Opts=Ind_Opts
    )

# print(df)

print('==============================================================================')
print('Percent Reduction at -1D: %.3f%%' %(df.loc[df.x1 == -1*D].u_ind/U0*100),end='')
print('\t|\tu/U0 at -1D: %.3f%%' %(df.loc[df.x1 == -1*D].u/U0*100))
print('Percent Reduction at -2D: %.3f%%' %(df.loc[df.x1 == -2*D].u_ind/U0*100),end='')
print('\t|\tu/U0 at -2D: %.3f%%' %(df.loc[df.x1 == -2*D].u/U0*100))
print('Percent Reduction at -3D: %.3f%%' %(df.loc[df.x1 == -3*D].u_ind/U0*100),end='')
print('\t|\tu/U0 at -3D: %.3f%%' %(df.loc[df.x1 == -3*D].u/U0*100))
print('Percent Reduction at -5D: %.3f%%' %(df.loc[df.x1 == -5*D].u_ind/U0*100),end='')
print('\t|\tu/U0 at -5D: %.3f%%' %(df.loc[df.x1 == -5*D].u/U0*100))
print('Percent Reduction at -8D: %.3f%%' %(df.loc[df.x1 == -8*D].u_ind/U0*100),end='')
print('\t|\tu/U0 at -8D: %.3f%%' %(df.loc[df.x1 == -8*D].u/U0*100))
print('==============================================================================')

UInd = df.u_ind/U0*100

# Plot and Show
fig, ax = plt.subplots()
ax.grid(axis='both',zorder=1)
# ax.scatter(df.x1/D,UInd,zorder=2)
ax.plot(df.x1/D,UInd,zorder=2)
ax.set_title('Streamwise Induction', fontsize=18)
ax.set_xlabel('X/D [-]',fontsize=12)
ax.set_ylabel('Blockage Vel Reduction/U0 [%]',fontsize=12)
# ax.set_xticks(-np.arange(11))
# ax.set_yticks(np.arange(math.ceil(UInd.min()),math.ceil(UInd.max())+1))


# Normalized streamwise induction with free stream
# UInd = (df.u_ind+U0) / U0
# Normalized streamwise induction with free stream rotated in the direction normal to the rotor plane
# UInd = np.sqrt(((df.u_ind+U0)*np.cos(yaw))**2)/(U0*np.cos(yaw))
# Normalized streamwise induction, free stream and spanwise induction rotated in the direction normal to the rotor plane
UInd = ((df.u_ind + U0)*np.cos(yaw) + (df.v_ind)*np.sin(yaw)) /(U0*np.cos(yaw))

# Plot and Show
fig, ax = plt.subplots()
ax.grid(axis='both',zorder=1)
ax.plot(df.x1/D,UInd,zorder=2)
ax.set_title('Streamwise Induction', fontsize=18)
ax.set_xlabel('X/D [-]',fontsize=12)
ax.set_ylabel('u/U0 [%]',fontsize=12)
plt.show()