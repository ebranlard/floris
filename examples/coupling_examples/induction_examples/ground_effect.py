"""
Plots induction velocity plots with and without ground effect.
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

minspeed = None #2
maxspeed = None #7

# Initialize floris interface object
fi = wfct.floris_interface.FlorisInterface(input_file)
fi.reinitialize_flow_field(layout_array=[[0],[0]])
fi2 = copy.deepcopy(fi)

# ===========================================================================================
# With Ground Effect
# ===========================================================================================

# Specify Blockage Model Parameters
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
Ind_Opts['Model'] ='VC'
Ind_Opts['nIter'] = 1
Ind_Opts['Ground'] = True
fi.IndOpts = Ind_Opts

# Calculate wake
fi.calculate_wake(Ind_Opts=Ind_Opts)
# Initialize the horizontal cut
hor_plane_g = fi.get_hor_plane(x_resolution = resolution.x1, y_resolution = resolution.x2)

# Create copy of horizontal plane to plot blockage velocities
block_hor_plane_g = copy.deepcopy(hor_plane_g)
# Set streamwise blockage velocity to u in hor_plane2 for plotting
block_hor_plane_g.df.u = block_hor_plane_g.df.u_ind

# Initialize horizontal cut at negative hub height
horplane_g = fi.get_hor_plane(height = -fi.floris.farm.turbines[0].hub_height, x_resolution = resolution.x1, y_resolution = resolution.x2)
# Create copy of horizontal plane to plot blockage velocities
# Set streamwise blockage velocity to u in horplane2 for plotting
block_horplane_g = copy.deepcopy(horplane_g)
block_horplane_g.df.u = block_horplane_g.df.u_ind

# Initialize Vertical Plane
y_plane_g = fi.get_y_plane(y_loc=0,x_resolution = resolution.x1, z_resolution = resolution.x2)
# Copy vertical-streamwise plane and set streamwise blockage velocity to u for plotting
ublock_y_plane_g = copy.deepcopy(y_plane_g)
ublock_y_plane_g.df.u = ublock_y_plane_g.df.u_ind
# Copy vertical-streamwise plane and set vertical blockage velocity to u for plotting
wblock_y_plane_g = copy.deepcopy(y_plane_g)
wblock_y_plane_g.df.u = wblock_y_plane_g.df.w_ind

# ==========================================================================================
# Ground Effect Removed
# ==========================================================================================
Ind_Opts['Ground'] = False
fi2.IndOpts = Ind_Opts

# Calculate wake
fi2.calculate_wake(Ind_Opts=Ind_Opts)
# Initialize the horizontal cut
hor_plane = fi2.get_hor_plane(x_resolution = resolution.x1, y_resolution = resolution.x2)
# Create copy of horizontal plane to plot blockage velocities
block_hor_plane = copy.deepcopy(hor_plane)
# Set streamwise blockage velocity to u in hor_plane2 for plotting
block_hor_plane.df.u = block_hor_plane.df.u_ind

# Initialize horizontal cut at negative hub height
horplane = fi2.get_hor_plane(height = -fi2.floris.farm.turbines[0].hub_height, x_resolution = resolution.x1, y_resolution = resolution.x2)
# Create copy of horizontal plane to plot blockage velocities at -hub height
block_horplane = copy.deepcopy(horplane)
# Set streamwise blockage velocity to u in horplane2 for plotting at -hub height
block_horplane.df.u = block_horplane.df.u_ind

# Initialize Vertical Plane
y_plane = fi2.get_y_plane(y_loc=0,x_resolution = resolution.x1, z_resolution = resolution.x2)
# Copy vertical-streamwise plane and set streamwise blockage velocity to u for plotting
ublock_y_plane = copy.deepcopy(y_plane)
ublock_y_plane.df.u = ublock_y_plane.df.u_ind
# Copy vertical-streamwise plane and set vertical blockage velocity to u for plotting
wblock_y_plane = copy.deepcopy(y_plane)
wblock_y_plane.df.u = wblock_y_plane.df.w_ind

# ==========================================================================================
# Plot Cut Planes
# ==========================================================================================
# Horizontal Plane Plot
fig, ax = plt.subplots(nrows=2,sharey=True) #sharex=True,
wfct.visualization.visualize_cut_plane(hor_plane_g, ax=ax[0],fig=fig,cbar=True)
ax[0].set_title('with Ground Effect')
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax[1],fig=fig,cbar=True)
ax[1].set_title('w/o Ground Effect')
fig.suptitle('Horizontal Plot at Hub Height (u)',fontsize = 14,fontweight="bold")
fig.tight_layout(rect = [0,0,1,0.95])


# Streamwise Induction only Horizontal Plots at hub height and negative hub height
fig, ax = plt.subplots(nrows=2,ncols=2)
wfct.visualization.visualize_cut_plane(block_hor_plane_g, ax=ax[0,0])
ax[0,0].set_title('Hub Height with Ground Effect')
wfct.visualization.visualize_cut_plane(block_hor_plane, ax=ax[0,1])
ax[0,1].set_title('Hub Height w/o Ground Effect')
wfct.visualization.visualize_cut_plane(block_horplane_g, ax=ax[1,0])
ax[1,0].set_title('[-] Hub Height with Ground Effect')
wfct.visualization.visualize_cut_plane(block_horplane, ax=ax[1,1])
ax[1,1].set_title('[-] Hub Height w/o Ground Effect')
fig.suptitle('Blockage Horizontal Plot (u_ind)', fontsize=14,fontweight="bold")
fig.tight_layout(rect = [0,0,1,0.95])

# Vertical-Streamwise Plot
fig, ax = plt.subplots(nrows=2,sharey=True) #sharex=True,
wfct.visualization.visualize_cut_plane(y_plane_g, ax=ax[0],fig=fig,cbar=True)
ax[0].set_title('with Ground Effect')
wfct.visualization.visualize_cut_plane(y_plane, ax=ax[1],fig=fig,cbar=True)
ax[1].set_title('w/o Ground Effect')
fig.suptitle('Vertical-Streamwise Velocity Plot at Y=0',fontsize = 14,fontweight="bold")
fig.tight_layout(rect = [0,0,1,0.95])

# Blockage only Vertical-Streamwise Velocity Plot (u_ind)
fig, ax = plt.subplots(nrows=2,sharey=True) #sharex=True,
wfct.visualization.visualize_cut_plane(ublock_y_plane_g, ax=ax[0],fig=fig,cbar=True)
ax[0].set_title('with Ground Effect')
wfct.visualization.visualize_cut_plane(ublock_y_plane, ax=ax[1],fig=fig,cbar=True)
ax[1].set_title('w/o Ground Effect')
fig.suptitle('Blockage Only Vertical-Streamwise Velocity Plot \n (u_ind) at Y=0',fontsize = 14,fontweight="bold")
fig.tight_layout(rect = [0,0,1,0.9])

# Blockage only Vertical-Streamwise Velocity Plot (w_ind)
fig, ax = plt.subplots(nrows=2,sharey=True) #sharex=True,
wfct.visualization.visualize_cut_plane(wblock_y_plane_g, ax=ax[0],fig=fig,cbar=True)
ax[0].set_title('with Ground Effect')
wfct.visualization.visualize_cut_plane(wblock_y_plane, ax=ax[1],fig=fig,cbar=True)
ax[1].set_title('w/o Ground Effect')
fig.suptitle('Blockage Only Vertical-Streamwise Velocity Plot \n (w_ind) at Y=0',fontsize = 14,fontweight="bold")
fig.tight_layout(rect = [0,0,1,0.9])

plt.show()