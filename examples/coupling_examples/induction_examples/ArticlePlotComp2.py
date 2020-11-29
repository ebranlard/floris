"""
Plots induction velocity contour plots for specified wind farm layout.
"""

import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
# --- Local
import floris.tools as wfct
from floris.utilities import Vec3
import time
tstart = time.time()

input_file="../example_induction_input.json"
rcParams['contour.negative_linestyle'] = 'solid'
# --- Resolution Parameters
ny= 100
nx= 400
resolution=Vec3(nx, ny, 2)

minSpeed = 0.5
maxSpeed = 1.05
contoursf = np.sort([0.2,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1,1.01,1.4]) # Contour fill values
contours = np.sort([1.01,0.99,0.98,0.95,0.9,0.8,0.7,0.6,0.5]) # Countour lines

# Initialize floris interface object
fi = wfct.floris_interface.FlorisInterface(input_file)
D= fi.floris.farm.turbines[0].rotor_diameter
# fi.reinitialize_flow_field(layout_array=[[0],[0]])

# Set paramters for iteration test
sep = 5 # streamwise separation for turbines (*D)
sepy = 3 # spanwise spearation between turbines (*D)
# Creates a turbine field with n rows and m columns
n = 2
m = 1
layout_x = []
layout_y = []
for i in range(m):
    for j in range(n):
        layout_x.append(j*sep*D)
        layout_y.append(i*sepy*D)

fi.reinitialize_flow_field(layout_array=[layout_x,layout_y])

#====================================================================
fi2 = copy.deepcopy(fi)
# Read in induction options
Ind_Opts2 = fi2.floris.farm.flow_field.Ind_Opts
# Set induction to true to model blockage effect
Ind_Opts2['induction']=True
Ind_Opts2["Model"] = 'VC'
Ind_Opts2['nIter'] = 2

fi2.IndOpts = Ind_Opts2

# Calculate wake
fi2.calculate_wake(Ind_Opts=Ind_Opts2)

# Initialize the horizontal cut
hor_plane2 = fi2.get_hor_plane(Ind_Opts = Ind_Opts2, x_resolution = resolution.x1, y_resolution = resolution.x2)

# Plot and Show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane2, ax=ax)
wfct.visualization.plot_turbines_with_fi(ax,fi2)
ax.set_title('Floris with Induction')

turbct = []
turbVCct = []
for turbine in fi2.floris.farm.turbine_map.turbines:
    turbct.append(turbine.Ct)
    turbVCct.append(turbine.VC_Ct[0])
print('Time Elapsed: ', (time.time()-tstart))
#====================================================================

Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
Ind_Opts['Model'] ='VC'
Ind_Opts['nIter'] = 1
Ind_Opts['Ct_test'] = True
fi.IndOpts = Ind_Opts

fi.floris.farm.flow_field.set_ct(turbVCct)

fi.floris.farm.turbines[0].rotor_diameter = D

# Calculate wake
fi.calculate_wake(Ind_Opts=Ind_Opts)

bounds=[-3*D,14*D,-4*D/2,4*D/2,89,90] # xmin xmax .. zmin zmax
U0 = fi.floris.farm.initwind[0]
yaw = fi.floris.farm.turbines[0].yaw_pos

# Initialize the horizontal cut
hor_plane = fi.get_hor_plane(x_resolution = resolution.x1, y_resolution = resolution.x2, x_bounds=[-5*D,25*D])

hor_plane2 = copy.deepcopy(hor_plane)
# print(hor_plane2.df.head())

# Normalized streamwise induction with free stream
# hor_plane2.df.u = (hor_plane2.df.u_ind+U0) / U0
# Span wise induction (V direction)
# hor_plane2.df.u = hor_plane2.df.v_ind
# Normalized streamwise induction with free stream rotated in the direction normal to the rotor plane
# hor_plane2.df.u = np.sqrt(((hor_plane2.df.u_ind+U0)*np.cos(yaw))**2)/(U0*np.cos(yaw))
# Normalized streamwise induction, free stream and spanwise induction rotated in the direction normal to the rotor plane
hor_plane2.df.u = ((hor_plane2.df.u_ind + U0)*np.cos(yaw) + (hor_plane2.df.v_ind)*np.sin(yaw)) /(U0*np.cos(yaw))

# print(hor_plane2.df.head())
# # ===============================================================================================
# # Induction Only Plot
# # ===============================================================================================
# fig, ax = plt.subplots()
# # Plot induction
# wfct.visualization.visualize_cut_plane(hor_plane2, ax=ax, levels = contours,fig=fig,cbar=True)
# wfct.visualization.plot_turbines_with_fi(ax,fi)
# ax.set_title('Vortex Cylinder Velocity Field', fontsize = 16)
# ax.set_xlabel('x [-]', fontsize = 14)
# ax.set_ylabel('y [-]', fontsize = 14)

# ===============================================================================================
# Emmanuel Plot Formatting
# ===============================================================================================
# ---- ColorMap
def make_colormap(seq,values=None):
    """Return a LinearSegmentedColormap
    seq: RGB-tuples. 
    values: corresponding values (location betwen 0 and 1)
    """
    hasAlpha=len(seq[0])==4
    if hasAlpha:
        nComp=4
    else:
        nComp=3

    n=len(seq)
    if values is None:
        values=np.linspace(0,1,n)

    doubled     = list(itertools.chain.from_iterable(itertools.repeat(s, 2) for s in seq))
    doubled[0]  = (None,)* nComp
    doubled[-1] = (None,)* nComp
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha':[]}
    for i,v in enumerate(values):
        if hasAlpha:
            r1, g1, b1, a1 = doubled[2*i]
            r2, g2, b2, a2 = doubled[2*i + 1]
        else:
            r1, g1, b1 = doubled[2*i]
            r2, g2, b2 = doubled[2*i + 1]
        cdict['red'].append([v, r1, r2])
        cdict['green'].append([v, g1, g2])
        cdict['blue'].append([v, b1, b2])
        if hasAlpha:
            cdict['alpha'].append([v, a1, a2])
    #print(cdict)
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def get_cmap(minSpeed,maxSpeed,alpha=1):
    DS=0.001
    seq=[
    (63/255 ,63/255 ,153/255, alpha), # Dark Blue
    (159/255,159/255,204/255, alpha), # Light Blue
    (158/255,204/255,170/255, alpha), # Light Green
    (1,212/255,96/255, alpha),  # Light Orange
    (1,1,1,alpha),  # White
    (1,1,1,alpha),  # White
    (1,1,1,alpha),  # White
    (138/255 ,42/255 ,93/255,alpha), # DarkRed
    ]
    valuesOri=np.array([
    minSpeed,  # Dark Blue
    0.90,
    0.95,
    0.98,
    1.00-DS , # White
    1.00    , # White
    1.00+DS , # White
    maxSpeed         # DarkRed
    ])
    values=(valuesOri-min(valuesOri))/(max(valuesOri)-min(valuesOri))
    valuesOri=np.around(valuesOri[np.where(np.diff(valuesOri)>DS)[0]],2)
    cmap= make_colormap(seq,values=values)
    return cmap,np.concatenate((valuesOri,[maxSpeed]))

cmap,_=get_cmap(minSpeed,maxSpeed,alpha=0.6)

fig,ax = plt.subplots()
# im=ax.contourf(Z/R,X/R,Speed,levels=levelsContour, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)
im = ax.contourf(
    hor_plane2.df.x1.values.reshape(hor_plane2.resolution[1],hor_plane2.resolution[0]),
    hor_plane2.df.x2.values.reshape(hor_plane2.resolution[1],hor_plane2.resolution[0]),
    hor_plane2.df.u.values.reshape(hor_plane2.resolution[1],hor_plane2.resolution[0]),
    levels = contoursf, cmap=cmap,vmin=minSpeed,vmax=maxSpeed)

# cs=ax.contour(Z/R, X/R, Speed, levels=levelsLines, colors='k', linewidths=0.8, alpha=1.0, linestyles='solid')
cs = ax.contour(
    hor_plane2.df.x1.values.reshape(hor_plane2.resolution[1],hor_plane2.resolution[0]),
    hor_plane2.df.x2.values.reshape(hor_plane2.resolution[1],hor_plane2.resolution[0]),
    hor_plane2.df.u.values.reshape(hor_plane2.resolution[1],hor_plane2.resolution[0]),
    levels = contours, colors = 'k', linewidths = 0.8, alpha = 1.0, linestyles = 'solid')

wfct.visualization.plot_turbines_with_fi(ax,fi)
cb=fig.colorbar(im, fraction = 0.024, pad = 0.04)
cb.ax.tick_params(labelsize='large')
ax.set_title('Vortex Cylinder Velocity Field', fontsize = 22)
ax.set_xlabel('x [-]', fontsize = 18)
ax.set_ylabel('y [-]', fontsize = 18)
ax.set_aspect('equal')
# fig.savefig('../../../../../../Documents/Blockage Effect/Induction Velocity Field_2.0/Yawed_Decay_Ct_2/'+str(m)+'x'+str(n)+'induction_velocity_countour_yaw.png')

print('gamma_t: ', fi.floris.farm.turbines[0].gamma_t[0])
print('Time Elapsed: ', (time.time()-tstart))
plt.show()