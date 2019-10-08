# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# See read the https://floris.readthedocs.io for documentation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import floris.tools as wfct
from floris.utilities import Vec3

from pybra.clean_exceptions import *

# Initialize the FLORIS interface fi

minSpeed=1
maxSpeed=9
nx=232
ny=114

input_file="example_input.json"
resolution=Vec3(nx, ny, 3)


def get_HH_plane_vel(input_file, no_induction=True, resolution=Vec3(232, 114, 3)):
    fi = wfct.floris_utilities.FlorisInterface(input_file)
    # Calculate wake
    fi.calculate_wake(no_induction=no_induction)
    # # Initialize the horizontal cut
    hor_plane = wfct.cut_plane.HorPlane(
        fi.get_flow_data(resolution = resolution, no_induction=no_induction),
        fi.floris.farm.turbines[0].hub_height
    )
    u_mesh = hor_plane.u_mesh.reshape(hor_plane.resolution[1],
                                      hor_plane.resolution[0])
    return hor_plane.x1_lin,hor_plane.x2_lin, u_mesh

def plotPlane(x,y,u_mesh,ax,minSpeed=None,maxSpeed=None, cmap='coolwarm', levels=None, colors='w', linewidths=0.8, alpha=0.3):
    if minSpeed is None:
        minSpeed = u_mesh.min()
        maxSpeed = u_mesh.max()
    if not ax:
        fig, ax = plt.subplots()
    Zm = np.ma.masked_where(np.isnan(u_mesh), u_mesh)

    # Plot the cut-through
    im = ax.pcolormesh(x,y, Zm, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)

    rcParams['contour.negative_linestyle'] = 'solid'
    ct=ax.contour(x,y,Zm, levels=levels, colors=colors, linewidths=linewidths, alpha=alpha)
    ax.set_aspect('equal')

    return im


x0,y0,u0 = get_HH_plane_vel(input_file, True, resolution)
x1,y1,u1 = get_HH_plane_vel(input_file, False, resolution)



# Plot and show
fig = plt.figure()
ax0= plt.subplot(211)
ax1= plt.subplot(212)

im0 = plotPlane(x0,y0,u0,ax0,minSpeed=minSpeed,maxSpeed=maxSpeed)
im1 = plotPlane(x1,y1,u1,ax1,minSpeed=minSpeed,maxSpeed=maxSpeed)
fig.colorbar(im1, ax=ax1)
plt.show()

# 
