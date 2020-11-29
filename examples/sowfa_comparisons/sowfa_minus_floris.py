# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import numpy as np
import matplotlib.pyplot as plt

import floris.tools as wfct
import floris.tools.cut_plane as cp
import floris.tools.visualization as vis


# Define a minspeed and maxspeed to use across visualiztions
minspeed = -7 #4.0
maxspeed = 5.5 #8.5

# Load the SOWFA case in
si = wfct.sowfa_utilities.SowfaInterface("sowfa_example")
sowfa_flow_data = si.flow_data

# Load the FLORIS case in
fi = wfct.floris_interface.FlorisInterface("../example_input.json")
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
Ind_Opts['induction'] = True
Ind_Opts['Model'] = 'VC'
fi.IndOpts = Ind_Opts
fi.calculate_wake()

# Set the relevant FLORIS parameters to equal the SOWFA case
fi.reinitialize_flow_field(
    wind_speed=[si.precursor_wind_speed],
    wind_direction=[si.precursor_wind_dir],
    layout_array=(si.layout_x, si.layout_y),
)

# Set the yaw angles
fi.calculate_wake(yaw_angles=si.yaw_angles)

# Show projected and unprojected cut planes
x_loc = 600

# cut_plane_sowfa = si.get_cross_plane(x_loc)
cut_plane_sowfa = si.get_hor_plane(90.0)
cut_plane_sowfa.df.u = cut_plane_sowfa.df.u-8.0
# cut_plane_floris = fi.get_cross_plane(x_loc)
cut_plane_floris = fi.get_hor_plane(fi.floris.farm.turbines[0].hub_height)
cut_plane_floris.df.u = cut_plane_floris.df.u - 8.0
cut_plane_floris_project = cp.project_onto(cut_plane_floris, cut_plane_sowfa)
cut_plane_difference = cp.subtract(cut_plane_sowfa, cut_plane_floris_project)

print('SOWFA Cut Plane: \n', cut_plane_sowfa.df.head())
print('Floris Cut Plane: \n', cut_plane_floris_project.df.head())
print('Cut Plane Difference: \n', cut_plane_difference.df.head())

# print('SOWFA\n\tMax: %.4f, Min: %.4f' %(max(cut_plane_sowfa.df.u),min(cut_plane_sowfa.df.u)))
# print('Floris\n\tMax: %.4f, Min: %.4f' %(max(cut_plane_floris.df.u),min(cut_plane_floris.df.u)))
# print('Floris Project\n\tMax: %.4f, Min: %.4f' %(max(cut_plane_floris_project.df.u),min(cut_plane_floris_project.df.u)))
# print('Difference\n\tMax: %.4f, Min: %.4f' %(max(cut_plane_difference.df.u),min(cut_plane_difference.df.u)))

fig, axarr = plt.subplots(2, 2, figsize=(10, 5))

minspeed = min(cut_plane_floris.df.u)
maxspeed = max(cut_plane_floris.df.u)

# SOWFA
ax = axarr[0, 0]
wfct.visualization.visualize_cut_plane(
    cut_plane_sowfa, ax=ax, minSpeed=minspeed, maxSpeed=maxspeed
)
ax.set_title("SOWFA")

# FLORIS
ax = axarr[0, 1]
wfct.visualization.visualize_cut_plane(
    cut_plane_floris, ax=ax, minSpeed=minspeed, maxSpeed=maxspeed
)
ax.set_title("FLORIS")

# FLORIS Project
ax = axarr[1, 0]
wfct.visualization.visualize_cut_plane(
    cut_plane_floris_project, ax=ax, minSpeed=minspeed, maxSpeed=maxspeed
)
ax.set_title("FLORIS Projected")

# SOWFA - FLORIS
ax = axarr[1, 1]
wfct.visualization.visualize_cut_plane(
    cut_plane_difference, ax=ax, minSpeed=-1, maxSpeed=1
)
ax.set_title("SOWFA - FLORIS Projected")

# fig.tight_layout()

plt.show()
