"""
Example of a user specified wind farm layout with different number of calculate_induction iterations
Returns: Horizontal velocity plots of each field and induction velocity plots of each field. Also
returns printouts of induction streamline velocities.
"""

import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Local
from floris.tools.optimization.scipy.yaw import YawOptimization
import floris.tools as wfct
from floris.utilities import Vec3

# Set paramters for iteration test
itertest = [1,5,10] # will run tests using each iteration
input_file="../example_induction_input.json"

# Initialize the floris interface
fi = wfct.floris_interface.FlorisInterface(input_file)
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts

fig,ax = plt.subplots(nrows=len(itertest),ncols=2,sharex=True,sharey=True)

u_ind_df = pd.DataFrame()

for i in range(len(itertest)):
    Ind_Opts['nIter'] = itertest[i]
    fi.floris.farm.flow_field.Ind_Opts = Ind_Opts

    # Calculate wake and get horizontal plane at turbine height for yaw optimized farm field
    fi.calculate_wake()

    # Initialize the horizontal cut
    hor_plane = fi.get_hor_plane(x_resolution=400, y_resolution=100)

    # Copy hor_plane and replace u with u_ind for plotting just the induction field
    ind_plane = copy.deepcopy(hor_plane)
    ind_plane.df.u = ind_plane.df.u_ind

    u_ind_df[str(itertest[i])+' Iteration'] = ind_plane.df.u_ind

    # Plot and show
    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax[i,0])
    ax[i,0].set_title("%d Iterations" %itertest[i])

    wfct.visualization.visualize_cut_plane(ind_plane, ax=ax[i,1])
    ax[i,1].set_title("Induction Only: %d Iterations" %itertest[i])

for i in range(len(u_ind_df.columns)-1):
    u_ind_df[str(itertest[i+1])+','+str(itertest[i])+' difference'] = u_ind_df[str(itertest[i+1])+' Iteration'] - u_ind_df[str(itertest[i])+' Iteration']
    print('-------------------------------------------------')
    print('Max Difference: ', max(u_ind_df[str(itertest[i+1])+','+str(itertest[i])+' difference']))
    print('Min Difference: ', min(u_ind_df[str(itertest[i+1])+','+str(itertest[i])+' difference']))

print('-------------------------------------------------')
print(u_ind_df.head(15))

fig.tight_layout()
plt.show()