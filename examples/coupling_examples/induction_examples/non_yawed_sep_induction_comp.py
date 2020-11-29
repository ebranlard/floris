import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Local
from floris.tools.optimization.scipy.yaw import YawOptimization
import floris.tools as wfct
from floris.utilities import Vec3
"""
Compares the turbine parameters of a wind farm with and without modelling blockage effects in the 
induction zone of turbines for various separations.
"""

import numpy as np
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
# from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Local
from floris.tools.optimization.scipy.yaw import YawOptimization
import floris.tools as wfct
from floris.utilities import Vec3

"""
Returns the power outputs, input velocities and ct values of floris as a function of the number of calculate_induction iterations used.
Tests the convergence of the results for different seperation distances between turbines for a 
specified wind farm.
"""

# --- Plot options
horPlot = False # Plots horizontal cuts of wind farms for each wind farm layout

# --- Resolution Parameters
ny=100
nx=ny*4
resolution=Vec3(nx, ny, 2)

sep = [4,5,6,7,8,9,11] # List of streamwise separations between turbines (*D)
titles = [str(i)+'D separation' for i in sep]
titles2 = [str(i)+'D' for i in sep]

sepy = 3 # spanwise spearation between turbines (*D)
# Creates a turbine field with n rows and m columns
n = 5
m = 1

input_file="../OptLayout_2x3.json"
# Initialize the floris interface
fi = wfct.floris_interface.FlorisInterface(input_file)

# Read in induction options from input file
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
Ind_Opts['induction'] = False
fi.IndOpts = Ind_Opts

# Turbine parameters
D = fi.floris.farm.turbines[0].rotor_diameter
layout_y = []
for i in range(m):
    for j in range(n):
        layout_y.append(i*sepy*D)

# Initialize dataframe for windfarm power and array for turbine powers
wfpow = pd.DataFrame(index = titles, columns=['Baseline','Induction','Power Change'])
turbpows = []

for s in range(len(sep)):
    # Creates list of x coordinate locations for turbines 
    layout_x = []
    for i in range(m):
        for j in range(n):
            layout_x.append(j*sep[s]*D)

    # Reinitialize flow flow field with specified layout
    fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])

    fi_ind = copy.deepcopy(fi)
    Ind_Opts2 = copy.deepcopy(Ind_Opts)
    Ind_Opts2['induction'] = True
    Ind_Opts2['Model'] = 'VC'
    Ind_Opts['nIter'] = 2
    fi_ind.IndOpts = Ind_Opts2

    fi.calculate_wake()
    fi_ind.calculate_wake()

    base_power = fi.get_farm_power()
    ind_power = fi_ind.get_farm_power()
    powerchange = (100.0 * (ind_power - base_power) / base_power)

    wfpow.at[titles[s],'Baseline'] = base_power
    wfpow.at[titles[s],'Induction'] = ind_power
    wfpow.at[titles[s],'Power Change'] = powerchange

    base_turbpowers = np.array(fi.get_turbine_power())
    ind_turbpowers = np.array(fi_ind.get_turbine_power())

    turbPow = pd.DataFrame(index = range(len(base_turbpowers)), columns = ['Baseline','Induction','Relative Change'])
    turbPow['Baseline'] = base_turbpowers
    turbPow['Induction'] = ind_turbpowers
    turbPow['Relative Change'] = (ind_turbpowers-base_turbpowers)/base_turbpowers*100
    turbpows.append(turbPow)

    # Plot horizontal cut of the wind farm for final iteration for each turbine layout
    if horPlot:
        hor_plane = fi.get_hor_plane(x_resolution = resolution.x1, y_resolution = resolution.x2, Ind_Opts = Ind_Opts)
        hor_plane2 = fi_ind.get_hor_plane(x_resolution = resolution.x1, y_resolution = resolution.x2, Ind_Opts = Ind_Opts2)
        # Plot and show
        fig, axs = plt.subplots(nrows = 2, ncols=1)
        wfct.visualization.visualize_cut_plane(hor_plane, ax=axs[0], fig=fig, cbar=True)
        wfct.visualization.plot_turbines_with_fi(axs[0],fi)
        axs[0].set_title("FLORIS Baseline", fontsize=16)
        wfct.visualization.visualize_cut_plane(hor_plane2, ax=axs[1], fig=fig, cbar=True)
        wfct.visualization.plot_turbines_with_fi(axs[1],fi_ind)
        axs[1].set_title("FLORIS with Blockage Effects", fontsize=16)
        fig.suptitle(titles[s])
        fig.tight_layout(rect=[0,0,1,0.95])

print('=====================================================================')
print('Relative Change in Turbine Power (%)')
print('---------------------------------------------------------------------')
print('\t\t', end="")
for i in range(len(titles2)):
    print(titles2[i],end="\t")
print('\n---------------------------------------------------------------------')
for i in range(len(base_turbpowers)):
    print('Turbine %d: ' %i, end='')
    for j in range(len(titles)):
            # print('\tTest',end='')
            print('\t%.2f' %turbpows[j]['Relative Change'][i],end='')
    print()

fig,ax = plt.subplots()
x = np.arange(len(base_turbpowers))
for i in range(len(titles)):
    ax.plot(x, turbpows[i]['Relative Change'])
    ax.scatter(x, turbpows[i]['Relative Change'])
ax.set_xticks(x)
ax.set_xticklabels(['T'+str(i) for i in x],fontsize=14)
ax.tick_params(axis='y',labelsize=12)
ax.set_ylabel('Relative Power Change \nDue to Blockage (%)',fontsize=14)
ax.set_xlabel('Turbine',fontsize=14)
ax.legend(titles)
ax.plot(x,np.zeros(len(x)),linestyle='dashed',linewidth=0.75,color = 'black')
ax.set_title('%dx%d Turbine Blockage Comparison' %(m,n) ,fontsize=20)
fig.tight_layout()

plt.show()