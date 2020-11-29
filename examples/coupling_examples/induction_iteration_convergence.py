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
Induction = True # Include induction
horPlot = False # Plots horizontal cuts of wind farms for each wind farm layout
IterPlots = False # Plots powers ,velocities and Ct values as a function of iteration
Legends = True # Includes turbine legends in IterPlots
TurbineIterPlots = True # Plots turbine power power,velocity, and Ct values individually per iteration
ReadOuts = False # Prints dataframes of turbine powers as a function of iterations used

# --- Resolution Parameters
ny=100
nx=ny*4

sep = [3,5,7] # List of streamwise separations between turbines (*D)
titles = [str(i)+'D separation' for i in sep]

# Create list of iterations to be tested
nIter = np.arange(1,6)

sepy = 3 # spanwise spearation between turbines (*D)
# Creates a turbine field with n rows and m columns
n = 5
m = 1

input_file="OptLayout_2x3.json"
input_dict=None
resolution=Vec3(nx, ny, 2)

# Initialize the floris interface
fi = wfct.floris_interface.FlorisInterface(input_file, input_dict)

# Read in induction options from input file
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts

if Induction:
    print(">>> Blockage effect is modeled")
    Ind_Opts['induction'] = True
else:
    Ind_Opts['induction'] = False

# Turbine parameters
D = fi.floris.farm.turbines[0].rotor_diameter
# layout_y = [0,0,0,0,0,3*D,3*D,3*D,3*D,3*D]
layout_y = []
for i in range(m):
    for j in range(n):
        layout_y.append(i*sepy*D)

# Initalize list of dataframes to store turbine parameters for each layout
power_df = []
velocity_df = []
ct_df = []

for s in range(len(sep)):
    # Creates list of x coordinate locations for turbines 
    layout_x = []
    for i in range(m):
        for j in range(n):
            layout_x.append(j*sep[s]*D)

    # Reinitialize flow flow field with specified layout
    fi.reinitialize_flow_field(layout_array=[layout_x, layout_y])

    # Initialize blank dataframes with iterations as index and turbines as columns
    sepdf = pd.DataFrame(index=nIter,columns = ['turbine '+str(i) for i in range(len(layout_y))])
    veldf = pd.DataFrame(index=nIter,columns = ['turbine '+str(i) for i in range(len(layout_y))])
    ctdf = pd.DataFrame(index=nIter,columns = ['turbine '+str(i) for i in range(len(layout_y))])

    for i in range(len(nIter)):
        Ind_Opts['nIter'] = nIter[i]
        fi.IndOpts = Ind_Opts
        # print('Number of iterations:', Ind_Opts['nIter'])

        # Calculate wake and get horizontal plane at turbine height for original farm field
        fi.calculate_wake(Ind_Opts=Ind_Opts)

        # Store value for total wind farm power
        power_initial = fi.get_farm_power()
        turbine_powers = fi.get_turbine_power()

        # Store value of turbine inflow velocity and thrust coefficient
        turbine_vel = []
        turbine_ct = []
        for turbine in fi.floris.farm.turbine_map.turbines:
            turbine_vel.append(turbine.average_velocity)
            turbine_ct.append(turbine.Ct)
        
        # Store turbine results in dataframe for each iteration tested
        sepdf.iloc[i] = turbine_powers
        veldf.iloc[i] = turbine_vel
        ctdf.iloc[i] = turbine_ct
    
    # Appends each layout's dataframes to list of corresponding dataframes
    power_df.append(sepdf)
    velocity_df.append(veldf)
    ct_df.append(ctdf)
    
    # Plot horizontal cut of the wind farm for final iteration for each turbine layout
    if horPlot:
        hor_plane = fi.get_hor_plane(x_resolution = resolution.x1, y_resolution = resolution.x2, Ind_Opts = Ind_Opts)
    
        fig, ax = plt.subplots()
        ax.title.set_text(titles[s])
        wfct.visualization.visualize_cut_plane(hor_plane,ax)
        wfct.visualization.plot_turbines_with_fi(ax,fi)

def normalize(df):
    # Performs columnwise normalization for each column in df
    norm_df = copy.deepcopy(df)
    for i in norm_df.columns:
        norm_df[i] /= norm_df[i].max()
    return norm_df

if IterPlots:
    def iterationplot(df_list,sep,ylabel):
        # Plots the convergence for each turbine variable for each separtion
        # Plots the normalized variable for each turbine in the field as a function of the number of iterations
        # fig,axs = plt.subplots(1, len(sep), sharey=True, sharex=True, figsize=(12.0,5.0))
        fig,axs = plt.subplots(len(sep), 1, sharey=True, sharex=False, figsize=(6.0,9.0))
        for i in range(len(sep)):
            if len(sep) == 1:
                ax = axs
                # ax.set_ylabel(ylabel,fontsize=16)
                ax.set_xlabel('Iterations',fontsize=15)
            else:
                ax = axs[i]
                # axs[0].set_ylabel(ylabel,fontsize=16)
                axs[len(sep)-1].set_xlabel('Iterations',fontsize=15)
            ax.plot(normalize(df_list[0]))
            ax.set_title('%dD Separation' %sep[i], fontsize=16)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            
            if Legends:
                if len(sep) == 1:
                    ax.legend(df_list[0].columns, loc="lower right", fontsize=13) # TODO find better location for legend
                else:
                    axs[len(sep)-1].legend(df_list[0].columns, loc="lower right",fontsize=13) # TODO find better location for legend
        
        fig.suptitle('Normalized Iteration Plot: %dx%d Layout' %(m,n), fontsize=22)
        # fig.text(0.5,0.04,'Iterations', ha='center',fontsize=16)
        fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', fontsize=15)
        # fig.tight_layout(rect=[0,0.05,1,0.95])
        fig.tight_layout(rect=[0.05,0,1,0.95])

    iterationplot(power_df,sep,'Normalized Power')
    iterationplot(velocity_df,sep,'Normalized Velocity')
    iterationplot(ct_df,sep,'Normalized Ct')
    
if TurbineIterPlots:
    def turbineiteration(df_list,sep,ylabel):
        # Plots the convergence for each turbine separtion simulation
        # Plots the normalized variable for each turbine in the field as a function of the number of iterations
        fig,axs = plt.subplots(1, len(layout_y), sharex=True, figsize=(15.0,4.0))
        # fig,axs = plt.subplots(len(layout_y), 1, figsize=(5.0,12.0))
        for (i,col) in enumerate(df_list[0].columns):
            ax = axs[i]
            ax.set_title(col)
            ax.plot(df_list[0][col])
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axs[0].set_ylabel(ylabel)
        # axs[len(layout_y)-1].set_xlabel('Number of Iterations')
        fig.suptitle('Induction Iteration Test for %dD Separation' %sep[0], fontsize=16)
        fig.text(0.5,0.04,'Number of Iterations', ha='center')
        # fig.text(0.04,0.5, ylabel, rotation='vertical', va='center', fontsize=14)
        fig.tight_layout(rect = [0.0,0.05,1,0.96])

    turbineiteration(power_df,sep,'Power')
    turbineiteration(velocity_df,sep,'Velocity')
    turbineiteration(ct_df,sep,'Ct')

if ReadOuts:
    print('Total Number of Iterations Tested:', Ind_Opts['nIter'])

    print('=========================================')
    print('Turbine Powers')
    print('-----------------------------------------')
    for i in range(len(sep)):
        print('%dD Separation Between Turbines:\n'%sep[i],power_df[i])
    print('-----------------------------------------')
    print('Turbine Velocities: ')
    print('-----------------------------------------')
    for i in range(len(sep)):
        print('%dD Separation Between Turbines:\n'%sep[i],velocity_df[i])
    print('-----------------------------------------')
    print('Turbine Ct: ')
    print('-----------------------------------------')
    for i in range(len(sep)):
        print('%dD Separation Between Turbines:\n'%sep[i],ct_df[i])


plt.show()