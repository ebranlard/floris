import numpy as np
import copy
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Local
from floris.tools.optimization.scipy.yaw import YawOptimization
import floris.tools as wfct
from floris.utilities import Vec3

"""
Plots the induction effects of the turbine field to ensure induction is not being over
calculated/estimated.
"""

# --- Resolution Parameters
ny= 100
nx=ny*4
resolution=Vec3(nx, ny, 2)

input_file="../OptLayout_2x3.json"
# Initialize the floris interface
fi = wfct.floris_interface.FlorisInterface(input_file)

D = fi.floris.farm.turbines[0].rotor_diameter

layout_x = [0,5*D]
layout_y = [0,0]

fi.reinitialize_flow_field(layout_array=[layout_x,layout_y])

# Read in induction options from input file
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts
Ind_Opts['induction'] = True
Ind_Opts['nIter'] = 5

fi.IndOpts = Ind_Opts

# print("FI Ind_Opt:", fi.IndOpts)

fi2 = copy.deepcopy(fi)

# Calculate wake and get horizontal plane at turbine height for original farm field
fi.calculate_wake(Ind_Opts=Ind_Opts)
hor_plane = fi.get_hor_plane(
        x_resolution = resolution.x1,
        y_resolution = resolution.x2,
        Ind_Opts = Ind_Opts
        )

# Store value for total wind farm power
power_initial = fi.get_farm_power()

turbine_vel = []
turbine_ct = []
for turbine in fi.floris.farm.turbine_map.turbines:
    turbine_vel.append(turbine.average_velocity)
    turbine_ct.append(turbine.Ct)

# Read in induction options from input file
Ind_Opts = fi2.floris.farm.flow_field.Ind_Opts
Ind_Opts['nIter']=1
Ind_Opts['Ct_test'] = True
# print('FI2 Ind_Opts',Ind_Opts)

fi2.IndOpts = Ind_Opts

fi2.floris.farm.flow_field.set_ct(turbine_ct)
fi2.calculate_wake(Ind_Opts=Ind_Opts)

hor_plane2 = fi2.get_hor_plane(
        x_resolution = resolution.x1,
        y_resolution = resolution.x2,
        Ind_Opts = Ind_Opts
        )

# Store value for total wind farm power
power_test = fi2.get_farm_power()

fig, axs = plt.subplots(nrows=2)
wfct.visualization.visualize_cut_plane(hor_plane, ax=axs[0])
wfct.visualization.plot_turbines_with_fi(axs[0],fi)
axs[0].set_title("Baseline Case for U = 8 m/s, Wind Direction = 270$^\circ$")

wfct.visualization.visualize_cut_plane(hor_plane2, ax=axs[1])
wfct.visualization.plot_turbines_with_fi(axs[1],fi2)
axs[1].set_title("Test Case for U = 8 m/s, Wind Direction = 270$^\circ$")

print("==========================================")
print('Turbine Powers:')
print("------------------------------------------")
print('\t\tBaseline\tTest')
print("------------------------------------------")
turb_powers = fi.get_turbine_power()
turb2_powers = fi2.get_turbine_power()
for i in range(len(turb_powers)):
    print('Turbine %d: \t%.2f\t%.2f' %(i, turb_powers[i],turb2_powers[i]))

print("==========================================")
print('Velocities Seen By Each Turbine:')
print("------------------------------------------")
print('\t\tBaseline\tTest')
print("------------------------------------------")
turbine2_vel = []
turbine2_ct = []
for turbine in fi2.floris.farm.turbine_map.turbines:
    turbine2_vel.append(turbine.average_velocity)
    turbine2_ct.append(turbine.Ct)

for i in range(len(turbine_vel)):
    print('Turbine %d: \t%.2f\t\t%.2f' %(i,turbine_vel[i],turbine2_vel[i]))

print("==========================================")
print('Ct Seen By Each Turbine:')
print("------------------------------------------")
print('\t\tBaseline\tTest')
print("------------------------------------------")
for i in range(len(turbine_ct)):
    print('Turbine %d: \t%.2f\t\t%.2f' %(i,turbine_ct[i],turbine2_ct[i]))

# Calculate the induction/wake effects for plotting
hor_plane.df.u = hor_plane.df.u - 8.0
hor_plane2.df.u = hor_plane2.df.u - 8.0

fig, axs = plt.subplots(nrows=2)
wfct.visualization.visualize_cut_plane(hor_plane, ax=axs[0])
axs[0].set_title("Induction for Baseline Case")

wfct.visualization.visualize_cut_plane(hor_plane2, ax=axs[1])
axs[1].set_title("Induction for Test Case")

plt.show()