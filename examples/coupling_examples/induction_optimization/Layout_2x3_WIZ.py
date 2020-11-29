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

fontsize = 12
plt.rc('font', family='serif')
plt.rc('font', size=12)

# --- Plot options
nStreamlines=0
U0 = 8
minSpeed=0.5
maxSpeed=1.03
levelsLines=np.sort([1.05,1.0,0.99,0.98,0.95,0.9,0.5])

# --- Resolution Parameters
ny=100 # 200
nx=ny*4

input_file="../OptLayout_2x3.json"
input_dict=None
D=126
resolution=Vec3(nx, ny, 2)
bounds=[-4*D,16*D,-2*D-10,5*D+10,89,90] # xmin xmax .. zmin zmax
Induction = True

# Helper function to plot wind farm velocity fields
def plotPlane(x,y,u,v,ax,minSpeed=None,maxSpeed=None, cmap='coolwarm', colors='w', linewidths=0.8, alpha=0.3,nStreamlines=0,levelsContour=None,levelsLines=None,axial=False):
    if axial:
        Speed=u
    else:
        Speed=np.sqrt(u**2+v**2)
    if minSpeed is None:
        minSpeed = Speed.min()
        maxSpeed = Speed.max()

    if not ax:
        fig, ax = plt.subplots()
    Z = np.ma.masked_where(np.isnan(Speed), Speed)

    # Plot the cut-through
    im = ax.pcolormesh(x, y, Z, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)
    if levelsLines is not None:
        rcParams['contour.negative_linestyle'] = 'solid'
        cs=ax.contour(x, y, Z, levels=levelsLines, colors=colors, linewidths=linewidths, alpha=alpha)

    if nStreamlines>0:
        yseed=np.linspace(min(y)*0.9,max(y)*0.9,nStreamlines)
        start=np.array([yseed*0,yseed])
        sp=ax.streamplot(x,y,u,v,color='k',start_points=start.T,linewidth=0.7,density=30,arrowstyle='-')
        # qv=streamQuiver(ax,sp,n=5,scale=40,angles='xy')
    ax.set_aspect('equal')

    return im

# Initialize the floris interface
fi = wfct.floris_interface.FlorisInterface(input_file, input_dict)

# D = fi.floris.farm.turbines[0].rotor_diameter
# U0 = fi.floris.farm.initwind[0]
# layout_x = [0,7*D,14*D,0,7*D,14*D]
# layout_y = [0,0,0,3*D,3*D,3*D]

# Make a copy for optimization instance
# fi_opt = copy.deepcopy(fi)
fi_opt = wfct.floris_interface.FlorisInterface(input_file, input_dict)
# Read in induction options from input file
Ind_Opts = fi.floris.farm.flow_field.Ind_Opts

if Induction:
    print(">>> Blockage effect is modeled")
    Ind_Opts['induction'] = True
else:
    Ind_Opts['induction'] = False

fi.IndOpts = Ind_Opts
fi_opt.IndOpts = Ind_Opts

# Calculate wake and get horizontal plane at turbine height for original farm field
fi.calculate_wake()
hor_plane = fi.get_hor_plane(
        height = fi.floris.farm.turbines[0].hub_height,
        x_resolution = resolution.x1,
        y_resolution = resolution.x2,
        x_bounds = tuple(bounds[0:2]),
        y_bounds = tuple(bounds[2:4]),
        Ind_Opts = Ind_Opts
        )
u_mesh = hor_plane.df.u.values.reshape(resolution.x2,resolution.x1)
v_mesh = hor_plane.df.v.values.reshape(resolution.x2,resolution.x1)

if U0 is not None:
        u_mesh=u_mesh/U0
        v_mesh=v_mesh/U0

plane_x, plane_y = np.unique(hor_plane.df.x1),np.unique(hor_plane.df.x2)

# Store value for total wind farm power
power_initial = fi.get_farm_power()

# ==================================================================================
# Yaw angle optimization
# ==================================================================================

# Instantiate optimization object
yaw_opt = YawOptimization(fi_opt, minimum_yaw_angle=0.0, maximum_yaw_angle=25.0)
# Perform Optimization
yaw_angles = yaw_opt.optimize()

#TODO need to plot baseline before this

# Calculate wake and get horizontal plane at turbine height for yaw optimized farm field
fi_opt.calculate_wake(yaw_angles=yaw_angles)
hor_plane_opt = fi_opt.get_hor_plane(
        height = fi_opt.floris.farm.turbines[0].hub_height,
        x_resolution = resolution.x1,
        y_resolution = resolution.x2,
        x_bounds = tuple(bounds[0:2]),
        y_bounds = tuple(bounds[2:4]),
        Ind_Opts = Ind_Opts
        )
u_optmesh = hor_plane_opt.df.u.values.reshape(resolution.x2,resolution.x1)
v_optmesh = hor_plane_opt.df.v.values.reshape(resolution.x2,resolution.x1)
if U0 is not None:
        u_optmesh=u_optmesh/U0
        v_optmesh=v_optmesh/U0

opt_x, opt_y = np.unique(hor_plane_opt.df.x1),np.unique(hor_plane_opt.df.x2)
power_opt = fi_opt.get_farm_power()

# ==================================================================================
# Plot Results
# ==================================================================================

titles=[]
titles.append('FLORIS (baseline)')
titles.append('FLORIS (optimized)')

# --- Plot and show
fig, axes = plt.subplots(nrows=len(titles), ncols=1, sharex=True, sharey=True, figsize=(12.0, 6.0))
ax = axes[0]
im = plotPlane(plane_x/D,plane_y/D,u_mesh,v_mesh,ax,minSpeed=minSpeed,maxSpeed=maxSpeed,
            nStreamlines=nStreamlines,levelsLines=levelsLines,axial=True,colors ='k')
ax.title.set_text(titles[0])
# wfct.visualization.plot_turbines_with_fi(ax,fi)
ax.set_ylabel('r/D [-]')
ax.set_xlim([-4,16])
ax.set_ylim([-2,5])

ax = axes[1]
im_opt = plotPlane(opt_x/D,opt_y/D,u_optmesh,v_optmesh,ax,minSpeed=minSpeed,maxSpeed=maxSpeed,
            nStreamlines=nStreamlines,levelsLines=levelsLines,axial=True,colors ='k')
ax.title.set_text(titles[1])
# wfct.visualization.plot_turbines_with_fi(ax,fi_opt)
ax.set_ylabel('r/D [-]')
ax.set_xlabel('x/D [-]')
ax.set_xlim([-4,16])
ax.set_ylim([-2,5])


print("==========================================")
print("optimized yaw angles = ")
for i in range(len(yaw_angles)):
    print("Turbine ", i, "=", yaw_angles[i], " deg")

print("==========================================")
print('Original Power:', power_initial)
print('Optimized Power:', power_opt)

print("==========================================")
print("Total Power Gain = %.1f%%" %(100.0 * (power_opt - power_initial) / power_initial))

print("==========================================")
print('Turbine Powers:')
print("------------------------------------------")
print('\t\tBaseline\tOptimized')
print("------------------------------------------")
init_powers = fi.get_turbine_power()
opt_powers = fi_opt.get_turbine_power()
for i in range(len(init_powers)):
    print('Turbine %d: \t%.2f\t%.2f' %(i, init_powers[i],opt_powers[i]))

print("==========================================")
print('Velocities Seen By Each Turbine:')
print("------------------------------------------")
print('\t\tBaseline\tOptimized')
print("------------------------------------------")
# for i,turbine in enumerate(fi.floris.farm.turbine_map):
#     print('Turbine %d: \t%.1f\t%.1f' %(i,turbine.average_velocity))
turbine_vel = []
optTurbine_vel = []
for turbine in fi.floris.farm.turbine_map.turbines:
    turbine_vel.append(turbine.average_velocity)
for turbine in fi_opt.floris.farm.turbine_map.turbines:
    optTurbine_vel.append(turbine.average_velocity)

for i in range(len(turbine_vel)):
    print('Turbine %d: \t%.2f\t\t%.2f' %(i,turbine_vel[i],optTurbine_vel[i]))

plt.show()