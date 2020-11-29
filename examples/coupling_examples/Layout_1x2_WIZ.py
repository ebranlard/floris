import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helper_functions import *

"""
Plots horizontal plane for 1x2 turbine array with and without blockage effects.
Used to compare to original wiz code. Uses helper functions to plot rather than
FLORIS visualization
"""

fontsize = 12
plt.rc('font', family='serif')
plt.rc('font', size=12)

# --- Plot options
bExport=False
bCompute=True
bColorBar=False
nStreamlines=0
U0 =6
minSpeed=0.5
maxSpeed=1.03
levelsLines=np.sort([1.05,1.0,0.99,0.98,0.95,0.9,0.5])

# ---
ny=100 # 30
nx=ny*4

input_file="Layout_1x2.json"
input_dict=None
D=126
resolution=Vec3(nx, ny, 2)
bounds=[-4*D,14*D,-2*D-10,2*D+10,89,90] # xmin xmax .. zmin zmax
#Induction = False # Default for inclusion of the induction zone


titles=[]
titles.append('FLORIS (original)')
titles.append('FLORIS (with blockage)')
# titles.append('With induction and blending')

# --- Parametric computation
if bCompute:
    planes=[]
    print('-----------------------------------------------------------------')
    Induction = False
    planes.append(get_HH_plane_vel(input_file, input_dict, Induction=Induction, resolution=resolution, bounds_to_set=bounds))
    #plane_x,plane_y,u_mesh,v_mesh, fi = get_HH_plane_vel(input_file, input_dict, Ind_Opts, resolution, bounds_to_set=bounds)
    savePlane(planes[-1],'_data/Layout12_0_',U0=U0)

    print('-----------------------------------------------------------------')
    Induction = True
    planes.append(get_HH_plane_vel(input_file, input_dict, Induction=Induction, resolution=resolution, bounds_to_set=bounds))
    savePlane(planes[-1],'_data/Layout12_1_',U0=U0)

planes=[]
planes.append(loadPlane('_data/Layout12_0_'))
planes.append(loadPlane('_data/Layout12_1_'))


# --- Plot and show
fig, axes = plt.subplots(nrows=len(titles), ncols=1, sharex=True, sharey=True, figsize=(12.0, 6.0))
for i,(ax,p,t) in enumerate(zip(axes.flat,planes,titles)):
    x=p[0]
    y=p[1]
    u=p[2]
    v=p[3]
    im = plotPlane(x/D,y/D,u,v,ax,minSpeed=minSpeed,maxSpeed=maxSpeed,
            nStreamlines=nStreamlines,levelsLines=levelsLines, axial=True, colors='black')
    ax.title.set_text(t)
    #wfct.visualization.plot_turbines_with_fi(ax,fi)
    ax.set_ylabel('r/D [-]',fontsize=18)
    # ax.title.set_text(t)
    ax.set_title(t, fontsize=24)
    ax.tick_params(direction='in')

    if i==1:
        ax.set_xlabel('x/D [-]',fontsize=18)

ax.set_xlim([-3,10])
ax.set_ylim([-2,2])

if bColorBar:
    fig.subplots_adjust(left=0.08, right=0.83, top=0.93, bottom=0.11,hspace=0.17)
    cbar_ax = fig.add_axes([0.88, 0.11, 0.04, 0.82])
    cbar=fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks(levelsLines)
    cbar.set_ticklabels([str(v) if v not in [0.99] else '' for v in levelsLines])
    cbar.ax.tick_params(axis='both', direction='in',length=18,color=(0.5,0.5,0.5))
else:
    fig.subplots_adjust(left=0.035, right=0.990, top=0.96, bottom=0.08,hspace=0.17)

fig.tight_layout()

plt.show()