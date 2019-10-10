# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
from ..utilities import Vec3
from ..utilities import cosd, sind, tand
from scipy.interpolate import griddata

from vortexcylinder.Solver import Ct_const_cutoff
from floris.VC import options_dict

# from pybra.colors import *
# def get_cmap(minSpeed,maxSpeed):
#     DS=0.0001
#     # MathematicaDarkRainbow=[(60 /255,86 /255,146/255), (64 /255,87 /255,142/255), (67 /255,107/255,98 /255), (74 /255,121/255,69 /255), (106/255,141/255,61 /255), (159/255,171/255,67 /255), (207/255,195/255,77 /255), (223/255,186/255,83 /255), (206/255,128/255,76 /255), (186/255,61 /255,58 /255)]
# 
#     #     ManuDarkOrange  = np.array([198 ,106,1   ])/255.;
#     #     ManuLightOrange = np.array([255.,212,96  ])/255.;
#     # (1,212/255,96/255),  # Light Orange
#     # (159/255,159/255,204/255), # Light Blue
#     #     MathematicaLightGreen = np.array([158,204,170 ])/255.;
#     # (159/255,159/255,204/255), # Light Blue
#     seq=[
#     (63/255 ,63/255 ,153/255), # Dark Blue
#     (159/255,159/255,204/255), # Light Blue
#     (158/255,204/255,170/255), # Light Green
#     (1,212/255,96/255),  # Light Orange
#     (1,1,1),  # White
#     (1,1,1),  # White
#     (1,1,1),  # White
#     (138/255 ,42/255 ,93/255), # DarkRed
#     ]
#     valuesOri=np.array([
#     minSpeed,  # Dark Blue
#     0.90,
#     0.95,
#     0.98,
#     1.00-DS , # White
#     1.00    , # White
#     1.00+DS , # White
#     maxSpeed         # DarkRed
#     ])
#     values=(valuesOri-min(valuesOri))/(max(valuesOri)-min(valuesOri))
# 
#     valuesOri=np.around(valuesOri[np.where(np.diff(valuesOri)>DS)[0]],2)
# 
#     cmap= make_colormap(seq,values=values)
#     return cmap,valuesOri



class FlowField():
    """
    Object containing flow field information.

    FlowField is at the core of the FLORIS package. This class handles 
    the domain creation and initialization and computes the flow field 
    based on the input wake model and turbine map. It also contains 
    helper functions for quick flow field visualization.

    Args:
        wind_speed: A float that is the wind speed.
        wind_direction: A float that is the wind direction.
        wind_shear: A float that is the wind shear coefficient.
        wind_veer: A float that is the amount of veer across the rotor.
        turbulence_intensity: A float that is the decimal percentage of 
            turbulence.
        wake: A container class :py:class:`floris.simulation.wake` with 
            wake model information used to calculate the flow field.
        wake_combination: A container class 
            :py:class:`floris.simulation.wake_combination` with wake 
            combination information.
        turbine_map: A :py:obj:`floris.simulation.turbine_map` object 
            that holds turbine information.

    Returns:
        FlowField: An instantiated FlowField object.
    """

    def __init__(self,
                 wind_speed,
                 wind_direction,
                 wind_shear,
                 wind_veer,
                 turbulence_intensity,
                 air_density,
                 wake,
                 turbine_map):

        self.reinitialize_flow_field(
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            turbulence_intensity=turbulence_intensity,
            air_density=air_density,
            wake=wake,
            turbine_map=turbine_map,
            with_resolution=wake.velocity_model.model_grid_resolution
        )

    def _discretize_turbine_domain(self):
        """
        Create grid points at each turbine
        """
        xt = [coord.x1 for coord in self.turbine_map.coords]
        rotor_points = int(
            np.sqrt(self.turbine_map.turbines[0].grid_point_count))
        x_grid = np.zeros((len(xt), rotor_points, rotor_points))
        y_grid = np.zeros((len(xt), rotor_points, rotor_points))
        z_grid = np.zeros((len(xt), rotor_points, rotor_points))

        for i, (coord, turbine) in enumerate(self.turbine_map.items):
            xt = [coord.x1 for coord in self.turbine_map.coords]
            yt = np.linspace(
                coord.x2 - turbine.rotor_radius,
                coord.x2 + turbine.rotor_radius,
                rotor_points
            )
            zt = np.linspace(
                coord.x3 - turbine.rotor_radius,
                coord.x3 + turbine.rotor_radius,
                rotor_points
            )

            for j in range(len(yt)):
                for k in range(len(zt)):
                    x_grid[i, j, k] = xt[i]
                    y_grid[i, j, k] = yt[j]
                    z_grid[i, j, k] = zt[k]

                    xoffset = x_grid[i, j, k] - coord.x1
                    yoffset = y_grid[i, j, k] - coord.x2
                    x_grid[i, j, k] = xoffset * cosd(-1 * self.wind_direction) - \
                        yoffset * sind(-1 * self.wind_direction) + coord.x1
                    y_grid[i, j, k] = yoffset * cosd(-1 * self.wind_direction) + \
                        xoffset * sind(-1 * self.wind_direction) + coord.x2

        return x_grid, y_grid, z_grid

    def _discretize_freestream_domain(self, xmin, xmax, ymin, ymax, zmin, zmax, resolution):
        """
        Generate a structured grid for the entire flow field domain.
        resolution: Vec3
        """
        x = np.linspace(xmin, xmax, int(resolution.x1))
        y = np.linspace(ymin, ymax, int(resolution.x2))
        z = np.linspace(zmin, zmax, int(resolution.x3))
        return np.meshgrid(x, y, z, indexing="ij")

    def _compute_initialized_domain(self, with_resolution=None):
        if with_resolution is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self.domain_bounds
            self.x, self.y, self.z = self._discretize_freestream_domain(
                xmin, xmax, ymin, ymax, zmin, zmax, with_resolution)
        else:
            self.x, self.y, self.z = self._discretize_turbine_domain()

        self.u_initial = self.wind_speed * \
            (self.z / self.specified_wind_height)**self.wind_shear
        self.v_initial = np.zeros(np.shape(self.u_initial))
        self.w_initial = np.zeros(np.shape(self.u_initial))

        self.u = self.u_initial.copy()
        self.v = self.v_initial.copy()
        self.w = self.w_initial.copy()

    def _compute_turbine_velocity_deficit(self, x, y, z, turbine, coord, deflection, wake, flow_field):
        return self.wake.velocity_function(x, y, z, turbine, coord, deflection, wake, flow_field)

    def _compute_turbine_wake_deflection(self, x, y, turbine, coord, flow_field):
        return self.wake.deflection_function(x, y, turbine, coord, flow_field)

    def _rotated_grid(self, angle, center_of_rotation):
        xoffset = self.x - center_of_rotation.x1
        yoffset = self.y - center_of_rotation.x2
        rotated_x = xoffset * \
            cosd(angle) - yoffset * \
            sind(angle) + center_of_rotation.x1
        rotated_y = xoffset * \
            sind(angle) + yoffset * \
            cosd(angle) + center_of_rotation.x2
        return rotated_x, rotated_y, self.z

    def _rotated_dir(self, angle, center_of_rotation, rotated_map):

        # get new boundaries for the wind farm once rotated
        x_coord = []
        y_coord = []
        for coord in rotated_map.coords:
            x_coord.append(coord.x1)
            y_coord.append(coord.x2)

        if str(self.wake.velocity_model) == 'curl':
            # re-setup the grid for the curl model
            xmin = np.min(x_coord) - 2 * self.max_diameter
            xmax = np.max(x_coord) + 10 * self.max_diameter
            ymin = np.min(y_coord) - 2 * self.max_diameter
            ymax = np.max(y_coord) + 2 * self.max_diameter
            zmin = 0.1
            zmax = 6 * self.specified_wind_height

            # Save these bounds
            self._xmin = xmin
            self._xmax = xmax
            self._ymin = ymin
            self._ymax = ymax
            self._zmin = zmin
            self._zmax = zmax

            resolution = self.wake.velocity_model.model_grid_resolution

            self.x, self.y, self.z = self._discretize_freestream_domain(
                xmin, xmax, ymin, ymax, zmin, zmax, resolution)
            rotated_x, rotated_y, rotated_z = self._rotated_grid(
                0.0, center_of_rotation)
        else:
            rotated_x, rotated_y, rotated_z = self._rotated_grid(
                self.wind_direction, center_of_rotation)

        return rotated_x, rotated_y, rotated_z

    def _calculate_area_overlap(self, wake_velocities, freestream_velocities, turbine):
        """
        compute wake overlap based on the number of points that are not freestream velocity, i.e. affected by the wake
        """
        count = np.sum(freestream_velocities - wake_velocities <= 0.05)
        return (turbine.grid_point_count - count) / turbine.grid_point_count

    # Public methods

    def set_bounds(self, bounds_to_set=None):
        """
        A method that will set the domain bounds for the wake model.

        This method allows a user to customzie the domain bounds for 
        the current wake model being used, unless the wake model is the 
        Curl model, then a predefined domain is specified and used. If 
        the bounds are not specified, then a pre-defined set of bounds 
        will be used. The bounds consist of the minimum and maximum 
        values in the x-, y-, and z-directions.

        Args:
            bounds_to_set: A list of values representing the mininum 
                and maximum values for the domain 
                [xmin, xmax, ymin, ymax, zmin, zmax] 
                (default is *None*).

        Returns:
            *None* -- The flow field is updated directly in the 
            :py:class:`floris.simulation.floris.flow_field` object.
        """

        # For the curl model, bounds are hard coded
        if self.wake.velocity_model.model_string == 'curl':
            coords = self.turbine_map.coords
            x = [coord.x1 for coord in coords]
            y = [coord.x2 for coord in coords]
            eps = 0.1
            self._xmin = min(x) - 2 * self.max_diameter
            self._xmax = max(x) + 10 * self.max_diameter
            self._ymin = min(y) - 2 * self.max_diameter
            self._ymax = max(y) + 2 * self.max_diameter
            self._zmin = 0 + eps
            self._zmax = 6 * self.specified_wind_height

        # Else, if none provided, use a shorter boundary for other models
        elif bounds_to_set is None:
            coords = self.turbine_map.coords
            x = [coord.x1 for coord in coords]
            y = [coord.x2 for coord in coords]
            eps = 0.1
            self._xmin = min(x) - 2 * self.max_diameter
            self._xmax = max(x) + 10 * self.max_diameter
            self._ymin = min(y) - 2 * self.max_diameter
            self._ymax = max(y) + 2 * self.max_diameter
            self._zmin = 0 + eps
            self._zmax = 2 * self.specified_wind_height

        else:  # Set the boundaries
            self._xmin = bounds_to_set[0]
            self._xmax = bounds_to_set[1]
            self._ymin = bounds_to_set[2]
            self._ymax = bounds_to_set[3]
            self._zmin = bounds_to_set[4]
            self._zmax = bounds_to_set[5]

    def reinitialize_flow_field(self,
                                wind_speed=None,
                                wind_direction=None,
                                wind_shear=None,
                                wind_veer=None,
                                turbulence_intensity=None,
                                air_density=None,
                                wake=None,
                                turbine_map=None,
                                with_resolution=None,
                                bounds_to_set=None):
        """
        Reiniaitilzies the flow field when a parameter needs to be 
        updated.

        This method allows for changing/updating a variety of flow 
        related parameters. This would typically be used in loops or 
        optimizations where the user is calculating AEP over a wind 
        rose or investigating wind farm performance at different 
        conditions.

        Args:
            wind_speed: A float that is the wind speed (default is 
                *None*).
            wind_direction: A float that is the wind direction (default 
                is *None*).
            wind_shear: A float that is the wind shear coefficient 
                (default is *None*).
            wind_veer: A float that is the amount of veer across the 
                rotor (default is *None*).
            turbulence_intensity: A float that is a decimal percentage 
                of turbulence (default is *None*).
            air_density: A float that is the air density (default is 
                *None*).
            wake: A container class :py:class:`floris.simulation.wake` 
                with wake model information used to calculate the flow 
                field (default is *None*).
            turbine_map: A :py:obj:`floris.simulation.turbine_map` 
                object that holds turbine information (default is 
                *None*).
            with_resolution: A :py:class:`floris.utilities.Vec3` object 
                that defines the flow field resolution at which to 
                calculate the wake (default is *None*).

        Returns:
            *None* -- The flow field is updated directly in the 
            :py:class:`floris.simulation.floris` object.
        """
        # reset the given parameters
        if turbine_map is not None:
            self.turbine_map = turbine_map
        if wind_speed is not None:
            self.wind_speed = wind_speed
        if wind_direction is not None:
            # frame of reference is west
            self.wind_direction = wind_direction - 270
        if wind_shear is not None:
            self.wind_shear = wind_shear
        if wind_veer is not None:
            self.wind_veer = wind_veer
        if turbulence_intensity is not None:
            self.turbulence_intensity = turbulence_intensity
        if air_density is not None:
            self.air_density = air_density
            for turbine in self.turbine_map.turbines:
                turbine.air_density = self.air_density
        if wake is not None:
            self.wake = wake
        if with_resolution is None:
            with_resolution = self.wake.velocity_model.model_grid_resolution

        # initialize derived attributes and constants
        self.max_diameter = max(
            [turbine.rotor_diameter for turbine in self.turbine_map.turbines])
        self.specified_wind_height = self.turbine_map.turbines[0].hub_height

        # Set the domain bounds
        self.set_bounds(bounds_to_set=bounds_to_set)

        # reinitialize the flow field
        self._compute_initialized_domain(with_resolution=with_resolution)

        # reinitialize the turbines
        for turbine in self.turbine_map.turbines:
            turbine.reinitialize_turbine(self.turbulence_intensity)

    def calculate_wake(self, no_wake=False, VC_Opts=None):
        """
        Updates the flow field based on turbine activity.

        This method rotates the turbine farm such that the wind 
        direction is coming from 270 degrees. It then loops over the 
        turbines, updating their velocities, calculating the wake 
        deflection/deficit, and combines the wake with the flow field.

        Args:
            no_wake: A bool that when *True* updates the turbine 
                quantities without calculating the wake or adding the 
                wake to the flow field.

        Returns:
            *None* -- The flow field and turbine properties are updated 
            directly in the :py:class:`floris.simulation.floris` object.
        """

        # define the center of rotation with reference to 270 deg
        center_of_rotation = Vec3(0, 0, 0)

        # Rotate the turbines such that they are now in the frame of reference
        # of the wind direction simpifying computing the wakes and wake overlap
        rotated_map = self.turbine_map.rotated(
            self.wind_direction, center_of_rotation)

        # rotate the discrete grid and turbine map
        rotated_x, rotated_y, rotated_z = self._rotated_dir(
            self.wind_direction, center_of_rotation, rotated_map)


        # sort the turbine map
        sorted_map = rotated_map.sorted_in_x_as_list()

        # --- VORTEX CYLINDER
        if VC_Opts is None:
            VC_Opts=options_dict()
        if not VC_Opts['no_induction']:
            for coord, turbine in sorted_map:
                #print(type(coord.tolist()))
                turbine.VC_WT.update_position(coord.tolist())
                turbine.VC_WT.R*=VC_Opts['Rfact'] # HACK to increase rotor size
            #print('Turbine',turbine.VC_WT.tostring())

        # calculate the velocity deficit and wake deflection on the mesh
        u_wake = np.zeros(np.shape(self.u))
        v_wake = np.zeros(np.shape(self.u))
        w_wake = np.zeros(np.shape(self.u))
        print(u_wake.shape)

        u_wake_vc = np.zeros(np.shape(self.u))
        v_wake_vc = np.zeros(np.shape(self.u))
        w_wake_vc = np.zeros(np.shape(self.u))

        #local_wind_speed = self.u_initial - u_wake
        for i,(coord, turbine) in enumerate(sorted_map):

            # update the turbine based on the velocity at its hub
            turbine.update_velocities(
                u_wake, coord, self, rotated_x, rotated_y, rotated_z)


            # get the wake deflecton field
            deflection = self._compute_turbine_wake_deflection(
                rotated_x, rotated_y, turbine, coord, self)


            # get the velocity deficit accounting for the deflection
            turb_u_wake, turb_v_wake, turb_w_wake = self._compute_turbine_velocity_deficit(
                rotated_x, rotated_y, rotated_z, turbine, coord, deflection, self.wake, self)

            #  compute vortex cylinder induction
            if not VC_Opts['no_induction']:
                # update vortex cylinder velocity and loading 
                U0=turbine.average_velocity
                turbine.VC_WT.update_wind([U0,0,0]) # NOTE: rotated wind along x in FLORIS
                r_bar_cut = 0.01
                CT0       = turbine.Ct
                R         = turbine.rotor_diameter/2* VC_Opts['Rfact']
                nCyl      = 1 # For now
                Lambda    = 30 # if >20 then no swirl
                vr_bar    = np.linspace(0,1.0,100)
                Ct_AD     = Ct_const_cutoff(CT0,r_bar_cut,vr_bar) # TODO change me to distributed
                turbine.VC_WT.R = R 
                turbine.VC_WT.update_loading(r=vr_bar*R, Ct=Ct_AD, Lambda=Lambda, nCyl=nCyl)
                turbine.VC_WT.gamma_t= turbine.VC_WT.gamma_t*VC_Opts['GammaFact']
                print('VC induction - U0={:.2f} - Ct={:.2f} - gamma_t_bar={:.3f} - {}/{}'.format(U0,CT0,turbine.VC_WT.gamma_t[0]/U0,i+1,len(sorted_map)))
                root  = False
                longi = False
                tang  = True 
                ux,uy,uz = turbine.VC_WT.compute_u(rotated_x,rotated_y,rotated_z,root=root,longi=longi,tang=tang, only_ind=True, no_wake=True)
                u_wake_vc += ux
                v_wake_vc += uy
                w_wake_vc += uz
#                 if uz.shape[0]>10:
# #                     uz1 = np.squeeze(uz[:,:,0], axis=(2,))
# #                     ux1 = np.squeeze(ux[:,:,0], axis=(2,))
#                     uz1 = ux[:,:,0]
#                     ux1 = uy[:,:,0]
#                     print(ux1.shape)
#                     print(rotated_z.shape)
#                     Z   = rotated_x[:,:,0]
#                     X   = rotated_y[:,:,0]
#                     import matplotlib.pyplot as plt
#                     fig=plt.figure()
#                     ax=fig.add_subplot(111)
#                     #Speed=np.sqrt(uz1**2+ux1**2)
#                     Speed=(U0+uz1)/U0
#                     cmap,_=get_cmap(0.5,1.1)
#                     im=ax.contourf(Z/R,X/R,Speed,levels=30, vmin=0.5, vmax=1.1, cmap=cmap)
#                     cb=fig.colorbar(im)
#                     plt.show()
            # include turbulence model for the gaussian wake model from Porte-Agel
            if self.wake.velocity_model.model_string == 'gauss':

                # compute area overlap of wake on other turbines and update downstream turbine turbulence intensities
                for coord_ti, turbine_ti in sorted_map:

                    if coord_ti.x1 > coord.x1 and np.abs(coord.x2 - coord_ti.x2) < 2*turbine.rotor_diameter:
                        # only assess the effects of the current wake

                        freestream_velocities = turbine_ti.calculate_swept_area_velocities(
                            self.wind_direction,
                            self.u_initial,
                            coord_ti,
                            rotated_x,
                            rotated_y,
                            rotated_z)

                        wake_velocities = turbine_ti.calculate_swept_area_velocities(
                            self.wind_direction,
                            self.u_initial - turb_u_wake,
                            coord_ti,
                            rotated_x,
                            rotated_y,
                            rotated_z)

                        area_overlap = self._calculate_area_overlap(
                            wake_velocities, freestream_velocities, turbine)
                        if area_overlap > 0.0:
                            turbine_ti.turbulence_intensity = turbine_ti.calculate_turbulence_intensity(
                                self.turbulence_intensity,
                                self.wake.velocity_model,
                                coord_ti,
                                coord,
                                turbine
                            )

            # combine this turbine's wake into the full wake field
            if not no_wake:
                # TODO: why not use the wake combination scheme in every component?
                u_wake = self.wake.combination_function(u_wake, turb_u_wake)
                v_wake = (v_wake + turb_v_wake)
                w_wake = (w_wake + turb_w_wake)
        


#         if VC_Opts['blend']:
#             #u_wake = (u_wake - u_wake_vc)
#             v_wake = (v_wake + v_wake_vc)
#             w_wake = (w_wake + w_wake_vc)
#         else:
        u_wake = (u_wake - u_wake_vc)
        v_wake = (v_wake + v_wake_vc)
        w_wake = (w_wake + w_wake_vc)




        # apply the velocity deficit field to the freestream
        if not no_wake:
            # TODO: are these signs correct?
            self.u = self.u_initial - u_wake
            self.v = self.v_initial + v_wake
            self.w = self.w_initial + w_wake

#         if self.u.shape[0]>10:
#             Z   = rotated_x[:,:,1]
#             X   = rotated_y[:,:,1]
#             import matplotlib.pyplot as plt
#             fig=plt.figure()
#             ax=fig.add_subplot(111)
#             U0=turbine.average_velocity
#             print('U0',U0,'Ct',turbine.Ct)
#             Speed=(self.u[:,:,1])/U0
#             Speed=(self.u[:,:,1])/U0
#             print(np.min(Speed.ravel()), np.max(Speed.ravel()))
#             cmap,valuesOri=get_cmap(0.5,1.1)
#             #im=ax.contourf(Z,X,Speed,levels=100, vmin=0.5, vmax=1.1, cmap=cmap)
#             im = ax.pcolormesh(Z, X, Speed, cmap=cmap, vmin=0.5, vmax=1.1)
#             cb=fig.colorbar(im)
#             cb.set_ticks(valuesOri)
#             cb.set_ticklabels([str(v) for v in valuesOri])
#             plt.show()
# 



        # rotate the grid if it is curl
        if self.wake.velocity_model.model_string == 'curl':
            self.x, self.y, self.z = self._rotated_grid(
                -1 * self.wind_direction, center_of_rotation)

    # Getters & Setters
    @property
    def domain_bounds(self):
        """
        Property that returns the bounds of the flow field domain.

        Returns:
            floats: xmin, xmax, ymin, ymax, zmin, zmax

            The mininmum and maxmimum values of the domain in the x, y, and z directions.

        Examples:
            To get the domain bounds:

            >>> xmin, xmax, ymin, ymax, zmin, zmax = 
            ... floris.farm.flow_field.domain_bounds()
        """
        return self._xmin, self._xmax, self._ymin, self._ymax, self._zmin, self._zmax
