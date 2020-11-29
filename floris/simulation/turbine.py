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


import math

import numpy as np
from scipy.stats import norm
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d

from ..utilities import cosd, sind, tand
from ..logging_manager import LoggerBase

from floris.simulation.wake_vortex.VortexCylinder import vc_tang_u, vc_longi_u, vc_root_u, vcs_tang_u, vcs_longi_u, vc_tang_u_doublet
from floris.simulation.wake_vortex.VortexDoublet  import doublet_line_u
from floris.simulation.wake_vortex.SelfSimilar    import ss_u
from floris.simulation.wake_vortex.VortexCylinderSkewed import svc_tang_u, svc_longi_u, svc_root_u, svcs_tang_u, svcs_longi_u
from floris.simulation.wake_vortex.Solver import Ct_const_cutoff, WakeVorticityFromCt, WakeVorticityFromGamma

class Turbine(LoggerBase):
    """
    Turbine is a class containing objects pertaining to the individual
    turbines.

    Turbine is a model class representing a particular wind turbine. It
    is largely a container of data and parameters, but also contains
    methods to probe properties for output.

    Args:
        instance_dictionary: A dictionary that is generated from the
            input_reader; it should have the following key-value pairs:

            -   **description** (*str*): A string containing a description of
                the turbine.
            -   **properties** (*dict*): A dictionary containing the following
                key-value pairs:

                -   **rotor_diameter** (*float*): The rotor diameter (m).
                -   **hub_height** (*float*): The hub height (m).
                -   **blade_count** (*int*): The number of blades.
                -   **pP** (*float*): The cosine exponent relating the yaw
                    misalignment angle to power.
                -   **pT** (*float*): The cosine exponent relating the rotor
                    tilt angle to power.
                -   **generator_efficiency** (*float*): The generator
                    efficiency factor used to scale the power production.
                -   **power_thrust_table** (*dict*): A dictionary containing the
                    following key-value pairs:

                    -   **power** (*list(float)*): The coefficient of power at
                        different wind speeds.
                    -   **thrust** (*list(float)*): The coefficient of thrust
                        at different wind speeds.
                    -   **wind_speed** (*list(float)*): The wind speeds for
                        which the power and thrust values are provided (m/s).

                -   **yaw_angle** (*float*): The yaw angle of the turbine
                    relative to the wind direction (deg). A positive value
                    represents a counter-clockwise rotation relative to the
                    wind direction.
                -   **tilt_angle** (*float*): The tilt angle of the turbine
                    (deg). Positive values correspond to a downward rotation of
                    the rotor for an upstream turbine.
                -   **TSR** (*float*): The tip-speed ratio of the turbine. This
                    parameter is used in the "curl" wake model.
                -   **ngrid** (*int*, optional): The square root of the number
                    of points to use on the turbine grid. This number will be
                    squared so that the points can be evenly distributed.
                    Defaults to 5.
                -   **rloc** (*float, optional): A value, from 0 to 1, that determines
                    the width/height of the grid of points on the rotor as a ratio of
                    the rotor radius.
                    Defaults to 0.5.

                    Need to Update _________________________________________
                    - R: rotor radius
                    - r_hub: position of the turbine hub in global coordinate system
                    - e_shaft_yaw0: unit vector along the shaft (untitled for now), going downwind, when the turbine has zero yaw
                    - e_vert: unit vertical vector, about which positive yawing is done
                    - U0: Free stream velocity in global coordinates (can be changerd with `update_wind`)
                    - Ct: Thrust coefficient (can be changed with `update_loading`)
                    - Ground: Include ground effect in calculations
                    - Model: one of ['VC','VCFF', 'VD', 'SS']
                            'VCFF': Vortex cylinder with far-field approximation (fastest)
                            'VC': Vortex cylinder
                            'SS': Self similar model of Troldborg et al.  (not good close to rotor)
                            'VD': Self similar model of Troldborg et al.  (not good close to rotor)

Returns:
        Turbine: An instantiated Turbine object.
    """

    def __init__(self, instance_dictionary):
        self.description = instance_dictionary["description"]

        properties = instance_dictionary["properties"]
        self.rotor_diameter = properties["rotor_diameter"]
        self.hub_height = properties["hub_height"]
        self.blade_count = properties["blade_count"]
        self.pP = properties["pP"]
        self.pT = properties["pT"]
        self.generator_efficiency = properties["generator_efficiency"]
        self.power_thrust_table = properties["power_thrust_table"]
        self.yaw_angle = properties["yaw_angle"]
        self.tilt_angle = properties["tilt_angle"]
        self.tsr = properties["TSR"]

        # Vortex turbine (for induction computation) parameters
        self.R = self.rotor_diameter/2
        self.r_hub = [0,0,self.hub_height]
        self.e_shaft_yaw0 = [1,0,0]
        self.e_vert = [0,0,1]

        """ 
            Specifies vectors to define coordinate notations for transformation
            matrices between vortex turbine cylindrical corrdinates and global coordinates
        """
        # Specify global coordinate system [TODO ??? need to check]
        self.e_shaft_g0 = np.asarray([1,0,0]).reshape(3,1)
        self.e_vert_g = np.asarray([0,0,1]).reshape(3,1)
        self.e_horz_g = np.asarray([1.,0.,0.]).reshape(3,1)
        # Transformation matrix from cylindrical to wind turbine coordinate system
        self.T_c2wt = np.asarray([[0,0,1,1,0,0,0,1,0]]).reshape(3,3)
        
        self.set_yaw_angle(self.yaw_angle)
        self.update_position(self.r_hub)
        self.U0_g = np.asarray([10,0,0]).ravel().reshape(3,1)
        #self.update_wind([0,0,10])
        self.name=''
        self.r=None
        self.gamma_t=None
        self.gamma_t=None
        self.Gamma_r=None
        self.Lambda=np.inf
        self.Ground=False# Ground effect will be included in calculation of induced velocity
        self.chi=None
        self.Model='VC'

        # initialize to an invalid value until calculated
        self.air_density = -1
        self.use_turbulence_correction = False

        # Initiate to False unless specifically set
        # For the following parameters, use default values if not user-specified
        self.ngrid = int(properties["ngrid"]) if "ngrid" in properties else 5
        self.rloc = float(properties["rloc"]) if "rloc" in properties else 0.5
        if "use_points_on_perimeter" in properties:
            self.use_points_on_perimeter = bool(properties["use_points_on_perimeter"])
        else:
            self.use_points_on_perimeter = False

        self._initialize_turbine()

        # The indices for this Turbine instance's points from the FlowField
        # are set in `FlowField._discretize_turbine_domain` and stored
        # in this variable.
        self.flow_field_point_indices = None

    # Private methods

    def _initialize_turbine(self):
        # Initialize the turbine given saved parameter settings

        # Precompute interps
        wind_speed = self.power_thrust_table["wind_speed"]

        cp = self.power_thrust_table["power"]
        self.fCpInterp = interp1d(wind_speed, cp, fill_value="extrapolate")

        ct = self.power_thrust_table["thrust"]
        self.fCtInterp = interp1d(wind_speed, ct, fill_value="extrapolate")

        # constants
        self.grid_point_count = self.ngrid * self.ngrid
        if np.sqrt(self.grid_point_count) % 1 != 0.0:
            raise ValueError("Turbine.grid_point_count must be the square of a number")

        self.reset_velocities()

        # initialize derived attributes
        self.grid = self._create_swept_area_grid()

        # Compute list of inner powers
        inner_power = np.array([self._power_inner_function(ws) for ws in wind_speed])
        self.powInterp = interp1d(wind_speed, inner_power, fill_value="extrapolate")

    def _create_swept_area_grid(self):
        # TODO: add validity check:
        # rotor points has a minimum in order to always include points inside
        # the disk ... 2?
        #
        # the grid consists of the y,z coordinates of the discrete points which
        # lie within the rotor area: [(y1,z1), (y2,z2), ... , (yN, zN)]

        # update:
        # using all the grid point because that how roald did it.
        # are the points outside of the rotor disk used later?

        # determine the dimensions of the square grid
        num_points = int(np.round(np.sqrt(self.grid_point_count)))

        pt = self.rloc * self.rotor_radius
        # syntax: np.linspace(min, max, n points)
        horizontal = np.linspace(-pt, pt, num_points)

        vertical = np.linspace(-pt, pt, num_points)

        # build the grid with all of the points
        grid = [(h, vertical[i]) for i in range(num_points) for h in horizontal]

        # keep only the points in the swept area
        if self.use_points_on_perimeter:
            grid = [
                point
                for point in grid
                if np.hypot(point[0], point[1]) <= self.rotor_radius
            ]
        else:
            grid = [
                point
                for point in grid
                if np.hypot(point[0], point[1]) < self.rotor_radius
            ]

        return grid

    def _power_inner_function(self, yaw_effective_velocity):
        """
        This method calculates the power for an array of yaw effective wind
        speeds without the air density and turbulence correction parameters.
        This is used to initialize the power interpolation method used to
        compute turbine power.
        """

        # Now compute the power
        cptmp = self._fCp(
            yaw_effective_velocity
        )  # Note Cp is also now based on yaw effective velocity
        return (
            0.5
            * (np.pi * self.rotor_radius ** 2)
            * cptmp
            * self.generator_efficiency
            * yaw_effective_velocity ** 3
        )

    def _fCp(self, at_wind_speed):
        wind_speed = self.power_thrust_table["wind_speed"]
        if at_wind_speed < min(wind_speed):
            return 0.0
        else:
            _cp = self.fCpInterp(at_wind_speed)
            if _cp.size > 1:
                _cp = _cp[0]
            return float(_cp)

    def _fCt(self, at_wind_speed):
        wind_speed = self.power_thrust_table["wind_speed"]
        if at_wind_speed < min(wind_speed):
            return 0.99
        else:
            _ct = self.fCtInterp(at_wind_speed)
            if _ct.size > 1:
                _ct = _ct[0]
            if _ct > 1.0:
                _ct = 0.9999
            return float(_ct)

    # Public methods

    def change_turbine_parameters(self, turbine_change_dict):
        """
        Change a turbine parameter and call the initialize function.

        Args:
            turbine_change_dict (dict): A dictionary of parameters to change.
        """
        for param in turbine_change_dict:
            self.logger.info(
                "Setting {} to {}".format(param, turbine_change_dict[param])
            )
            setattr(self, param, turbine_change_dict[param])
        self._initialize_turbine()

    def calculate_swept_area_velocities(
        self, local_wind_speed, coord, x, y, z, additional_wind_speed=None
    ):
        """
        This method calculates and returns the wind speeds at each
        rotor swept area grid point for the turbine, interpolated from
        the flow field grid.

        Args:
            wind_direction (float): The wind farm wind direction (deg).
            local_wind_speed (np.array): The wind speed at each grid point in
                the flow field (m/s).
            coord (:py:obj:`~.utilities.Vec3`): The coordinate of the turbine.
            x (np.array): The x-coordinates of the flow field grid.
            y (np.array): The y-coordinates of the flow field grid.
            z (np.array): The z-coordinates of the flow field grid.

        Returns:
            np.array: The wind speed at each rotor grid point
            for the turbine (m/s).
        """
        u_at_turbine = local_wind_speed

        # TODO:
        # # PREVIOUS METHOD========================
        # # UNCOMMENT IF ANY ISSUE UNCOVERED WITH NEW MOETHOD
        # x_grid = x
        # y_grid = y
        # z_grid = z

        # yPts = np.array([point[0] for point in self.grid])
        # zPts = np.array([point[1] for point in self.grid])

        # # interpolate from the flow field to get the flow field at the grid
        # # points
        # dist = [np.sqrt((coord.x1 - x_grid)**2 \
        #      + (coord.x2 + yPts[i] - y_grid) **2 \
        #      + (self.hub_height + zPts[i] - z_grid)**2) \
        #      for i in range(len(yPts))]
        # idx = [np.where(dist[i] == np.min(dist[i])) for i in range(len(yPts))]
        # data = [np.mean(u_at_turbine[idx[i]]) for i in range(len(yPts))]
        # # PREVIOUS METHOD========================

        # Use this if no saved points (curl)
        if self.flow_field_point_indices is None:
            # # NEW METHOD========================
            # Sort by distance
            flow_grid_points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

            # Set up a grid array
            y_array = np.array(self.grid)[:, 0] + coord.x2
            z_array = np.array(self.grid)[:, 1] + self.hub_height
            x_array = np.ones_like(y_array) * coord.x1
            grid_array = np.column_stack([x_array, y_array, z_array])

            ii = np.argmin(distance_matrix(flow_grid_points, grid_array), axis=0)
        else:
            ii = self.flow_field_point_indices

        # return np.array(data)
        if additional_wind_speed is not None:
            return (
                np.array(u_at_turbine.flatten()[ii]),
                np.array(additional_wind_speed.flatten()[ii]),
            )
        else:
            return np.array(u_at_turbine.flatten()[ii])

    def return_grid_points(self, coord):
        """
        Retrieve the x, y, and z grid points on the rotor.

        Args:
            coord (:py:obj:`~.utilities.Vec3`): The coordinate of the turbine.

        Returns:
            np.array, np.array, np.array:

                - x grid points on the rotor.
                - y grid points on the rotor.
                - xzgrid points on the rotor.
        """
        y_array = np.array(self.grid)[:, 0] + coord.x2
        z_array = np.array(self.grid)[:, 1] + self.hub_height
        x_array = np.ones_like(y_array) * coord.x1

        return x_array, y_array, z_array

    def update_velocities(
        self, u_wake, coord, flow_field, rotated_x, rotated_y, rotated_z
    ):
        """
        This method updates the velocities at the rotor swept area grid
        points based on the flow field freestream velocities and wake
        velocities.

        Args:
            u_wake (np.array): The wake deficit velocities at all grid points
                in the flow field (m/s).
            coord (:py:obj:`~.utilities.Vec3`): The coordinate of the turbine.
            flow_field (:py:class:`~.flow_field.FlowField`): The flow field.
            rotated_x (np.array): The x-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
            rotated_y (np.array): The y-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
            rotated_z (np.array): The z-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
        """
        # reset the waked velocities
        local_wind_speed = flow_field.u_initial - u_wake
        self.velocities = self.calculate_swept_area_velocities(
            local_wind_speed, coord, rotated_x, rotated_y, rotated_z
        )
        #print('velo shape',self.velocities.shape,len(self.grid))
        #yPts = np.array([point[0] for point in self.grid])
        #zPts = np.array([point[1] for point in self.grid])
        #print(self.grid)
        #print(yPts)
        #print(zPts)
        #import matplotlib.pyplot  as  plt
        #plt.plot(yPts,self.velocities)
        #plt.plot(zPts,self.velocities)
        #plt.show()

    def reset_velocities(self):
        """
        This method sets the velocities at the turbine's rotor swept
        area grid points to zero.
        """
        self.velocities = np.array([0.0] * self.grid_point_count)

    def set_yaw_angle(self, yaw_angle):
        """
        This method sets the turbine's yaw angle.

        Args:
            yaw_angle (float): The new yaw angle (deg).

        Examples:
            To set a turbine's yaw angle:

            >>> floris.farm.turbines[0].set_yaw_angle(20.0)
        """
        self._yaw_angle = yaw_angle

        # Vortex wind turbine
        # print('>>> turbine.py : set yaw VC_WT') 
        self.yaw_pos = yaw_angle * np.pi/180 # Convert from degrees to radians
        # print('Yaw Angle',yaw_angle)
        # print('Yaw_pos',self.yaw_pos)

        # Transformation matrix for rotating vector around yaw angle
        c,s=np.cos(self.yaw_pos),np.sin(self.yaw_pos)
        self.T_wt2g  = np.asarray([c,-s,0,s,c,0,0,0,1]).reshape(3,3)
        # Rotating the shaft vector so that its coordinate follow the new yaw position
        self.e_shaft_g=np.dot(self.T_wt2g , self.e_shaft_g0)

    def update_position(self,r_hub):
        self.r_hub=np.asarray(r_hub).ravel().reshape(3,1)

    def compute_induction(self, Ind_Opts, rotated_x, rotated_y, rotated_z, CT0=None):
        """ 
        Computes induction from the turbine as a result of the blockage effect. Applied to velocity 
        field to simulate the induction zone of a turbine.

        INPUTS:
            Ind_Opts (dict): Dictionary of inputs to model the resulting 
                turbine induction zone as a result of the blockage effect.
            rotated_x (np.array): The x-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
            rotated_y (np.array): The y-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
            rotated_z (np.array): The z-coordinates of the flow field grid
                rotated so the new x axis is aligned with the wind direction.
        """
        self.Ind_Opts = Ind_Opts

        if Ind_Opts['induction']: # Can remove (won't be called unless induction)
            if Ind_Opts['Ct_test']:
                print('Ct-test')
                # update vortex cylinder velocity and loading
                r_bar_cut = 0.11
                r_bar_tip = 0.9
                if CT0 is None:
                    CT0       = self.Ct
                print('CT0: ', CT0)
                self.R = self.rotor_diameter/2*Ind_Opts['Rfact']
                nCyl      = 1 # For now
                Lambda    = np.inf
                vr_bar    = np.linspace(0,1.0,100)
                Ct_AD     = Ct_const_cutoff(CT0,r_bar_cut,vr_bar,r_bar_tip) # TODO change me to distributed
                gamma_t_Ct = None
                self.update_loading(r=vr_bar*self.R, VC_Ct=Ct_AD, Lambda=Lambda, nCyl=nCyl, gamma_t_Ct=gamma_t_Ct)
                self.gamma_t= self.gamma_t*Ind_Opts['GammaFact']
                root  = False
                longi = False
                tang  = True
                # print('.',end='')
                ux,uy,uz = self.compute_u(rotated_x,rotated_y,rotated_z,root=root,longi=longi,tang=tang, only_ind=True, no_wake=False, Decay=False, Model = Ind_Opts['Model'], ground=Ind_Opts['Ground'],R_far_field=Ind_Opts['R_far_field'])
            else:
                # update vortex cylinder velocity and loading
                r_bar_cut = 0.01
                # r_bar_cut = 0.11
                # r_bar_tip = 0.9
                # print("------Ct:", self.Ct)
                if CT0 is None:
                    CT0       = self.Ct
                # print('CT0: ', CT0)
                self.R = self.rotor_diameter/2*Ind_Opts['Rfact']
                nCyl      = 1 # For now
                Lambda    = 30 # if >20 then no swirl
                # Lambda    = np.inf
                vr_bar    = np.linspace(0,1.0,100)
                Ct_AD     = Ct_const_cutoff(CT0,r_bar_cut,vr_bar) # TODO change me to distributed
                # Ct_AD     = Ct_const_cutoff(CT0,r_bar_cut,vr_bar,r_bar_tip) # TODO change me to distributed
                gamma_t_Ct = None
                self.update_loading(r=vr_bar*self.R, VC_Ct=Ct_AD, Lambda=Lambda, nCyl=nCyl, gamma_t_Ct=gamma_t_Ct)
                self.gamma_t= self.gamma_t*Ind_Opts['GammaFact']
                # print('gamma_t: ', self.gamma_t)
                root  = False
                longi = False
                tang  = True
                # print('.',end='')
                ux,uy,uz = self.compute_u(rotated_x,rotated_y,rotated_z,root=root,longi=longi,tang=tang, only_ind=True, no_wake=True, Decay=True, Model = Ind_Opts['Model'], ground=Ind_Opts['Ground'],R_far_field=Ind_Opts['R_far_field'])

        return ux,uy,uz

    def update_loading(self,r=None,VC_Ct=None,Gamma=None,Lambda=None,nCyl=1,gamma_t_Ct=None):
        """
        VC_Ct differs from Ct in that for a vortex cylinder VC_Ct is constant along the blade and
        zero at the root and the tip
        """
        """ 
        Computes relevant parameters when the turbine loading is updated, mainly, gamma_t, 
        the intensity of the tangential vorticity sheet.
        The ditributon will be determined based on the inputs, with one these three approaches:
           1. VC_Ct(r) distribution
           2. Gamma(r) distribution
           3. gamma_t(VC_Ct(r)) function

        INPUTS:
          r: radial coordinates at which VC_Ct or Gamma are provided
          VC_Ct: local thrust coefficient (VC_Ct(r), array), or total thrust coefficient (CT, scalar)
          Gamma:  bound circulation (Gamma(r), array), or total rotor circulation (Gamma_tot, scalar)
          Lambda: tip speed ratio (assumed infinite if None)
          nCyl : number of cylindrical model used in the spanwise direction (default is 1)
                 The circulation (gamma_t) will be determined for each of the radial cylinder
          gamma_t_Ct: function that provides gamma_t as function of VC_Ct (or gamma_t as function of CT)
        """
        # Update vortex cylinder average velocity at turbine
        self.U0_g = np.asarray([self.average_velocity,0,0]).ravel().reshape(3,1)

        U0=np.linalg.norm(self.U0_g)
        # print('Turbineprint('Turbine Avg U:',self.average_velocity)

        # --- Reinterpolating loading to number of cylinders if needed
        if nCyl is not None:
            if nCyl==1:
                vr0= np.array([0.995*self.R])
                if VC_Ct is not None:
                    VC_Ct =np.array([np.mean(VC_Ct)])
                if Gamma is not None:
                    Gamma =np.array([np.mean(Gamma)])
            else:
                vr0= np.linspace(0.005,0.995,nCyl)*self.R
                if VC_Ct is not None:
                    VC_Ct = np.interp(vr0,r,VC_Ct)
                else:
                    Gamma = np.interp(vr0,r,Gamma)
            r=vr0

        # Updating Lambda
        if Lambda is None:
            Lambda=self.Lambda
        if Lambda is None:
            raise Exception('Provide `Lambda` for update_loading. (Note: `Lambda=np.Inf` supported) ')
        Omega = Lambda*U0/self.R
        #print('U0',U0)
        #print('VC_Ct',VC_Ct)

        # Computing and storing gamma distribution and loading
        if gamma_t_Ct is not None:
            if VC_Ct is None:
                raise Exception('Provide `Ct` along `gamma_t_Ct`')
            self.gamma_t = gamma_t_Ct(VC_Ct)
            self.gamma_l=None # TODO
            self.Gamma_r=None # TODO
        elif VC_Ct is not None:
            self.gamma_t,self.gamma_l,self.Gamma_r,misc=WakeVorticityFromCt(r,VC_Ct,self.R,U0,Omega)
        elif Gamma is not None:
            self.gamma_t,self.gamma_l,self.Gamma_r,misc=WakeVorticityFromGamma(r,Gamma,self.R,U0,Omega)
        else:
            raise Exception('Unknown loading spec')
        #self.gamma_t=self.gamma_t*1.06
        #print('gamma_t    ',self.gamma_t)
        #print('gamma_l    ',self.gamma_l)
        #print('Gamma_r    ',self.Gamma_r)
        #print('Gamma_/2piR',-self.Gamma_r/(2*np.pi*self.R))
        #print(misc)
        self.Lambda=Lambda
        self.r=r
        self.VC_Ct=VC_Ct

    def compute_u(self, Xg, Yg, Zg, only_ind=False, longi=False, tang=True, root=False, no_wake=False, ground=None, Decay=False, Model=None, R_far_field=6): 
        """ 
        INPUTS:
            Xg, Yg, Zg: Control points in global coordinates where the flow is to be computed. 
            only_ind: if true, only induction is returned (without the free stream)

            longi, tang, root: booleans specifying which component of vorticity is considered.
                               Default is `tang` only
            no_wake: boolean, if true: the induced velocity in the wake is set to 0. 
                     Typically set to true when combining with wake models.

            Model : string in ['VC','VCFF','SS','VD']
                   'VCFF': Vortex cylinder with far-field approximation (fastest)
                   'VC': Vortex cylinder
                   'SS': Self similar model of Troldborg et al.  (not good close to rotor)
                   'VD': Self similar model of Troldborg et al.  (not good close to rotor)
        """
        # --- Optional argument overriding self
        if ground is None:
            ground=self.Ground
        if Model is None:
            Model=self.Model

        # Control points in "Cylinder coordinate system" (rotation only)
        T_c2g=np.dot(self.T_wt2g,self.T_c2wt)
        Xc,Yc,Zc = transform_T(T_c2g, Xg,Yg,Zg)
        # Detecting whether our vertical convention match, and define chi
        e_vert_c = np.dot(T_c2g.T , self.e_vert_g)
        # if self.chi is None:
        #     # TODO TODO chi needs induction effect!
        #     self.chi= np.sign(e_vert_c.ravel()[1])* (self.yaw_wind-self.yaw_pos)
        
        # TODO TODO chi needs induction effect!
        # self.chi= np.sign(e_vert_c.ravel()[1])* (self.yaw_wind-self.yaw_pos)
        # print('Chi: ', self.chi)
        if self.VC_Ct > 1:
            self.VC_Ct = 1
        self.chi= np.sign(e_vert_c.ravel()[1])* (self.yaw_wind-self.yaw_pos) * (1+0.3*(1-np.sqrt(1-self.VC_Ct[0])))
        # print('Chi_: ', self.chi)
        # self.chi = self.chi*1.5
        # print('Chi: ', self.chi)

        if self.gamma_t is None:
            raise Exception('Please set loading with `update_loading` before calling `compute_u`')

        uxc = np.zeros(Xg.shape)
        uyc = np.zeros(Xg.shape)
        uzc = np.zeros(Xg.shape)
        m=np.tan(self.chi)
        # Cylinder position in "Cylinder coordinate system) (rotation only)
        Xcyl, Ycyl, Zcyl = transform_T(T_c2g,np.array([self.r_hub[0]]), np.array([self.r_hub[1]]),  np.array([self.r_hub[2]]))
        # Translate control points such that origin is at rotor center. NOTE: not all routines use this
        Xc0,Yc0,Zc0=Xc-Xcyl[0],Yc-Ycyl[0],Zc-Zcyl[0]
        if ground:
            # Mirror control points are two time the hub height above the cylinder
            Yc0mirror=Yc0+2*Ycyl[0]
            Ylist=[Yc0,Yc0mirror]
            #print('>>> Ground effect',Ycyl[0])
        else:
            Ylist=[Yc0]

        # --- Root vortex influence
        if root and  (self.Gamma_r is not None) and self.Gamma_r!=0:
            for Y in Ylist:
                if np.abs(self.chi)>1e-7:
                    uxc0,uyc0,uzc0 = svc_root_u(Xc0,Y,Zc0,Gamma_r=self.Gamma_r,m=m,polar_out=False)
                else:
                    uxc0,uyc0,uzc0 =  vc_root_u(Xc0,Y,Zc0,Gamma_r=self.Gamma_r,polar_out=False)
                uxc += uxc0
                uyc += uyc0
                uzc += uzc0


        if len(self.gamma_t)==1:
            # --- Tangential and longi - ONE Cylinder only
            for iY,Y in enumerate(Ylist):
                if tang and (self.gamma_t!=0):
                    if np.abs(self.chi)>1e-7:
                        if Model =='VC':
                            uxc0,uyc0,uzc0 = svc_tang_u(Xc0,Y,Zc0,gamma_t=self.gamma_t,R=self.R,m=m,polar_out=False)
                            # print('-----------------Vortex Cylinder Skewed Model------------------')
                        else:
                            raise NotImplementedError('Model '+Model + ', with yaw.')
                    else:
                        if Model =='VC':
                                uxc0,uyc0,uzc0 = vc_tang_u        (Xc0,Y,Zc0, gamma_t=self.gamma_t, R=self.R, polar_out=False)
                        elif Model =='VCFF':
                            uxc0,uyc0,uzc0 = vc_tang_u_doublet(Xc0,Y,Zc0, gamma_t=self.gamma_t, R=self.R, polar_out=False,r_bar_Cut=R_far_field)
                        elif Model =='VD':
                            uxc0,uyc0,uzc0 = doublet_line_u(Xc0, Y, Zc0, dmz_dz = self.gamma_t * self.R**2 * np.pi)
                        elif Model =='SS':
                            uzc0 = ss_u          (Xc0, Y, Zc0, gamma_t=self.gamma_t, R=self.R)
                            uxc0=uzc0*0
                            uyc0=uzc0*0
                        else:
                            raise NotImplementedError('Model'+Model)
                    uxc += uxc0
                    uyc += uyc0
                    uzc += uzc0
                if longi and (self.gamma_l is not None) and self.gamma_l!=0 :
                    if np.abs(self.chi)>1e-7:
                        if Model =='VC':
                            uxc0,uyc0,uzc0 = svc_longi_u(Xc0,Y,Zc0,gamma_l=self.gamma_l,R=self.R,m=m,polar_out=False)
                        else:
                            raise NotImplementedError('Model '+Model + ', longi component.')
                    else:
                        if Model =='VC':
                            uxc0,uyc0,uzc0 = vc_longi_u (Xc0,Y,Zc0,gamma_l=self.gamma_l,R=self.R    ,polar_out=False)
                        else:
                            raise NotImplementedError('Model'+Model + ', longi component.')
                    uxc += uxc0
                    uyc += uyc0
                    uzc += uzc0
        else:
            # --- Tangential and longi - MULTI Cylinders
            if Model =='VC':
                nr   = len(self.r)
                nWT = 1
                # Control points are directly translated by routine
                gamma_t = self.gamma_t.reshape((nWT,nr))
#                 print('r      ',self.r)
#                 print('gamma_t',gamma_t)
                if self.gamma_l is not None:
                    gamma_l = self.gamma_l.reshape((nWT,nr))
                vR      = self.r.reshape((nWT,nr))
                vm       = m* np.ones((nWT,nr))
                if tang:
                    if np.abs(self.chi)>1e-7:
                        uxc0,uyc0,uzc0 = svcs_tang_u(Xc,Yc,Zc,gamma_t=gamma_t,R=vR,m=vm,Xcyl=Xcyl,Ycyl=Ycyl,Zcyl=Zcyl,Ground=ground)
                    else:
                        uxc0,uyc0,uzc0 = vcs_tang_u (Xc,Yc,Zc,gamma_t=gamma_t,R=vR    ,Xcyl=Xcyl,Ycyl=Ycyl,Zcyl=Zcyl, Ground=ground)
                    uxc += uxc0
                    uyc += uyc0
                    uzc += uzc0
                if longi and (self.gamma_l is not None):
                    if np.abs(self.chi)>1e-7:
                        uxc0,uyc0,uzc0 = svcs_longi_u(Xc,Yc,Zc,gamma_l=gamma_l,R=vR,m=vm,Xcyl=Xcyl,Ycyl=Ycyl,Zcyl=Zcyl, Ground=ground)
                    else:
                        uxc0,uyc0,uzc0 = vcs_longi_u (Xc,Yc,Zc,gamma_l=gamma_l,R=vR      ,Xcyl=Xcyl,Ycyl=Ycyl,Zcyl=Zcyl, Ground=ground)
                    uxc += uxc0
                    uyc += uyc0
                    uzc += uzc0
            else:
                raise NotImplementedError('Model'+Model, 'with multiple cylinders')
#         if no_wake:
# #             uxc[:]=0
# #             uyc[:]=0
# #             uzc[:]=1
#             # Zero wake induction
#             bDownStream=Zc0>=-0.20*self.R
# #             bDownStream=Zc0>=0
#             Rc = np.sqrt(Xc0**2 + Yc0**2)
#             bRotorTube = Rc<self.R*1.001 # we give a margin since VD and VC have fields very dissimilar at R+/-eps
#             bSelZero = np.logical_and(bRotorTube,bDownStream)
#             uxc[bSelZero]=0
#             uyc[bSelZero]=0
#             uzc[bSelZero]=0

        # Transform back to global
        uxg = T_c2g[0,0]*uxc+T_c2g[0,1]*uyc+T_c2g[0,2]*uzc
        uyg = T_c2g[1,0]*uxc+T_c2g[1,1]*uyc+T_c2g[1,2]*uzc
        uzg = T_c2g[2,0]*uxc+T_c2g[2,1]*uyc+T_c2g[2,2]*uzc

        # Decay
        if Decay:
            bDownStream=Xg>=(Yg-self.r_hub[1])*np.tan(-self.yaw_pos)-0.20*self.R+self.r_hub[0]
            XDecay = np.ones(uxg.shape)
            # XDecay[bDownStream] = np.exp(-((Xg[bDownStream]-self.r_hub[0])/(self.R*2))**2)
            # XDecay[bDownStream] = np.exp(-((Xg[bDownStream]-self.r_hub[0])/(self.R*2)*self.VC_Ct)**2)
            # XDecay[bDownStream] = np.exp(-(((Xg[bDownStream]-self.r_hub[0])*np.cos(-self.yaw_pos)-(Yg[bDownStream]-self.r_hub[1])*np.sin(-self.yaw_pos))/(self.R*2))**2)
            XDecay[bDownStream] = np.exp(-(((Xg[bDownStream]-self.r_hub[0])*np.cos(-self.yaw_pos)-(Yg[bDownStream]-self.r_hub[1])*np.sin(-self.yaw_pos))/(self.R*2)*self.VC_Ct)**2)
            uxg*=XDecay
            uyg*=XDecay
            uzg*=XDecay

        if no_wake:
            # Zero wake induction
            # Remove wake downstream of turbine (include small region in front of turbine to ensure induction does not affect free stream velocity)
            bDownStream=Xg>=(Yg-self.r_hub[1])*np.tan(-self.yaw_pos)-0.20*self.R+self.r_hub[0]
            # Only remove wake if within vortex cylinder radius
            # Rc = np.sqrt((Yg-self.r_hub[1])**2 + (Zg-self.hub_height)**2)
            vortex_vector = np.sqrt((-(Zg-self.r_hub[2]))**2+((Yg-self.r_hub[1])-(Xg-self.r_hub[0])*np.tan(self.chi+self.yaw_pos))**2) 
            Rc = vortex_vector/np.linalg.norm(np.array([1,np.tan(self.chi+self.yaw_pos),0]))
            bRotorTube = Rc<self.R*1.001 # we give a margin since VD and VC have fields very dissimilar at R+/-eps
            # Check if point is both downstream and within vortex cylinder radius
            bSelZero = np.logical_and(bRotorTube,bDownStream)
            uxg[bSelZero]=0
            uyg[bSelZero]=0
            uzg[bSelZero]=0

            # # Removes ground effect vortex cylinder wake
            # # Remove wake downstream of turbine (include small region in front of turbine to ensure induction does not affect free stream velocity)
            # bDownStream=Xg>=(Yg+self.r_hub[1])*np.tan(-self.yaw_pos)-0.20*self.R+self.r_hub[0]
            # vortex_vector = np.sqrt((-(Zg+self.r_hub[2]))**2+((Yg+self.r_hub[1])-(Xg-self.r_hub[0])*np.tan(self.chi+self.yaw_pos))**2) 
            # Rc = vortex_vector/np.linalg.norm(np.array([1,np.tan(self.chi+self.yaw_pos),0]))        
            # bRotorTube = Rc<self.R*1.001 # we give a margin since VD and VC have fields very dissimilar at R+/-eps
            # # Check if point is both downstream and within vortex cylinder radius
            # bSelZero = np.logical_and(bRotorTube,bDownStream)
            # uxg[bSelZero]=0
            # uyg[bSelZero]=0
            # uzg[bSelZero]=0

        # Add free stream if requested
        if not only_ind:
            uxg += self.U0_g[0]
            uyg += self.U0_g[1]
            uzg += self.U0_g[2]
        return uxg,uyg,uzg
    def TKE_to_TI(self, turbulence_kinetic_energy):
        """
        Converts a list of turbulence kinetic energy values to
        turbulence intensity.
        Args:
            turbulence_kinetic_energy (list): Values of turbulence kinetic
                energy in units of meters squared per second squared.
            wind_speed (list): Measurements of wind speed in meters per second.
        Returns:
            list: converted turbulence intensity values expressed as a decimal
            (e.g. 10%TI -> 0.10).
        """
        total_turbulence_intensity = (
            np.sqrt((2 / 3) * turbulence_kinetic_energy)
        ) / self.average_velocity
        return total_turbulence_intensity

    def TI_to_TKE(self):
        """
        Converts TI to TKE.
        Args:
            wind_speed (list): Measurements of wind speed in meters per second.
        Returns:
            list: converted TKE values
        """
        return ((self.average_velocity * self.current_turbulence_intensity) ** 2) / (2 / 3)

    def u_prime(self):
        """
        Converts a TKE to horizontal deviation component.
        Args:
            wind_speed (list): Measurements of wind speed in meters per second.
        Returns:
            list: converted u_prime values in meters per second
        """
        tke = self.TI_to_TKE()
        return np.sqrt(2 * tke)

    # Getters & Setters
    @property
    def yaw_wind(self):
        """ NOTE: this is wind angle not wind direction, measured with same convention as yaw:
            - around the axis e_vert
        """
        u_horz = self.U0_g - np.dot(self.U0_g.T,self.e_vert_g)*self.e_vert_g
        e_w    = u_horz/np.linalg.norm(u_horz)
        sign = np.sign  ( np.dot(np.cross(self.e_horz_g.T, self.U0_g.T),self.e_vert_g) )
        if sign==0:
            yaw_wind = np.arccos(np.dot(e_w.T,self.e_horz_g))
        else:
            yaw_wind = sign * np.arccos(np.dot(e_w.T,self.e_horz_g))
        return yaw_wind.ravel()[0]

    @property
    def turbulence_parameter(self):
        """
        This property calculates and returns the turbulence correction
        parameter for the turbine, a value used to account for the
        change in power output due to the effects of turbulence.

        Returns:
            float: The value of the turbulence parameter.
        """
        if not self.use_turbulence_correction:
            return 1.0
        else:
            # define wind speed, ti, and power curve components
            ws = np.array(self.power_thrust_table["wind_speed"])
            cp = np.array(self.power_thrust_table["power"])
            ws = ws[np.where(cp != 0)]
            ciws = ws[0]  # cut in wind speed
            cows = ws[len(ws) - 1]  # cut out wind speed
            speed = self.average_velocity
            ti = self.current_turbulence_intensity

            if ciws >= speed or cows <= speed or ti == 0.0 or math.isnan(speed):
                return 1.0
            else:
                # define mean and standard deviation to create normalized pdf with sum = 1
                mu = speed
                sigma = ti * mu
                if mu + sigma >= cows:
                    xp = np.linspace((mu - sigma), cows, 100)
                else:
                    xp = np.linspace((mu - sigma), (mu + sigma), 100)
                pdf = norm.pdf(xp, mu, sigma)
                npdf = np.array(pdf) * (1 / np.sum(pdf))

                # calculate turbulence parameter (ratio of corrected power to original power)
                return np.sum([npdf[k] * self.powInterp(xp[k]) for k in range(100)]) / (
                    self.powInterp(mu)
                )

    @property
    def current_turbulence_intensity(self):
        """
        This method returns the current turbulence intensity at
        the turbine expressed as a decimal fraction.

        **Note:** This is a virtual property used to "get" or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Examples:
            To get the turbulence intensity for a turbine:

            >>> current_turbulence_intensity = floris.farm.turbines[0].turbulence_intensity()
        """
        return self._turbulence_intensity

    @current_turbulence_intensity.setter
    def current_turbulence_intensity(self, value):
        self._turbulence_intensity = value

    @property
    def rotor_radius(self):
        """
        This method returns the rotor radius of the turbine (m).

        **Note:** This is a virtual property used to "get" a value.

        Returns:
            float: The rotor radius of the turbine.

        Examples:
            To get the rotor radius for a turbine:

            >>> rotor_radius = floris.farm.turbines[0].rotor_radius()
        """
        return self.rotor_diameter / 2.0

    @property
    def yaw_angle(self):
        """
        This method gets or sets the turbine's yaw angle.

        **Note:** This is a virtual property used to "get"  or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Examples:
            To set the yaw angle for each turbine in the wind farm:

            >>> yaw_angles = [20.0, 10.0, 0.0]
            >>> for yaw_angle, turbine in
            ... zip(yaw_angles, floris.farm.turbines):
            ...     turbine.yaw_angle = yaw_angle

            To get the current yaw angle for each turbine in the wind
            farm:

            >>> yaw_angles = []
            >>> for i, turbine in enumerate(floris.farm.turbines):
            ...     yaw_angles.append(turbine.yaw_angle())
        """
        return self._yaw_angle

    @yaw_angle.setter
    def yaw_angle(self, value):
        self._yaw_angle = value

    @property
    def tilt_angle(self):
        """
        This method gets the turbine's tilt angle.

        **Note:** This is a virtual property used to "get"  or "set" a value.

        Args:
            value (float): Value to set.

        Returns:
            float: Value currently set.

        Examples:
            To get the current tilt angle for a turbine:

            >>> tilt_angle = floris.farm.turbines[0].tilt_angle()
        """
        return self._tilt_angle

    @tilt_angle.setter
    def tilt_angle(self, value):
        self._tilt_angle = value

    @property
    def average_velocity(self):
        """
        This property calculates and returns the cube root of the
        mean cubed velocity in the turbine's rotor swept area (m/s).

        Returns:
            float: The average velocity across a rotor.

        Examples:
            To get the average velocity for a turbine:

            >>> avg_vel = floris.farm.turbines[0].average_velocity()
        """
        # remove all invalid numbers from interpolation
        data = self.velocities[np.where(~np.isnan(self.velocities))]
        avg_vel = np.cbrt(np.mean(data ** 3))
        if np.isnan(avg_vel):
            avg_vel = 0
        elif np.isinf(avg_vel):
            avg_vel = 0

        return avg_vel

    @property
    def Cp(self):
        """
        This property returns the power coeffcient of a turbine.

        This property returns the coefficient of power of the turbine
        using the rotor swept area average velocity, interpolated from
        the coefficient of power table. The average velocity is
        calculated as the cube root of the mean cubed velocity in the
        rotor area.

        **Note:** The velocity is scalled to an effective velocity by the yaw.

        Returns:
            float: The power coefficient of a turbine at the current
            operating conditions.

        Examples:
            To get the power coefficient value for a turbine:

            >>> Cp = floris.farm.turbines[0].Cp()
        """
        # Compute the yaw effective velocity
        pW = self.pP / 3.0  # Convert from pP to pW
        yaw_effective_velocity = self.average_velocity * cosd(self.yaw_angle) ** pW

        return self._fCp(yaw_effective_velocity)

    @property
    def Ct(self):
        """
        This property returns the thrust coefficient of a turbine.

        This method returns the coefficient of thrust of the yawed
        turbine, interpolated from the coefficient of power table,
        using the rotor swept area average velocity and the turbine's
        yaw angle. The average velocity is calculated as the cube root
        of the mean cubed velocity in the rotor area.

        Returns:
            float: The thrust coefficient of a turbine at the current
            operating conditions.

        Examples:
            To get the thrust coefficient value for a turbine:

            >>> Ct = floris.farm.turbines[0].Ct()
        """
        return self._fCt(self.average_velocity) * cosd(self.yaw_angle)  # **self.pP

    @property
    def power(self):
        """
        This property returns the power produced by turbine (W),
        adjusted for yaw and tilt.

        Returns:
            float: Power of a turbine in watts.

        Examples:
            To get the power for a turbine:

            >>> power = floris.farm.turbines[0].power()
        """
        # Update to power calculation which replaces the fixed pP exponent with
        # an exponent pW, that changes the effective wind speed input to the power
        # calculation, rather than scaling the power.  This better handles power
        # loss to yaw in above rated conditions
        #
        # based on the paper "Optimising yaw control at wind farm level" by
        # Ervin Bossanyi

        # Compute the yaw effective velocity
        pW = self.pP / 3.0  # Convert from pP to w
        yaw_effective_velocity = self.average_velocity * cosd(self.yaw_angle) ** pW

        # Now compute the power
        return (
            self.air_density
            * self.powInterp(yaw_effective_velocity)
            * self.turbulence_parameter
        )

    @property
    def aI(self):
        """
        This property returns the axial induction factor of the yawed
        turbine calculated from the coefficient of thrust and the yaw
        angle.

        Returns:
            float: Axial induction factor of a turbine.

        Examples:
            To get the axial induction factor for a turbine:

            >>> aI = floris.farm.turbines[0].aI()
        """
        return (
            0.5
            / cosd(self.yaw_angle)
            * (1 - np.sqrt(1 - self.Ct * cosd(self.yaw_angle)))
        )

# --------------------------------------------------------------------------------}
# --- Helper functions for geometry 
# --------------------------------------------------------------------------------{
def transform_T(T_a2b,Xb,Yb,Zb):
    Xa=T_a2b[0,0]*Xb+T_a2b[1,0]*Yb+T_a2b[2,0]*Zb
    Ya=T_a2b[0,1]*Xb+T_a2b[1,1]*Yb+T_a2b[2,1]*Zb
    Za=T_a2b[0,2]*Xb+T_a2b[1,2]*Yb+T_a2b[2,2]*Zb
    return Xa,Ya,Za