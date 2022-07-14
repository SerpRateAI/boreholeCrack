"""
class for making grids around the borehole
"""

import numpy as np


class Grid:
    """
    Grid for calculating theoretical arrival times
    
    All length measurements are in meters
    All time measurements are in decimal seconds
    
    Parameters
    -------------------
    hydrophone_N_location : float
        depth location of hydrophone. In meters.
    radius : float 
        radius from borehole cube should extend to. In meters.
    depth : float
        depth of borehole. In meters. default to 400m.
    gridscale : float
        scale of grid to be created.
    velocity_model : object
        grid of velocity model, if input is float then a grid will be created
        that is the same velocity everywhere, if grid is input it will be checked
        to make sure it's the same size and then this grid will be used.
    dr : dict
        dictionary containing distance grid calculation from grid point to hydrophone. keys are hydrophone keys.
        
    Functions
    -------------------
    make_grid
    calc_distance_grid
    get_grid_slice
    get_grid_midslice
    """
    
    def __init__(self, radius, gridscale, velocity_model, depth=400.,):
        self.hydrophone_locations = {
            'h1':30
            ,'h2':100
            ,'h3':170
            ,'h4':240
            ,'h5':310
            ,'h6':380
        }

        self.radius = radius
        self.depth = depth
        self.gridscale = gridscale
        
        # TODO : make grid
        self.grid = self.make_grid()
        self.xx = self.grid[0]
        self.yy = self.grid[1]
        self.zz = self.grid[2]
        
        self.dr = {
            'h1':self.calc_distance_grid(hydrophone_location=self.hydrophone_locations['h1'])
            ,'h2':self.calc_distance_grid(hydrophone_location=self.hydrophone_locations['h2'])
            ,'h3':self.calc_distance_grid(hydrophone_location=self.hydrophone_locations['h3'])
            ,'h4':self.calc_distance_grid(hydrophone_location=self.hydrophone_locations['h4'])
            ,'h5':self.calc_distance_grid(hydrophone_location=self.hydrophone_locations['h5'])
            ,'h6':self.calc_distance_grid(hydrophone_location=self.hydrophone_locations['h6'])
            }
        
        if isinstance(velocity_model, float):
            print('creating constant velocity model. vp = {vp} m/s'.format(vp=velocity_model))
            self.velocity = velocity_model
            self.velocity_model = np.zeros_like(self.grid) + self.velocity
        else:
            print('using provided velocity model.')
            self.velocity_model = velocity_model

        self.theoretical_t = {
             'h1':self.calc_theoretical_travel_times(hydrophone_distance_grid=self.dr['h1'])
            ,'h2':self.calc_theoretical_travel_times(hydrophone_distance_grid=self.dr['h2'])
            ,'h3':self.calc_theoretical_travel_times(hydrophone_distance_grid=self.dr['h3'])
            ,'h4':self.calc_theoretical_travel_times(hydrophone_distance_grid=self.dr['h4'])
            ,'h5':self.calc_theoretical_travel_times(hydrophone_distance_grid=self.dr['h5'])
            ,'h6':self.calc_theoretical_travel_times(hydrophone_distance_grid=self.dr['h6'])
        }
     
    def make_grid(self):
        """
        Creates cartesian grid around borehole
        
        Converts cylindrical coordinates of radius and depth into cartesian coordinates
        
        Returns
        --------------------
        grid : numpy 3D array
        """
        self.x = np.arange(-self.radius, self.radius, self.gridscale)
        self.y = np.arange(0, self.depth, self.gridscale)
        self.z = np.arange(-self.radius, self.radius, self.gridscale)
        return np.meshgrid(self.x, self.y, self.z)
    
    def calc_distance_grid(self, hydrophone_location):
        """
        calculates the distance from every grid point to every hydrophone location
        
        Returns
        ---------------------
        numpy.array : float
            an array of each grid point calculated distance to the selected hydrophone.
        """
        return np.sqrt(self.xx**2 + (self.yy - hydrophone_location)**2 + self.zz**2)
        
    def calc_theoretical_travel_times(self, hydrophone_distance_grid):
        """
        calculates the theoretical travel time for an event for each grid point to the selected hydrophone.
        
        Returns
        ---------------------
        numpy.array : float
            an array of each grid point calcuated travel time to the selected hydrophone.
        """
        return hydrophone_distance_grid/self.velocity_model
    
    def get_grid_slice(self, grid, dimension, slice):
        """
        returns grid slice through specified dimension and slice
        
        Returns
        ---------------------
        numpy.array : float
            a 2d array that represents a slice of the grid around the borehole
        """
        dimension = dimension.upper()
        if dimension == 'X':
            return grid[slice, :, :]
        elif dimension == 'Y':
            return grid[:, slice, :]
        elif dimension == 'Z':
            return grid[:, :, slice]
        else:
            raise ValueError('{G} is not a dimension.'.format(G=dimension))
    
    def get_grid_midslice(self, grid):
        """
        Same as get_grid_slice but gets the middle slice only. This grid slice transects the borehole.
        
        Returns
        ---------------------
        numpy.array : float
            a 2d array that represents the middle slice of the grid around the borehole
        """
        midpoint = self.radius
        return self.get_grid_slice(grid=grid, dimension='y', slice=midpoint)
    
    def _calc_travel_time_for_grid_point(self, i, j, k, good_picks, event_times, uncertainty=0.1):
        """
        Calculates the root mean square of the estimate of the location based on the estimated travel time from grid point (i, j, k) to each hydrophone.
        
        Uses beginning of the hour to pick a 'close enough' starting point, then minimizes uses least squares the best fit line between the selected hydrophone and the grid point to caluclate the root mean square error of this calculation for grid point (i, j, k).
        
        Params
        --------------------
        i : int
            index for x component of theoretical arrival array
        j : int
            index for y component of theoretical arrival array
        k : int
            index for k component of theoretical arrival array
        good_picks : list
            list of 1s and 0s ordered by hydrophones (top to bottom). 1s indicate good picks.
        uncertainty : float
            the uncertainty in the measurement for travel times. In seconds.
        hydrophone : str
            hydrophone id.
        
        Returns
        --------------------
        travel_time : float
        """
        
        theoretical_arrivals = [
             self.theoretical_t['h1'][i, j, k]
            ,self.theoretical_t['h2'][i, j, k]
            ,self.theoretical_t['h3'][i, j, k]
            ,self.theoretical_t['h4'][i, j, k]
            ,self.theoretical_t['h5'][i, j, k]
            ,self.theoretical_t['h6'][i, j, k]
        ]
        
        # create diagonal matrix for arrival times identified as 
        # reasonable picks based on their signal to noise ratio
        good_picks = np.diag(good_picks)
        
        # data vector, difference between true times and the theoretical times
        # event times
        dvec = np.vstack(np.array(event_times)) - np.vstack(np.array(theoretical_arrivals))
        
        # weights the measurements
        weight_matrix = np.vstack(uncertainty*good_picks)
        
        # matrix of ones
        gvec = np.ones_like(good_picks).astype(float)
        
        # create top and bottom
        t0_top = gvec.transpose().dot(weight_matrix).dot(gvec)
        t0_bottom = gvec.dot(weight_matrix).dot(dvec)
        
        # Computes the vector x that approximately solves the equation a @ x = b. The @ sign designates reverse division.
        # # https://stackoverflow.com/questions/7160162/left-matrix-division-and-numpy-solve
        t0 = np.linalg.lstsq(t0_top, t0_bottom)[0]
        
        # error vector
        evec = np.zeros_like(good_picks)
        evec = dvec - gvec.dot(t0)
        rms_ijk = np.sqrt(evec.transpose().dot(weight_matrix).dot(evec) / np.sum(np.diag(weight_matrix)))[0][0]
        return rms_ijk
    
    def _calc_travel_times_serial(self, ijk, good_picks, event_times, uncertainty=0.1):
        """
        Calculates the RMS grid serialized.
        
        Wrapper function for calc_travel_time_for_grid_point to apply to all grid points.
        
        Returns
        --------------------
        None.
        """
        i_s, j_s, k_s = ijk
        
        self.rms_grid = np.zeros_like(self.theoretical_t['h3'])
        
        for i in i_s:
            for j in j_s:
                for k in k_s:
                    self.rms_grid[i, j, k] = self._calc_travel_time_for_grid_point(i=i, j=j, k=k, good_picks=good_picks, event_times=event_times, uncertainty=uncertainty)
    
    def _calc_travel_times_multiprocessing(self):
        """
        docstring
        """
    
    def calc_travel_times(self, event, method, centered):
        """
        Wrapper function to calculate travel times.
        
        Parameters
        ------------------
        method : str
            toggle switch to calculate RMS grid in a serial fashion or in parallel. Options: 'serial', 'multiprocessing'.
        centered : binary
            toggle switch to calculate on full 3d grid or on a centered slice.
        """
        
        good_picks = event['good_picks']
        event_times = event['event_times']
        
        i_s = np.arange(0, self.theoretical_t['h3'].shape[0], 1)
        
        if centered == True:
            # only calculate on the mid point slice
            j_s = (self.radius,)
            
        else:
            j_s = np.arange(0, self.theoretical_t['h3'].shape[1], 1)   
                             
        k_s = np.arange(0, self.theoretical_t['h3'].shape[2], 1)
                             
        if method == 'serial':
            self._calc_travel_times_serial(ijk=(i_s, j_s, k_s), good_picks=good_picks, event_times=event_times, uncertainty=0.1)
                             
        elif method == 'multiprocessing':
            self._calc_travel_times_multiprocessing()
                             
        else:
            raise ValueError('{m} is not a method for calculating rms.'.format(m=method))
        print('rms_grid calculated.')
        
#     def calc_rms_min(self, centered=True):
#         """
#         calculates rms minimum from rms grid.
        
#         assumes
        
#         Returns
#         ------------------
#         rms_min : float
#             minimum of rms_grid.
#         """
#         if centered == True:
#             return self.rms_grid[:, self.radius, :].min()
        
    def calc_location(self):
        """
        returns the location derived from the minimized RMS grid
        
        Returns
        ------------------
        location : tuple
            (radius, depth)
        """
        # centered_matrix_location = np.unravel_index(self.rms_grid[:, 50, :].argmin(), self.rms_grid[:, 50, :].shape)
        centered_matrix_location = np.unravel_index(self.rms_grid[:, self.radius, :].argmin(), self.rms_grid[:, self.radius, :].shape)
        
        radius = self.x[centered_matrix_location[1]]
        depth = -self.y[centered_matrix_location[0]]
        return radius, depth
        
    def plot_rms_grid(self):
        """
        makes matplotlib pcolormesh plot of RMS grid
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # cbar = ax.pcolormesh(self.x, -self.y, self.rms_grid[:,50,:], cmap='nipy_spectral_r', shading='auto')
        cbar = ax.pcolormesh(self.x, -self.y, self.rms_grid[:,self.radius,:], cmap='nipy_spectral_r', shading='auto')
        fig.colorbar(cbar, label='Root Mean Squared Error')
        
        hdepths = np.array(list(self.hydrophone_locations.values()))
        ax.plot(np.zeros_like(hdepths), -hdepths, color='black', marker='s')
        hlabels = self.hydrophone_locations.keys()
        for h, y in zip(hlabels, hdepths):
            ax.text(s=h, x=0, y=-y)