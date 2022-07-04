"""
class for making grids around the borehole
"""

import numpy as np


class Grid:
    """
    Grid for calculating theoretical arrival times
    
    Parameters
    -------------------
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
        
    Functions
    -------------------
    make_grid
    get_grid_slice
    get_grid_midslice
    """
    
    def __init__(self, radius, gridscale, velocity_model, depth=400.,):
        self.radius = radius
        self.depth = depth
        
        # TODO : make grid
        
        if isinstance(velocity_model, float):
            print('creating constant velocity model. vp = {vp} m/s'.format(vp=velocity_model))
            self.velocity = velocity_model
            # TODO : implement grid maker for velocity model
        else:
            print('using provided velocity model.')
            self.velocity_model = velocity_model
            
    def make_grid(self):
        """
        Creates cartesian grid around borehole
        """
        
    def get_grid_slice(self):
        pass
    
    def get_grid_midslice(self):
        pass