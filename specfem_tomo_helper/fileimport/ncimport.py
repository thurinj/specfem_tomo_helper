#!/usr/bin/env python

import netCDF4
import numpy as np


class Nc_model:
    """ netCDF model container """

    def __init__(self, path):
        self.path = path
        self.dataset = None

        self.lat = []
        self.lon = []
        self.depth = []

        self.load_ncfile()

    def load_ncfile(self):
        """ Load netCDF file from model path

        Notes
        ----------
        The function also check that the bundled model from IRIS EMC contains
        latitude, longitude and depth parameters defined. Most models on the IRIS
        EMC database should follow this convention. Custom coordinates names is note
        yet implemented
        The raw netCDF is stored in self.dataset
        """

        # Safety block parameters definition
        wanted_params = ['latitude', 'longitude', 'depth']
        ncfile = netCDF4.Dataset(self.path)
        varkeys = [varkey for varkey in ncfile.variables.keys()]
        for param in wanted_params:
            if not param in varkeys:
                raise AssertionError('Missing '+param+' key in netCDF variables')
        self.dataset = ncfile

    def load_coordinates(self):
        """ Load model coordinates as numpy.array()

        Output format: lon, lat, depth
        """
        if self.dataset.variables['depth'].positive == 'up':
            self.depth = np.array(self.dataset.variables['depth'])
        else:
            self.depth = np.array(self.dataset.variables['depth'])*-1
        self.lat = np.array(self.dataset.variables['latitude'])
        self.lon = np.array(self.dataset.variables['longitude'])
        return self.lon, self.lat, self.depth

    def load_variable(self, varname, fill_nan=False):
        """ Load model parameters

        Returns a Model_array object (values, name). the fill_nan flag is used to fill the nan values from the bottom-up. It proceeds with a sweep from the deepest part of the model and replaces all the nan values encountered from by the model value directly bellow.
        """
        dims = self.dataset.variables[str(varname)].dimensions
        var_data = np.array(self.dataset.variables[str(varname)][:, :, :].data)
        if dims == ('depth', 'latitude', 'longitude'):
            var_data = np.rot90(var_data.T,k=1)
        if self.dataset.variables[str(varname)].mask:
            var_mask = np.array(self.dataset.variables[str(varname)][:, :, :].mask)
            var_data[var_mask] = np.nan
        if fill_nan is True:
            for depth_id in len(self.depth)-2-np.arange(len(self.depth)-1):
                var_data[:,:,depth_id][np.isnan(var_data[:,:,depth_id])] = var_data[:,:,depth_id+1][np.isnan(var_data[:,:,depth_id])]

        return Model_array(varname, var_data)



class Model_array:
    """ netCDF model parameter container


    Notes
    ----------
    self.values return a numpy.array() with shape (lat x lon x depth)
    self.name contains the variable name used during the import
    """


    def __init__(self, name, values):
        self.name = name
        self.values = values
