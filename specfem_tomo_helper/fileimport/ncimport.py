#!/usr/bin/env python

import netCDF4
import numpy as np
from scipy.interpolate import NearestNDInterpolator


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
        if 'up' in [self.dataset.geospatial_vertical_positive, self.dataset.variables['depth'].positive]:
            self.depth = np.array(self.dataset.variables['depth'])
        else:
            self.depth = np.array(self.dataset.variables['depth'])*-1
        self.lat = np.array(self.dataset.variables['latitude'])
        self.lon = np.array(self.dataset.variables['longitude'])
        return self.lon, self.lat, self.depth

    def load_variable(self, varname, fill_nan=False):
        """ Load model parameters

        Returns a Model_array object (values, name). The fill_nan flag can be set to 'vertical' to fill NaN values from bottom-up (deepest to shallowest). Any remaining NaN values will be extrapolated during interpolation using nearest neighbor.
        """
        dims = self.dataset.variables[str(varname)].dimensions
        var_data = np.array(self.dataset.variables[str(varname)][:, :, :].data)
        # Check if mask is set to True, and mask actually spans all model cells.
        if (self.dataset.variables[str(varname)].mask and
            np.shape(self.dataset.variables[str(varname)]) == np.shape(self.dataset.variables[str(varname)][:,:,:].mask)):
            var_mask = np.array(self.dataset.variables[str(varname)][:, :, :].mask)
            var_data[var_mask] = np.nan
        if dims == ('depth', 'latitude', 'longitude'):
            var_data = np.moveaxis(var_data, 0, -1)
            # Move depth axis to the last column.
        
        # Replace missing values with np.nan if needed
        var = self.dataset.variables[str(varname)]
        missing_value = getattr(var, 'missing_value', None)
        if missing_value is not None:
            var_data = np.where(var_data == missing_value, np.nan, var_data)

        # Robust density unit check and conversion
        if 'rho' in varname:
            units = getattr(var, 'units', '').lower()
            # Convert if units are g/cm^3
            if 'g/cm3' in units or 'g/cm^3' in units:
                var_data = var_data * 1000
            elif 'kg/m3' in units or 'kg/m^3' in units:
                pass  # Already in SI
            else:
                # Fallback: check median of valid values
                valid = var_data[np.isfinite(var_data)]
                if valid.size > 0:
                    median_val = np.median(valid)
                    # If median is less than 50, likely g/cm^3
                    if median_val < 50:
                        var_data = var_data * 1000
            
        if fill_nan == 'vertical':
            for depth_id in len(self.depth)-2-np.arange(len(self.depth)-1):
                var_data[:,:,depth_id][np.isnan(var_data[:,:,depth_id])] = var_data[:,:,depth_id+1][np.isnan(var_data[:,:,depth_id])]

        # Note: 'lateral' and 'horizontal' options removed since NaN values 
        # are automatically extrapolated during interpolation using nearest neighbor

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
