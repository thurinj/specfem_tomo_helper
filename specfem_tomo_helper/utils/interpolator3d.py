#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.interpolate
import matplotlib.pyplot as plt

class trilinear_interpolator():
    """ 3D interpolator. """

    def __init__(self, model, projection):
        # Initialise the various paramters of the interpolator, mainly the coordinates that will be used later.
        self.utm_x = None
        self.utm_y = None
        self.tomo_xyz = []
        self.lon, self.lat, self.depth = model.load_coordinates()
        self.y, self.x, self.z = np.meshgrid(self.lat, self.lon, self.depth, indexing='ij')
        utm_x, utm_y = projection(self.x, self.y)
        self.utm_x = utm_x.flatten()
        self.utm_y = utm_y.flatten()

    def interpolation_parameters(self, xspecfem_min, xspecfem_max, dx, yspecfem_min, yspecfem_max, dy, zspecfem_min, zspecfem_max, dz):
        """ Initialise the interpolation grid from a set of coordinates parameters. """
        # xspecfem_min/max corresponds to the Longitude min/max in Mesh_Par_File
        self.xspecfem_min = xspecfem_min
        self.xspecfem_max = xspecfem_max
        # dx is the easting sampling rate in m
        self.dx = dx
        # yspecfem_min/max corresponds to the Latitude min/max in Mesh_Par_File
        self.yspecfem_min = yspecfem_min
        self.yspecfem_max = yspecfem_max
        # dy is the northing sampling rate in m
        self.dy = dy
        # zmin and zmax are the desired interpolation bounds in km. zmin should be the positive topographic values (the "top" of the model), zmax is the maximum depth (should be negative in the vast majorities of instances).
        self.zmin = zspecfem_min
        self.zmax = zspecfem_max
        # dz is the depth sampling rate in m
        self.dz = dz
        # Here are defined the interpolation coordinates as a regular grid in utm coordinates.
        self.x_interp_coordinates = np.arange(self.xspecfem_min-self.dx,
                                              self.xspecfem_max+self.dx*2, self.dx)
        self.y_interp_coordinates = np.arange(self.yspecfem_min-self.dy,
                                              self.yspecfem_max+self.dy*2, self.dy)
        self.z_interp_coordinates = np.arange(self.zmin*1e3, self.zmax*1e3+self.dz, self.dz)
        # Create the meshgrid from the previously computed coordinates
        Z, Y, X = np.meshgrid(self.z_interp_coordinates,
                              self.y_interp_coordinates,
                              self.x_interp_coordinates, indexing='ij')
        # Transform the (nz, ny, nx) meshgrid arrays, to 3 vectors.
        self.Y_grid = Y.flatten()
        self.X_grid = X.flatten()
        self.Z_grid = Z.flatten()

    def interpolation_grid(self):
        """ This function can be used if you wish to simply return the interpolation grid coordinates (only the easting and northing). It might be useful to vizualise the interpolation points on a map. """
        self.x_interp_coordinates = np.arange(self.xspecfem_min-self.dx,
                                              self.xspecfem_max+self.dx*2, self.dx)
        self.y_interp_coordinates = np.arange(self.yspecfem_min-self.dy,
                                              self.yspecfem_max+self.dy*2, self.dy)
        Y, X = np.meshgrid(self.y_interp_coordinates, self.x_interp_coordinates, indexing='ij')
        self.Y_grid = Y.flatten()
        self.X_grid = X.flatten()
        return (self.x_interp_coordinates, self.y_interp_coordinates)

    def interpolate(self, model_param):
        """ Computes the interpolated model on a regular UTM grid from the predefined interpolation grid and the netCDF model_param (model paramters such as vp, vs, rho).
        Prior to using this function, the trilinear_interpolator.interpolation_parameters() function must be run. After that, the interpolation grid and coordinates are generated withing the interpolator object.

        ------------
        example:
        interpolator.interpolation_parameters(longitude_min, longitude_max, dx,
                                              latitude_min, latitude_max, dy,
                                              z_min, z_max)
        tomography_xyz = interpolator.interpolate([vp, vs, rho])
        """
        # Check that the model parameters are within a list, and create a list from the input values, to avoid syntax errors.
        if not type(model_param) is list:
            model_param = [model_param]
        param_names = [param.name for param in model_param]
        # Compute the original model sampling, and define the required padding lenght to make sure the interpolation domain lies within the original model.
        model_pad_y = np.diff(self.lat[0:2])[0]*111000
        model_pad_x = np.diff(self.lon[0:2])[0]*111000
        # Set model parameters array to vector list to create the required data structure
        model_values = [param.values.flatten() for param in model_param]
        data_array = np.vstack(
            (self.x.flatten(), self.y.flatten(), self.z.flatten()*1e3, model_values, self.utm_x, self.utm_y)).T
        # Create a pandas dataframe from the data_array above.
        dframe = pd.DataFrame(data=data_array, columns=[
                             'lon', 'lat', 'depth'] + param_names + ['utm_x', 'utm_y'])
        # Select the desired model bounds for interpolation, with the pandas.DataFrame filtering option. This reduces the cost of interpolation withing a dense 2D model by getting rid of unecessary coordinates.
        filtered_df = dframe[(dframe['utm_y'] >= self.Y_grid.min()-model_pad_y) & (dframe['utm_y'] <= self.Y_grid.max()+model_pad_y) & (
            dframe['utm_x'] >= self.X_grid.min()-model_pad_x) & (dframe['utm_x'] <= self.X_grid.max()+model_pad_x)]
        # Define the interpolation parameters:
        # 1) Taking the coordinates vector
        utm_xyz = np.vstack((filtered_df['utm_x'], filtered_df['utm_y'], filtered_df['depth'])).T
        # 2) Taking the correspondig parameters to interpolate
        filtered_model_param = np.asarray(filtered_df[param_names]).T
        # Actual interpolation for each params in the model_param list.
        interpolated_params = [scipy.interpolate.LinearNDInterpolator(utm_xyz, param)(
            self.X_grid, self.Y_grid, self.Z_grid)
            for param in filtered_model_param]
        # Create the final array with the structure required for specfem (lon, lat, depth, model parameters)
        self.tomo_xyz = np.vstack(
            (self.X_grid, self.Y_grid, self.Z_grid, interpolated_params)).T

        return self.tomo_xyz
