#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.interpolate

class trilinear_interpolator():
    """ 3D interpolator. """

    def __init__(self, model, projection):
        self.utm_x = None
        self.utm_y = None
        self.tomo_xyz = []
        self.lon, self.lat, self.depth = model.load_coordinates()
        self.y, self.x, self.z = np.meshgrid(self.lat, self.lon, self.depth, indexing='ij')
        utm_x, utm_y = projection(self.x, self.y)
        self.utm_x = utm_x.flatten()
        self.utm_y = utm_y.flatten()

    def interpolation_parameters(self, xspecfem_min, xspecfem_max, dx, yspecfem_min, yspecfem_max, dy, zspecfem_min, zspecfem_max, dz):
        self.xspecfem_min = xspecfem_min
        self.xspecfem_max = xspecfem_max
        self.dx = dx
        self.yspecfem_min = yspecfem_min
        self.yspecfem_max = yspecfem_max
        self.dy = dy
        self.zmin = zspecfem_min
        self.zmax = zspecfem_max
        self.dz = dz
        self.x_interp_coordinates = np.arange(self.xspecfem_min-self.dx,
                                              self.xspecfem_max+self.dx*2, self.dx)
        self.y_interp_coordinates = np.arange(self.yspecfem_min-self.dy,
                                              self.yspecfem_max+self.dy*2, self.dy)
        self.z_interp_coordinates = np.arange(self.zmin*1e3, self.zmax*1e3+self.dz, self.dz)
        Z, Y, X = np.meshgrid(self.z_interp_coordinates,
                              self.y_interp_coordinates,
                              self.x_interp_coordinates, indexing='ij')
        self.Y_grid = Y.flatten()
        self.X_grid = X.flatten()
        self.Z_grid = Z.flatten()

    def interpolation_grid(self):
        #
        self.x_interp_coordinates = np.arange(self.xspecfem_min-self.dx,
                                              self.xspecfem_max+self.dx*2, self.dx)
        self.y_interp_coordinates = np.arange(self.yspecfem_min-self.dy,
                                              self.yspecfem_max+self.dy*2, self.dy)
        Y, X = np.meshgrid(self.y_interp_coordinates, self.x_interp_coordinates, indexing='ij')
        self.Y_grid = Y.flatten()
        self.X_grid = X.flatten()
        return (self.x_interp_coordinates, self.y_interp_coordinates)

    def interpolate(self, model_param):
        if not type(model_param) is list:
            model_param = [model_param]
        param_names = [param.name for param in model_param]

        model_pad_y = np.diff(self.lat[0:2])[0]*111000
        model_pad_x = np.diff(self.lon[0:2])[0]*111000

        model_values = [param.values.flatten() for param in model_param]
        data_array = np.vstack(
            (self.x.flatten(), self.y.flatten(), self.z.flatten()*1e3, model_values, self.utm_x, self.utm_y)).T
        frame = pd.DataFrame(data=data_array, columns=[
                             'lon', 'lat', 'depth'] + param_names + ['utm_x', 'utm_y'])

        df = frame[(frame['utm_y'] >= self.Y_grid.min()-model_pad_y) & (frame['utm_y'] <= self.Y_grid.max()+model_pad_y) & (
            frame['utm_x'] >= self.X_grid.min()-model_pad_x) & (frame['utm_x'] <= self.X_grid.max()+model_pad_x)]

        utm_xyz = np.vstack((df['utm_x'], df['utm_y'], df['depth'])).T
        filtered_model_param = np.asarray(df[param_names]).T
        interpolated_params = [scipy.interpolate.LinearNDInterpolator(utm_xyz, param)(
            self.X_grid, self.Y_grid, self.Z_grid)
            for param in filtered_model_param]

        self.tomo_xyz = np.vstack(
            (self.X_grid, self.Y_grid, self.Z_grid, interpolated_params)).T
        return self.tomo_xyz
