#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.interpolate

class bilinear_interpolator():
    """ 2D interpolator. The model is interpolated over each of the defined depth slices and does not interpolate vertically. """

    def __init__(self, model, projection):
        self.utm_x = None
        self.utm_y = None
        self.tomo_xyz = []
        self.lon, self.lat, self.depth = model.load_coordinates()
        self.x, self.y = np.meshgrid(self.lon, self.lat, indexing='ij')
        utm_x, utm_y = projection(self.x, self.y)
        self.utm_x = utm_x.flatten()
        self.utm_y = utm_y.flatten()

    def interpolation_parameters(self, xspecfem_min, xspecfem_max, dx, yspecfem_min, yspecfem_max, dy, zmin, zmax):
        self.xspecfem_min = xspecfem_min
        self.xspecfem_max = xspecfem_max
        self.dx = dx
        self.yspecfem_min = yspecfem_min
        self.yspecfem_max = yspecfem_max
        self.dy = dy
        self.zmin = zmin
        self.zmax = zmax
        self.x_interp_coordinates = np.arange(self.xspecfem_min-self.dx,
                                         self.xspecfem_max+self.dx*2, self.dx)
        self.y_interp_coordinates = np.arange(self.yspecfem_min-self.dy,
                                         self.yspecfem_max+self.dy*2, self.dy)
        Y, X = np.meshgrid(self.y_interp_coordinates, self.x_interp_coordinates, indexing='ij')
        self.Y_grid = Y.flatten()
        self.X_grid = X.flatten()

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

    def interpolate(self,model_param):
        if not type(model_param) is list:
            model_param = [model_param]
        param_names = [param.name for param in model_param]

        self.z_interp_coordinates = []

        model_pad_y = np.diff(self.lat[0:2])[0]*111000
        model_pad_x = np.diff(self.lon[0:2])[0]*111000

        for depth_id, depth_val in enumerate(self.depth):
            if depth_val <= self.zmax and depth_val >= self.zmin:
                print(depth_id, depth_val)
                self.z_interp_coordinates.append(depth_val)
                z_model_slices = [param.values[:, :, depth_id].T.flatten() for param in model_param]
                Z_grid = np.ones_like(self.X_grid)*depth_val*1e3
                data_array = np.vstack(
                    (self.x.flatten(), self.y.flatten(), z_model_slices, self.utm_x, self.utm_y)).T
                frame = pd.DataFrame(data=data_array, columns=[
                                     'lon', 'lat'] + param_names + ['utm_x', 'utm_y'])

                df = frame[(frame['utm_y'] >= self.Y_grid.min()-model_pad_y) & (frame['utm_y'] <= self.Y_grid.max()+model_pad_y) & (
                    frame['utm_x'] >= self.X_grid.min()-model_pad_x) & (frame['utm_x'] <= self.X_grid.max()+model_pad_x)]

                utm_xy = np.vstack((df['utm_x'], df['utm_y'])).T
                filtered_model_param = np.asarray(df[param_names]).T
                interpolated_params = [scipy.interpolate.LinearNDInterpolator(utm_xy, param)(
                    self.X_grid, self.Y_grid)
                    for param in filtered_model_param]

                slice_interpolated_map = np.vstack(
                    (self.X_grid, self.Y_grid, Z_grid, interpolated_params)).T
                self.tomo_xyz.insert(0, slice_interpolated_map)
        self.tomo_xyz = np.concatenate(self.tomo_xyz)
        return self.tomo_xyz
