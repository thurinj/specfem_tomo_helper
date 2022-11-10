#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.interpolate import interp1d

class linear_interpolator1d2d():
    """ 1D+2D interpolator. """

    def __init__(self, model, projection):
        # Initialise the various paramters of the interpolator, mainly the coordinates that will be used later.
        self.utm_x = None
        self.utm_y = None
        self.tomo_xyz = []
        self.lon, self.lat, self.depth = model.load_coordinates()
        self.x, self.y = np.meshgrid(self.lon, self.lat, indexing='ij')
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
        self.z_interp_coordinates = np.flip(np.arange(self.zmax*1e3, self.zmin*1e3+self.dz, -self.dz))
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


    def interpolate(self,model_param):
        if not type(model_param) is list:
            model_param = [model_param]
        param_names = [param.name for param in model_param]

        z_coordinates = []

        model_pad_y = np.diff(self.lat[0:2])[0]*111000
        model_pad_x = np.diff(self.lon[0:2])[0]*111000

        for depth_id, depth_val in enumerate(self.depth):
            if depth_val <= self.zmax and depth_val >= self.zmin:
                print(depth_id, depth_val)
                z_coordinates.append(depth_val)
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

        nlay=len(self.x_interp_coordinates)*len(self.y_interp_coordinates)
        dep_int=np.unique(self.z_interp_coordinates)*1000
        tomo_lay=np.empty((nlay,3+len(param_names),len(dep_int)))
        for i,dep,in enumerate(dep_int):
            tomo_lay[:,:,i]=self.tomo_xyz[i*nlay:(i+1)*nlay,:]
        tomo_lay=np.swapaxes(tomo_lay,1,2)
        #interp with new depth
        self.tomo_xyz=np.empty((nlay,len(self.z_interp_coordinates),6))
        f_dep_interp=interp1d(dep_int,tomo_lay,axis=1)
        for i,zint in enumerate(dep_int):
            self.tomo_xyz[:,i,:]=f_dep_interp(zint)
        self.tomo_xyz= self.tomo_xyz.reshape(-1, self.tomo_xyz.shape[-1],order='F')
        return self.tomo_xyz