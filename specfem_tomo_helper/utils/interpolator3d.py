import numpy as np
from scipy.interpolate import LinearNDInterpolator

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
        """ Initialize the interpolation grid from a set of coordinates parameters. """
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
        self.zmin = zspecfem_min * 1e3  # Convert to meters
        self.zmax = zspecfem_max * 1e3  # Convert to meters
        # dz is the depth sampling rate in m
        self.dz = dz
        # Here are defined the interpolation coordinates as a regular grid in utm coordinates.
        self.x_interp_coordinates = np.arange(self.xspecfem_min, self.xspecfem_max + self.dx, self.dx)
        self.y_interp_coordinates = np.arange(self.yspecfem_min, self.yspecfem_max + self.dy, self.dy)
        self.z_interp_coordinates = np.arange(self.zmin, self.zmax + self.dz, self.dz)
        self.X_grid, self.Y_grid, self.Z_grid = np.meshgrid(self.x_interp_coordinates, self.y_interp_coordinates, self.z_interp_coordinates, indexing='ij')

    def interpolate(self, model_param):
        model_values = [param.values.flatten() for param in model_param]

        utm_x, utm_y = self.utm_x, self.utm_y
        lon, lat, depth = self.x.flatten(), self.y.flatten(), self.z.flatten() * 1e3

        model_pad_y = np.diff(self.lat[:2])[0] * 111000
        model_pad_x = np.diff(self.lon[:2])[0] * 111000

        valid_indices = np.where(
            (utm_y >= self.Y_grid.min() - model_pad_y) &
            (utm_y <= self.Y_grid.max() + model_pad_y) &
            (utm_x >= self.X_grid.min() - model_pad_x) &
            (utm_x <= self.X_grid.max() + model_pad_x)
        )[0]

        utm_x, utm_y, lon, lat, depth = utm_x[valid_indices], utm_y[valid_indices], lon[valid_indices], lat[valid_indices], depth[valid_indices]
        model_values = [values[valid_indices] for values in model_values]

        utm_xyz = np.vstack((utm_x, utm_y, depth)).T

        interpolated_params = []
        interp_points = np.vstack((self.X_grid.flatten(), self.Y_grid.flatten(), self.Z_grid.flatten())).T
        for values in model_values:
            interpolator = LinearNDInterpolator(utm_xyz, values)
            interpolated_params.append(interpolator(interp_points).reshape(self.X_grid.shape))

        tomo_volume = np.stack(interpolated_params, axis=-1)

        final_tomo_xyz = np.column_stack((
            self.X_grid.flatten(order='F'), 
            self.Y_grid.flatten(order='F'), 
            self.Z_grid.flatten(order='F')
        ))

        for param in interpolated_params:
            final_tomo_xyz = np.column_stack((final_tomo_xyz, param.flatten(order='F')))

        return final_tomo_xyz
