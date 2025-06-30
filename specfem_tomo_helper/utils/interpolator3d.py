import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

class trilinear_interpolator():
    """
    3D interpolator using RegularGridInterpolator in lat-lon-depth space,
    with the final output on a uniform UTM x-y-depth grid.
    """

    def __init__(self, model, projection):
        self.projection = projection

        # Load coordinate arrays (1D)
        self.lon, self.lat, self.depth_km = model.load_coordinates()

        # For RegularGridInterpolator
        self.lat_array = self.lat
        self.lon_array = self.lon
        self.depth_array_m = self.depth_km * 1e3

        # Target grid placeholders
        self.x_interp_coordinates = None
        self.y_interp_coordinates = None
        self.z_interp_coordinates = None
        self.X_grid = None
        self.Y_grid = None
        self.Z_grid = None

        # Also store these for reference
        self.xspecfem_min = None
        self.xspecfem_max = None
        self.dx = None
        self.yspecfem_min = None
        self.yspecfem_max = None
        self.dy = None
        self.zmin = None
        self.zmax = None
        self.dz = None

    def interpolation_parameters(self,
                                 xspecfem_min, xspecfem_max, dx,
                                 yspecfem_min, yspecfem_max, dy,
                                 zspecfem_min, zspecfem_max, dz):
        self.xspecfem_min = xspecfem_min
        self.xspecfem_max = xspecfem_max
        self.dx = dx

        self.yspecfem_min = yspecfem_min
        self.yspecfem_max = yspecfem_max
        self.dy = dy

        self.zmin = zspecfem_min * 1e3
        self.zmax = zspecfem_max * 1e3
        self.dz = dz


        nx = int(np.round((self.xspecfem_max - self.xspecfem_min) / self.dx)) + 1
        ny = int(np.round((self.yspecfem_max - self.yspecfem_min) / self.dy)) + 1
        nz = int(np.round((self.zmax - self.zmin) / self.dz)) + 1
        
        self.x_interp_coordinates = np.linspace(self.xspecfem_min, self.xspecfem_max, nx)
        self.y_interp_coordinates = np.linspace(self.yspecfem_min, self.yspecfem_max, ny)
        self.z_interp_coordinates = np.linspace(self.zmin, self.zmax, nz)

        self.X_grid, self.Y_grid, self.Z_grid = np.meshgrid(
            self.x_interp_coordinates,
            self.y_interp_coordinates,
            self.z_interp_coordinates,
            indexing='ij'
        )

    def interpolate(self, model_param_list, chunk_size=200000):
        """
        Interpolate a list of 3D model parameters [param1_3d, param2_3d, ...]
        each shaped (n_lat, n_lon, n_depth), onto the uniform UTM x-y-depth grid.

        Optionally, specify `chunk_size` to process queries in batches
        and print progress for large grids.

        Returns:
            final_tomo_xyz : array of shape (N, 3 + n_params)
        """
        # Flatten out the target grid in 'F' order
        xq = self.X_grid.ravel(order='F')
        yq = self.Y_grid.ravel(order='F')
        zq = self.Z_grid.ravel(order='F')

        # Inverse projection: (UTM_x, UTM_y) -> (lon_new, lat_new)
        lon_new, lat_new = self.projection(xq, yq, inverse=True)

        # Build final coordinate array for RegularGridInterpolator queries
        query_points = np.column_stack((lat_new, lon_new, zq))
        n_total = query_points.shape[0]

        interpolated_params = []

        # For each parameter in model_param_list
        for param_idx, param in enumerate(model_param_list, start=1):
            if hasattr(param, "values"):
                data = param.values
            else:
                data = param

            # Build the interpolator for this parameter
            f = RegularGridInterpolator(
                (self.lat_array, self.lon_array, self.depth_array_m),
                data,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )

            # Now do chunked interpolation to show progress
            vals = np.empty(n_total, dtype=float)
            for start in range(0, n_total, chunk_size):
                end = min(start + chunk_size, n_total)
                vals[start:end] = f(query_points[start:end])
                # Print simple progress
                percent = (end / n_total) * 100
                print(f"  Param {param_idx}/{len(model_param_list)}: "
                      f"Interpolated {end}/{n_total} points ({percent:.1f}%)")

            if np.any(np.isnan(vals)):
                print(f"  Param {param_idx}/{len(model_param_list)}: "
                      f"Found NaN values in interpolation results: \n  Please make sure you are happy with the chosen bounds (we're extrapolating what's missing).")
                # Fill NaNs using nearest neighbor interpolation
                coord_grid_flat = np.column_stack((
                    self.X_grid.flatten(order='F'),
                    self.Y_grid.flatten(order='F'),
                    self.Z_grid.flatten(order='F'),
                ))

                vals = fill_nan_from_nearest_interpolated(coord_grid_flat, vals)

            interpolated_params.append(vals)

        # Assemble final array
        final_tomo_xyz = np.column_stack((
            self.X_grid.flatten(order='F'),
            self.Y_grid.flatten(order='F'),
            self.Z_grid.flatten(order='F')
        ))

        for arr in interpolated_params:
            final_tomo_xyz = np.column_stack((final_tomo_xyz, arr))

        return final_tomo_xyz


def fill_nan_from_nearest_interpolated(grid_flat, values_flat):
    """
    grid_flat: (N, 3) array of X, Y, Z coordinates
    values_flat: (N,) array of interpolated values, some of which are NaN
    """
    nan_mask = np.isnan(values_flat)
    not_nan_mask = ~nan_mask

    if not np.any(nan_mask):
        return values_flat  # nothing to fill

    # Build KD-tree on valid points
    tree = cKDTree(grid_flat[not_nan_mask])
    
    # Query nearest neighbor for NaN points
    _, idx = tree.query(grid_flat[nan_mask], k=1)
    
    # Replace NaNs with the value from the nearest valid neighbor
    values_flat[nan_mask] = values_flat[not_nan_mask][idx]
    return values_flat