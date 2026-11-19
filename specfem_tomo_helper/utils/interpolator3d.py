import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


class trilinear_interpolator():
    """
    3D interpolator using RegularGridInterpolator in lat-lon-depth space.

    The final output is on a UTM x-y grid. The depth grid can either be:
      1. Uniformly spaced using dz, or
      2. Taken directly from the input model when variable_spacing=True.
    """

    def __init__(self, model, projection):
        self.projection = projection

        # Load coordinate arrays (1D)
        self.lon, self.lat, self.depth_km = model.load_coordinates()

        # For RegularGridInterpolator
        self.lat_array = np.asarray(self.lat)
        self.lon_array = np.asarray(self.lon)
        self.depth_array_m = np.asarray(self.depth_km) * 1e3

        # Target grid placeholders
        self.x_interp_coordinates = None
        self.y_interp_coordinates = None
        self.z_interp_coordinates = None
        self.X_grid = None
        self.Y_grid = None
        self.Z_grid = None

        # Store parameters for reference
        self.xspecfem_min = None
        self.xspecfem_max = None
        self.dx = None

        self.yspecfem_min = None
        self.yspecfem_max = None
        self.dy = None

        self.zmin = None
        self.zmax = None
        self.dz = None

        self.variable_spacing = False

    def interpolation_parameters(
        self,
        xspecfem_min, xspecfem_max, dx,
        yspecfem_min, yspecfem_max, dy,
        zspecfem_min, zspecfem_max, dz,
        variable_spacing=False
    ):
        """
        Define the target interpolation grid.

        Parameters
        ----------
        xspecfem_min, xspecfem_max : float
            UTM X limits in meters.

        dx : float
            Horizontal X grid spacing in meters.

        yspecfem_min, yspecfem_max : float
            UTM Y limits in meters.

        dy : float
            Horizontal Y grid spacing in meters.

        zspecfem_min, zspecfem_max : float
            Depth limits in meters.

        dz : float
            Uniform depth spacing in meters. Used only when
            variable_spacing=False.

        variable_spacing : bool
            If True, use the input model's depth coordinates instead
            of generating a uniform depth grid.
        """

        self.xspecfem_min = xspecfem_min
        self.xspecfem_max = xspecfem_max
        self.dx = dx

        self.yspecfem_min = yspecfem_min
        self.yspecfem_max = yspecfem_max
        self.dy = dy

        self.zmin = zspecfem_min * 1e3
        self.zmax = zspecfem_max * 1e3
        self.dz = dz
        self.variable_spacing = variable_spacing

        # ---------------------------------------------------------
        # X grid
        # ---------------------------------------------------------
        nx = int(np.round(
            (self.xspecfem_max - self.xspecfem_min) / self.dx
        )) + 1

        self.x_interp_coordinates = np.linspace(
            self.xspecfem_min,
            self.xspecfem_max,
            nx
        )

        # ---------------------------------------------------------
        # Y grid
        # ---------------------------------------------------------
        ny = int(np.round(
            (self.yspecfem_max - self.yspecfem_min) / self.dy
        )) + 1

        self.y_interp_coordinates = np.linspace(
            self.yspecfem_min,
            self.yspecfem_max,
            ny
        )

        # ---------------------------------------------------------
        # Z grid
        # ---------------------------------------------------------
        if self.variable_spacing:
            # Use the actual depth coordinates from the input model.
            depth = np.asarray(self.depth_array_m, dtype=float)

            # Keep only depths inside requested range.
            z_mask = (
                (depth >= self.zmin) &
                (depth <= self.zmax)
            )

            self.z_interp_coordinates = depth[z_mask]

            if self.z_interp_coordinates.size == 0:
                raise ValueError(
                    "Variable depth spacing is enabled, but no input "
                    f"depth coordinates fall within the requested "
                    f"range [{zspecfem_min}, {zspecfem_max}] km."
                )

            # Make sure depths are sorted and unique.
            self.z_interp_coordinates = np.unique(
                self.z_interp_coordinates
            )

            verbose_message = (
                f"Using variable depth spacing from input model: "
                f"{len(self.z_interp_coordinates)} depth levels"
            )
            print(verbose_message)

        else:
            # Existing uniform-depth behavior.
            nz = int(np.round(
                (self.zmax - self.zmin) / self.dz
            )) + 1

            self.z_interp_coordinates = np.linspace(
                self.zmin,
                self.zmax,
                nz
            )

        # ---------------------------------------------------------
        # Build 3D grid
        # ---------------------------------------------------------
        self.X_grid, self.Y_grid, self.Z_grid = np.meshgrid(
            self.x_interp_coordinates,
            self.y_interp_coordinates,
            self.z_interp_coordinates,
            indexing='ij'
        )

    def interpolate(self, model_param_list, chunk_size=200000):
        """
        Interpolate a list of 3D model parameters
        [param1_3d, param2_3d, ...] onto the target
        UTM x-y-depth grid.

        Returns
        -------
        final_tomo_xyz : ndarray
            Shape (N, 3 + n_params)
        """

        # Flatten target grid in F order.
        xq = self.X_grid.ravel(order='F')
        yq = self.Y_grid.ravel(order='F')
        zq = self.Z_grid.ravel(order='F')

        # Inverse projection:
        # UTM x/y -> longitude/latitude
        lon_new, lat_new = self.projection(
            xq,
            yq,
            inverse=True
        )

        query_points = np.column_stack((
            lat_new,
            lon_new,
            zq
        ))

        n_total = query_points.shape[0]

        interpolated_params = []

        for param_idx, param in enumerate(
            model_param_list,
            start=1
        ):
            if hasattr(param, "values"):
                data = param.values
            else:
                data = param

            f = RegularGridInterpolator(
                (
                    self.lat_array,
                    self.lon_array,
                    self.depth_array_m
                ),
                data,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )

            vals = np.empty(n_total, dtype=float)

            for start in range(0, n_total, chunk_size):
                end = min(
                    start + chunk_size,
                    n_total
                )

                vals[start:end] = f(
                    query_points[start:end]
                )

                percent = (end / n_total) * 100

                print(
                    f"  Param {param_idx}/"
                    f"{len(model_param_list)}: "
                    f"Interpolated {end}/{n_total} "
                    f"points ({percent:.1f}%)"
                )

            if np.any(np.isnan(vals)):
                print(
                    f"  Param {param_idx}/"
                    f"{len(model_param_list)}: "
                    f"Found NaN values in interpolation results."
                )

                print(
                    "  Filling missing values using nearest "
                    "neighbor interpolation."
                )

                coord_grid_flat = np.column_stack((
                    self.X_grid.flatten(order='F'),
                    self.Y_grid.flatten(order='F'),
                    self.Z_grid.flatten(order='F'),
                ))

                vals = fill_nan_from_nearest_interpolated(
                    coord_grid_flat,
                    vals
                )

            interpolated_params.append(vals)

        # Assemble coordinates.
        final_tomo_xyz = np.column_stack((
            self.X_grid.flatten(order='F'),
            self.Y_grid.flatten(order='F'),
            self.Z_grid.flatten(order='F')
        ))

        # Append model parameters.
        for arr in interpolated_params:
            final_tomo_xyz = np.column_stack((
                final_tomo_xyz,
                arr
            ))

        return final_tomo_xyz


def fill_nan_from_nearest_interpolated(
    grid_flat,
    values_flat
):
    """
    Fill NaN values using the nearest valid grid point.
    """

    nan_mask = np.isnan(values_flat)
    not_nan_mask = ~nan_mask

    if not np.any(nan_mask):
        return values_flat

    if not np.any(not_nan_mask):
        raise ValueError(
            "All interpolated values are NaN; cannot perform "
            "nearest-neighbor filling."
        )

    tree = cKDTree(
        grid_flat[not_nan_mask]
    )

    _, idx = tree.query(
        grid_flat[nan_mask],
        k=1
    )

    values_flat[nan_mask] = (
        values_flat[not_nan_mask][idx]
    )

    return values_flat

