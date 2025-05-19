import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import cm
import matplotlib.colors as colors 
from scipy.ndimage import gaussian_filter
from scipy.interpolate import LinearNDInterpolator
import pyproj
from netCDF4 import Dataset
import urllib.request

# -- ~ ~ Defining colors ~ ~ --
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256)) 
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256)) 
all_colors = np.vstack((colors_undersea, colors_land)) 
terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

class TopographyProcessor:
    def __init__(self, interpolator, myProj, save_dir="./topography_analysis", smoothing_sigma=1, tomo=None, mesh_processor=None):
        """
        Initializes the processor with an interpolator and parameters.
        :param interpolator: Instance of `trilinear_interpolator` with grid setup.
        :param save_dir: Directory where results will be saved.
        :param smoothing_sigma: Smoothing sigma for Gaussian filter.
        """
        self.interpolator = interpolator
        self.utm_zone = myProj.crs.utm_zone
        self.x_interp = interpolator.x_interp_coordinates
        self.y_interp = interpolator.y_interp_coordinates
        self.save_dir = save_dir
        self.smoothing_sigma = smoothing_sigma
        self.tomo = tomo
        self.mesh_processor = mesh_processor
        os.makedirs(self.save_dir, exist_ok=True)
        self.filename = None

    def download_etopo_netcdf(self):
        """Downloads the ETOPO1 NetCDF file if not already present in the package's download directory."""
        import importlib.resources

        url = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/netcdf/ETOPO1_Bed_g_gmt4.grd.gz"
        # Locate the data directory within the installed package
        data_dir = os.path.join(os.path.dirname(__file__), '../download')
        os.makedirs(data_dir, exist_ok=True)
        self.filename = os.path.join(data_dir, "ETOPO1_Ice_c_gmt4.grd")

        if not os.path.isfile(self.filename):
            print("Downloading ETOPO1 NetCDF data...")
            gz_filename = self.filename + ".gz"
            urllib.request.urlretrieve(url, gz_filename, self._show_progress)
            print("\nDownload complete.")
            os.system(f"gunzip {gz_filename}")
            print("Unzip complete.")
        else:
            print("ETOPO1 NetCDF data already downloaded.")
        return self.filename

    @staticmethod
    def _show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        progress = downloaded / total_size if total_size > 0 else 0
        bar_length = 50
        bar = '#' * int(progress * bar_length) + '-' * (bar_length - int(progress * bar_length))
        print(f"\rDownloading: |{bar}| {progress:.2%}", end='')

    def get_etopo_data_netcdf(self, lon_min, lon_max, lat_min, lat_max):
        """Reads and crops the ETOPO1 data."""
        print("Reading and cropping ETOPO1 NetCDF data...")
        dataset = Dataset(self.filename, 'r')
        lon = dataset.variables['x'][:]
        lat = dataset.variables['y'][:]
        topo = dataset.variables['z'][:]

        lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]

        lon_cropped = lon[lon_inds]
        lat_cropped = lat[lat_inds]
        topo_cropped = topo[np.ix_(lat_inds, lon_inds)]

        print("Data cropping complete.")
        return lon_cropped, lat_cropped, topo_cropped



    def interpolate_topography(self):
        """
        Interpolates topography for the grid specified by the interpolator.
        """
        Xi, Yi = np.meshgrid(self.x_interp, self.y_interp)
        proj = pyproj.Proj(proj="utm", zone=self.utm_zone[:-1], datum="WGS84")
        Xloni, Ylati = proj(Xi.ravel(), Yi.ravel(), inverse=True)
        self.download_etopo_netcdf()
        lon_min, lon_max = np.min(Xloni), np.max(Xloni)
        lat_min, lat_max = np.min(Ylati), np.max(Ylati)
        Xlon, Ylat, Zt = self.get_etopo_data_netcdf(
            lon_min - 0.01 * np.abs(lon_min), lon_max + 0.01 * np.abs(lon_max),
            lat_min - 0.01 * np.abs(lat_min), lat_max + 0.01 * np.abs(lat_max)
        )
        Xlon, Ylat = np.meshgrid(Xlon, Ylat)
        interp = LinearNDInterpolator((Xlon.ravel(), Ylat.ravel()), Zt.ravel())
        Zli = interp(Xloni.ravel(), Ylati.ravel()).reshape(Xi.shape)
        if self.smoothing_sigma > 0:
            Zli = gaussian_filter(Zli, sigma=self.smoothing_sigma)
        return Xi, Yi, Zli

    def calculate_slope(self, Zi, dx, dy):
        """Calculates slope from the topography."""
        dFdx, dFdy = np.gradient(Zi, dx, dy)
        Fgrad = np.sqrt(dFdx**2 + dFdy**2)
        return np.arctan(Fgrad) * 180 / np.pi

    def save_results(self, slope_thresholds=[10, 15, 20], doubling_layers=None):
        """
        Saves all results: topography, slopes, histogram, and doubling layers.
        """
        Xi, Yi, Zi = self.interpolate_topography()
        dx, dy = Xi[0, 1] - Xi[0, 0], Yi[1, 0] - Yi[0, 0]
        Yslope = self.calculate_slope(Zi, dx, dy)

        # Save topography data and image
        np.savetxt(os.path.join(self.save_dir, "topography.txt"), Zi.flatten(), fmt="%.1f")
        plt.figure()

        print('vmin', np.min(Zi))
        print('vmax', np.max(Zi))
        if np.min(Zi) < 0 and np.max(Zi) < 0:  # Fully underwater case
            divnorm = colors.TwoSlopeNorm(vmin=np.min(Zi), vcenter=np.mean(Zi), vmax=np.max(Zi))
            colors_water_only = plt.cm.terrain(np.linspace(0, 0.17, 256))  # Use only water part of the colormap
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', colors_water_only)
            print('vcenter', np.mean(Zi))
        elif np.min(Zi) < 0 and np.max(Zi) > 0 :  # Mixed case
            divnorm = colors.TwoSlopeNorm(vmin=np.min(Zi), vcenter=0, vmax=np.max(Zi))
            # Combine water and land parts, excluding the turquoise region
            colors_combined = np.vstack((plt.cm.terrain(np.linspace(0, 0.17, 128)), plt.cm.terrain(np.linspace(0.25, 1, 128))))
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', colors_combined)
            print('vcenter', 0)
        else:  # Fully above water
            divnorm = colors.TwoSlopeNorm(vmin=np.min(Zi), vcenter=np.mean(Zi), vmax=np.max(Zi))
            colors_land_only = plt.cm.terrain(np.linspace(0.25, 1, 256))  # Use only land part of the colormap
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', colors_land_only)
            print('vcenter', np.mean(Zi))

        # Plot topography
        plt.pcolor(Xi, Yi, Zi, shading="auto", cmap=terrain_map, norm=divnorm)
        plt.colorbar(label="Elevation (m)")
        plt.title("Topography")
        plt.savefig(os.path.join(self.save_dir, "topography_visualization.png"))

        # Save a version of the image re-projected in lat/lon
        plt.figure()
        Xi_lon, Yi_lat = pyproj.Proj(proj="utm", zone=self.utm_zone[:-1], datum="WGS84")(Xi, Yi, inverse=True)
        plt.pcolor(Xi_lon, Yi_lat, Zi, shading="auto", cmap=terrain_map, norm=divnorm)
        plt.colorbar(label="Elevation (m)")
        plt.title("Topography (Lat/Lon)")
        plt.savefig(os.path.join(self.save_dir, "topography_visualization_latlon.png"))

        # Save slope data and visualizations
        plt.figure()
        norm = Normalize(vmin=0, vmax=np.max(Yslope))
        plt.pcolor(Xi, Yi, Yslope, shading="auto", cmap="viridis", norm=norm)
        plt.colorbar(label="Slope (degrees)")
        plt.title("Slope")

        # Generate colors dynamically based on the number of thresholds
        cmap = plt.cm.get_cmap("autumn", len(slope_thresholds))
        threshold_colors = [cmap(i) for i in range(len(slope_thresholds))]

        # Plot points with dynamically generated colors for each threshold
        for i, threshold in enumerate(slope_thresholds):
            mask = (Yslope > threshold) & (Yslope <= (slope_thresholds[i + 1] if i + 1 < len(slope_thresholds) else np.max(Yslope)))
            if np.any(mask):  # Only add to legend if there are points in this range
                plt.scatter(
                    Xi[mask],
                    Yi[mask],
                    color=threshold_colors[i],
                    s=5,
                    label=f"Slope > {threshold} degrees"
                )

        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.save_dir, "slope_visualization.png"))

        results = {}
        for threshold in slope_thresholds:
            mask = Yslope > threshold
            count = np.sum(mask)
            results[threshold] = {
                'count': count,
                'mask': mask
            }
            print(f'{count} points with slope > {threshold} degrees')

        # Save slope histogram
        plt.figure()
        plt.hist(Yslope.flatten(), bins=30, edgecolor="black", color="skyblue")
        plt.xlabel("Slope (degrees)")
        plt.ylabel("Frequency")
        plt.title("Slope Histogram")
        plt.savefig(os.path.join(self.save_dir, "slope_histogram.png"))

        # Save doubling layers
        n_layers = 0
        if doubling_layers:
            mean_topo = np.mean(Zi)
            for i, (base, factor) in enumerate(doubling_layers, 1):
                layer = base + (Zi - mean_topo) * factor
                np.savetxt(os.path.join(self.save_dir, f"layer_{i}.txt"), layer.flatten(), fmt="%.1f")
                print(f"Saved doubling layer {i}")
            n_layers = len(doubling_layers)
        else:
            # If not using doubling, just save one layer
            np.savetxt(os.path.join(self.save_dir, "layer_1.txt"), Zi.flatten(), fmt="%.1f")
            n_layers = 1

        print("All results saved successfully!")

        # Write interfaces.txt using the Jinja2 template
        # You may want to pass the actual mesh config if available
        try:
            from ..utils.mesh_processor import MeshProcessor
            mesh = getattr(self, 'mesh_processor', None)
            if mesh is not None and hasattr(mesh, '_vertical_cache') and mesh._vertical_cache is not None:
                vertical_config = mesh._vertical_cache
                horizontal_config = mesh.horizontal_dict()
            else:
                # Fallback: minimal config
                vertical_config = {'NREGIONS': n_layers, 'REGIONS': [(1, 1, 1, 1, 1, 1, -1)]*n_layers}
                horizontal_config = {'NEX_XI': Zi.shape[1]-1, 'NEX_ETA': Zi.shape[0]-1}
            self.write_interfaces_jinja(
                os.path.join(self.save_dir, "interfaces.txt"),
                vertical_config,
                horizontal_config,
                n_layers
            )
        except Exception as e:
            print(f"Could not write interfaces.txt with Jinja2: {e}")

    def generate_interface_file(self, interface_file_path, vertical_elements):
        """
        Generates an interface file for the mesh processor.

        Parameters:
        - interface_file_path: Path to save the interface file.
        - vertical_elements: List of integers specifying the number of vertical elements per layer.
        """
        Xi, Yi, Zi = self.interpolate_topography()
        dx, dy = Xi[0, 1] - Xi[0, 0], Yi[1, 0] - Yi[0, 0]

        # Save topography layers as text files
        layer_files = []
        for i, layer in enumerate(vertical_elements, start=1):
            layer_file = os.path.join(self.save_dir, f"layer_{i}.txt")
            np.savetxt(layer_file, Zi.flatten(), fmt="%.1f")
            layer_files.append(layer_file)

        # Write the interface file
        with open(interface_file_path, "w") as f:
            f.write(f"# number of interfaces\n {len(vertical_elements)}\n")
            f.write("#\n")
            f.write("# We describe each interface below, structured as a 2D-grid, with several parameters :\n")
            f.write("# number of points along XI and ETA, minimal XI ETA coordinates\n")
            f.write("# and spacing between points which must be constant.\n")
            f.write("# Then the records contain the Z coordinates of the NXI x NETA points.\n")
            f.write("#\n")

            for i, layer_file in enumerate(layer_files, start=1):
                f.write(f"# interface number {i}\n")
                f.write(f" .true. {Xi.shape[1]} {Yi.shape[0]} {Xi[0, 0]:.1f} {Yi[0, 0]:.1f} {dx:.1f} {dy:.1f}\n")
                f.write(f" {layer_file}\n")

            f.write("#\n")
            f.write("# for each layer, we give the number of spectral elements in the vertical direction\n")
            f.write("#\n")

            for i, elements in enumerate(vertical_elements, start=1):
                f.write(f"# layer number {i}\n {elements}\n")

        print(f"Interface file generated at {interface_file_path}")

    def generate_interface_file_with_doubling(self, interface_file_path, depth, doubling_layers):
        """
        Generates an interface file for the mesh processor, considering doubling layers.

        Parameters:
        - interface_file_path: Path to save the interface file.
        - depth: Total depth of the model in meters.
        - doubling_layers: List of depths (in meters) where doubling occurs.
        """
        from .mesh_processor import calculate_adjusted_vertical_elements

        Xi, Yi, Zi = self.interpolate_topography()
        dx, dy = Xi[0, 1] - Xi[0, 0], Yi[1, 0] - Yi[0, 0]

        # Calculate adjusted vertical elements
        vertical_elements = calculate_adjusted_vertical_elements(depth, doubling_layers)

        # Save topography layers as text files
        layer_files = []
        for i, layer in enumerate(vertical_elements, start=1):
            layer_file = os.path.join(self.save_dir, f"layer_{i}.txt")
            np.savetxt(layer_file, Zi.flatten(), fmt="%.1f")
            layer_files.append(layer_file)

        # Write the interface file
        with open(interface_file_path, "w") as f:
            f.write(f"# number of interfaces\n {len(vertical_elements)}\n")
            f.write("#\n")
            f.write("# We describe each interface below, structured as a 2D-grid, with several parameters :\n")
            f.write("# number of points along XI and ETA, minimal XI ETA coordinates\n")
            f.write("# and spacing between points which must be constant.\n")
            f.write("# Then the records contain the Z coordinates of the NXI x NETA points.\n")
            f.write("#\n")

            for i, layer_file in enumerate(layer_files, start=1):
                f.write(f"# interface number {i}\n")
                f.write(f" .true. {Xi.shape[1]} {Yi.shape[0]} {Xi[0, 0]:.1f} {Yi[0, 0]:.1f} {dx:.1f} {dy:.1f}\n")
                f.write(f" {layer_file}\n")

            f.write("#\n")
            f.write("# for each layer, we give the number of spectral elements in the vertical direction\n")
            f.write("#\n")

            for i, elements in enumerate(vertical_elements, start=1):
                f.write(f"# layer number {i}\n {elements}\n")

        print(f"Interface file with doubling layers generated at {interface_file_path}")

    def write_interfaces_jinja(self, output_path, interface_grid_info, n_layers, vertical_counts=None):
        """
        Write interfaces.txt using the Jinja2 template and the current mesh configuration.
        interface_grid_info: list of dicts, one per interface, with keys:
            - nxi, neta, x0, y0, dx, dy, filename
        vertical_counts: list of ints, number of vertical elements per layer (bottom to top)
        """
        from jinja2 import Environment, FileSystemLoader
        template_dir = os.path.join(os.path.dirname(__file__), '../templates')
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template('interfaces_template.j2')
        # Prepare the context for the template
        context = {
            'interfaces': interface_grid_info,
            'n_layers': n_layers,
            'vertical_counts': vertical_counts or [1]*n_layers
        }
        rendered = template.render(**context)
        with open(output_path, 'w') as f:
            f.write(rendered)
        print(f"interfaces.txt written using Jinja2 template at {output_path}")

    def write_all_outputs(self, interfaces=None, slope_thresholds=[10, 15, 20]):
        """
        Flexible, user-friendly entry point to write topography, layers, and interfaces.txt.
        interfaces: list of interface definitions. Can be:
            - 'topography' (str): always writes topography.txt as a layer
            - (base, factor) tuples for doubling layers
            - custom arrays (future extension)
        """
        Xi, Yi, Zi = self.interpolate_topography()
        dx, dy = Xi[0, 1] - Xi[0, 0], Yi[1, 0] - Yi[0, 0]
        x0, y0 = Xi[0, 0], Yi[0, 0]
        nxi, neta = Xi.shape[1], Xi.shape[0]
        # Always write topography
        np.savetxt(os.path.join(self.save_dir, "topography.txt"), Zi.flatten(), fmt="%.1f")
        # Slope and plots (reuse existing logic)
        Yslope = self.calculate_slope(Zi, dx, dy)

        # Topography visualization (define divnorm here)
        if np.min(Zi) < 0 and np.max(Zi) < 0:  # Fully underwater case
            divnorm = colors.TwoSlopeNorm(vmin=np.min(Zi), vcenter=np.mean(Zi), vmax=np.max(Zi))
            colors_water_only = plt.cm.terrain(np.linspace(0, 0.17, 256))
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', colors_water_only)
        elif np.min(Zi) < 0 and np.max(Zi) > 0 :  # Mixed case
            divnorm = colors.TwoSlopeNorm(vmin=np.min(Zi), vcenter=0, vmax=np.max(Zi))
            colors_combined = np.vstack((plt.cm.terrain(np.linspace(0, 0.17, 128)), plt.cm.terrain(np.linspace(0.25, 1, 128))))
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', colors_combined)
        else:  # Fully above water
            divnorm = colors.TwoSlopeNorm(vmin=np.min(Zi), vcenter=np.mean(Zi), vmax=np.max(Zi))
            colors_land_only = plt.cm.terrain(np.linspace(0.25, 1, 256))
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', colors_land_only)

        # Save a version of the image re-projected in lat/lon
        plt.figure()
        Xi_lon, Yi_lat = pyproj.Proj(proj="utm", zone=self.utm_zone[:-1], datum="WGS84")(Xi, Yi, inverse=True)
        plt.pcolor(Xi_lon, Yi_lat, Zi, shading="auto", cmap=terrain_map, norm=divnorm)
        plt.colorbar(label="Elevation (m)")
        plt.title("Topography (Lat/Lon)")
        plt.savefig(os.path.join(self.save_dir, "topography_visualization_latlon.png"))

        # Save slope data and visualizations
        plt.figure()
        norm = Normalize(vmin=0, vmax=np.max(Yslope))
        plt.pcolor(Xi, Yi, Yslope, shading="auto", cmap="viridis", norm=norm)
        plt.colorbar(label="Slope (degrees)")
        plt.title("Slope")

        # Generate colors dynamically based on the number of thresholds
        cmap = plt.cm.get_cmap("autumn", len(slope_thresholds))
        threshold_colors = [cmap(i) for i in range(len(slope_thresholds))]

        # Plot points with dynamically generated colors for each threshold
        for i, threshold in enumerate(slope_thresholds):
            mask = (Yslope > threshold) & (Yslope <= (slope_thresholds[i + 1] if i + 1 < len(slope_thresholds) else np.max(Yslope)))
            if np.any(mask):  # Only add to legend if there are points in this range
                plt.scatter(
                    Xi[mask],
                    Yi[mask],
                    color=threshold_colors[i],
                    s=5,
                    label=f"Slope > {threshold} degrees"
                )

        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.save_dir, "slope_visualization.png"))

        results = {}
        for threshold in slope_thresholds:
            mask = Yslope > threshold
            count = np.sum(mask)
            results[threshold] = {
                'count': count,
                'mask': mask
            }
            print(f'{count} points with slope > {threshold} degrees')

        # Save slope histogram
        plt.figure()
        plt.hist(Yslope.flatten(), bins=30, edgecolor="black", color="skyblue")
        plt.xlabel("Slope (degrees)")
        plt.ylabel("Frequency")
        plt.title("Slope Histogram")
        plt.savefig(os.path.join(self.save_dir, "slope_histogram.png"))

        # Layer writing logic
        layer_files = []
        region_names = []
        n_layers = 0
        if interfaces is None:
            interfaces = ['topography']
        for i, iface in enumerate(interfaces, 1):
            if iface == 'topography':
                fname = f"topography.txt"
                np.savetxt(os.path.join(self.save_dir, fname), Zi.flatten(), fmt="%.1f")
                layer_files.append(fname)
                region_names.append(fname)
            elif isinstance(iface, tuple) and len(iface) == 2:
                base, factor = iface
                mean_topo = np.mean(Zi)
                layer = base + (Zi - mean_topo) * factor
                fname = f"layer_{i}.txt"
                np.savetxt(os.path.join(self.save_dir, fname), layer.flatten(), fmt="%.1f")
                layer_files.append(fname)
                region_names.append(fname)
            # ...future: handle custom arrays...
        n_layers = len(layer_files)

        # Build a list of (depth, filename, grid_info) for sorting
        interface_depths = []
        for i, iface in enumerate(interfaces, 1):
            if iface == 'topography':
                depth = 0.0
                fname = "topography.txt"
                grid_info = {
                    'nxi': nxi,
                    'neta': neta,
                    'x0': x0,
                    'y0': y0,
                    'dx': dx,
                    'dy': dy,
                    'filename': fname
                }
            elif isinstance(iface, tuple) and len(iface) == 2:
                depth = float(iface[0])
                fname = f"layer_{i}.txt"
                grid_info = {
                    'nxi': nxi,
                    'neta': neta,
                    'x0': x0,
                    'y0': y0,
                    'dx': dx,
                    'dy': dy,
                    'filename': fname
                }
            else:
                continue  # skip unsupported
            interface_depths.append((depth, fname, grid_info))

        # Sort by depth (deepest first, topography last)
        interface_depths.sort(key=lambda x: x[0])

        # Get vertical element counts from mesh config (bottom to top)
        mesh = getattr(self, 'mesh_processor', None)
        if mesh is not None and hasattr(mesh, '_vertical_cache') and mesh._vertical_cache is not None:
            regions = mesh._vertical_cache['REGIONS']
            vertical_counts = [r[5] - r[4] + 1 for r in regions]
        else:
            vertical_counts = [1] * n_layers

        # Build interface_grid_info and filenames in sorted order
        interface_grid_info = [x[2] for x in interface_depths]
        # Do NOT reverse vertical_counts; use as is for bottom-to-top order
        # vertical_counts = list(reversed(vertical_counts)) if len(vertical_counts) == len(interface_grid_info) else vertical_counts

        self.write_interfaces_jinja(
            os.path.join(self.save_dir, "interfaces.txt"),
            interface_grid_info,
            n_layers,
            vertical_counts
        )
        print("All outputs written (topography, layers, interfaces.txt)")


# Example Usage
if __name__ == "__main__":
    # Assuming interpolator is already defined and initialized
    processor = TopographyProcessor(interpolator, myProj, save_dir="./topography_analysis")
    doubling_layers = [
        16.0,  # Layer 1 in km
        4.0,   # Layer 2 in km
        50.0,  # Layer 3 in km
    ]
    processor.save_results(slope_thresholds=[20], doubling_layers=doubling_layers)

