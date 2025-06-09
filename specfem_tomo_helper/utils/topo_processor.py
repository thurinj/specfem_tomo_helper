import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib import colors
from scipy.ndimage import gaussian_filter
from scipy.interpolate import LinearNDInterpolator
from netCDF4 import Dataset
from typing import Optional, List, Tuple, Any, Dict
import pyproj
import logging

logging.basicConfig(level=logging.INFO)

class TopographyProcessor:
    """
    Processes and interpolates topography data, generates slope maps, and writes interface files for SPECFEM3D.
    """
    def __init__(
        self,
        interpolator: Any,
        myProj: pyproj.Proj,
        save_dir: str = "./topography_analysis",
        smoothing_sigma: float = 1,
        mesh_processor: Optional[Any] = None,
        doubling_depth: Optional[list] = None
    ) -> None:
        """
        Initialize the TopographyProcessor.

        Parameters
        ----------
        interpolator : Any
            Interpolator object with x_interp_coordinates and y_interp_coordinates attributes.
        myProj : pyproj.Proj
            UTM projection object (from pyproj).
        save_dir : str, optional
            Directory to save outputs. Default is './topography_analysis'.
        smoothing_sigma : float, optional
            Sigma for Gaussian smoothing of topography. Default is 1.
        mesh_processor : Optional[Any], optional
            MeshProcessor instance, if available.
        doubling_depth : Optional[list], optional
            List of doubling layer depths (in km, negative down; will be converted to meters internally if used).
        """
        self.interpolator = interpolator
        self.utm_zone = getattr(myProj.crs, 'utm_zone', None)
        self.x_interp = interpolator.x_interp_coordinates
        self.y_interp = interpolator.y_interp_coordinates
        self.save_dir = save_dir
        self.smoothing_sigma = smoothing_sigma
        self.mesh_processor = mesh_processor
        # Convert doubling_layers from km to meters if provided
        self.doubling_layers = [dl * 1000.0 for dl in doubling_depth] if doubling_depth is not None else None
        os.makedirs(self.save_dir, exist_ok=True)
        self.filename: Optional[str] = None

    def _show_progress(self, block_num, block_size, total_size):
        """
        Show download progress bar in the terminal.
        """
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        bar_len = 40
        filled_len = int(bar_len * percent // 100)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        print(f'\r[Downloading] |{bar}| {percent:.1f}% ', end='')
        if downloaded >= total_size:
            print()  # Newline at end

    def download_etopo_netcdf(self) -> str:
        """
        Downloads the ETOPO1 NetCDF file if not already present in the package's download directory.
        Supports resuming interrupted downloads.

        Returns
        -------
        str
            Path to the downloaded or existing ETOPO1 NetCDF file.
        """
        import requests
        import gzip
        import shutil
        import time
        url = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/netcdf/ETOPO1_Bed_g_gmt4.grd.gz"
        data_dir = os.path.join(os.path.dirname(__file__), '../download')
        os.makedirs(data_dir, exist_ok=True)
        unzipped_filename = os.path.join(data_dir, "ETOPO1_Bed_g_gmt4.grd")
        gzipped_filename = unzipped_filename + ".gz"
        self.filename = unzipped_filename
        max_attempts = 5
        attempt = 0
        while not os.path.isfile(unzipped_filename):
            attempt += 1
            try:
                resume_byte_pos = 0
                headers = {}
                if os.path.exists(gzipped_filename):
                    resume_byte_pos = os.path.getsize(gzipped_filename)
                    if resume_byte_pos > 0:
                        headers['Range'] = f'bytes={resume_byte_pos}-'
                logging.info(f"ETOPO1 NetCDF file not found at {unzipped_filename}. Downloading from {url} (attempt {attempt})...")
                with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                    r.raise_for_status()
                    mode = 'ab' if resume_byte_pos > 0 else 'wb'
                    total_size = int(r.headers.get('Content-Range', '').split('/')[-1] or r.headers.get('Content-Length', 0))
                    if not total_size:
                        total_size = None
                    downloaded = resume_byte_pos
                    bar_len = 40
                    with open(gzipped_filename, mode) as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size:
                                    percent = min(100, downloaded * 100 / total_size)
                                    filled_len = int(bar_len * percent // 100)
                                    bar = '=' * filled_len + '-' * (bar_len - filled_len)
                                    print(f'\r[Downloading] |{bar}| {percent:.1f}% ', end='')
                    if total_size:
                        print()
                logging.info(f"Downloaded {gzipped_filename}. Decompressing...")
                with gzip.open(gzipped_filename, 'rb') as f_in:
                    with open(unzipped_filename, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(gzipped_filename)
                logging.info(f"ETOPO1 NetCDF file ready at {unzipped_filename}.")
            except Exception as e:
                logging.warning(f"Download failed (attempt {attempt}): {e}")
                if os.path.exists(gzipped_filename):
                    os.remove(gzipped_filename)
                if attempt >= max_attempts:
                    raise RuntimeError(f"Failed to download ETOPO1 after {max_attempts} attempts.") from e
                time.sleep(2)  # Wait before retrying
        return unzipped_filename

    def get_etopo_data_netcdf(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reads and crops the ETOPO1 data.

        Parameters
        ----------
        lon_min, lon_max, lat_min, lat_max : float
            Longitude and latitude bounds for cropping.

        Returns
        -------
        lon_cropped : np.ndarray
            Cropped longitude array.
        lat_cropped : np.ndarray
            Cropped latitude array.
        topo_cropped : np.ndarray
            Cropped topography array.
        """
        logging.info("Reading and cropping ETOPO1 NetCDF data...")
        dataset = Dataset(self.filename, 'r')
        lon = dataset.variables['x'][:]
        lat = dataset.variables['y'][:]
        topo = dataset.variables['z'][:]
        lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]
        lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_cropped = lon[lon_inds]
        lat_cropped = lat[lat_inds]
        topo_cropped = topo[np.ix_(lat_inds, lon_inds)]
        logging.info("Data cropping complete.")
        return lon_cropped, lat_cropped, topo_cropped

    def interpolate_topography(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolates topography for the grid specified by the interpolator.

        Returns
        -------
        Xi, Yi : np.ndarray
            Meshgrid arrays for X and Y coordinates.
        Zli : np.ndarray
            Interpolated and optionally smoothed topography array.
        """
        Xi, Yi = np.meshgrid(self.x_interp, self.y_interp)
        proj = pyproj.Proj(proj="utm", zone=self.utm_zone[:-1] if self.utm_zone else "32", datum="WGS84")
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

    def calculate_slope(self, Zi: np.ndarray, dx: float, dy: float) -> np.ndarray:
        """
        Calculates slope from the topography.

        Parameters
        ----------
        Zi : np.ndarray
            Topography array.
        dx, dy : float
            Grid spacing in X and Y directions.

        Returns
        -------
        np.ndarray
            Slope array in degrees.
        """
        dFdx, dFdy = np.gradient(Zi, dx, dy)
        Fgrad = np.sqrt(dFdx**2 + dFdy**2)
        return np.arctan(Fgrad) * 180 / np.pi

    def generate_interface_file(
        self,
        interface_file_path: str,
        vertical_elements: List[int]
    ) -> None:
        """
        Generates an interface file for the mesh processor.

        Parameters
        ----------
        interface_file_path : str
            Path to save the interface file.
        vertical_elements : list of int
            Number of vertical elements per layer.
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

        logging.info(f"Interface file generated at {interface_file_path}")


    def write_interfaces_jinja(
        self,
        output_path: str,
        interface_grid_info: List[Dict[str, Any]],
        n_layers: int,
        vertical_counts: Optional[List[int]] = None
    ) -> None:
        """
        Write interfaces.txt using the Jinja2 template and the current mesh configuration.

        Parameters
        ----------
        output_path : str
            Path to write interfaces.txt.
        interface_grid_info : list of dict
            One per interface, with keys: nxi, neta, x0, y0, dx, dy, filename.
        n_layers : int
            Number of layers.
        vertical_counts : list of int, optional
            Number of vertical elements per layer (bottom to top).
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
        logging.info(f"interfaces.txt written using Jinja2 template at {output_path}")

    def write_all_outputs(
        self,
        interfaces: Optional[List[Any]] = None,
        slope_thresholds: Optional[List[float]] = None
    ) -> None:
        """
        Flexible, user-friendly entry point to write topography, layers, and interfaces.txt.

        Parameters
        ----------
        interfaces : list, optional
            List of interface definitions. Can be 'topography' (str), (base, factor) tuples, or custom arrays.
        slope_thresholds : list of float, optional
            Slope thresholds for visualization.
        """
        
        # Put the interfaces as a list of tuples, followed by 'topography'
        doubling_depths = None
        if self.mesh_processor and hasattr(self.mesh_processor, 'doubling_layers'):
            doubling_depths = self.mesh_processor.doubling_layers
        elif self.doubling_layers is not None:
            doubling_depths = self.doubling_layers
        if interfaces is None:
            interfaces = ['topography']
            if doubling_depths is not None:
                for _i, depth in enumerate(doubling_depths, 1):
                    factor = -1 * depth/1e3 * (-1/self.mesh_processor.max_depth) + 1 if self.mesh_processor and hasattr(self.mesh_processor, 'max_depth') else 1.0
                    interfaces.append((depth, factor))

        Xi, Yi, Zi = self.interpolate_topography()
        dx, dy = Xi[0, 1] - Xi[0, 0], Yi[1, 0] - Yi[0, 0]
        x0, y0 = Xi[0, 0], Yi[0, 0]
        nxi, neta = Xi.shape[1], Xi.shape[0]
        # Always write topography
        np.savetxt(os.path.join(self.save_dir, "topography.txt"), Zi.flatten(), fmt="%.1f")
        # Slope and plots (reuse existing logic)
        Yslope = self.calculate_slope(Zi, dx, dy)

        # Topography visualization (define divnorm here)
        vmin = np.min(Zi)
        vmax = np.max(Zi)
        # Ensure vmin < vcenter < vmax for TwoSlopeNorm
        if vmin < 0 and vmax < 0:  # Fully underwater case
            vcenter = (vmin + vmax) / 2
            if not (vmin < vcenter < vmax):
                vcenter = vmin + (vmax - vmin) / 2  # fallback
            divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            colors_water_only = plt.cm.terrain(np.linspace(0, 0.17, 256))
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', colors_water_only)
        elif vmin < 0 and vmax > 0 :  # Mixed case
            vcenter = 0
            # Ensure vmin < vcenter < vmax
            if not (vmin < vcenter < vmax):
                vcenter = (vmin + vmax) / 2
            divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            colors_combined = np.vstack((plt.cm.terrain(np.linspace(0, 0.17, 128)), plt.cm.terrain(np.linspace(0.25, 1, 128))))
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', colors_combined)
        else:  # Fully above water
            vcenter = (vmin + vmax) / 2
            if not (vmin < vcenter < vmax):
                vcenter = vmin + (vmax - vmin) / 2
            divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            colors_land_only = plt.cm.terrain(np.linspace(0.25, 1, 256))
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map', colors_land_only)

        # Save a version of the image re-projected in lat/lon
        plt.figure()
        Xi_lon, Yi_lat = pyproj.Proj(proj="utm", zone=self.utm_zone[:-1], datum="WGS84")(Xi, Yi, inverse=True)
        plt.pcolor(Xi_lon, Yi_lat, Zi, shading="auto", cmap=terrain_map, norm=divnorm)
        plt.colorbar(label="Elevation (m)")
        plt.title("Topography (Lat/Lon)")
        plt.savefig(os.path.join(self.save_dir, "topography_visualization_latlon.png"))

        # Save a version of the image in UTM coordinates
        plt.figure()
        plt.pcolor(Xi, Yi, Zi, shading="auto", cmap=terrain_map, norm=divnorm)
        plt.colorbar(label="Elevation (m)")
        plt.title("Topography (UTM)")
        plt.savefig(os.path.join(self.save_dir, "topography_visualization_utm.png"))
        
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
            logging.info(f'{count} points with slope > {threshold} degrees')

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

        self.write_interfaces_jinja(
            os.path.join(self.save_dir, "interfaces.txt"),
            interface_grid_info,
            n_layers,
            vertical_counts
        )
        logging.info("All outputs written (topography, layers, interfaces.txt)")
    
    def auto_smooth_topography(self, Zi, Xi, Yi, min_slope, max_iter=50, dt=0.2):
        """
        Iteratively apply explicit diffusion smoothing to Zi until all slopes are below min_slope.
        Uses Neumann (zero-gradient) boundary conditions to avoid edge effects.
        Returns the smoothed Zi.
        """
        Zi_smoothed = Zi.copy()
        for _ in range(max_iter):
            # Pad with edge values to enforce Neumann (zero-gradient) BCs
            Zi_pad = np.pad(Zi_smoothed, 1, mode='edge')
            laplacian = (
                Zi_pad[2:, 1:-1] + Zi_pad[:-2, 1:-1] +
                Zi_pad[1:-1, 2:] + Zi_pad[1:-1, :-2] -
                4 * Zi_pad[1:-1, 1:-1]
            )
            Zi_smoothed += dt * laplacian
            Yslope = self.calculate_slope(Zi_smoothed, Xi[0, 1] - Xi[0, 0], Yi[1, 0] - Yi[0, 0])
            if np.all(Yslope <= min_slope):
                break
        return Zi_smoothed

