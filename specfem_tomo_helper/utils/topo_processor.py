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
    def __init__(self, interpolator, myProj, save_dir="./topography_analysis", smoothing_sigma=1):
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

    def save_results(self, slope_thresholds=[10,15,20], doubling_layers=None):
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
        if doubling_layers:
            mean_topo = np.mean(Zi)
            for i, (base, factor) in enumerate(doubling_layers, 1):
                layer = base + (Zi - mean_topo) * factor
                np.savetxt(os.path.join(self.save_dir, f"layer_{i}.txt"), layer.flatten(), fmt="%.1f")
                print(f"Saved doubling layer {i}")

        print("All results saved successfully!")



# Example Usage
if __name__ == "__main__":
    # Assuming interpolator is already defined and initialized
    processor = TopographyProcessor(interpolator, myProj, save_dir="./topography_analysis")
    doubling_layers = [
        (-16000, 0.33),  # Layer 1
        (-4000, 0.66),   # Layer 2
        (-50000, 0.0),   # Layer 3
    ]
    processor.save_results(slope_thresholds=[20], doubling_layers=doubling_layers)

