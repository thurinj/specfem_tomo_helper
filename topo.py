import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pyproj
import urllib.request
import os
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def download_etopo_netcdf():
    url = "https://www.ngdc.noaa.gov/thredds/fileServer/etopo/etopo1_bedrock.nc"
    filename = "ETOPO1_Ice_c_gmt4.grd"
    if not os.path.isfile(filename):
        print("Downloading ETOPO1 NetCDF data...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print("ETOPO1 NetCDF data already downloaded.")
    return filename

def get_etopo_data_netcdf(filename, lon_min, lon_max, lat_min, lat_max):
    print("Reading and cropping ETOPO1 NetCDF data...")
    dataset = Dataset(filename, 'r')
    lon = dataset.variables['x'][:]
    lat = dataset.variables['y'][:]
    topo = dataset.variables['z'][:]

    # Find indices for the desired cropping region
    lon_inds = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    lat_inds = np.where((lat >= lat_min) & (lat <= lat_max))[0]

    lon_cropped = lon[lon_inds]
    lat_cropped = lat[lat_inds]
    topo_cropped = topo[np.ix_(lat_inds, lon_inds)]

    print("Data cropping complete.")
    return lon_cropped, lat_cropped, topo_cropped

def utm2ll_pyproj(x, y, zone):
    proj = pyproj.Proj(proj='utm', zone=zone[:-1], datum='WGS84')
    lon, lat = proj(x, y, inverse=True)
    return lon, lat

def ll2utm_pyproj(lon, lat, zone):
    proj = pyproj.Proj(proj='utm', zone=zone[:-1], datum='WGS84')
    x, y = proj(lon, lat)
    return x, y

def utmgrid2topo(xvi, yvi, szone, smoothing_sigma=0):
    xmin, xmax = np.min(xvi), np.max(xvi)
    ymin, ymax = np.min(yvi), np.max(yvi)
    print(f'interpolated grid is UTM zone {szone} with bounds {xmin, xmax, ymin, ymax}')

    Xi, Yi = np.meshgrid(xvi, yvi)
    ny, nx = Xi.shape

    Xloni, Ylati = utm2ll_pyproj(Xi.ravel(), Yi.ravel(), szone)
    Xloni, Ylati = np.array(Xloni).reshape(ny, nx), np.array(Ylati).reshape(ny, nx)

    filename = download_etopo_netcdf()
    lon_min, lon_max = np.min(Xloni), np.max(Xloni)
    lat_min, lat_max = np.min(Ylati), np.max(Ylati)
    Xlon, Ylat, Zt = get_etopo_data_netcdf(filename, lon_min-(lon_min*0.01), lon_max+(lon_max*0.01), lat_min-(lat_min*0.01), lat_max+(lat_max*0.01))
    
    # Interpolating using LinearNDInterpolator
    print('Interpolating topography data...')
    from scipy.interpolate import LinearNDInterpolator
    Xlon, Ylat = np.meshgrid(Xlon, Ylat)
    interp = LinearNDInterpolator((Xlon.ravel(), Ylat.ravel()), Zt.ravel())
    Zli = interp(Xloni.ravel(), Ylati.ravel())
    Zli = Zli.reshape(ny, nx)

    # Apply Gaussian smoothing if smoothing_sigma is greater than 0
    if smoothing_sigma > 0:
        Zli = gaussian_filter(Zli, sigma=smoothing_sigma)

    # Reproject back to UTM
    Xutm, Yutm = ll2utm_pyproj(Xloni.ravel(), Ylati.ravel(), szone)
    Xutm, Yutm, Zutm = Xutm.reshape(ny, nx), Yutm.reshape(ny, nx), Zli

    # Debug plot: Interpolated grid
    plt.figure()
    plt.pcolor(Xutm, Yutm, Zutm, shading='auto', vmin=0)
    plt.colorbar()
    plt.scatter(168077.6080353111, 4158078.943310803, c='r')
    plt.title('Interpolated Topography Grid in UTM')
    plt.xlabel('UTM X')
    plt.ylabel('UTM Y')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()

    ztmin, ztmax = np.nanmin(Zutm), np.nanmax(Zutm)
    print(f'topo in this region: min = {ztmin:.0f} m, max = {ztmax:.0f} m')

    return Xutm, Yutm, Zutm

def calculate_slope(Zi, dx, dy):
    dFdx, dFdy = np.gradient(Zi, dx, dy)
    Fgrad = np.sqrt(dFdx**2 + dFdy**2)
    Yslope = np.arctan(Fgrad) * 180 / np.pi
    return Yslope

def check_topo(Xi, Yi, Zi, slope_thresholds=[20, 30, 40]):
    dx = np.abs(Xi[0, 1] - Xi[0, 0])
    dy = np.abs(Yi[1, 0] - Yi[0, 0])
    
    Yslope = calculate_slope(Zi, dx, dy)
    maxslope = np.max(Yslope)
    
    results = {}
    for threshold in slope_thresholds:
        mask = Yslope > threshold
        count = np.sum(mask)
        results[threshold] = {
            'count': count,
            'mask': mask
        }
        print(f'{count} points with slope > {threshold} degrees')
    
    # Plotting results
    plt.figure()
    plt.pcolor(Xi, Yi, Yslope, shading='auto', cmap='viridis')
    plt.colorbar(label='Slope (degrees)')
    for threshold, result in results.items():
        plt.contour(Xi, Yi, result['mask'], colors='red', linewidths=0.5)
    plt.axis('equal')
    plt.title(f'Slope of Topography (max = {maxslope:.1f} degrees)')
    plt.show()
    
    # Histogram of slopes
    plt.figure()
    plt.hist(Yslope.flatten(), bins=30, edgecolor='black')
    plt.xlabel('Slope (degrees)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Slope Angles')
    plt.show()
    
    return maxslope, Yslope

if __name__ == '__main__':
    latitude_min = 4005708.6
    latitude_max = 4355708.6
    longitude_min = -226960.5
    longitude_max = 473039.5
    dx = 500  # grid spacing, meters

    xvi = np.arange(longitude_min, longitude_max + dx, dx)
    yvi = np.arange(latitude_min, latitude_max + dx, dx)
    szone = '54N'
    smoothing_sigma = 1  # Set the smoothing parameter (sigma) here
    Xi, Yi, Zi = utmgrid2topo(xvi, yvi, szone, smoothing_sigma=1)
    
    maxslope, Yslope = check_topo(Xi, Yi, Zi)
    
    # Whe using doubling layers, we need to project the topography to the depth of the interface
    # usually reducing the amplitude of the topography progressively
    double_layers = True
    if double_layers:
        demean_Zi = Zi - np.mean(Zi)
        interface2 = -16000 + demean_Zi*0.33
        interface3 = -4000 + demean_Zi*0.66
        bottom = np.ones_like(interface2) * -50000.0

    # Write as a textfile with a single column of values
    # Format in m with one decimal 
    Zi_to_save = Zi.flatten()

    print('File length is: ', len(Zi_to_save))

    print("Shape is", np.shape(Zi))
    lx = np.shape(Zi)[1]
    ly = np.shape(Zi)[0]

    np.savetxt('topo_'+str(lx)+'_'+str(ly)+'.txt', Zi_to_save, fmt='%.1f')
    print('topo_'+str(lx)+'_'+str(ly)+'.txt', 'file saved')
    np.savetxt('interface2_1964.txt', interface2.flatten(), fmt='%.1f')
    print('interface2_1964.txt file saved')
    np.savetxt('interface3_1964.txt', interface3.flatten(), fmt='%.1f')
    print('interface3_1964.txt file saved')
    np.savetxt('interface1_1964.txt', bottom.flatten(), fmt='%.1f')
