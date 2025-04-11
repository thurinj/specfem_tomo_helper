#!/usr/bin/env python

from specfem_tomo_helper.projection import vp2rho, vp2vs, vs2vp, define_utm_projection
from specfem_tomo_helper.fileimport import Nc_model
from specfem_tomo_helper.utils import maptool, trilinear_interpolator, write_tomo_file, TopographyProcessor
import matplotlib.pyplot as plt
import numpy as np

# INPUTS ARE HERE.
path = '../data/csem-europe-2019.12.01.nc'
dx = 10000 # in m
dy = 10000 # in m
z_min = -250  # in km
z_max = 0  # in km
dz = 10000  # in m


# NO INPUT NEEDED PAST THIS LINE
# load netCDF model
nc_model = Nc_model(path)
# extract coordinates
lon, lat, depth = nc_model.load_coordinates()
# load model parameters
# In this case the interpolated model is vsv.
vsv = nc_model.load_variable('vsv', fill_nan='vertical')

# define pyproj custom projection
myProj = define_utm_projection(34, 'N')
# Second mode is here, with graphical area selection for interpolation.
# The graphical selection tool takes an initial projection as argument. It can be modified using the GUI if it was not correctly specified.
gui_parameters = maptool(nc_model, myProj)
interpolator = trilinear_interpolator(nc_model, gui_parameters.projection)
interpolator.interpolation_parameters(gui_parameters.extent[0], gui_parameters.extent[1], dx,
                                      gui_parameters.extent[2], gui_parameters.extent[3], dy,
                                      z_min, z_max, dz)
tomo = interpolator.interpolate([vsv])

# As we want Vp, Vs and Rho to write the specfem files, we perform the following.
coordinates = tomo[:, 0:3]
vsv = tomo[:, 3]
vp = vs2vp(vsv)  # Convert vs to vp with empirical law
rho = vp2rho(vp)  # Converte vp to rho with empirical law
param = np.vstack((vp, vsv, rho)).T  # Stack the three params

# Recombine the coordinates and the params in to the tomo array, [lon, lat, depth, vp, vs, rho] required to write.
tomo = np.hstack((coordinates, param))
# Write the tomography_file.xyz in "./" directory. It uses the tomography array and the interpolator parameters to produce the HEADER.
write_tomo_file(tomo, interpolator, './')

processor = TopographyProcessor(interpolator, gui_parameters.projection, save_dir="./topography_analysis")
doubling_layers = [
    (-8000, 0.33),  # Layer 1
    (-4000, 0.66),   # Layer 2
    (-20000, 0.0),   # Layer 3
]
processor.save_results(slope_thresholds=[20], doubling_layers=doubling_layers)


# Display the interpolated model.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tomo[:, 0], tomo[:, 1], tomo[:, 2], c=tomo[:, 3])
plt.show()
