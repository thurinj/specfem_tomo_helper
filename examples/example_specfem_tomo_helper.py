#!/usr/bin/env python

from specfem_tomo_helper.projection import vp2rho, vp2vs, define_utm_projection
from specfem_tomo_helper.fileimport import Nc_model
from specfem_tomo_helper.utils import maptool, bilinear_interpolator, write_tomo_file
import subprocess
import os

path = '../data/SCEC-CVM-H-v15-1-n4.nc'

if not os.path.isfile(path):
    subprocess.call(['wget', '-P', '../data/', 'http://ds.iris.edu/files/products/emc/emc-files/SCEC-CVM-H-v15-1-n4.nc'])

# Mandatory inputs:
dy = 1000  # in m
dx = 1000  # in m
# In the 2D interpolation scheme, each depth value of the model is used as depth value. Therefore there is no dz parameters.
z_min = -50  # in km
z_max = 4  # in km

# load netCDF model
nc_model = Nc_model(path)
# extract coordinates
lon, lat, depth = nc_model.load_coordinates()
# Load the 3 model parameters from the netCDF model
vp = nc_model.load_variable('vp', fill_nan=True)
vs = nc_model.load_variable('vs', fill_nan=True)
rho = nc_model.load_variable('rho', fill_nan=True)
# fill_nan is set to True here, as the shallow layers of the model contain nan values

# define pyproj custom projection. 11 North for South California
myProj = define_utm_projection(11, 'N')

# Here are two example modes possible (1 or 2)
# 1 for direct input, 2 for GUI map input
mode = 2
if mode == 1:
    # Direct input akin to specfem Mesh_Par_File
    latitude_min = 3537939.0
    latitude_max = 3937939.0
    longitude_min = 330704.1
    longitude_max = 830704.1
    # Initialize interpolator with model and UTM projection
    interpolator = bilinear_interpolator(nc_model, myProj)
    interpolator.interpolation_parameters(longitude_min, longitude_max, dx,
                                          latitude_min, latitude_max, dy,
                                          z_min, z_max)

if mode == 2:
    # Second mode with graphical area selection for interpolation.
    # The GUI window might freeze during the interpolation, instead of closing. Don't panic!
    gui_parameters = maptool(nc_model, myProj)
    interpolator = bilinear_interpolator(nc_model, myProj)
    interpolator.interpolation_parameters(
        gui_parameters.extent[0], gui_parameters.extent[1], dx,
        gui_parameters.extent[2], gui_parameters.extent[3], dy,
        z_min, z_max)

# Compute the tomography array
tomo = interpolator.interpolate([vp, vs, rho])
# Write the tomography_file.xyz in "./" directory. It uses the tomography array and the interpolator parameters to produce the HEADER.
write_tomo_file(tomo, interpolator, './')
